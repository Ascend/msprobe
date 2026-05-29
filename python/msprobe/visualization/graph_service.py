# pylint: disable=too-many-lines
# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

import os
import signal
import time
import threading
from copy import deepcopy
from dataclasses import dataclass
from multiprocessing import cpu_count, Pool, Manager
from typing import Callable, Optional

from tqdm import tqdm
from msprobe.core.common.file_utils import (
    check_file_type,
    create_directory,
    FileChecker,
    check_file_or_directory_path,
    find_proc_dir,
)
from msprobe.core.common.const import FileCheckConst, Const
from msprobe.core.common.utils import CompareException, get_dump_mode
from msprobe.visualization.compare.graph_comparator import GraphComparator
from msprobe.visualization.utils import (
    GraphConst,
    check_directory_content,
    SerializableArgs,
    load_parallel_param,
    sort_rank_number_strings,
    validate_parallel_param,
    get_step_or_rank_int,
    monitor_progress,
    ProgressInfo,
    calculate_list,
    get_log_msg_wrapper,
    post_process_db_pbar,
)
from msprobe.visualization.builder.graph_builder import GraphBuilder, GraphExportConfig, GraphInfo, BuildGraphTaskInfo
from msprobe.core.common.log import logger, BaseLogger
from msprobe.visualization.graph.node_colors import NodeColors
from msprobe.core.compare.layer_mapping import generate_api_mapping_by_layer_mapping
from msprobe.core.compare.utils import check_and_return_dir_contents
from msprobe.core.common.utils import detect_framework_by_dump_json
from msprobe.visualization.graph.distributed_analyzer import distributed_analyse
from msprobe.visualization.builder.graph_merger import GraphMerger
from msprobe.visualization.db_utils import post_process_db

current_time = time.strftime("%Y%m%d%H%M%S")
build_output_db_name = f'build_{current_time}.vis.db'
compare_output_db_name = f'compare_{current_time}.vis.db'


def _pool_initializer():
    """Reset SIGTERM handler to default in worker processes.

    When mindspeed/torch_npu is installed, the NPU runtime may register a SIGTERM
    handler in worker processes (via fork inheritance). This handler can deadlock
    during pool.terminate() because cleanup operations on the forked NPU context
    hang indefinitely. Resetting to SIG_DFL ensures workers exit immediately.
    """
    signal.signal(signal.SIGTERM, signal.SIG_DFL)


def _compare_graph(graph_n: GraphInfo, graph_b: GraphInfo, input_param, args, pbar_info=None):
    dump_path_param = {
        'npu_path': graph_n.data_path,
        'bench_path': graph_b.data_path,
        'stack_path': graph_n.stack_path,
        'is_print_compare_log': input_param.get("is_print_compare_log", False),
    }
    mapping_dict = {}
    if args.layer_mapping:
        try:
            mapping_dict = generate_api_mapping_by_layer_mapping(
                graph_n.data_path, graph_b.data_path, args.layer_mapping
            )
        except Exception:
            logger.warning('The layer mapping file parsing failed, please check file format, mapping is not effective.')
    is_cross_framework = detect_framework_by_dump_json(graph_n.data_path) != detect_framework_by_dump_json(
        graph_b.data_path
    )
    if is_cross_framework and not args.layer_mapping:
        logger.error(
            'The cross_frame graph comparison failed. '
            'Please specify -lm or --layer_mapping when performing cross_frame graph comparison.'
        )
        raise CompareException(CompareException.CROSS_FRAME_ERROR)

    graph_comparator = GraphComparator(
        [graph_n.graph, graph_b.graph],
        dump_path_param,
        args,
        is_cross_framework,
        mapping_dict=mapping_dict,
        pbar_info=pbar_info,
    )
    graph_comparator.compare()
    return graph_comparator


def _compare_graph_result(input_param, args, pbar_info=None):
    logger.info('Start building model graphs...')
    # 对两个数据进行构图
    graph_n = _build_graph_info(input_param.get('npu_path'), args, pbar_info=pbar_info)
    graph_b = _build_graph_info(input_param.get('bench_path'), args, pbar_info=pbar_info)
    logger.info('Model graphs built successfully, start comparing graphs...')
    # 基于graph、stack和data进行比较
    graph_comparator = _compare_graph(graph_n, graph_b, input_param, args, pbar_info=pbar_info)
    # 增加micro step标记
    micro_steps = graph_n.graph.paging_by_micro_step(graph_b.graph)
    # 开启溢出检测
    if args.overflow_check:
        graph_n.graph.overflow_check()
        graph_b.graph.overflow_check()

    return CompareGraphResult(graph_n.graph, graph_b.graph, graph_comparator, micro_steps)


def _export_compare_graph_result(args, result, pbar_info=None):
    graphs = [result.graph_n, result.graph_b]
    graph_comparator = result.graph_comparator
    micro_steps = result.micro_steps
    logger.info(f'Start exporting compare graph result, file name: {compare_output_db_name}...')
    output_db_path = os.path.join(args.output_path, compare_output_db_name)
    task = GraphConst.GRAPHCOMPARE_MODE_TO_DUMP_MODE_TO_MAPPING.get(graph_comparator.ma.compare_mode)
    export_config = GraphExportConfig(
        graphs[0],
        graphs[1],
        graph_comparator.ma.get_tool_tip(),
        NodeColors.get_node_colors(graph_comparator.ma.compare_mode),
        micro_steps,
        task,
        args.overflow_check,
        graph_comparator.ma.compare_mode,
        result.step,
        result.rank,
        args.step_list if hasattr(args, 'step_list') else [0],
        args.rank_list if hasattr(args, 'rank_list') else [0],
    )
    try:
        GraphBuilder.to_db(output_db_path, export_config, pbar_info=pbar_info)
        logger.info(f'Exporting compare graph result successfully, the result file is saved in {output_db_path}')
        return ''
    except RuntimeError as e:
        logger.error(f'Failed to export compare graph result, file: {compare_output_db_name}, error: {e}')
        return compare_output_db_name


def _build_graph_info(dump_path, args, graph=None, pbar_info=None):
    construct_path = FileChecker(
        os.path.join(dump_path, GraphConst.CONSTRUCT_FILE), FileCheckConst.FILE, FileCheckConst.READ_ABLE
    ).common_check()
    data_path = FileChecker(
        os.path.join(dump_path, GraphConst.DUMP_FILE), FileCheckConst.FILE, FileCheckConst.READ_ABLE
    ).common_check()
    stack_path = FileChecker(
        os.path.join(dump_path, GraphConst.STACK_FILE), FileCheckConst.FILE, FileCheckConst.READ_ABLE
    ).common_check()
    if not graph:
        graph = GraphBuilder.build(construct_path, data_path, stack_path, pbar_info=pbar_info)
    return GraphInfo(graph, construct_path, data_path, stack_path)


def _build_graph_result(dump_path, args, pbar_info=None):
    logger.info('Start building model graphs...')
    graph = _build_graph_info(dump_path, args, pbar_info=pbar_info).graph
    # 增加micro step标记
    micro_steps = graph.paging_by_micro_step()
    # 开启溢出检测
    if args.overflow_check:
        graph.overflow_check()
    return BuildGraphResult(graph, micro_steps)


def _run_build_graph_compare(input_param, args, nr, br, pbar_info=None):
    logger.info(f'Start building graph for {nr}...')
    graph_n = _build_graph_info(input_param.get('npu_path'), args, pbar_info=pbar_info)
    graph_b = _build_graph_info(input_param.get('bench_path'), args, pbar_info=pbar_info)
    logger.info(f'Building graph for {nr} finished.')
    return BuildGraphTaskInfo(graph_n, graph_b, nr, br, current_time)


def _run_build_graph_single(dump_ranks_path, rank, step, args, pbar_info=None):
    logger.info(f'Start building graph for {rank}...')
    dump_path = os.path.join(dump_ranks_path, rank)
    result = _build_graph_result(dump_path, args, pbar_info=pbar_info)
    if rank != Const.RANK:
        result.rank = get_step_or_rank_int(rank, True)
    logger.info(f'Building graph for step: {step}, rank: {rank} finished.')
    return result


def _run_build_graph_and_export(dump_ranks_path, rank, step, args, pbar_info=None):
    result = _run_build_graph_single(dump_ranks_path, rank, step, args, pbar_info)
    if step is not None:
        result.step = get_step_or_rank_int(step)
    create_directory(args.output_path)

    return _export_build_graph_result(args, result, pbar_info)


def _run_graph_compare(graph_task_info, input_param, args, pbar_info=None):
    logger.info(f'Start comparing data for {graph_task_info.npu_rank}...')
    graph_n = graph_task_info.graph_info_n
    graph_b = graph_task_info.graph_info_b
    nr = graph_task_info.npu_rank
    graph_comparator = _compare_graph(graph_n, graph_b, input_param, args, pbar_info=pbar_info)
    micro_steps = graph_n.graph.paging_by_micro_step(graph_b.graph)
    # 开启溢出检测
    if args.overflow_check:
        graph_n.graph.overflow_check()
        graph_b.graph.overflow_check()
    graph_result = CompareGraphResult(graph_n.graph, graph_b.graph, graph_comparator, micro_steps)
    if nr != Const.RANK:
        graph_result.rank = get_step_or_rank_int(nr, True)
    logger.info(f'Comparing data for {graph_task_info.npu_rank} finished.')
    return graph_result


def _export_build_graph_result(args, result, pbar_info=None):
    out_path = args.output_path
    graph = result.graph
    micro_steps = result.micro_steps
    overflow_check = args.overflow_check

    if getattr(args, 'file_type', 'db') == 'json':
        logger.info('Start exporting graph to JSON files...')
        try:
            GraphBuilder.to_json(out_path, graph, result.rank, result.step, pbar_info=pbar_info)
            logger.info(f'Model graph exported successfully as JSON files to {out_path}')
            return None
        except (RuntimeError, OSError) as e:
            logger.error(f'Failed to export model graph as JSON, error: {e}')
            return 'json_export_error'

    logger.info(f'Start exporting graph for {build_output_db_name}...')
    output_db_path = os.path.join(out_path, build_output_db_name)
    config = GraphExportConfig(
        graph,
        micro_steps=micro_steps,
        overflow_check=overflow_check,
        rank=result.rank,
        step=result.step,
        rank_list=args.rank_list if hasattr(args, 'rank_list') else [0],
        step_list=args.step_list if hasattr(args, 'step_list') else [0],
    )
    try:
        GraphBuilder.to_db(output_db_path, config, pbar_info=pbar_info)
        logger.info(f'Model graph exported successfully, the result file is saved in {output_db_path}')
        return None
    except RuntimeError as e:
        logger.error(f'Failed to export model graph, file: {build_output_db_name}, error: {e}')
        return build_output_db_name


def is_real_data_compare(input_param, npu_ranks, bench_ranks):
    dump_rank_n = input_param.get('npu_path')
    dump_rank_b = input_param.get('bench_path')
    has_real_data = False
    for nr, br in zip(npu_ranks, bench_ranks):
        dump_path_param = {
            'npu_path': FileChecker(
                os.path.join(dump_rank_n, nr, GraphConst.DUMP_FILE), FileCheckConst.FILE, FileCheckConst.READ_ABLE
            ).common_check(),
            'bench_path': FileChecker(
                os.path.join(dump_rank_b, br, GraphConst.DUMP_FILE), FileCheckConst.FILE, FileCheckConst.READ_ABLE
            ).common_check(),
        }
        has_real_data |= get_dump_mode(dump_path_param) == Const.ALL
    return has_real_data


def _mp_compare(input_param, serializable_args, nr, br, pbar_info=None):
    graph_task_info = _run_build_graph_compare(input_param, serializable_args, nr, br, pbar_info=pbar_info)
    return _run_graph_compare(graph_task_info, input_param, serializable_args, pbar_info=pbar_info)


def _mp_compare_and_export(input_param, args, rank, step, pbar_info=None):
    graph_result = _mp_compare(input_param, args, rank, rank, pbar_info=pbar_info)
    if step is not None:
        graph_result.step = get_step_or_rank_int(step)
    create_directory(args.output_path)

    return _export_compare_graph_result(args, graph_result, pbar_info=pbar_info)


def _compare_graph_ranks(input_param, args, step=None, pbar_info=None):
    dump_rank_n = input_param.get('npu_path')
    dump_rank_b = input_param.get('bench_path')
    npu_ranks = sort_rank_number_strings(check_and_return_dir_contents(dump_rank_n, Const.RANK, skip_wrong_dir=True))
    bench_ranks = sort_rank_number_strings(check_and_return_dir_contents(dump_rank_b, Const.RANK, skip_wrong_dir=True))
    if not npu_ranks and not bench_ranks:
        npu_ranks = [os.path.basename(find_proc_dir(dump_rank_n))]
        bench_ranks = [os.path.basename(find_proc_dir(dump_rank_b))]
    elif npu_ranks != bench_ranks:
        intersection_ranks = sort_rank_number_strings(list(set(npu_ranks) & set(bench_ranks)))
        if not intersection_ranks:
            logger.error('The ranks in the two runs are completely different. Unable to match the ranks.')
            raise CompareException(CompareException.INVALID_PATH_ERROR)
        npu_ranks = intersection_ranks
        bench_ranks = intersection_ranks
    args.rank_list = [get_step_or_rank_int(rank, True) for rank in npu_ranks]
    serializable_args = SerializableArgs(args)

    with Pool(processes=max(int((cpu_count() + 1) // 4), 1), initializer=_pool_initializer) as pool:

        def err_call(err):
            logger.error(f'Error occurred while comparing graph ranks: {err}')

        if is_real_data_compare(input_param, npu_ranks, bench_ranks):
            # 真实数据模式，考虑到tensor比对过程会使用进程池启用多进程，为了避免嵌套进程池，graph比对使用串行
            compare_graph_results = []
            mp_task_dict = {}
            for nr, br in zip(npu_ranks, bench_ranks):
                input_param['npu_path'] = os.path.join(dump_rank_n, nr)
                input_param['bench_path'] = os.path.join(dump_rank_b, br)
                build_key = f'{step}_{nr}' if step else f'{nr}'
                input_param_copy = deepcopy(input_param)
                pbar_info_copy = PbarInfo.update_task_id(pbar_info, nr)
                mp_task_dict[build_key] = pool.apply_async(
                    _run_build_graph_compare,
                    args=(input_param_copy, serializable_args, nr, br, pbar_info_copy),
                    error_callback=err_call,
                )
            mp_res_dict = {k: v.get() for k, v in mp_task_dict.items()}
            for build_key, mp_res in mp_res_dict.items():
                if pbar_info:
                    if Const.REPLACEMENT_CHARACTER in build_key:
                        build_key = build_key.split(Const.REPLACEMENT_CHARACTER)[-1]
                    pbar_info.task_id = build_key
                compare_graph_results.append(_run_graph_compare(mp_res, input_param, serializable_args, pbar_info))
            if step is not None:
                for result in compare_graph_results:
                    result.step = get_step_or_rank_int(step)

            export_res_task_list = []
            create_directory(args.output_path)
            for result in compare_graph_results:
                export_res_task_list.append(
                    pool.apply_async(
                        _export_compare_graph_result,
                        args=(serializable_args, result),
                        error_callback=err_call,
                    )
                )
            export_res_list = [res.get() for res in export_res_task_list]
        else:
            compare_graph_tasks = []
            for nr, br in zip(npu_ranks, bench_ranks):
                input_param['npu_path'] = os.path.join(dump_rank_n, nr)
                input_param['bench_path'] = os.path.join(dump_rank_b, br)
                input_param_copy = deepcopy(input_param)
                pbar_info_copy = PbarInfo.update_task_id(pbar_info, nr)
                compare_graph_tasks.append(
                    pool.apply_async(
                        _mp_compare_and_export,
                        args=(input_param_copy, serializable_args, nr, step, pbar_info_copy),
                        error_callback=err_call,
                    )
                )
            export_res_list = [res.get() for res in compare_graph_tasks]
        if any(export_res_list):
            failed_names = list(filter(lambda x: x, export_res_list))
            logger.error(f'Unable to export compare graph results: {", ".join(failed_names)}.')
        else:
            logger.info('Successfully exported compare graph results.')


def _compare_graph_steps(input_param, args, pbar_info=None):
    dump_step_n = input_param.get('npu_path')
    dump_step_b = input_param.get('bench_path')

    npu_steps = calculate_list(dump_step_n, dump_step_b, Const.STEP)

    args.step_list = sorted([get_step_or_rank_int(step) for step in npu_steps])

    for i, folder_step in enumerate(npu_steps):
        logger.info(f'Start processing data for {folder_step}...')
        input_param['npu_path'] = os.path.join(dump_step_n, folder_step)
        input_param['bench_path'] = os.path.join(dump_step_b, folder_step)

        if pbar_info:
            pbar_info.step = i

        if not args.parallel_merge:
            _compare_graph_ranks(input_param, args, step=folder_step, pbar_info=pbar_info)
        else:
            _compare_graph_ranks_parallel(input_param, args, step=folder_step, pbar_info=pbar_info)


def _build_graph_ranks_parallel(args, step=None, pbar_info=None):
    dump_ranks_path = os.path.join(args.target_path, step) if step is not None else args.target_path
    if step is None:
        base_name = os.path.basename(os.path.normpath(args.target_path))
        if base_name.startswith(Const.STEP) and base_name[len(Const.STEP) :].isdigit():
            step = base_name
    ranks = sort_rank_number_strings(check_and_return_dir_contents(dump_ranks_path, Const.RANK, skip_wrong_dir=True))
    if not ranks:
        ranks = [os.path.basename(find_proc_dir(dump_ranks_path))]
    serializable_args = SerializableArgs(args)
    with Pool(processes=max(int((cpu_count() + 1) // 4), 1), initializer=_pool_initializer) as pool:

        def err_call(err):
            logger.error(f'Error occurred while comparing graph ranks: {err}')
            try:
                pool.close()
            except OSError as e:
                logger.error(f'Error occurred while terminating the pool: {e}')

        build_graph_tasks = []
        if pbar_info and pbar_info.step:
            PbarInfo.reset_progress_and_current_stage(pbar_info, ranks)
        for rank in ranks:
            pbar_info_copy = PbarInfo.update_task_id(pbar_info, rank)
            build_graph_tasks.append(
                pool.apply_async(
                    _run_build_graph_single,
                    args=(dump_ranks_path, rank, step, serializable_args, pbar_info_copy),
                    error_callback=err_call,
                )
            )
        build_graph_results = [task.get() for task in build_graph_tasks]

        if step is not None:
            for result in build_graph_results:
                result.step = get_step_or_rank_int(step)

        if args.parallel_params:
            validate_parallel_param(args.parallel_params[0], dump_ranks_path)
            build_graph_results = GraphMerger(
                build_graph_results, args.parallel_params[0], pbar_info=pbar_info
            ).merge_graph()
            if pbar_info:
                PbarInfo.del_progress_dict_item(
                    pbar_info, ranks, [f'{Const.RANK}{result.rank}' for result in build_graph_results]
                )

        create_directory(args.output_path)
        export_build_graph_tasks = []
        serializable_args.rank_list = [result.rank for result in build_graph_results]
        for result in build_graph_results:
            pbar_info_copy = PbarInfo.update_task_id(pbar_info, f'{Const.RANK}{result.rank}')
            export_build_graph_tasks.append(
                pool.apply_async(
                    _export_build_graph_result,
                    args=(serializable_args, result, pbar_info_copy),
                    error_callback=err_call,
                )
            )
        export_build_graph_result = [task.get() for task in export_build_graph_tasks]
        if any(export_build_graph_result):
            failed_names = list(filter(lambda x: x, export_build_graph_result))
            logger.error(f'Unable to export build graph results: {failed_names}.')
        else:
            logger.info('Successfully exported build graph results.')


def _build_graph_ranks(args, step=None, pbar_info=None):
    dump_ranks_path = os.path.join(args.target_path, step) if step is not None else args.target_path
    ranks = sort_rank_number_strings(check_and_return_dir_contents(dump_ranks_path, Const.RANK, skip_wrong_dir=True))
    if not ranks:
        ranks = [os.path.basename(find_proc_dir(dump_ranks_path))]
    args.rank_list = [get_step_or_rank_int(rank, True) for rank in ranks]
    serializable_args = SerializableArgs(args)
    with Pool(processes=max(int((cpu_count() + 1) // 4), 1), initializer=_pool_initializer) as pool:

        def err_call(err):
            logger.error(f'Error occurred while comparing graph ranks: {err}')

        tasks = []
        for rank in ranks:
            pbar_info_copy = PbarInfo.update_task_id(pbar_info, rank)
            tasks.append(
                pool.apply_async(
                    _run_build_graph_and_export,
                    args=(dump_ranks_path, rank, step, serializable_args, pbar_info_copy),
                    error_callback=err_call,
                )
            )
        results = [task.get() for task in tasks]
        if any(results):
            failed_names = list(filter(lambda x: x, results))
            logger.error(f'Unable to export build graph results: {failed_names}.')
        else:
            logger.info('Successfully exported build graph results.')


def _build_graph_steps(args, pbar_info=None):
    steps = sorted(check_and_return_dir_contents(args.target_path, Const.STEP))
    args.step_list = sorted([get_step_or_rank_int(step) for step in steps])

    for i, step in enumerate(steps):
        logger.info(f'Start processing data for {step}...')
        if pbar_info:
            pbar_info.step = i
        if not args.parallel_merge:
            _build_graph_ranks(args, step, pbar_info=pbar_info)
        else:
            _build_graph_ranks_parallel(args, step, pbar_info=pbar_info)


def _compare_and_export_graph(graph_task_info, input_param, args, step=None, pbar_info=None):
    result = _run_graph_compare(graph_task_info, input_param, args, pbar_info=pbar_info)
    if step is not None:
        result.step = get_step_or_rank_int(step)
    return _export_compare_graph_result(args, result, pbar_info=pbar_info)


def _compare_graph_ranks_parallel(input_param, args, step=None, pbar_info=None):
    args.fuzzy_match = True
    npu_path = input_param.get('npu_path')
    bench_path = input_param.get('bench_path')
    ranks_n = sort_rank_number_strings(check_and_return_dir_contents(npu_path, Const.RANK, skip_wrong_dir=True))
    ranks_b = sort_rank_number_strings(check_and_return_dir_contents(bench_path, Const.RANK, skip_wrong_dir=True))
    if not ranks_n and not ranks_b:
        ranks_n = [os.path.basename(find_proc_dir(npu_path))]
        ranks_b = [os.path.basename(find_proc_dir(bench_path))]
    parallel_params = args.parallel_params
    if len(parallel_params) != 2:
        raise RuntimeError('Parallel params error in compare graph!')
    validate_parallel_param(parallel_params[0], npu_path)
    validate_parallel_param(parallel_params[1], bench_path, '[Bench]')
    serializable_args = SerializableArgs(args)

    with Pool(processes=max(int((cpu_count() + 1) // 4), 1), initializer=_pool_initializer) as pool:

        def err_call(err):
            logger.error(f'Error occurred while comparing graph ranks: {err}')
            try:
                pool.close()
            except OSError as e:
                logger.error(f'Error occurred while terminating the pool: {e}')

        # 1.并行构图
        build_graph_tasks_n = []
        build_graph_tasks_b = []
        if pbar_info and pbar_info.step:
            PbarInfo.reset_progress_and_current_stage(pbar_info, list(set(ranks_n) | set(ranks_b)))
        for rank in ranks_n:
            pbar_info_copy = PbarInfo.update_task_id(pbar_info, rank)
            build_graph_tasks_n.append(
                pool.apply_async(
                    _run_build_graph_single,
                    args=(npu_path, rank, step, serializable_args, pbar_info_copy),
                    error_callback=err_call,
                )
            )
        for rank in ranks_b:
            pbar_info_copy = PbarInfo.update_task_id(pbar_info, rank)
            build_graph_tasks_b.append(
                pool.apply_async(
                    _run_build_graph_single,
                    args=(bench_path, rank, step, serializable_args, pbar_info_copy),
                    error_callback=err_call,
                )
            )
        graph_results_n = [task.get() for task in build_graph_tasks_n]
        graph_results_b = [task.get() for task in build_graph_tasks_b]

        # 2.图合并
        build_graph_results_n = GraphMerger(graph_results_n, parallel_params[0], pbar_info=pbar_info).merge_graph()
        build_graph_results_b = GraphMerger(
            graph_results_b, parallel_params[1], True, pbar_info=pbar_info
        ).merge_graph()

        if len(build_graph_results_n) != len(build_graph_results_b):
            raise RuntimeError(
                f'Parallel merge failed because the dp of npu: {len(build_graph_results_n)} '
                f'is inconsistent with that of bench: {len(build_graph_results_b)}!'
            )
        serializable_args.rank_list = [result.rank for result in build_graph_results_n]
        if pbar_info:
            PbarInfo.del_progress_dict_item(
                pbar_info,
                list(set(ranks_n) | set(ranks_b)),
                [f'{Const.RANK}{result.rank}' for result in build_graph_results_n],
            )
        # 3.并行图比对和输出
        export_res_task_list = []
        create_directory(args.output_path)
        for i, result_n in enumerate(build_graph_results_n):
            graph_n = result_n.graph
            graph_b = build_graph_results_b[i].graph
            graph_task_info = BuildGraphTaskInfo(
                _build_graph_info(os.path.join(npu_path, f'rank{graph_n.root.rank}'), args, graph_n),
                _build_graph_info(os.path.join(bench_path, f'rank{graph_b.root.rank}'), args, graph_b),
                f'rank{graph_n.root.rank}',
                f'rank{graph_b.root.rank}',
                current_time,
            )
            pbar_info_copy = PbarInfo.update_task_id(pbar_info, f'{Const.RANK}{result_n.rank}')
            export_res_task_list.append(
                pool.apply_async(
                    _compare_and_export_graph,
                    args=(graph_task_info, input_param, serializable_args, step, pbar_info_copy),
                    error_callback=err_call,
                )
            )
        export_res_list = [res.get() for res in export_res_task_list]
        if any(export_res_list):
            failed_names = list(filter(lambda x: x, export_res_list))
            logger.error(f'Unable to export compare graph results: {", ".join(failed_names)}.')
        else:
            logger.info('Successfully exported compare graph results.')


def _graph_service_parser(parser):
    # -------------------------- 基础必填参数 --------------------------
    parser.add_argument(
        "-tp", "--target_path", dest="target_path", type=str, help="<Required> The target path.", required=True
    )
    parser.add_argument(
        "-o",
        "--output_path",
        dest="output_path",
        type=str,
        help="<Required> The visualization task result out path.",
        required=True,
    )
    # -------------------------- 基础可选参数 --------------------------
    parser.add_argument(
        "-gp", "--golden_path", dest="golden_path", type=str, help="<Optional> The golden path.", required=False
    )
    parser.add_argument(
        "-lm",
        "--layer_mapping",
        dest="layer_mapping",
        type=str,
        nargs='?',
        const=True,
        help="<Optional> The layer mapping file path.",
        required=False,
    )
    parser.add_argument(
        "-oc",
        "--overflow_check",
        dest="overflow_check",
        action="store_true",
        help="<Optional> whether open overflow_check for graph.",
        required=False,
    )
    parser.add_argument(
        "-fm",
        "--fuzzy_match",
        dest="fuzzy_match",
        action="store_true",
        help="<Optional> whether to perform a fuzzy match on the api name.",
        required=False,
    )
    parser.add_argument(
        "-tensor_log",
        "--is_print_compare_log",
        dest="is_print_compare_log",
        action="store_true",
        help="<Optional> whether print tensor compare log for visualization task.",
        required=False,
    )
    parser.add_argument(
        "-progress_log",
        "--is_print_progress_log",
        dest="is_print_progress_log",
        action="store_true",
        help="<Optional> whether print progress log for visualization task.",
        required=False,
    )

    # -------------------------- 不同并行切分策略合并可选参数 --------------------------
    group_n = parser.add_argument_group(
        "Parallel Parameters, used for graph merging under different parallel partitioning strategies"
    )

    group_n.add_argument(
        "--rank_size", type=int, nargs='+', help="<Optional> The rank size of dump path.", required=False
    )
    group_n.add_argument(
        "--tp",
        type=int,
        nargs='+',
        help="<Optional, but required if rank_size is not empty> The tp size of dump path.",
        required=False,
    )
    group_n.add_argument(
        "--pp",
        type=int,
        nargs='+',
        help="<Optional, but required if rank_size is not empty> The pp size of dump path.",
        required=False,
    )
    group_n.add_argument(
        "--vpp", type=int, nargs='+', default=[1], help="<Optional> The vpp size of dump path.", required=False
    )
    group_n.add_argument(
        "--order",
        type=str,
        nargs='+',
        default=['tp-cp-ep-dp-pp'],
        help="<Optional> The order of dump path.",
        required=False,
    )

    # -------------------------- 输出文件类型可选参数 --------------------------
    parser.add_argument(
        "--file_type",
        dest="file_type",
        type=str,
        default="db",
        choices=["db", "json"],
        help="<Optional> Output file type, default is db, can be json. Only supported in graph merge build scenarios.",
        required=False,
    )


def _graph_service_command(args):
    try:
        if args.is_print_progress_log:
            # 往ProgressInfo中记录error日志，用于前端展示
            BaseLogger.error = get_log_msg_wrapper(BaseLogger.error)
        npu_path = args.target_path
        bench_path = args.golden_path
        ProgressInfo.print_progress_log = args.is_print_progress_log
        args.parallel_merge = bool(args.rank_size)
        args.parallel_params = load_parallel_param(args) if args.parallel_merge else None
        if args.file_type == 'json' and bench_path:
            logger.error(
                'The --file_type json parameter is not supported in graph comparison mode. '
                'Please remove the -gp/--golden_path parameter or use --file_type db.'
            )
            raise CompareException(CompareException.INVALID_COMPARE_MODE)
        if args.file_type == 'json' and not args.parallel_merge:
            logger.warning(
                'The --file_type json parameter is only supported in graph merge build scenarios. '
                'Falling back to default db output.'
            )
            args.file_type = 'db'
        check_file_or_directory_path(npu_path, isdir=True)
        if bench_path:
            check_file_or_directory_path(bench_path, isdir=True)
        if check_file_type(npu_path) == FileCheckConst.DIR and not bench_path:
            content = check_directory_content(npu_path)
            if content == GraphConst.RANKS:
                _build_graph_ranks_with_pbar(args)
            elif content == GraphConst.STEPS:
                _build_graph_steps_with_pbar(args)
            else:
                if args.parallel_merge and args.file_type == 'json':
                    logger.error(
                        'You have specified the "--rank_size" parameter for data merging, '
                        'so the "-tp" parameter must pass in multiple rank data.'
                    )
                    raise CompareException(CompareException.INVALID_PARAM_ERROR)
                _build_graph_with_pbar(npu_path, args)
        elif check_file_type(npu_path) == FileCheckConst.DIR and check_file_type(bench_path) == FileCheckConst.DIR:
            content_n = check_directory_content(npu_path)
            content_b = check_directory_content(bench_path)
            if content_n != content_b:
                raise ValueError('The directory structures of npu_path and bench_path are inconsistent.')
            input_param = {
                'npu_path': args.target_path,
                'bench_path': args.golden_path,
                'is_print_compare_log': args.is_print_compare_log,
            }
            if content_n == GraphConst.RANKS:
                _compare_graph_ranks_with_pbar(input_param, args)
            elif content_n == GraphConst.STEPS:
                _compare_graph_steps_with_pbar(input_param, args)
            else:
                _compare_graph_with_pbar(input_param, args)
        else:
            logger.error("The npu_path or bench_path should be a folder.")
            raise CompareException(CompareException.INVALID_COMPARE_MODE)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user, terminating processes and cleaning up...")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise e
    finally:
        ProgressInfo.update_process_running(False)


@dataclass
class ProgressConfig:
    core_func: Callable
    get_ranks: Callable
    db_name: str
    pbar_info_kwargs: dict = None
    use_monitor_thread: bool = True
    tqdm_total: Optional[int] = None


def _run_with_progress(param, args, config: ProgressConfig):
    """通用进度条处理"""

    monitor_thread = None
    pbar_info = None
    ranks = None

    try:
        if config.use_monitor_thread:
            manager = Manager()
            progress_dict = manager.dict()
            pbar_info = PbarInfo(progress_dict=progress_dict, **config.pbar_info_kwargs)
            ranks = config.get_ranks(args)
        else:
            pbar_info = PbarInfo(**config.pbar_info_kwargs)

        tqdm_args = {
            "desc": GraphConst.PBAR_DESC_PREFIX,
            "total": config.tqdm_total if config.tqdm_total is not None else pbar_info.total,
            "bar_format": GraphConst.BAR_FORMAT,
        }

        with tqdm(**tqdm_args) as pbar:
            # 单进程场景直接更新pbar，多进程场景需要通过monitor thread从共享dict中获取进度更新pbar
            if config.use_monitor_thread:
                monitor_thread = threading.Thread(
                    target=monitor_progress, args=(pbar_info, pbar, ranks, args.parallel_merge)
                )
                monitor_thread.start()
            else:
                pbar_info.pbar = pbar

            if param:
                config.core_func(param, args, pbar_info=pbar_info)
            else:
                config.core_func(args, pbar_info=pbar_info)

            if getattr(args, 'file_type', 'db') != 'json':
                db_path = os.path.join(args.output_path, config.db_name)
                post_process_db(db_path, pbar_info=pbar_info, is_parallel_merge=args.parallel_merge)

                if not args.parallel_merge and config.use_monitor_thread:
                    distributed_analyse(db_path, args.overflow_check, pbar_info=pbar_info)
            elif config.use_monitor_thread and pbar_info:
                # For JSON output, signal completion to the monitor thread directly
                post_process_db_pbar(pbar_info, is_parallel_merge=args.parallel_merge)

            if config.use_monitor_thread and monitor_thread:
                monitor_thread.join(timeout=5)

    except KeyboardInterrupt:
        logger.warning("Interrupted by user, terminating processes and cleaning up...")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise e
    finally:
        ProgressInfo.update_process_running(False)
        if config.use_monitor_thread and pbar_info:
            pbar_info.stop_monitor = True


def _build_graph_ranks_with_pbar(args):
    def core_func(args, pbar_info):
        if args.parallel_merge:
            _build_graph_ranks_parallel(args, pbar_info=pbar_info)
        else:
            _build_graph_ranks(args, pbar_info=pbar_info)

    def get_ranks(args):
        ranks = check_and_return_dir_contents(args.target_path, Const.RANK, skip_wrong_dir=True)
        if not ranks:
            ranks = [os.path.basename(find_proc_dir(args.target_path))]
        return ranks

    stage_total = _get_parallel_stage_total(args) if args.parallel_merge else GraphConst.BUILD_STAGES_TOTAL

    _run_with_progress(
        param=None,
        args=args,
        config=ProgressConfig(
            core_func=core_func,
            get_ranks=get_ranks,
            pbar_info_kwargs={"stage_total": stage_total},
            db_name=build_output_db_name,
        ),
    )


def _build_graph_steps_with_pbar(args):
    steps = check_and_return_dir_contents(args.target_path, Const.STEP)

    def get_ranks(args):
        ranks = check_and_return_dir_contents(os.path.join(args.target_path, steps[0]), Const.RANK, skip_wrong_dir=True)
        if not ranks:
            ranks = [os.path.basename(find_proc_dir(os.path.join(args.target_path, steps[0])))]
        return ranks

    stage_total = _get_parallel_stage_total(args, steps) if args.parallel_merge else GraphConst.BUILD_STAGES_TOTAL

    _run_with_progress(
        param=None,
        args=args,
        config=ProgressConfig(
            core_func=_build_graph_steps,
            get_ranks=get_ranks,
            pbar_info_kwargs={"step_total": len(steps), "stage_total": stage_total},
            db_name=build_output_db_name,
        ),
    )


def _build_graph_with_pbar(npu_path, args):
    def core_func(param, args, pbar_info):
        result = _build_graph_result(param, args, pbar_info)
        create_directory(args.output_path)
        file_name = _export_build_graph_result(args, result, pbar_info)
        if file_name:
            logger.error('Failed to export model build graph.')

    _run_with_progress(
        param=npu_path,
        args=args,
        config=ProgressConfig(
            core_func=core_func,
            get_ranks=lambda x: None,
            pbar_info_kwargs={},
            db_name=build_output_db_name,
            use_monitor_thread=False,
            tqdm_total=GraphConst.PBAR_TOTAL,
        ),
    )


def _compare_graph_ranks_with_pbar(input_param, args):
    def core_func(param, args, pbar_info):
        if args.parallel_merge:
            _compare_graph_ranks_parallel(param, args, pbar_info=pbar_info)
        else:
            _compare_graph_ranks(param, args, pbar_info=pbar_info)

    def get_ranks(args):
        if args.parallel_merge:
            return calculate_list(args.target_path, args.golden_path, mode=GraphConst.UNION)
        return calculate_list(args.target_path, args.golden_path)

    stage_total = (
        _get_parallel_stage_total(args, is_compare=True) if args.parallel_merge else GraphConst.COMPARE_STAGES_TOTAL
    )

    _run_with_progress(
        param=input_param,
        args=args,
        config=ProgressConfig(
            core_func=core_func,
            get_ranks=get_ranks,
            pbar_info_kwargs={"stage_total": stage_total},
            db_name=compare_output_db_name,
        ),
    )


def _compare_graph_steps_with_pbar(input_param, args):
    steps = calculate_list(args.target_path, args.golden_path, Const.STEP)

    def get_ranks(args):
        rank_path_t = os.path.join(args.target_path, steps[0])
        rank_path_g = os.path.join(args.golden_path, steps[0])
        if args.parallel_merge:
            return calculate_list(rank_path_t, rank_path_g, mode=GraphConst.UNION)
        return calculate_list(rank_path_t, rank_path_g)

    stage_total = (
        _get_parallel_stage_total(args, steps, is_compare=True)
        if args.parallel_merge
        else GraphConst.COMPARE_STAGES_TOTAL
    )

    _run_with_progress(
        param=input_param,
        args=args,
        config=ProgressConfig(
            core_func=_compare_graph_steps,
            get_ranks=get_ranks,
            pbar_info_kwargs={"stage_total": stage_total, "step_total": len(steps)},
            db_name=compare_output_db_name,
        ),
    )


def _compare_graph_with_pbar(input_param, args):
    def core_func(param, args, pbar_info):
        result = _compare_graph_result(param, args, pbar_info=pbar_info)
        create_directory(args.output_path)
        file_name = _export_compare_graph_result(args, result, pbar_info=pbar_info)
        if file_name:
            logger.error('Failed to export model compare graph.')

    _run_with_progress(
        param=input_param,
        args=args,
        config=ProgressConfig(
            core_func=core_func,
            get_ranks=lambda x: None,
            pbar_info_kwargs={"pbar": None, "stage_total": GraphConst.COMPARE_STAGES_TOTAL},
            db_name=compare_output_db_name,
            use_monitor_thread=False,
            tqdm_total=GraphConst.PBAR_TOTAL,
        ),
    )


def _get_parallel_stage_total(args, steps=None, is_compare=False):
    """
    获取不同并行切分策略的任务阶段数
    """
    parallel_params = args.parallel_params
    if not is_compare and (not parallel_params or len(parallel_params) != 1):
        raise RuntimeError('Parallel params error in build graph!')
    if is_compare and (not parallel_params or len(parallel_params) != 2):
        raise RuntimeError('Parallel params error in compare graph!')
    target_path = os.path.join(args.target_path, steps[0]) if steps else args.target_path
    validate_parallel_param(parallel_params[0], target_path)

    if is_compare:
        golden_path = os.path.join(args.golden_path, steps[0]) if steps else args.golden_path
        validate_parallel_param(parallel_params[1], golden_path, '[Bench]')

    stage_count_map = {
        "TPMerger": lambda param: param.rank_size // param.tp,
        "PPMerger": lambda param: param.rank_size // param.pp,
        "VPPMerger": lambda param: param.rank_size // param.pp,
        "TPPPMerger": lambda param: param.rank_size // param.pp + param.rank_size // param.pp // param.tp,
        "FullMerger": lambda param: param.rank_size // param.pp + param.rank_size // param.pp // param.tp,
        "NoParallelMerger": 0,
    }

    def _get_stage_count(parallel_param, merger_name: str) -> int:
        rule = stage_count_map.get(merger_name, 0)
        return rule(parallel_param) if callable(rule) else rule

    merger_name_t = GraphMerger([], parallel_params[0]).strategy.__class__.__name__
    stage_count_target = _get_stage_count(parallel_params[0], merger_name_t)

    if is_compare:
        merger_name_g = GraphMerger([], parallel_params[1]).strategy.__class__.__name__
        stage_count_golden = _get_stage_count(parallel_params[1], merger_name_g)
        return GraphConst.COMPARE_STAGES_TOTAL + stage_count_target + stage_count_golden

    return GraphConst.BUILD_STAGES_TOTAL + stage_count_target


class CompareGraphResult:
    def __init__(self, graph_n, graph_b, graph_comparator, micro_steps, rank=0, step=0):
        self.graph_n = graph_n
        self.graph_b = graph_b
        self.graph_comparator = graph_comparator
        self.micro_steps = micro_steps
        self.rank = rank
        self.step = step


class BuildGraphResult:
    def __init__(self, graph, micro_steps=0, rank=0, step=0):
        self.graph = graph
        self.micro_steps = micro_steps
        self.rank = rank
        self.step = step


class PbarInfo:
    def __init__(
        self,
        pbar=None,
        progress_dict=None,
        task_id=None,
        step=0,
        step_total=1,
        stage_total=GraphConst.BUILD_STAGES_TOTAL,
    ):
        self.pbar = pbar
        self.progress_dict = progress_dict
        self.task_id = task_id
        self.step = step
        self.step_total = step_total
        self.total = GraphConst.PBAR_TOTAL * step_total
        self.stage_total = stage_total * step_total  # 有几个阶段
        self.current_stage_dict = Manager().dict()  # 当前阶段，进程共享
        self.stage_progress = round(self.total / self.stage_total, 2)  # 每个阶段的最大进度
        self.stop_monitor = False
        self.wait_monitor = False
        self.continue_monitor = True

    def __deepcopy__(self, memo):
        new_obj = PbarInfo()
        new_obj.progress_dict = self.progress_dict
        new_obj.task_id = self.task_id
        new_obj.step = self.step
        new_obj.step_total = self.step_total
        new_obj.stage_total = self.stage_total
        new_obj.current_stage_dict = self.current_stage_dict
        new_obj.stage_progress = self.stage_progress
        new_obj.total = self.total
        new_obj.stop_monitor = self.stop_monitor
        new_obj.wait_monitor = self.wait_monitor
        new_obj.continue_monitor = self.continue_monitor
        return new_obj

    @staticmethod
    def update_task_id(pbar_info, task_id):
        """
        在进程池中，实例作为入参，修改实例属性，需要深拷贝实例使修改生效
        """
        if pbar_info:
            pbar_info.task_id = task_id
            return deepcopy(pbar_info)
        return pbar_info

    @staticmethod
    def del_progress_dict_item(pbar_info, origin_ranks, merged_ranks):
        """
        不同并行切分策略的图合并场景下，graph合并到一些rank中，剩余的rank作为task_id不再需要
        """
        diff_ranks = list(set(origin_ranks) - set(merged_ranks))
        for rank in diff_ranks:
            if rank in pbar_info.progress_dict:
                del pbar_info.progress_dict[rank]

    @staticmethod
    def reset_progress_and_current_stage(pbar_info, task_ids):
        """
        不同并行切分策略的图合并场景下，每个step需要重置进度信息
        """
        for task_id in task_ids:
            pbar_info.progress_dict[task_id] = GraphConst.PBAR_TOTAL * pbar_info.step
            pbar_info.current_stage_dict[task_id] = pbar_info.stage_total // pbar_info.step_total * pbar_info.step

    def set_continue_monitor(self, value: bool):
        self.continue_monitor = value
        self.wait_monitor = not value

    def set_wait_monitor(self, value: bool):
        self.wait_monitor = value
        self.continue_monitor = not value
