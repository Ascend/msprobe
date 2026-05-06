# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025-2026 Huawei Technologies Co.,Ltd.
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
from msprobe.core.common.file_utils import check_file_type, check_file_or_directory_path
from msprobe.core.common.const import FileCheckConst, Const
from msprobe.core.common.utils import CompareException
from msprobe.core.common.log import logger
from msprobe.core.compare.utils import get_paired_dirs
from msprobe.core.compare.utils import get_compare_framework
from msprobe.core.compare.utils import compare_distributed_inner


def compare_auto_mode(args, depth=1):
    """
    Auto-detect comparison mode based on input paths
    This is the original compare_cli logic
    """

    if depth > 2:
        logger.error("Recursive compare error, depth exceeds 2.")
        raise CompareException(CompareException.RECURSION_LIMIT_ERROR)

    common_kwargs = {
        "fuzzy_match": args.fuzzy_match,
        "data_mapping": args.data_mapping,
        "diff_analyze": args.diff_analyze,
        "is_print_compare_log": args.is_print_compare_log,
        "xlsx": args.xlsx,
        "cell_mapping": args.cell_mapping,
        "layer_mapping": args.layer_mapping,
        "consistent_check": args.consistent_check,
        "backend": args.backend
    }

    tp_file_type = check_file_type(args.target_path)
    gp_file_type = check_file_type(args.golden_path)

    # ===================== FILE vs FILE =====================
    if tp_file_type == FileCheckConst.FILE and gp_file_type == FileCheckConst.FILE:
        check_file_or_directory_path(args.target_path)
        check_file_or_directory_path(args.golden_path)

        frame_name = get_compare_framework(args.target_path, args.golden_path)

        input_param = {
            "npu_path": args.target_path,
            "bench_path": args.golden_path,
            "is_print_compare_log": args.is_print_compare_log
        }

        # 所有框架都基于 common_kwargs
        kwargs = dict(common_kwargs)

        if frame_name == Const.PT_FRAMEWORK:
            if args.api_mapping is not None:
                logger.error("Argument -am is not supported in PyTorch framework")
                raise CompareException(CompareException.INVALID_TASK_ERROR)
            from msprobe.pytorch.compare.pt_compare import pt_compare
            pt_compare(input_param, args.output_path, **kwargs)
        else:
            kwargs["api_mapping"] = args.api_mapping
            from msprobe.mindspore.compare.ms_compare import ms_compare
            ms_compare(input_param, args.output_path, **kwargs)

    # ===================== DIR vs DIR =====================
    elif tp_file_type == FileCheckConst.DIR and gp_file_type == FileCheckConst.DIR:
        check_file_or_directory_path(args.target_path, isdir=True)
        check_file_or_directory_path(args.golden_path, isdir=True)

        # depth=1 的混合比较
        if depth == 1:
            if mix_compare(args, depth):
                return

        # rank 静态图比较模式优先
        if args.rank is not None:
            from msprobe.mindspore.compare.distributed_compare import ms_graph_compare
            ms_graph_compare(args)
            return

        # diff_analyze 首差异节点模式
        if common_kwargs.get('diff_analyze', False):
            logger.info("Start finding first diff node......")
            from msprobe.core.compare.find_first.analyzer import DiffAnalyzer
            DiffAnalyzer(args.target_path, args.golden_path, args.output_path).analyze()
            return

        # 默认多卡比较
        kwargs = dict(common_kwargs)
        kwargs["api_mapping"] = args.api_mapping
        compare_distributed_inner(args.target_path, args.golden_path, args.output_path, **kwargs)

    else:
        logger.error("The target_path and golden_path need to be of the same type.")
        raise CompareException(CompareException.INVALID_COMPARE_MODE)


def mix_compare(args, depth):
    npu_path = args.target_path
    bench_path = args.golden_path

    npu_bench_same_dirs_set = set(get_paired_dirs(npu_path, bench_path))
    compare_cross_set = npu_bench_same_dirs_set & Const.MIX_DUMP_NAMES

    if compare_cross_set:
        logger.info("Start mix compare.")
        origin_output = args.output_path

        for folder_name in list(compare_cross_set):
            new_npu_path = os.path.join(npu_path, folder_name)
            new_bench_path = os.path.join(bench_path, folder_name)
            paired_steps = get_paired_dirs(new_npu_path, new_bench_path)

            for step_name in paired_steps:
                logger.info(f"[mix compare] Start comparing {folder_name}/{step_name}")
                npu_dir = os.path.join(new_npu_path, step_name)
                bench_dir = os.path.join(new_bench_path, step_name)
                args.target_path = npu_dir
                args.golden_path = bench_dir
                args.output_path = os.path.join(origin_output, folder_name, step_name)
                compare_auto_mode(args, depth + 1)
        return True
    return False
