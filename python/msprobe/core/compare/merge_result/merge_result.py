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
import re
import multiprocessing
from functools import partial

import pandas as pd
from tqdm import tqdm

from msprobe.core.common.file_utils import (
    load_yaml,
    logger,
    FileChecker,
    save_excel,
    read_xlsx,
    create_directory,
    remove_path,
)
from msprobe.core.common.const import FileCheckConst, Const, CompareConst
from msprobe.core.common.utils import CompareException, add_time_with_xlsx
from msprobe.core.compare.utils import table_value_is_valid
from msprobe.core.compare.merge_result.utils import replace_compare_index_dict, check_config


def check_compare_result_name(file_name):
    """
    check whether the compare result name is as expected
    """
    single_rank_pattern = r"^compare_result_(rank|rank-rank)_\d{14}\.xlsx$"
    multi_ranks_pattern = r"^compare_result_rank(\d+)(?:-rank\1)?_\d{14}\.xlsx$"
    if re.match(multi_ranks_pattern, file_name):
        return True
    if re.match(single_rank_pattern, file_name):
        logger.warning("Single rank compare result do not need to be merged.")
        return False
    logger.error(f"Wrong compare result name: {file_name}, please check!")
    raise CompareException(CompareException.MERGE_COMPARE_RESULT_ERROR)


def reorder_path(compare_result_path_list):
    """
    reorder compare results by rank num
    """
    rank_pattern = r"compare_result_rank(\d+)"
    reorder_path_list = sorted(
        compare_result_path_list, key=lambda path: int(re.search(rank_pattern, os.path.basename(path)).group(1))
    )
    return reorder_path_list


def get_result_path(input_dir):
    """
    get rank ordered compare result file path list
    """
    compare_result_path_list = [
        os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(FileCheckConst.XLSX_SUFFIX)
    ]
    filt_compare_result_path_list = []
    for file_path in compare_result_path_list:
        FileChecker(file_path, FileCheckConst.FILE, FileCheckConst.READ_ABLE).common_check()
        file_name = os.path.basename(file_path)
        if check_compare_result_name(file_name):
            compare_result_path_checker = FileChecker(file_path, FileCheckConst.FILE, FileCheckConst.READ_ABLE)
            compare_result_path = compare_result_path_checker.common_check()
            filt_compare_result_path_list.append(compare_result_path)

    filt_compare_result_path_list = reorder_path(filt_compare_result_path_list)  # 多卡比对结果按rank序号重新排序

    if len(filt_compare_result_path_list) < 2:
        logger.warning("Number of compare result is no more than 1, no need to merge.")  # 单卡结果无需合并，直接退出
        raise CompareException(CompareException.MERGE_COMPARE_RESULT_ERROR)
    return filt_compare_result_path_list


def get_dump_mode(result_df, rank_num):
    """
    get dump mode from header of first compare result table
    """
    header = result_df.columns.tolist()
    if header in [
        CompareConst.COMPARE_RESULT_HEADER + [CompareConst.DATA_NAME],
        CompareConst.COMPARE_RESULT_HEADER_STACK + [CompareConst.DATA_NAME],
    ]:
        return Const.ALL
    elif header in [CompareConst.SUMMARY_COMPARE_RESULT_HEADER, CompareConst.SUMMARY_COMPARE_RESULT_HEADER_STACK]:
        return Const.SUMMARY
    elif header in [CompareConst.MD5_COMPARE_RESULT_HEADER, CompareConst.MD5_COMPARE_RESULT_HEADER_STACK]:
        return Const.MD5
    else:
        logger.warning(
            f"A valid dump task can not be identified from rank{rank_num} compare result, please check! "
            f"The compare result will not be shown in merged result."
        )
        return ""


def check_index_dump_mode_consistent(dump_mode, rank_num):
    """
    check compare index to merge is consistent with dump mode
    if compare_index_list is None, return all compare_indexes of dump mode
    """
    if dump_mode == Const.MD5:
        logger.warning(
            f"Rank{rank_num} compare result is 'md5' dump task and does not support merging result, please "
            f"check! The compare result will not be shown in merged result."
        )
        return []

    dump_mode_compare_index_map = {
        Const.ALL: CompareConst.ALL_COMPARE_INDEX + [CompareConst.REQ_GRAD_CONSIST],
        Const.SUMMARY: CompareConst.SUMMARY_COMPARE_INDEX + [CompareConst.REQ_GRAD_CONSIST],
    }
    valid_compare_index = dump_mode_compare_index_map.get(dump_mode)

    share_list = list(share_compare_index_list)

    # 如果传入的compare_index_list为空，则比对指标为dump_mode对应的全部比对指标
    if not share_list:
        share_compare_index_list.extend(valid_compare_index)
        return list(share_compare_index_list)
    if set(share_list).issubset(valid_compare_index):
        return share_list
    else:
        invalid_compare_index = set(valid_compare_index) - set(share_list)
        logger.warning(
            f"Compare indexes in rank{rank_num} compare result are not consistent with "
            f"those in other compare results, please check!"
        )
        logger.warning("The compare result will not be shown in merged result.")
        logger.warning(f"The invalid compare indexes: {invalid_compare_index}")
        return []


def extract_api_full_name(api_list, result_df, rank_num):
    """
    find api full name from compare result according to api list
    """
    api_full_name_list = []
    for api in api_list:
        api_pat = api + Const.SEP
        escaped_api_pat = api_pat.replace('.', r'\.')
        single_api_full_name_list = result_df.loc[
            result_df[CompareConst.NPU_NAME].str.contains(escaped_api_pat, na=False), CompareConst.NPU_NAME
        ].tolist()
        if len(single_api_full_name_list) == 0:
            logger.warning(f"{api} not found in rank{rank_num} compare result.")
            continue
        api_full_name_list.extend(single_api_full_name_list)
    return api_full_name_list


def search_api_index_result(api_list, compare_index_list, result_df, rank_num, compare_index_dict):
    """
    parsing single rank compare result into the intermediate target dict
    {
        compare_index1: {
            api_full_name1:{
                rank1: value,
            },
            api_full_name2,
            ...
        },
        compare_index2: {},
        ...
    }
    """
    api_full_name_list = extract_api_full_name(api_list, result_df, rank_num)
    for compare_index in compare_index_list:
        api_index_dict = {}
        for api_full_name in api_full_name_list:
            table_value_check(api_full_name)
            row_num = result_df.index[result_df[CompareConst.NPU_NAME] == api_full_name].tolist()[0]
            index_value = result_df.loc[row_num, compare_index]
            table_value_check(index_value)
            api_index_dict.setdefault(api_full_name, {})[rank_num] = index_value  # update api_index_dict
        compare_index_dict[compare_index] = api_index_dict

    compare_index_dict = replace_compare_index_dict(compare_index_dict, compare_index_list, rank_num)
    return compare_index_dict


def table_value_check(value):
    if not table_value_is_valid(value):
        raise RuntimeError(f"Malicious value [{value}] is not allowed to be written into the merged xlsx.")


def result_process(compare_result_path_list, api_list):
    """
    process compare results into target intermediate dict list
    """
    compare_index_dict_list = []
    rank_num_list = []
    compare_index_list = []

    for compare_result_path in compare_result_path_list:
        compare_index_dict = {}
        result_df = read_xlsx(compare_result_path)

        rank_pattern = r"compare_result_rank(\d+)"
        rank_num = int(re.search(rank_pattern, os.path.basename(compare_result_path)).group(1))
        logger.info(f"Parsing rank{rank_num} compare result...")
        if not result_df.empty:
            dump_mode = get_dump_mode(result_df, rank_num)
            if dump_mode == "":
                return [], [], []
            # 因为compare_index是指定的，固定不变，所以一旦compare_index是确定的，dump_mode也是确定的，
            # 所以只要校验compare_index和dump_mode一致性就能保证所有rank的结果都是dump_mode一致的
            compare_index_list = check_index_dump_mode_consistent(dump_mode, rank_num)
            if len(compare_index_list) == 0:
                return [], [], []
            compare_index_list.extend([CompareConst.NPU_MAX, CompareConst.BENCH_MAX])
            compare_index_dict = search_api_index_result(
                api_list, compare_index_list, result_df, rank_num, compare_index_dict
            )
            compare_index_dict_list.append(compare_index_dict)
            rank_num_list.append(rank_num)
            compare_index_list.pop()
            compare_index_list.pop()
        else:
            logger.warning(f"Rank{rank_num} compare result is empty and will not shown in merged result.")

    return compare_index_dict_list, rank_num_list, compare_index_list


def handle_multi_process(func, func_args, lock):
    compare_result_path_list, api_list = func_args

    result_num = len(compare_result_path_list)
    process_num = int((multiprocessing.cpu_count() + 1) / 2)
    if result_num <= process_num:
        process_num = result_num
        chunks = [[compare_result_path] for compare_result_path in compare_result_path_list]
    else:
        chunk_size = result_num // process_num
        chunks = [compare_result_path_list[i : i + chunk_size] for i in range(0, result_num, chunk_size)]

    pool = multiprocessing.Pool(  # pylint: disable=consider-using-with
        process_num,
        initializer=init_worker,
        initargs=(share_compare_index_list,),
    )

    def err_call(args):
        logger.error('Multiprocess merge result failed! Reason: {}'.format(args))
        try:
            pool.close()
        except OSError:
            logger.error("Pool terminate failed")

    progress_bar = tqdm(total=result_num, desc="Compare Result Parsing Process", unit="num", ncols=100)

    def update_progress(size, progress_lock, extra_param=None):
        with progress_lock:
            progress_bar.update(size)

    results = []
    for chunk in chunks:
        chunk_size = len(chunk)
        result = pool.apply_async(
            func,  # pool.apply_async立即返回ApplyResult对象，因此results中结果是顺序的
            args=(chunk, api_list),
            error_callback=err_call,
            callback=partial(update_progress, chunk_size, lock),
        )
        results.append(result)

    all_compare_index_dict_list = []
    all_rank_num_list = []
    all_compare_index_list_list = []
    for result in results:
        compare_index_dict, rank_num_list, compare_index_list = result.get()
        all_compare_index_dict_list.append(compare_index_dict)
        all_rank_num_list.append(rank_num_list)
        all_compare_index_list_list.append(compare_index_list)

    pool.close()
    pool.join()

    if not any(all_compare_index_dict_list):
        logger.warning("Nothing to merge.")
        raise CompareException(CompareException.MERGE_COMPARE_RESULT_ERROR)

    return all_compare_index_dict_list, all_rank_num_list, all_compare_index_list_list


def generate_result_df(api_index_dict, header):
    """
    Generates a DataFrame from the given api_index_dict and header.
    api_index_dict:
    {
        api_full_name1:{
            rank1: value,
            },
        api_full_name2:{
            rank1: value
            },
        ...
    }
    """
    result = []
    for api_full_name, rank_value_dict in api_index_dict.items():
        result_item = [api_full_name]
        result_item.extend(rank_value_dict.values())
        result.append(result_item)
    return pd.DataFrame(result, columns=header, dtype="object")


def generate_merge_result(all_compare_index_dict_list, all_rank_num_list, all_compare_index_list_list, output_dir):
    """
    generate merge result from the intermediate dict.
    one compare index, one sheet
    """
    file_name = add_time_with_xlsx("multi_ranks_compare_merge")
    output_path = os.path.join(output_dir, file_name)

    compare_index_list = None
    for item in all_compare_index_list_list:
        if item:
            compare_index_list = item
            break
    if not compare_index_list:
        logger.error("No compare index recognized, please check!")
        raise CompareException(CompareException.MERGE_COMPARE_RESULT_ERROR)

    all_result_df_list = []
    for compare_index_dict_list, rank_num_list in zip(all_compare_index_dict_list, all_rank_num_list):
        for compare_index_dict, rank_num in zip(compare_index_dict_list, rank_num_list):
            header = [CompareConst.NPU_NAME, "rank" + str(rank_num)]
            result_df_list = []
            for _, api_index_dict in compare_index_dict.items():
                result_df = generate_result_df(api_index_dict, header)
                result_df_list.append(result_df)
            all_result_df_list.append(result_df_list)

    merge_df_list = df_merge(all_result_df_list)
    final_result_df_list = []
    for i, df in enumerate(merge_df_list):
        # merge_df_list中df与compare_index_list中compare_index一一对应
        final_result_df_list.append((df, compare_index_list[i]))

    if os.path.exists(output_path):
        logger.warning(f"{output_path} will be deleted.")
        remove_path(output_path)
    save_excel(output_path, final_result_df_list)
    logger.info(f"The compare results of the multi-ranks are merged and saved in: {output_path}.")


def df_merge(all_result_df_list):
    """
    merge different rank result_df
    """
    if len(all_result_df_list) == 0:
        logger.warning("Nothing to merge.")
        raise CompareException(CompareException.MERGE_COMPARE_RESULT_ERROR)
    if len(all_result_df_list) == 1:
        logger.info("Only one compare result gets merge data.")
    merge_df_base = all_result_df_list[0]
    for sublist in all_result_df_list[1:]:
        for i, sub_df in enumerate(sublist):
            merge_df_base[i] = pd.merge(merge_df_base[i], sub_df, on=CompareConst.NPU_NAME, how='outer')
    for i, value in enumerate(merge_df_base):
        merge_df_base[i] = value.reindex(
            columns=[CompareConst.NPU_NAME] + [col for col in value.columns if col != CompareConst.NPU_NAME]
        )
    return merge_df_base


share_compare_index_list = []


def init_worker(share_index_list):
    """
    pool worker initializer, pass the shared compare index list to worker process explicitly.
    forkserver/spawn start method workers do not inherit module globals from parent process,
    so the shared list must be delivered through Pool initargs (python3.14+ defaults to forkserver on linux).
    """
    global share_compare_index_list
    share_compare_index_list = share_index_list


def initialize_compare_index(config):
    global share_compare_index_list
    manager = multiprocessing.Manager()
    share_compare_index_list = manager.list(config.get("compare_index", []))  # 创建共享全局列表


def merge_result(input_dir, output_dir, config_path):
    input_dir = FileChecker(input_dir, FileCheckConst.DIR, FileCheckConst.READ_ABLE).common_check()
    create_directory(output_dir)

    compare_result_path_list = get_result_path(
        input_dir
    )  # 获得的input_dir中所有比对结果件的全路径，数量少于2，便提示退出

    config = load_yaml(config_path)
    config = check_config(config)
    api_list = config.get('api')

    # 初始化共享全局变量share_compare_index_list
    initialize_compare_index(config)

    func_args = (compare_result_path_list, api_list)
    all_compare_index_dict_list, all_rank_num_list, all_compare_index_list_list = handle_multi_process(
        result_process, func_args, multiprocessing.Manager().RLock()
    )

    generate_merge_result(all_compare_index_dict_list, all_rank_num_list, all_compare_index_list_list, output_dir)
