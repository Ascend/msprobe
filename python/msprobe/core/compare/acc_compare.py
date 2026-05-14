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
import re
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from msprobe.core.common.const import CompareConst, Const
from msprobe.core.common.exceptions import FileCheckException
from msprobe.core.common.file_utils import load_json, remove_path, create_directory, save_excel, save_json, \
    write_df_to_csv
from msprobe.core.common.log import logger
from msprobe.core.common.utils import CompareException, add_time_with_csv, add_time_with_xlsx, \
    check_op_str_pattern_valid, set_dump_path, get_dump_mode, check_compare_param, load_stack_json, \
    get_file_type, add_time_with_json
from msprobe.core.compare.check import check_dump_json_str, check_stack_json_str, cross_dtype_mapping, \
    check_configuration_param, check_consistent_param
from msprobe.core.compare.utils import print_compare_ends_info, read_op, set_stack_json_path, reorder_index, \
    parse_op_data, load_pt_in_structure
from msprobe.core.compare.config import ModeConfig, MappingConfig, MappingDict
from msprobe.core.compare.multiprocessing_compute import CompareRealData
from msprobe.core.compare.diff_analyze.first_diff_analyze import FirstDiffAnalyze
from msprobe.core.compare.indicator_analysis.calculator import calculate_excel_result_df
from msprobe.core.compare.stats_diff_calc import ValType, ALL_TYPES
from msprobe.core.compare.verl.mapping import FSDP_MODULE_MAP, MEGATRON_MODULE_MAP
from msprobe.core.common.output_postprocess.processor import should_postprocess_output_for_compare, extract_valid_len


@dataclass
class ComparisonConfig:
    dump_mode: str
    stack_mode: bool
    fuzzy_match: bool
    data_mapping: dict
    suffix: str
    cell_mapping: dict
    api_mapping: dict
    layer_mapping: dict
    xlsx: bool
    compared_file_type: str
    first_diff_analyze: bool
    is_print_compare_log: bool
    consistent_check: bool
    backend: str


class Comparator:

    def __init__(self, file_reader, mode_config: ModeConfig, mapping_config: MappingConfig, is_cross_framework=False):
        self.file_reader = file_reader
        self.mode_config = mode_config
        self.mapping_config = mapping_config
        self.cross_frame = is_cross_framework
        self.mapping_dict = MappingDict(mapping_config)

    @staticmethod
    def restore_dump_order(match_result):
        if match_result.empty:
            return match_result

        # 检查是否包含 NPU_PARSE_ORDER 列
        has_npu_order = CompareConst.NPU_PARSE_ORDER in match_result.columns
        has_bench_order = CompareConst.BENCH_PARSE_ORDER in match_result.columns

        # 如果没有相关列，则直接返回原始数据
        if not has_npu_order and not has_bench_order:
            return match_result

        if has_npu_order:
            match_result[CompareConst.NPU_PARSE_ORDER] = pd.to_numeric(match_result[CompareConst.NPU_PARSE_ORDER],
                                                                       errors='coerce')

        # 使用 COMPARE_ORDER 临时列进行排序
        if has_npu_order:
            match_result[CompareConst.COMPARE_ORDER] = match_result[CompareConst.NPU_PARSE_ORDER]
        else:
            match_result[CompareConst.COMPARE_ORDER] = np.nan

        match_result = match_result.sort_values(by=CompareConst.COMPARE_ORDER, na_position="last",
                                                kind="stable").reset_index(
            drop=True)
        match_result.drop(columns=[CompareConst.COMPARE_ORDER], inplace=True, errors='ignore')
        return match_result

    def process_output_file(self, output_path, suffix, compared_file_type):
        file_name_prefix_mapping = {
            Const.DUMP_JSON_FILE: "compare_result",
            Const.DEBUG_JSON_FILE: "debug_compare_result"
        }
        file_name_prefix = file_name_prefix_mapping.get(compared_file_type, "compare_result")
        if self.mode_config.first_diff_analyze:
            file_name = add_time_with_json("compare_result" + suffix)
        elif self.mode_config.xlsx:
            file_name = add_time_with_xlsx(file_name_prefix + suffix)
        else:
            file_name = add_time_with_csv(file_name_prefix + suffix)
        file_path = os.path.join(os.path.realpath(output_path), file_name)
        if os.path.exists(file_path):
            logger.warning(f"{file_path} will be deleted.")
            remove_path(file_path)
        return file_path

    def compare_core(self, input_param, output_path, **kwargs):
        """
        Compares data from multiple JSON files and generates a comparison report.

        Args:
            input_param (dict): A dictionary containing paths to JSON files ("npu_path", "bench_path",
                                "stack_path").
            output_path (str): The path where the output Excel report will be saved.
            **kwargs: Additional keyword arguments including:
            - stack_mode (bool, optional): Enables stack mode comparison. Defaults to False.
            - auto_analyze (bool, optional): If True, triggers automatic analysis after comparison. Defaults to True.
            - suffix (str, optional): Suffix to append to the output file name. Defaults to ''.
            - fuzzy_match (bool, optional): Enables fuzzy matching during comparison. Defaults to False.
            - dump_mode (str): ALL, SUMMARY, MD5.

        Returns:
        """
        logger.info("Please check whether the input data belongs to you. If not, there may be security risks.")

        # get kwargs or set default value
        suffix = kwargs.get('suffix', '')
        rank = suffix[1:]
        # process output file
        file_path = self.process_output_file(output_path, suffix, self.mode_config.compared_file_type)

        # initialize the compare result table and compare general data(name, dtype, shape, statistics/md5, etc.)
        npu_json = input_param.get("npu_path")
        bench_json = input_param.get("bench_path")
        stack_json = input_param.get("stack_path")
        data_dirs = {
            CompareConst.DEVICE_NPU: input_param.get(CompareConst.NPU_DUMP_DATA_DIR),
            CompareConst.DEVICE_BENCH: input_param.get(CompareConst.BENCH_DUMP_DATA_DIR)
        }
        parse_data = ParseData(self.mode_config, rank, data_dirs)  # load and parse json data
        npu_df, bench_df = parse_data.parse([npu_json, bench_json, stack_json])

        logger.info("Matching APIs/Modules in progress...")
        result_df = self.compare_statistics(npu_df, bench_df)
        logger.info("APIs/Modules match done.")
        if result_df.empty:
            logger.warning("Can`t match any op. No compare result file generated.")
            return

        if self.mode_config.first_diff_analyze:
            # add P2POp additional info from npu_df and bench_df to result_df
            result_df['NPU P2POp op'] = npu_df['op']
            result_df['Bench P2POp op'] = bench_df['op']
            result_df['NPU P2POp peer'] = npu_df['peer']
            result_df['Bench P2POp peer'] = bench_df['peer']

            first_diff_analyze = FirstDiffAnalyze(self.mode_config, rank)
            check_result = first_diff_analyze.check(result_df)
            save_json(file_path, check_result, indent=4)
            logger.info(f"Saving json file to disk: {file_path}")
            return

        # compare real data
        if self.mode_config.dump_mode == Const.ALL:
            logger.info("Compare real data in progress...")
            compare_real_data = CompareRealData(self.file_reader, self.mode_config, self.cross_frame)
            result_df = result_df.reset_index(drop=True)
            result_df = compare_real_data.do_multi_process(input_param, result_df)
            result_df.drop(columns=CompareConst.ALL_MODE_DROP_COLUMNS, inplace=True, errors='ignore')
            logger.info("Compare real data done.")

        # calculate Indicators
        logger.info("Calculating comparison indicators in progress...")
        calculate_excel_result_df(result_df, self.mode_config.dump_mode, self.mode_config.backend,
                                  self.mode_config.consistent_check)
        result_df.drop(columns=['state', 'api_origin_name'], inplace=True)  # 删除中间数据，两列不落盘
        logger.info("Comparison indicators calculation done.")

        # save result file
        logger.info("Saving result file in progress...")
        logger.info(f"The result file path is: {file_path}.")
        if self.mode_config.xlsx:
            save_excel(file_path, result_df)
        else:
            # float类型的inf转成空格+inf的字符串，解决excel打开csv，inf不被识别显示#NAME?的问题
            result_df = result_df.replace({np.inf: " inf", -np.inf: " -inf"})
            write_df_to_csv(result_df, file_path)

        print_compare_ends_info()

    def compare_statistics(self, npu_df, bench_df):
        npu_df[[Const.DTYPE, Const.SHAPE]] = npu_df[[Const.DTYPE, Const.SHAPE]].astype(str)
        bench_df[[Const.DTYPE, Const.SHAPE]] = bench_df[[Const.DTYPE, Const.SHAPE]].astype(str)
        npu_df[CompareConst.NPU_PARSE_ORDER] = range(len(npu_df))
        bench_df[CompareConst.BENCH_PARSE_ORDER] = range(len(bench_df))

        # create new columns for compare op_name and shape
        # process npu_df's COMPARE_KEY whether same or different framework
        process_df = ProcessDf(self.mode_config, self.mapping_config, self.mapping_dict)
        if self.mode_config.consistent_check:
            npu_df, bench_df = process_df.process_consistent_df(npu_df, bench_df)
        else:
            # 统一反向序号重计算和非反向序号重计算，后续cmp_key统一使用op_name_update, 默认保持原始值
            npu_df[CompareConst.OP_NAME_UPDATE] = npu_df[CompareConst.OP_NAME]
            bench_df[CompareConst.OP_NAME_UPDATE] = bench_df[CompareConst.OP_NAME]
            # 处理重计算对应的backward的序号。反向重计算序号调整属于精确匹配，与模糊匹配互斥。
            # 训推一致性比对不考虑反向
            # 单点数据不考虑反向序号重计算
            if not self.mode_config.fuzzy_match and self.mode_config.compared_file_type != Const.DEBUG_JSON_FILE:
                npu_df = process_df.update_backward_call(npu_df)
                bench_df = process_df.update_backward_call(bench_df)
            npu_df, bench_df = process_df.process_compare_key_and_shape(npu_df, bench_df)

        # match npu and bench, match_result contains both npu_info and bench_info
        match = Match(self.mode_config, self.mapping_config, self.cross_frame)
        match_result = match.match_api_infos(npu_df, bench_df)
        if not self.mode_config.consistent_check:
            # 比对主逻辑，训推一致性另外单独处理
            match_result = self.restore_dump_order(match_result)
            # 筛选出npu_name存在的行并填充筛选出行中的缺失值为N/A
            match_result = match_result[match_result['op_name_x'].notna()].fillna(CompareConst.N_A)
            bench_columns = [i + '_y' for i in bench_df.columns]
            match_result_columns = match_result.columns.tolist()
            new_bench_columns = []
            for col in bench_columns:
                if col in match_result_columns:
                    new_bench_columns.append(col)
            for col in new_bench_columns:
                match_result[col] = match_result[col].astype(object)
            match_result.loc[~match.gen_dtype_condition(match_result), new_bench_columns] = CompareConst.N_A

        # organize compare result table by renaming columns
        if self.mode_config.dump_mode == Const.ALL and self.mode_config.first_diff_analyze:
            self.mode_config.dump_mode = Const.SUMMARY
        match_result.drop(columns=[CompareConst.NPU_PARSE_ORDER, CompareConst.BENCH_PARSE_ORDER], inplace=True,
                          errors='ignore')
        create_table = CreateTable(self.mode_config)
        result_df, header = create_table.make_result_df(match_result)

        # calculate statistics diff
        calc_stats_diff = CalcStatsDiff(self.mode_config)
        return calc_stats_diff.calc_accuracy(result_df, header)


class ParseData:
    def __init__(self, mode_config: ModeConfig, rank, data_dirs: dict):
        self.mode_config = mode_config
        self.rank = rank
        self.data_dirs = data_dirs  # {"NPU": npu_dump_data_dir, "Bench": bench_dump_data_dir}

    def split_data_name_components(self, data_name):
        """
        解析 data_name，返回：
            op_no_number
            call_direction
            direction

        支持格式：
            1. op_no_number + forward/backward/数字组合（api级别数字在前，模块级别数字在后）
            2. op_no_number + parameters_grad
        """
        # 如果单点数据比对模式，不做处理，直接返回
        if self.mode_config.compared_file_type == Const.DEBUG_JSON_FILE:
            return data_name, data_name, data_name

        parts = data_name.split(Const.SEP)
        if not parts:
            return '', '', ''

        last = parts[-1]
        second_last = parts[-2] if len(parts) >= 2 else ''

        # parameters_grad
        if Const.PARAMS_GRAD in (last, second_last):
            return (
                Const.SEP.join(parts[:-1]) if len(parts) > 1 else '',
                Const.PARAMS_GRAD,
                Const.PARAMS_GRAD
            )

        # forward / backward
        for direction in (Const.FORWARD, Const.BACKWARD):
            if direction in (last, second_last):
                return (
                    Const.SEP.join(parts[:-2]) if len(parts) > 2 else '',
                    Const.SEP.join((second_last, last)),
                    direction
                )

        return '', '', ''

    def parse(self, file_list):
        npu_json_path, bench_json_path, stack_json_path = file_list
        npu_json_data = load_json(npu_json_path)
        bench_json_data = load_json(bench_json_path)
        stack_json_data = load_stack_json(stack_json_path) if self.mode_config.stack_mode else None

        # parse json data and generate df
        npu_df = self.gen_data_df(npu_json_data, stack_json_data, CompareConst.DEVICE_NPU)
        bench_df = self.gen_data_df(bench_json_data, stack_json_data, CompareConst.DEVICE_BENCH)

        return npu_df, bench_df

    def init_result(self):
        result = {
            CompareConst.OP_NAME: [],
            Const.DTYPE: [],
            Const.SHAPE: [],
            Const.SUMMARY: [],
            Const.STACK_INFO: [],
            Const.STATE: [],
            Const.API_ORIGIN_NAME: [],
            Const.REQ_GRAD: [],
            Const.DIRECTION: [],  # 目前三种选择：'forward', 'backward', None
            Const.CALL_DIRECTION: [],
            Const.OP_NO_NUMBER: [],  # 删除调用序号
            Const.FORWARD_CALL_ORDER: [],  # 计算op前向调用顺序，初始化为0
            Const.BACKWARD_CALL_ORDER: [],  # 计算op反向调用顺序，初始化为0
            Const.SUFFIX: []
        }
        if self.mode_config.dump_mode == Const.ALL:
            result[Const.DATA_NAME] = []
            result[Const.DIRTY_VALID_LEN] = []
        elif self.mode_config.dump_mode == Const.MD5:
            result[Const.MD5] = []
        return result

    def create_progress_bar(self, api_nums: int, device: str):
        desc = f'{device} API/Module Read Progress'
        if self.rank:
            desc = f'[{self.rank}]' + desc
        return tqdm(total=api_nums, desc=desc, unit="api/module", ncols=100)

    def gen_data_df(self, data_json, stack_json_data, device: str):
        result = self.init_result()
        op_forward_count = defaultdict(int)  # 记录forward调用次数，用于后续调用序更新
        op_backward_count = defaultdict(int)  # 记录backward调用次数，用于后续调用序更新

        apis_data = data_json.get('data', None)
        if not apis_data:
            logger.warning('No APIs found in dump.json.')
            return pd.DataFrame(result)

        progress_bar = self.create_progress_bar(len(apis_data), device)

        parse_flag = True  # 使用布尔值，默认为 True，表示解析开启

        # 从json中循环解析API数据，遍历所有API
        for op_name in apis_data:
            check_op_str_pattern_valid(op_name)

            parse_flag = self.should_parse_op(parse_flag, op_name, device)
            # 如果 parse_flag 为 False，则跳过当前数据项
            if not parse_flag:
                progress_bar.update(1)
                continue

            op_data = apis_data.get(op_name)
            op_parsed_list = self.gen_merge_list(op_data, op_name, stack_json_data)
            if not op_parsed_list:
                progress_bar.update(1)
                continue

            op_no_number, call_direction, direction = self.split_data_name_components(op_name)
            if not op_no_number:
                progress_bar.update(1)
                continue

            forward_call_order = op_forward_count.get(op_no_number, 0)
            backward_call_order = op_backward_count.get(op_no_number, 0)
            if direction == Const.FORWARD:
                op_forward_count[op_no_number] += 1
            if direction == Const.BACKWARD:
                op_backward_count[op_no_number] += 1

            stack_info = op_parsed_list[-1].get('full_info') if self.mode_config.stack_mode else None

            reordered_index_list = reorder_index(op_parsed_list)

            if self.mode_config.dump_mode == Const.ALL:
                valid_len = self.get_valid_len_with_postprocess(op_name, op_data, device)
                result[Const.DIRTY_VALID_LEN].extend([valid_len] * len(reordered_index_list))

            for i, index in enumerate(reordered_index_list):
                op_item = op_parsed_list[index]
                summary_data = [
                    str(op_item.get(key)) if op_item.get(key) is None else op_item.get(key)
                    for key in Const.SUMMARY_METRICS_LIST
                ]
                full_op_name = op_item.get(Const.FULL_OP_NAME)
                suffix = full_op_name.replace(op_name, '')

                # common key
                result[CompareConst.OP_NAME].append(full_op_name)
                result[Const.DTYPE].append(op_item.get(Const.DTYPE))
                result[Const.SHAPE].append(op_item.get(Const.SHAPE))
                result[Const.STATE].append(op_item.get(Const.STATE))
                result[Const.REQ_GRAD].append(op_item.get(Const.REQ_GRAD))
                result[Const.API_ORIGIN_NAME].append(op_name)
                result[Const.SUMMARY].append(summary_data)
                result[Const.DIRECTION].append(direction)
                result[Const.CALL_DIRECTION].append(call_direction)
                result[Const.OP_NO_NUMBER].append(op_no_number)
                result[Const.FORWARD_CALL_ORDER].append(forward_call_order)
                result[Const.BACKWARD_CALL_ORDER].append(backward_call_order)
                result[Const.SUFFIX].append(suffix)

                # dump_mode differ key
                if self.mode_config.dump_mode == Const.MD5:
                    result[Const.MD5].append(op_item.get(Const.MD5))
                if self.mode_config.dump_mode == Const.ALL:
                    result[Const.DATA_NAME].append(op_item.get(Const.DATA_NAME))

                # mode_config stack_mode addition key
                result[Const.STACK_INFO].append(stack_info if i == 0 else None)

                # mode_config first_diff_analyze addition key
                if self.mode_config.first_diff_analyze:
                    result.setdefault('op', []).append(op_item.get('op', str(None)))
                    result.setdefault('peer', []).append(op_item.get('peer', str(None)))

            progress_bar.update(1)
        progress_bar.close()
        return pd.DataFrame(result)

    def gen_merge_list(self, op_data, op_name, stack_json_data):
        if self.mode_config.compared_file_type == Const.DUMP_JSON_FILE:
            check_dump_json_str(op_data, op_name)
        op_parsed_list = read_op(op_data, op_name)

        if self.mode_config.stack_mode:
            stack_info = stack_json_data.get(op_name)
            if stack_info is not None:
                check_stack_json_str(stack_info, op_name)
        else:
            stack_info = None
        # always add stack_info whether stack_mode is True
        op_parsed_list.append({
            'full_op_name': op_name,
            'full_info': stack_info
        })
        return op_parsed_list

    def should_parse_op(self, parse_flag, op_name, device: str):
        """
        根据操作名判断是否应解析。

        :param parse_flag: 原是否解析标识
        :param op_name: 操作名字符串
        :param device: Npu/Bench
        :return: 返回解析标志，True表示解析开启，False表示解析关闭
        """
        if not self.mode_config.consistent_check:  # 非训推一致性比对均解析数据
            return True

        if device == CompareConst.DEVICE_BENCH:
            return True

        op_name_splits = op_name.split(Const.SEP)
        # 确保分割后列表长度符合预期
        if len(op_name_splits) < 3:
            logger.error(f"Length of op name in dump.json is not as expected. "
                         f"The division should be no less than 3. Op_name is {op_name}. Please check.")
            return False  # 确保在长度不符合时返回 False

        # 提取操作名的相关部分
        # 模块/子模块名
        module_name = op_name_splits[-3]
        # 前向/反向
        direction_name = op_name_splits[-2]

        # 根据操作名调整解析标志
        if module_name in CompareConst.VERL_STOP_PARSE_RULES.get(
                self.mode_config.backend) and direction_name == Const.BACKWARD:
            return False
        elif module_name in CompareConst.VERL_BEGIN_PARSE_RULES.get(
                self.mode_config.backend) and direction_name == Const.FORWARD:
            return True

        return parse_flag

    def get_valid_len_with_postprocess(self, op_name, op_data, device):
        backend = CompareConst.DEVICE_TO_BACKEND.get(device)  # 调用预检设计接口需要将device转成backend

        valid_len = -1  # 默认值，表示未成功提取到有效长度
        should_postprocess, matched_api_name = should_postprocess_output_for_compare(op_name, backend)

        if should_postprocess:
            parsed_op_info = parse_op_data(op_data)
            loaded_op_info = load_pt_in_structure(parsed_op_info, self.data_dirs, device)
            if parsed_op_info == loaded_op_info:
                logger.warning(f"Failed to get dirty clean info for output of {op_name}.")
            else:
                op_args = loaded_op_info.get("input_args", [])
                op_kwargs = loaded_op_info.get("input_kwargs", {})
                valid_len = extract_valid_len(matched_api_name, op_args, op_kwargs, backend)
                if valid_len is None:
                    valid_len = -1  # 如果提取失败，设置为 -1
        
        return valid_len


class ProcessDf:
    def __init__(self, mode_config: ModeConfig, mapping_config: MappingConfig, mapping_dict: MappingDict):
        self.mode_config = mode_config
        self.mapping_config = mapping_config
        self.mapping_dict = mapping_dict

    @staticmethod
    def get_api_name(api_list):
        try:
            api_name = api_list[0] + Const.SEP + api_list[1]
        except IndexError as error:
            logger.error('Failed to retrieve API name, please check if the dump data is reasonable')
            raise CompareException(CompareException.INDEX_OUT_OF_BOUNDS_ERROR) from error
        return api_name

    @staticmethod
    def update_forward_call(cmp_df):
        # ===============================
        # Step 1: 拆 call_direction 的两部分
        # ===============================
        cd_parts = cmp_df[Const.CALL_DIRECTION].str.rsplit(Const.SEP, n=1, expand=True)
        cd_parts.columns = ["cd_head", "cd_tail"]

        head_is_digit = cd_parts["cd_head"].str.isdigit()
        tail_is_digit = cd_parts["cd_tail"].str.isdigit()
        is_forward = cmp_df[Const.DIRECTION] == Const.FORWARD

        # ===============================
        # Step 2: 矢量化替换 forward_call_order
        # ===============================
        mask_head = is_forward & head_is_digit
        mask_tail = is_forward & tail_is_digit
        update_mask = mask_head | mask_tail

        cd_parts.loc[mask_head, "cd_head"] = (
            cmp_df.loc[mask_head, Const.FORWARD_CALL_ORDER].astype(str)
        )
        cd_parts.loc[mask_tail, "cd_tail"] = (
            cmp_df.loc[mask_tail, Const.FORWARD_CALL_ORDER].astype(str)
        )

        # 只更新需要更新的 call_direction
        cmp_df.loc[update_mask, Const.CALL_DIRECTION] = (
                cd_parts.loc[update_mask, "cd_head"]
                + Const.SEP
                + cd_parts.loc[update_mask, "cd_tail"]
        )

        return cmp_df

    @staticmethod
    def update_backward_call(cmp_df):
        # ===============================
        # Step 1: 拆 call_direction 的两部分
        # ===============================
        cd_parts = cmp_df[Const.CALL_DIRECTION].str.rsplit(Const.SEP, n=1, expand=True)
        cd_parts.columns = ["cd_head", "cd_tail"]

        head_is_digit = cd_parts["cd_head"].str.isdigit()
        tail_is_digit = cd_parts["cd_tail"].str.isdigit()
        is_backward = cmp_df[Const.DIRECTION] == Const.BACKWARD

        # ===============================
        # Step 2: 矢量化替换 backward_call_order
        # ===============================
        mask_head = is_backward & head_is_digit
        mask_tail = is_backward & tail_is_digit
        update_mask = mask_head | mask_tail

        cd_parts.loc[mask_head, "cd_head"] = (
            cmp_df.loc[mask_head, Const.BACKWARD_CALL_ORDER].astype(str)
        )
        cd_parts.loc[mask_tail, "cd_tail"] = (
            cmp_df.loc[mask_tail, Const.BACKWARD_CALL_ORDER].astype(str)
        )

        # 只更新需要更新的 call_direction
        cmp_df.loc[update_mask, Const.CALL_DIRECTION] = (
                cd_parts.loc[update_mask, "cd_head"]
                + Const.SEP
                + cd_parts.loc[update_mask, "cd_tail"]
        )

        # ===============================
        # Step 3: 初始化 + 局部更新 OP_NAME_UPDATE
        # ===============================
        # 只更新命中的行
        cmp_df.loc[update_mask, CompareConst.OP_NAME_UPDATE] = (
                cmp_df.loc[update_mask, Const.OP_NO_NUMBER]
                + Const.SEP
                + cmp_df.loc[update_mask, Const.CALL_DIRECTION]
                + cmp_df.loc[update_mask, Const.SUFFIX]
        )

        return cmp_df

    @staticmethod
    def get_op_layer(data_df):
        # 获取模块层数，对于非layer的op，layer设置默认值为-1
        layer_pattern = r'layers\.(\d+)'
        data_df[Const.LAYER] = data_df[Const.OP_NO_NUMBER].str.extract(layer_pattern).fillna(-1).astype(int)

    @staticmethod
    def get_op_module_and_class(data_df, engine: str, backend: str):
        """
        根据module_class列处理获取module和class
        """
        if engine == 'train':
            if backend == Const.FSDP:
                train_op_pattern = '_fsdp_wrapped_module.'
            elif backend == Const.MEGATRON:
                train_op_pattern = r'^.*layers\.\d+\.'
            data_df[Const.MODULE_CLASS] = data_df[Const.OP_NO_NUMBER].str.split(train_op_pattern).str[-1]
            # 处理非layer层
            data_df[Const.MODULE_CLASS] = data_df[Const.MODULE_CLASS].str.replace(r'^.*module.module\.', '', regex=True)
        elif engine == 'infer':
            infer_layer_pattern = r'^.*layers\.\d+\.'
            data_df[Const.MODULE_CLASS] = data_df[Const.OP_NO_NUMBER].str.replace(infer_layer_pattern, '', regex=True)
            # 处理非layer层
            data_df[Const.MODULE_CLASS] = data_df[Const.MODULE_CLASS].str.replace(r'^Module.model\.', '', regex=True)
        else:
            raise ValueError(f'Unsupported engine type: {engine}')

        # 使用 str.rsplit() 拆分，最多拆分一次，最后一个是class
        split_result = data_df[Const.MODULE_CLASS].str.rsplit(Const.SEP, n=1, expand=True)

        # 处理拆分后的结果，检查拆分后的列数
        data_df[Const.MODULE] = split_result[0].where(split_result[1].notna(), '')  # 如果没有拆分点，module为''
        data_df[Const.CLASS] = split_result[1].fillna(data_df[Const.MODULE_CLASS])  # 如果没有拆分点，class为原始字符串
        data_df[Const.MODULE_LEN] = data_df[Const.MODULE].str.split(Const.SEP).str.len()

    def process_module_mapping(self, data_df, engine: str, backend: str):
        """
        fsdp:仅映射训练数据，推理数据直接赋值
        megatron:训练数据和推理数据均涉及映射
        """
        if engine == 'infer' and backend == Const.FSDP:
            data_df[Const.MODULE_MAPPING] = data_df[Const.MODULE].apply(
                lambda x: str(x).split(Const.SEP)[-1] if pd.notna(x) and Const.SEP in str(x) else x)
            return
        # 直接为module_len == 1分配默认值
        len_1_mask = data_df[Const.MODULE_LEN] == 1
        data_df[Const.MODULE_PART_PREFIX] = ''
        data_df[Const.MODULE_LAST] = data_df[Const.MODULE]
        data_df[Const.MODULE_MAPPING] = data_df[Const.MODULE]

        # 对于 module_len > 1 的行，拆分module为prefix和last_part
        split_columns = data_df.loc[~len_1_mask, Const.MODULE].str.split(Const.SEP, n=1, expand=True)
        if split_columns.empty:
            return
        split_columns.columns = [Const.MODULE_PART_PREFIX, Const.MODULE_LAST]  # 设置列名对齐，split分割后默认列名是(0, 1)
        data_df.loc[~len_1_mask, [Const.MODULE_PART_PREFIX, Const.MODULE_LAST]] = split_columns

        # 映射module_last生成新列module_last_mapping, 使用map()进行矢量化替换
        mapping = {}
        if self.mode_config.backend == Const.FSDP:
            mapping = FSDP_MODULE_MAP
        elif self.mode_config.backend == Const.MEGATRON:
            mapping = MEGATRON_MODULE_MAP
        last_part_mapped = data_df[Const.MODULE_LAST].map(mapping)
        data_df[Const.MODULE_LAST_MAPPING] = last_part_mapped.fillna(data_df[Const.MODULE_LAST])

        # 拼接prefix和映射后的last_part
        data_df[Const.MODULE_MAPPING] = data_df[Const.MODULE_LAST_MAPPING]

    def process_compare_key_and_shape(self, npu_df, bench_df):
        npu_df = self.assign_npu_df_compare_key(npu_df, bench_df)
        npu_df[CompareConst.CMP_SHAPE] = npu_df[Const.SHAPE]
        bench_df[CompareConst.CMP_KEY] = bench_df[CompareConst.OP_NAME_UPDATE]
        bench_df[CompareConst.CMP_SHAPE] = bench_df[Const.SHAPE]
        return npu_df, bench_df

    def assign_npu_df_compare_key(self, npu_df, bench_df):
        """
        处理 npu_df 的 COMPARE_KEY 赋值逻辑

        :param npu_df: DataFrame，NPU 对比数据
        :param bench_df: DataFrame，Bench 对比数据
        :return: compare_key(name)处理后的 npu_df
        """
        # 处理api_mapping映射
        if self.mapping_config.api_mapping:
            # 如果用户不传api_mapping.yaml，先使用内置api_mapping.yaml替换npu_op_name
            npu_df[CompareConst.CMP_KEY] = npu_df[CompareConst.OP_NAME].apply(self.process_internal_api_mapping)
            # 如果用户传入api_mapping.yaml，再使用传入api_mapping.yaml进一步替换npu_op_name
            if isinstance(self.mapping_config.api_mapping, str):
                self.modify_compare_data_with_user_mapping(npu_df, bench_df)
        # 处理cell_mapping映射
        elif self.mapping_config.cell_mapping:
            npu_df[CompareConst.CMP_KEY] = npu_df[CompareConst.OP_NAME].apply(self.process_cell_mapping)
        # 处理data_mapping映射
        elif self.mapping_config.data_mapping:
            npu_df[CompareConst.CMP_KEY] = npu_df[CompareConst.OP_NAME].apply(self.process_data_mapping)
        else:
            npu_df[CompareConst.CMP_KEY] = npu_df[CompareConst.OP_NAME_UPDATE]
        return npu_df

    def process_internal_api_mapping(self, npu_op_name):
        # get api name & class name from op_name
        ms_api_name = self.get_api_name(npu_op_name.split(Const.SEP))
        class_name = ms_api_name.split(Const.SEP)[0]
        if class_name == "Mint":
            return npu_op_name.replace("Mint", "Torch")
        elif class_name == "MintFunctional":
            return npu_op_name.replace("MintFunctional", "Functional")
        elif self.mapping_dict.ms_to_pt_mapping.get(ms_api_name):
            return npu_op_name.replace(ms_api_name, self.mapping_dict.ms_to_pt_mapping.get(ms_api_name))
        else:
            return npu_op_name

    def modify_compare_data_with_user_mapping(self, npu_df, bench_df):
        def remove_prefix(string, prefix):
            if string.startswith(prefix):
                return string[len(prefix):]
            return string

        def gen_input_compare_key(pattern, term):
            is_unmatched = True
            for i, prefix in enumerate(mapping_dict.get(f'ms_{term}')):
                if remove_prefix(op_name, api_origin_name + pattern) == str(prefix):
                    npu_df.loc[index, CompareConst.CMP_KEY] = (
                        op_name.replace(pattern + str(prefix), pattern + str(mapping_dict.get(f'pt_{term}')[i])))
                    is_unmatched = False
            return is_unmatched

        ms_api_indices_dict = self.get_api_indices_dict(npu_df)
        pt_api_indices_dict = self.get_api_indices_dict(bench_df)

        for mapping_dict in self.mapping_dict.api_mapping_dict:
            all_length_equal = True
            for k1, k2 in CompareConst.API_MAPPING_KEYS_TO_COMPARE:
                if len(mapping_dict.get(k1, [])) != len(mapping_dict.get(k2, [])):
                    all_length_equal = False
            if not all_length_equal:
                logger.warning('The user-defined mapping table is incorrect,\
                                make sure that the number of parameters is equal')
                continue

            ms_api, pt_api = mapping_dict.get('ms_api'), mapping_dict.get('pt_api')
            if ms_api not in ms_api_indices_dict or pt_api not in pt_api_indices_dict:
                continue
            for index in ms_api_indices_dict.get(ms_api):
                op_name = npu_df.loc[index, CompareConst.OP_NAME].replace(ms_api, pt_api, 1)
                state = npu_df.loc[index, Const.STATE]
                api_origin_name = npu_df.loc[index, Const.API_ORIGIN_NAME].replace(ms_api, pt_api, 1)
                if state == Const.INPUT:
                    is_abandoned = gen_input_compare_key(CompareConst.INPUT_PATTERN, 'args')
                elif state == Const.KWARGS:
                    is_abandoned = gen_input_compare_key(CompareConst.KWARGS_PATTERN, 'args')
                elif state == Const.OUTPUT:
                    is_abandoned = gen_input_compare_key(CompareConst.OUTPUT_PATTERN, 'output')
                elif state == Const.PARAMS:
                    is_abandoned = gen_input_compare_key(CompareConst.PARAMS_PATTERN, 'parameters')
                elif state == Const.PARAMS_GRAD:
                    is_abandoned = gen_input_compare_key(CompareConst.PARAMS_GRAD_PATTERN, 'parameters_grad')
                else:
                    logger.error(f'Excepted op_name: {op_name}')
                    raise CompareException(CompareException.INVALID_DATA_ERROR)
                if is_abandoned:
                    npu_df.loc[index, CompareConst.CMP_KEY] = op_name + 'abandoned'

    def get_api_indices_dict(self, op_name_df):
        """
        生成多个api对应的各自的所有的input、output等的index的键值对字典
        示例：
        {'Functional.conv2d': [0, 1, 2, 3],
        'Functional.batch_norm': [4, 5, 6, 7, 8]
        }
        """
        api_indices_dict = defaultdict(list)
        for op_index, name in enumerate(op_name_df[CompareConst.OP_NAME]):
            api_name = self.get_api_name(name.split(Const.SEP))
            api_indices_dict[api_name].append(op_index)
        return api_indices_dict

    def process_cell_mapping(self, npu_op_name):
        if not npu_op_name:
            return CompareConst.N_A
        param_grad_flag = Const.PARAMS_GRAD in npu_op_name.split(Const.SEP)
        if not param_grad_flag and not re.search(Const.REGEX_FORWARD_BACKWARD, npu_op_name):
            return CompareConst.N_A
        npu_op_name = npu_op_name.replace("Cell", "Module", 1)
        if self.mapping_dict.cell_mapping_dict:
            # get cell name & class name from op_name
            # Cell.fc1.Dense.forward.0.input.0
            # npu_op_name.split(Const.SEP, 1)[-1] 表示Module或Cell字段后面的部分
            cell_name = re.split(r'\.(?:forward|backward|parameters_grad)\.', npu_op_name.split(Const.SEP, 1)[-1])[0]

            # 1. 精确整段匹配（保持原行为）
            if cell_name in self.mapping_dict.cell_mapping_dict:
                new_cell_name = self.mapping_dict.cell_mapping_dict[cell_name]
            # 2. 兜底：简单字符串替换（不做任何限制）
            else:
                new_cell_name = cell_name
                for target_name, golden_name in self.mapping_dict.cell_mapping_dict.items():
                    if target_name in new_cell_name:
                        new_cell_name = new_cell_name.replace(target_name, golden_name, 1)
            # 3. 应用替换
            if new_cell_name != cell_name:
                npu_op_name = npu_op_name.replace(cell_name, new_cell_name, 1)

        return npu_op_name

    def process_data_mapping(self, npu_op_name):
        return self.mapping_dict.data_mapping_dict.get(npu_op_name, npu_op_name)

    def process_consistent_df(self, npu_df, bench_df):
        self.get_op_layer(npu_df)
        self.get_op_layer(bench_df)
        self.get_op_module_and_class(npu_df, engine='train', backend=self.mode_config.backend)
        self.get_op_module_and_class(bench_df, engine='infer', backend=self.mode_config.backend)
        self.process_module_mapping(npu_df, engine='train', backend=self.mode_config.backend)
        self.process_module_mapping(bench_df, engine='infer', backend=self.mode_config.backend)
        self.update_forward_call(npu_df)
        return npu_df, bench_df


class Match:
    def __init__(self, mode_config: ModeConfig, mapping_config: MappingConfig, cross_frame):
        self.mode_config = mode_config
        self.mapping_config = mapping_config
        self.cross_frame = cross_frame

    @staticmethod
    def put_unmatched_in_table(match_result, npu_op_item):
        npu_columns = npu_op_item.index.tolist()[:-2]
        bench_columns = [name + '_y' for name in npu_columns]
        na_series = pd.Series([CompareConst.N_A] * len(bench_columns), index=bench_columns)
        new_result_item = pd.concat([npu_op_item, na_series]).to_frame().T
        new_result_item.columns = CompareConst.MATCH_RESULT_COLUMNS
        match_result = pd.concat([match_result, new_result_item])
        return match_result

    @staticmethod
    def put_matched_in_table(match_result, npu_op_item, bench_op_item):
        head_len = len(CompareConst.MATCH_RESULT_COLUMNS)
        new_result_item = pd.concat([npu_op_item, bench_op_item]).head(head_len).to_frame().T
        new_result_item.columns = CompareConst.MATCH_RESULT_COLUMNS
        match_result = pd.concat([match_result, new_result_item])
        return match_result

    @staticmethod
    def rename_api(op_name):
        """
        原api： {api_type}.{api_name}.{API调用次数}.{前向反向}.{input/output}.{参数序号}
        rename后： {api_type}.{api_name}.{前向反向}.{input/output}.{参数序号}
        """
        if Const.FORWARD not in op_name and Const.BACKWARD not in op_name:
            return op_name
        process = Const.FORWARD if Const.FORWARD in op_name else Const.BACKWARD
        name_split = op_name.split(process)
        try:
            torch_func_index, in_out = name_split[0], name_split[1]
        except IndexError as error:
            logger.error(f'{op_name} can not be split with {process}, please check!')
            raise CompareException(CompareException.INDEX_OUT_OF_BOUNDS_ERROR) from error
        torch_func_split = torch_func_index.rsplit(Const.SEP, 2)
        torch_func = str(torch_func_split[0]) + Const.SEP + process + str(in_out)
        return torch_func

    @staticmethod
    def dual_monotonic_sort(match_result, pos1: str, pos2: str):
        """
        对 df 按 (pos1, pos2) 排序，保证：
        - pos1（忽略 NaN）递增, 'npu_pos'
        - pos2（忽略 NaN）递增, 'bench_pos'

        若数据本身存在冲突，抛出 ValueError。
        复杂度：O(n log n)

        核心理解：
        - order1 和 order2 并不是两份不同的数据
        - 它们只是同一个 DataFrame 的“行索引”的两种排序结果
        - 元素集合完全相同（都是 0..n-1），只是顺序不同
        - 本算法是在合并这两种排序约束
        """

        # 索引重排
        match_result = match_result.reset_index(drop=True)

        # ------------------------------------------------
        # 分别按 pos1 / pos2 排序，得到“行索引顺序”
        #
        # 注意：
        # order1 和 order2 中的元素是完全一样的（都是 work 的行号），
        # 只是排序规则不同：
        #   order1: 按 pos1 排序后的行顺序
        #   order2: 按 pos2 排序后的行顺序
        # ------------------------------------------------
        order1 = match_result.sort_values(pos1, na_position="last").index.tolist()
        order2 = match_result.sort_values(pos2, na_position="last").index.tolist()

        used_rows = set()  # 已经选过的“行索引”
        i = j = 0  # 指针：分别在 order1 / order2 上前进
        result_idx = []  # 最终结果行顺序

        n = len(match_result)

        while len(result_idx) < n:
            picked = None

            # 跳过已经用过的行
            while i < n and order1[i] in used_rows:
                i += 1
            while j < n and order2[j] in used_rows:
                j += 1

            # 当前在 pos1 / pos2 意义下“最小的未使用行”
            min1 = order1[i] if i < n else None
            min2 = order2[j] if j < n else None

            # 在 min1 和 min2 中尝试选一个“合法行”
            for k in (min1, min2):
                if k is None or k in used_rows:
                    continue

                row = match_result.loc[k]
                ok = True

                # 如果该行在 pos1 上不是 NaN，
                # 它必须是当前 pos1 中的最小值对应行
                if not pd.isna(row[pos1]) and k != min1:
                    ok = False

                # 如果该行在 pos2 上不是 NaN，
                # 它必须是当前 pos2 中的最小值对应行
                if not pd.isna(row[pos2]) and k != min2:
                    ok = False

                if ok:
                    picked = k
                    break

            # 若两个最小候选都不合法，说明两列排序约束冲突
            if picked is None:
                raise ValueError("Conflicting comparison data:"
                                 " the two columns cannot be monotonically increasing at the same time.")

            used_rows.add(picked)
            result_idx.append(picked)

        # 按 result_idx 给出的顺序取行
        return match_result.loc[result_idx].reset_index(drop=True)

    def check_op_item(self, npu_op_item, bench_op_item):
        name_match = self.rename_api(npu_op_item[CompareConst.CMP_KEY]) == self.rename_api(
            bench_op_item[CompareConst.CMP_KEY])
        if name_match:
            return True
        else:
            npu_op_name = npu_op_item[CompareConst.OP_NAME]
            bench_op_name = bench_op_item[CompareConst.OP_NAME]
            check_op_str_pattern_valid(npu_op_name)
            check_op_str_pattern_valid(bench_op_name)
            logger.warning(f"{npu_op_name} and {bench_op_name} can not fuzzy match")
            return False

    def match_api_infos(self, npu_df, bench_df):
        """
        正常匹配和模糊匹配
        """
        if self.mapping_config.data_mapping:
            match_result = pd.merge(npu_df, bench_df, on=[CompareConst.CMP_KEY], how='left')

            # reorder match_result by op_name of npu
            op_name_order = npu_df[CompareConst.OP_NAME].tolist()
            match_result[CompareConst.OP_NAME_X] = pd.Categorical(match_result[CompareConst.OP_NAME_X],
                                                                  categories=op_name_order, ordered=True)
            match_result = match_result.sort_values(CompareConst.OP_NAME_X).reset_index(drop=True)
            match_result[CompareConst.OP_NAME_X] = match_result[CompareConst.OP_NAME_X].astype('object')
        elif self.mode_config.fuzzy_match:
            common_drop_list = [Const.DIRECTION, Const.CALL_DIRECTION, Const.OP_NO_NUMBER, Const.FORWARD_CALL_ORDER,
                                Const.BACKWARD_CALL_ORDER, Const.SUFFIX]
            npu_drop_list = common_drop_list + (
                [CompareConst.NPU_PARSE_ORDER] if CompareConst.NPU_PARSE_ORDER in npu_df.columns else []
            ) + (
                [CompareConst.OP_NAME_UPDATE] if CompareConst.OP_NAME_UPDATE in npu_df.columns else []
            )
            bench_drop_list = common_drop_list + (
                [CompareConst.BENCH_PARSE_ORDER] if CompareConst.BENCH_PARSE_ORDER in bench_df.columns else []
            ) + (
                [CompareConst.OP_NAME_UPDATE] if CompareConst.OP_NAME_UPDATE in bench_df.columns else []
            )
            npu_df.drop(columns=npu_drop_list, inplace=True, errors="ignore")
            bench_df.drop(columns=bench_drop_list, inplace=True, errors="ignore")
            match_result = self.process_fuzzy_match(npu_df, bench_df)

        # 非fsdp后端暂无layer、class、module_mapping等列，因此增加后端限制
        elif self.mode_config.consistent_check and (
                self.mode_config.backend == Const.FSDP or self.mode_config.backend == Const.MEGATRON):
            match_result = self.consistent_match(npu_df, bench_df)
        else:
            match_result = pd.merge(npu_df, bench_df, on=[CompareConst.CMP_KEY], how='outer')

        return match_result

    def consistent_match(self, npu_df, bench_df):
        keep_list = [CompareConst.OP_NAME, Const.DTYPE, Const.SHAPE, Const.SUMMARY, Const.STACK_INFO, Const.STATE,
                     Const.API_ORIGIN_NAME, Const.REQ_GRAD, Const.CALL_DIRECTION, Const.SUFFIX,
                     Const.LAYER, Const.CLASS, Const.MODULE_MAPPING]
        if self.mode_config.dump_mode == Const.ALL:
            keep_list.append(Const.DATA_NAME)

        npu_df = npu_df[keep_list]
        bench_df = bench_df[keep_list]
        op_name_npu = npu_df[CompareConst.OP_NAME]
        op_name_bench = bench_df[CompareConst.OP_NAME]
        npu_rank = {v: i for i, v in enumerate(op_name_npu)}
        bench_rank = {v: i for i, v in enumerate(op_name_bench)}

        # 1. 根据前向调用、后缀、层数、映射模块名匹配
        merge_df = pd.merge(
            npu_df, bench_df,
            on=[Const.CALL_DIRECTION, Const.SUFFIX, Const.LAYER, Const.MODULE_MAPPING],
            suffixes=('_x', '_y'),
            how='outer'
        )

        matched_df = merge_df

        # 统计量列表转str用于后续拼接
        matched_df['summary_x'] = matched_df['summary_x'].astype(str)
        matched_df['summary_y'] = matched_df['summary_y'].astype(str)

        # 非匹配行summary统一设置为 'N/A'
        npu_unmatched_mask = matched_df['op_name_x'].isna()
        bench_unmatched_mask = matched_df['op_name_y'].isna()
        matched_df.loc[npu_unmatched_mask, 'summary_x'] = CompareConst.N_A
        matched_df.loc[bench_unmatched_mask, 'summary_y'] = CompareConst.N_A

        # 2. 根据推理测op_name聚合
        if self.mode_config.backend == Const.FSDP:
            na_rows = matched_df[matched_df['op_name_y'].isna()]
            agg_rows = matched_df[matched_df['op_name_y'].notna()]
            na_rows.fillna(CompareConst.N_A, inplace=True)
            agg_rows.fillna(CompareConst.N_A, inplace=True)
            # 初始化聚合字典，默认所有列用 'first'  聚合
            agg_dict = {col: 'first' for col in agg_rows.columns}
            # 对 op_name_x 使用自定义聚合方法
            if self.mode_config.xlsx:
                agg_dict['op_name_x'] = lambda x: ';\n'.join(x)
            else:
                agg_dict['op_name_x'] = lambda x: ';  '.join(x)
            agg_dict['dtype_x'] = lambda x: ';'.join(x)
            agg_dict['shape_x'] = lambda x: ';'.join(x)
            agg_dict['summary_x'] = lambda x: ';'.join(x)  # 统计量先按字符串聚合，后续再拆提取

            if self.mode_config.dump_mode == Const.ALL:
                agg_dict['data_name_x'] = lambda x: ';'.join(x)  # 真实数据模式还需要对tensor名进行聚合拼接
            agg_result = agg_rows.groupby('op_name_y', as_index=False, sort=False).agg(agg_dict)
        elif self.mode_config.backend == Const.MEGATRON:
            na_rows = matched_df[matched_df['op_name_x'].isna()]
            agg_rows = matched_df[matched_df['op_name_x'].notna()]
            na_rows.fillna(CompareConst.N_A, inplace=True)
            agg_rows.fillna(CompareConst.N_A, inplace=True)
            # 初始化聚合字典，默认所有列用 'first'  聚合
            agg_dict = {col: 'first' for col in agg_rows.columns}
            # 对 op_name_x 使用自定义聚合方法
            if self.mode_config.xlsx:
                agg_dict['op_name_y'] = lambda x: ';\n'.join(x)
            else:
                agg_dict['op_name_y'] = lambda x: ';  '.join(x)
            agg_dict['dtype_y'] = lambda x: ';'.join(x)
            agg_dict['shape_y'] = lambda x: ';'.join(x)
            agg_dict['summary_y'] = lambda x: ';'.join(x)  # 统计量先按字符串聚合，后续再拆提取

            if self.mode_config.dump_mode == Const.ALL:
                agg_dict['data_name_y'] = lambda x: ';'.join(x)  # 真实数据模式还需要对tensor名进行聚合拼接
            # 根据推理测op_name聚合，聚合连接符为；
            agg_result = agg_rows.groupby('op_name_x', as_index=False, sort=False).agg(agg_dict)

        match_result = pd.concat([agg_result, na_rows])

        # 3. 调整op顺序
        if self.mode_config.backend == Const.FSDP:
            match_result['npu_pos'] = match_result['op_name_x'].str.split(';', n=1).str[0].map(npu_rank)
            match_result['bench_pos'] = match_result['op_name_y'].map(bench_rank)
        elif self.mode_config.backend == Const.MEGATRON:
            match_result['bench_pos'] = match_result['op_name_y'].str.split(';', n=1).str[0].map(bench_rank)
            match_result['npu_pos'] = match_result['op_name_x'].map(npu_rank)

        match_result = self.dual_monotonic_sort(match_result, 'npu_pos', 'bench_pos')
        match_result.drop(columns=['npu_pos', 'bench_pos'], inplace=True)

        return match_result

    def process_fuzzy_match(self, npu_df, bench_df):
        """
        模糊匹配通过循环方式匹配api
        """
        npu_ops_queue = []
        bench_ops_queue = []
        match_result = pd.DataFrame(columns=CompareConst.MATCH_RESULT_COLUMNS)

        max_len = max(len(npu_df), len(bench_df))
        min_len = min(len(npu_df), len(bench_df))
        for i in range(max_len):
            if i < min_len:
                npu_ops_queue.append(npu_df.iloc[i])
                bench_ops_queue.append(bench_df.iloc[i])
            else:
                try:
                    npu_ops_queue.append(npu_df.iloc[i])
                except IndexError:
                    pass
                try:
                    bench_ops_queue.append(bench_df.iloc[i])
                except IndexError:
                    pass

            # 如果append之后queue状态不一致，则判断结束
            if bool(npu_ops_queue) ^ bool(bench_ops_queue):
                break

            npu_match_point, bench_match_point = self.match_op(npu_ops_queue, bench_ops_queue)

            # 如果没有匹配到，数据放到队列中，跳过。直到后面匹配到，把匹配之前的api放到不匹配中
            if npu_match_point == -1 and bench_match_point == -1:
                continue

            npu_op_item = npu_ops_queue[npu_match_point]
            bench_op_item = bench_ops_queue[bench_match_point]
            unmatched_data = npu_ops_queue[0: npu_match_point]
            for op_item in unmatched_data:
                match_result = self.put_unmatched_in_table(match_result, op_item)
            match_result = self.put_matched_in_table(match_result, npu_op_item, bench_op_item)
            del npu_ops_queue[0: npu_match_point + 1]
            del bench_ops_queue[0: bench_match_point + 1]

        if npu_ops_queue:
            for op_item in npu_ops_queue:
                match_result = self.put_unmatched_in_table(match_result, op_item)

        match_result.reset_index(drop=True, inplace=True)
        return match_result

    def match_op(self, npu_queue, bench_queue):
        for b_index, b_op in enumerate(bench_queue[0: -1]):
            if self.check_op_item(npu_queue[-1], b_op):
                return len(npu_queue) - 1, b_index
        if self.check_op_item(npu_queue[-1], bench_queue[-1]):
            return len(npu_queue) - 1, len(bench_queue) - 1
        for n_index, n_op in enumerate(npu_queue[0: -1]):
            if self.check_op_item(n_op, bench_queue[-1]):
                return n_index, len(bench_queue) - 1
        return -1, -1

    def gen_dtype_condition(self, match_result):
        """
        dtype匹配条件为npu、bench的dtype一致或属于规定的映射关系
        """
        # 如果使用了data_mapping，不校验dtype，返回全True的DataFrame
        if self.mapping_config.data_mapping:
            return pd.Series(True, index=match_result.index)

        npu_dtype = match_result['dtype_x']
        bench_dtype = match_result['dtype_y']
        npu_dtype = self.process_cross_frame_dtype(npu_dtype)
        bench_dtype = self.process_cross_frame_dtype(bench_dtype)

        equal_condition = npu_dtype == bench_dtype
        match_condition = (
                (npu_dtype.isin(CompareConst.DTYPE_MATCH_GROUPS[0]) & bench_dtype.isin(
                    CompareConst.DTYPE_MATCH_GROUPS[0])) |
                (npu_dtype.isin(CompareConst.DTYPE_MATCH_GROUPS[1]) & bench_dtype.isin(
                    CompareConst.DTYPE_MATCH_GROUPS[1]))
        )
        return equal_condition | match_condition

    def process_cross_frame_dtype(self, dtype):
        if self.cross_frame:
            dtype = dtype.map(cross_dtype_mapping).fillna(dtype)
        return dtype


class CreateTable:
    def __init__(self, mode_config: ModeConfig):
        self.mode_config = mode_config

    @staticmethod
    def process_all_mode_addition_header(result):
        result['data_name_x'] = result.apply(lambda row: [row['data_name_x'], row['data_name_y']], axis=1)
        result['dirty_valid_len_x'] = result.apply(lambda row: [row['dirty_valid_len_x'], row['dirty_valid_len_y']], 
                                                   axis=1)
        return result

    @staticmethod
    def summary_list_str_to_list(s: str):
        if s.lower() == 'nan':
            return [CompareConst.NAN] * 4
        summary = []
        s = s.strip().strip("[]")
        parts = s.split(",")
        for p in parts:
            p = p.strip()
            try:
                p = float(p)
            except ValueError:
                p = p.strip('"').strip("'")
            summary.append(p)
        return summary

    @staticmethod
    def filter_numbers(lst):
        """只保留能转成数字的"""
        nums = []
        for x in lst:
            if isinstance(x, float):
                nums.append(x)
        return nums

    @staticmethod
    def fill_state_api_origin_name(result):
        """
        训推一致性比对场景，使用推理数据填充训练npu列的N/A，用于后续比对指标计算能找到唯一op
        state直接填回，api_origin_name根据映射得到对应训练op_name填回
        """
        mask = result['state_x'] == CompareConst.N_A
        result.loc[mask, 'state_x'] = result.loc[mask, 'state_y']

        # 先算 N/A 行的 mask
        mask_n_a = result['api_origin_name_x'] == CompareConst.N_A

        # 1. 用非 N/A 行构建映射字典
        mapping = (
            result.loc[~mask_n_a, ['api_origin_name_y', 'api_origin_name_x']]
            .drop_duplicates()
            .set_index('api_origin_name_y')['api_origin_name_x']
            .to_dict()
        )
        if CompareConst.N_A in mapping:
            del mapping[CompareConst.N_A]

        # 2. 用映射字典回填 N/A（如果字典里没有对应 key，就用 api_origin_name_y 本身）
        mapped = result.loc[mask_n_a, 'api_origin_name_y'].map(mapping)
        result.loc[mask_n_a, 'api_origin_name_x'] = mapped.fillna(result.loc[mask_n_a, 'api_origin_name_y'])

        return result

    def parse_statistic_str(self, s: str):
        """
        处理形如：
        '[2.3, -2.9, -3e-05, 49];[2.3, -2.9, -3e-05, 49];[2.3, -2.9, -3e-05, 49]'
        返回：
        [max_of_max, min_of_min, joined_mean, joined_norm]
        """
        if ';' not in s:
            summary_list = self.summary_list_str_to_list(s)
            return summary_list

        values = []
        parts = s.split(";")
        for p in parts:
            p_list = self.summary_list_str_to_list(p)
            if p_list == [CompareConst.NAN] * 4:
                return [CompareConst.NAN] * 4
            values.append(p_list)

        summary_cols = [list(x) for x in zip(*values)]
        if len(summary_cols) != len(Const.SUMMARY_METRICS_LIST):
            return [CompareConst.NAN] * 4

        # 第1列：取最大值
        col1_nums = self.filter_numbers(summary_cols[0])
        max_val = max(col1_nums) if col1_nums else CompareConst.NONE

        # 第2列：取最小值
        col2_nums = self.filter_numbers(summary_cols[1])
        min_val = min(col2_nums) if col2_nums else CompareConst.NONE

        # 第3、第4列直接用
        col3 = ';'.join([str(x) for x in summary_cols[2]])
        col4 = ';'.join([str(x) for x in summary_cols[3]])

        # 拼接结果
        result = [max_val, min_val, col3, col4]
        return result

    def set_summary(self, summary):
        if summary == CompareConst.N_A:
            return [CompareConst.N_A] * 4  # 4为统计值个数

        summary_list = []

        # 训推一致性场景将统计量列表转了字符串，需要转回来
        if isinstance(summary, str) or str(summary).lower() == 'nan':
            summary = self.parse_statistic_str(str(summary))

        for i in summary:
            if str(i).lower() == 'nan':
                summary_list.append(CompareConst.NAN)
            else:
                summary_list.append(i)
        return summary_list

    def make_result_df(self, result):
        # get header
        header = CompareConst.HEAD_OF_COMPARE_MODE[self.mode_config.dump_mode][:]
        if self.mode_config.stack_mode:
            header.append(CompareConst.STACK)
        if self.mode_config.dump_mode == Const.ALL:
            header.append(CompareConst.DATA_NAME)
            header.append(CompareConst.DIRTY_VALID_LEN)
            result = self.process_all_mode_addition_header(result)

        if self.mode_config.consistent_check:
            result = self.fill_state_api_origin_name(result)

        # rename match_result columns
        result.rename(columns={'op_name_x': CompareConst.NPU_NAME,
                               'op_name_y': CompareConst.BENCH_NAME,
                               'dtype_x': CompareConst.NPU_DTYPE,
                               'dtype_y': CompareConst.BENCH_DTYPE,
                               'shape_x': CompareConst.NPU_SHAPE,
                               'shape_y': CompareConst.BENCH_SHAPE,
                               'md5_x': CompareConst.NPU_MD5,
                               'md5_y': CompareConst.BENCH_MD5,
                               'data_name_x': CompareConst.DATA_NAME,
                               'stack_info_x': CompareConst.STACK,
                               'state_x': Const.STATE,
                               'api_origin_name_x': Const.API_ORIGIN_NAME,
                               'requires_grad_x': CompareConst.NPU_REQ_GRAD,
                               'requires_grad_y': CompareConst.BENCH_REQ_GRAD,
                               'dirty_valid_len_x': CompareConst.DIRTY_VALID_LEN
                               },
                      inplace=True)

        # process requires_grad
        result[CompareConst.REQ_GRAD_CONSIST] = result[CompareConst.NPU_REQ_GRAD] == result[CompareConst.BENCH_REQ_GRAD]

        # process summary data
        if result.empty:
            result[CompareConst.NPU_SUMMARY] = pd.DataFrame(columns=CompareConst.NPU_SUMMARY)
            result[CompareConst.BENCH_SUMMARY] = pd.DataFrame(columns=CompareConst.BENCH_SUMMARY)
        else:
            result[CompareConst.NPU_SUMMARY] = result['summary_x'].apply(self.set_summary).tolist()
            result[CompareConst.BENCH_SUMMARY] = result['summary_y'].apply(self.set_summary).tolist()

        header.extend([Const.STATE, Const.API_ORIGIN_NAME])
        result_df = pd.DataFrame(columns=header)
        for h in header:
            if h in result.columns:
                result_df[h] = result[h]
        return result_df, header


class CalcStatsDiff:
    def __init__(self, mode_config: ModeConfig):
        self.mode_config = mode_config
        self.rules = None
        self.build_rules()
        self.DEFAULT_RULE = self.static_diff(CompareConst.N_A)

    @staticmethod
    def is_same_value(a: pd.Series, b: pd.Series) -> pd.Series:
        """
        检查是否相等
        """
        return a.astype(str).eq(b.astype(str))

    @staticmethod
    def is_number(val: pd.Series) -> pd.Series:
        """
        检查是否为有效的数值，并排除布尔类型
        """
        val_str = val.astype(str)
        mask_bool = val_str.str.lower().eq('true') | val_str.str.lower().eq('false')
        mask_numeric = pd.to_numeric(val, errors='coerce').notna()
        return mask_numeric & ~mask_bool

    @staticmethod
    def is_nan(val: pd.Series) -> pd.Series:
        """
        检查是否为字符串形式的 'nan' (包括大小写) 或实际的 NaN
        """
        val_str = val.astype(str)
        return val.isna() | val_str.str.lower().eq('nan')

    @staticmethod
    def is_inf(val: pd.Series) -> pd.Series:
        """
        检查是否为正无穷 (inf)
        """
        val_str = val.astype(str)
        return (val == np.inf) | val_str.str.lower().eq('inf')

    @staticmethod
    def is_neg_inf(val: pd.Series) -> pd.Series:
        """
        检查是否为负无穷 (-inf)
        """
        val_str = val.astype(str)
        return (val == -np.inf) | val_str.str.lower().eq('-inf')

    @staticmethod
    def is_device(val: pd.Series) -> pd.Series:
        """
        检查是否包含 'npu', 'cpu' 或 'cuda' 字符串
        """
        return val.astype(str).str.contains('npu|cpu|cuda', case=False, na=False)

    @staticmethod
    def is_na(val: pd.Series) -> pd.Series:
        """
        检查是否为 N/A
        """
        return val.astype(str).eq(CompareConst.N_A)

    @staticmethod
    def rule_num_num(npu_num: pd.Series, bench_num: pd.Series):
        diff = npu_num - bench_num
        rel = pd.Series(CompareConst.INF, index=diff.index)
        mask_nonzero = bench_num != 0
        rel.loc[mask_nonzero] = (diff[mask_nonzero] / bench_num[mask_nonzero] * 100).abs().astype(str) + "%"
        return diff, rel

    @staticmethod
    def static_diff(diff: str, rel=None):
        if rel is None:
            rel = diff
        return diff, rel

    @staticmethod
    def get_number(val):
        return pd.to_numeric(val.astype(str), errors='coerce')

    def build_rules(self):
        """
        创建npu、bench不相等规则
        """
        # NUM × NUM
        self.rules = {(ValType.NUM, ValType.NUM): self.rule_num_num}

        # ---------- NAN 规则 ----------
        nan_rule = self.static_diff(CompareConst.NAN)
        nan_range = {ValType.NUM, ValType.NAN, ValType.INF, ValType.NEG_INF}
        for t in nan_range:
            self.rules[(ValType.NAN, t)] = nan_rule
            self.rules[(t, ValType.NAN)] = nan_rule

        # ---------- INF / -INF 规则 ----------
        pos_inf = self.static_diff(CompareConst.INF)
        neg_inf = self.static_diff(CompareConst.NEG_INF)
        # INF / -INF 在左
        for r in (ValType.NUM, ValType.NEG_INF):
            self.rules[(ValType.INF, r)] = pos_inf
        for r in (ValType.NUM, ValType.INF):
            self.rules[(ValType.NEG_INF, r)] = neg_inf
        # INF / -INF 在右
        self.rules[(ValType.NUM, ValType.INF)] = neg_inf
        self.rules[(ValType.NUM, ValType.NEG_INF)] = pos_inf

        # ---------- N/A 规则 ----------
        na_rule = self.static_diff(CompareConst.N_A)
        na_range = {ValType.NUM, ValType.NAN, ValType.INF, ValType.NEG_INF, ValType.DEVICE, ValType.NA, ValType.OTHER}
        for t in na_range:
            self.rules[(ValType.NA, t)] = na_rule
            self.rules[(t, ValType.NA)] = na_rule

        # ---------- DEVICE / OTHER 规则 ----------
        # 给diff，都是device给N/A
        diff_rule = self.static_diff(CompareConst.DIFF_FLAG)
        diff_range = {ValType.NUM, ValType.NAN, ValType.INF, ValType.NEG_INF, ValType.DEVICE, ValType.OTHER}
        for t in diff_range:
            self.rules[(ValType.DEVICE, t)] = diff_rule
            self.rules[(t, ValType.DEVICE)] = diff_rule
            self.rules[(ValType.OTHER, t)] = diff_rule
            self.rules[(t, ValType.OTHER)] = diff_rule
        self.rules[(ValType.DEVICE, ValType.DEVICE)] = na_rule

    def classify(self, val: pd.Series) -> pd.Series:
        result_values = np.select(
            [
                self.is_nan(val),
                self.is_inf(val),
                self.is_neg_inf(val),
                self.is_number(val),  # 'inf', '-inf'会被pandas认为是number，所以放在inf/-inf判断后面
                self.is_device(val),
                self.is_na(val),
            ],
            [
                ValType.NAN.value,
                ValType.INF.value,
                ValType.NEG_INF.value,
                ValType.NUM.value,
                ValType.DEVICE.value,
                ValType.NA.value,
            ],
            default=ValType.OTHER.value,
        )
        val_to_enum = {t.value: t for t in ValType}
        result_series = pd.Series(result_values, index=val.index).map(val_to_enum)
        return result_series

    def calc_summary_diff(self, result_df, stats_index: str):
        npu_val = result_df['NPU ' + stats_index]
        bench_val = result_df['Bench ' + stats_index]
        diff_name = stats_index.capitalize() + ' diff'
        rel_err_name = ('norm' if stats_index == 'l2norm' else stats_index).capitalize() + 'RelativeErr'

        # ---------------------- 初始化 ----------------------
        result_df[[diff_name, rel_err_name]] = CompareConst.N_A

        # ---------------------- 基础 mask ----------------------
        npu_num = self.get_number(npu_val)
        bench_num = self.get_number(bench_val)
        mask_equal = self.is_same_value(npu_val, bench_val)
        mask_unequal = ~mask_equal

        # ---------------------- npu, bench统计量相等 ----------------------
        result_df.loc[mask_equal, [diff_name, rel_err_name]] = '0'

        # ---------------------- npu, bench统计量不相等 ----------------------
        npu_type = self.classify(npu_val)
        bench_type = self.classify(bench_val)
        for t1 in ALL_TYPES:
            for t2 in ALL_TYPES:
                mask = mask_unequal & (npu_type == t1) & (bench_type == t2)
                if not mask.any():
                    continue

                rule = self.rules.get((t1, t2), self.DEFAULT_RULE)

                if callable(rule):
                    diff, rel = rule(npu_num[mask], bench_num[mask])
                else:
                    diff, rel = rule

                if isinstance(diff, str):
                    result_df.loc[mask, diff_name] = diff
                else:
                    result_df.loc[mask, diff_name] = diff.astype(str)
                if isinstance(rel, str):
                    result_df.loc[mask, rel_err_name] = rel
                else:
                    result_df.loc[mask, rel_err_name] = rel.astype(str)

    def calc_accuracy(self, result_df, header):
        # bench name N/A represents no bench data, err_msg adds "No bench data matched."
        condition_no_bench = result_df[CompareConst.BENCH_NAME] == CompareConst.N_A
        result_df[condition_no_bench] = result_df[condition_no_bench].fillna(CompareConst.N_A)
        result_df.loc[condition_no_bench, CompareConst.ERROR_MESSAGE] = CompareConst.NO_BENCH
        condition_req_grad_consist = result_df[CompareConst.NPU_REQ_GRAD] == result_df[CompareConst.BENCH_REQ_GRAD]

        if self.mode_config.dump_mode == Const.MD5:
            condition_md5_equal = result_df[CompareConst.NPU_MD5] == result_df[CompareConst.BENCH_MD5]
            result_df.loc[condition_md5_equal, CompareConst.RESULT] = CompareConst.PASS
            result_df.loc[~condition_md5_equal & ~condition_no_bench, CompareConst.RESULT] = CompareConst.DIFF
        elif self.mode_config.first_diff_analyze or self.mode_config.dump_mode == Const.SUMMARY:
            for stats_index in ['max', 'min', 'mean', 'l2norm']:
                self.calc_summary_diff(result_df, stats_index)
            result_df.loc[~condition_no_bench, [CompareConst.RESULT, CompareConst.ERROR_MESSAGE]] = ''
            result_df.loc[~condition_req_grad_consist, CompareConst.ERROR_MESSAGE] += 'Requires_grad inconsistent. '
        else:
            fill_cols = [CompareConst.COSINE, CompareConst.EUC_DIST,
                         CompareConst.MAX_ABS_ERR, CompareConst.MAX_RELATIVE_ERR,
                         CompareConst.ONE_THOUSANDTH_ERR_RATIO, CompareConst.FIVE_THOUSANDTHS_ERR_RATIO,
                         CompareConst.ERROR_MESSAGE]
            result_df.loc[~condition_no_bench, fill_cols] = ''  # 默认填充'', df默认省缺值为nan，不便后续处理，容易出现意外情况
            result_df.loc[~condition_req_grad_consist, CompareConst.ERROR_MESSAGE] = 'Requires_grad inconsistent. '

        return result_df[header]


def setup_comparison(input_param, output_path, **kwargs) -> ComparisonConfig:
    """公共的前置处理逻辑，返回封装后的 ComparisonConfig 对象"""
    try:
        config = ComparisonConfig(
            dump_mode='',
            stack_mode=False,
            fuzzy_match=kwargs.get('fuzzy_match', False),
            data_mapping=kwargs.get('data_mapping', {}),
            suffix=kwargs.get('suffix', ''),
            cell_mapping=kwargs.get('cell_mapping', {}),
            api_mapping=kwargs.get('api_mapping', {}),
            layer_mapping=kwargs.get('layer_mapping', {}),
            xlsx=kwargs.get('xlsx', False),
            first_diff_analyze=kwargs.get('first_diff_analyze', False),
            compared_file_type='',
            is_print_compare_log=kwargs.get('is_print_compare_log', False),
            consistent_check=kwargs.get('consistent_check', False),
            backend=kwargs.get('backend', '')
        )

        set_dump_path(input_param)
        config.dump_mode = get_dump_mode(input_param)
        config.compared_file_type = get_file_type(input_param.get("npu_path", None))

        # set stack_mode and set "stack_path" in input_param
        config.stack_mode, input_param = set_stack_json_path(input_param)

        check_configuration_param(config)
        create_directory(output_path)
        check_compare_param(input_param, output_path, config.dump_mode, config.stack_mode)
        check_consistent_param(config.consistent_check, config.backend)

        return config

    except (CompareException, FileCheckException) as error:
        logger.error('Compare failed. Please check the arguments and do it again!')
        raise CompareException(error.code) from error
