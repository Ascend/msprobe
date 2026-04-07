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

import re
import json
from typing import List

from msprobe.core.common.const import Const, CompareConst
from msprobe.core.common.log import logger
from msprobe.core.compare.indicator_analysis.utils import CompareMode, ResultLevel, divide_result_df, IgnoreInfo
from msprobe.core.compare.indicator_analysis.algorithm import BaseAlgorithm, TENSOR_CHECKERS, STATISTICS_CHECKERS, \
    MD5_CHECKERS, STATISTICS_CHECKERS_PARALLEL_MERGE, VERL_TENSOR_CHECKERS
from msprobe.core.compare.indicator_analysis.api_data import ApiData


class ApiIndicatorCalculator:
    RANK_SUFFIX_PATTERN = re.compile(r'_rank\d+$')

    def __init__(self, mode, backend, parallel_merge=False, consistent_check=False):
        self.mode = mode
        self.parallel_merge = parallel_merge
        self.consistent_check = consistent_check
        self.backend = backend
        self.all_ignore_set = {'empty', 'empty_like', 'numpy', 'to', '__setitem__', 'empty_with_format',
                               'new_empty_strided', 'new_empty', 'empty_strided'}
        self.input_ignore_set = {'_reduce_scatter_base', '_all_gather_base', 'all_to_all_single', 'batch_isend_irecv'}
        self.algorithms: List[BaseAlgorithm] = []
        self._add_algorithm()

    @staticmethod
    def get_api_indicator_and_msg(api_data: ApiData):
        """
        基于all_data_lists（api或模块的所有参数的数据）得到一个api或模块的指标和异常信息

        indicator取所有参数最差的（error）
        err msg取所有参数汇总

        Return:
            精度比对指标（pass/warning/error）
        """
        all_data_lists = api_data.input_data + api_data.output_data
        final_indicator = ResultLevel.PASS
        for data_list in all_data_lists:
            indicator = api_data.get_data_by_header(CompareConst.RESULT, data_list)
            if isinstance(indicator, ResultLevel):
                api_data.set_data_by_header(CompareConst.RESULT, data_list, indicator.value)
                if indicator > final_indicator:
                    final_indicator = indicator
            err_msg = api_data.get_data_by_header(CompareConst.ERROR_MESSAGE, data_list)
            if isinstance(err_msg, list):
                api_data.set_data_by_header(CompareConst.ERROR_MESSAGE, data_list, json.dumps(err_msg))

        return final_indicator.value

    def get_api_ignore_info(self, api_data: ApiData):
        """
        api是否需要忽略判断规则的情况
        """
        if not api_data.input_data:
            return IgnoreInfo.NO_IGNORE
        npu_param_name = api_data.get_data_by_header(CompareConst.NPU_NAME, api_data.input_data[0])
        name_split = npu_param_name.split(Const.SEP)
        if len(name_split) < 2:
            return IgnoreInfo.NO_IGNORE
        api_name = self.RANK_SUFFIX_PATTERN.sub('', name_split[1]) if self.parallel_merge else name_split[1]
        if api_name in self.all_ignore_set:
            return IgnoreInfo.ALL_IGNORE
        elif api_name in self.input_ignore_set:
            return IgnoreInfo.INPUT_IGNORE

        return IgnoreInfo.NO_IGNORE

    def calculate(self, raw_data_list: List[List]):
        """
        计算入口
        """
        api_data = ApiData(self.mode, raw_data_list)

        ignore_info = self.get_api_ignore_info(api_data)

        self.execute_all(api_data, ignore_info, self.backend)

        return self.get_api_indicator_and_msg(api_data)

    def add_algorithm(self, algorithm: BaseAlgorithm):
        if not isinstance(algorithm, BaseAlgorithm):
            msg = 'It must be an instance of a subclass of BaseAlgorithm.'
            logger.error(msg)
            raise TypeError(msg)
        self.algorithms.append(algorithm)

    def execute_all(self, api_data: ApiData, ignore_info: IgnoreInfo, backend: str):
        for algorithm in self.algorithms:
            try:
                algorithm.run(api_data, ignore_info, backend)
            except Exception as e:
                msg = f'Run algorithm failed.'
                logger.error(msg)
                raise RuntimeError(msg) from e

    def _add_algorithm(self):
        if self.mode == CompareMode.STATISTICS.value:
            checkers = STATISTICS_CHECKERS_PARALLEL_MERGE if self.parallel_merge else STATISTICS_CHECKERS
            for checker in checkers:
                self.add_algorithm(checker())
        elif self.mode == CompareMode.TENSOR.value:
            if self.consistent_check:
                checkers = VERL_TENSOR_CHECKERS
            else:
                checkers = TENSOR_CHECKERS
            for checker in checkers:
                self.add_algorithm(checker())
        elif self.mode == CompareMode.MD5.value:
            for checker in MD5_CHECKERS:
                self.add_algorithm(checker())


def calculate_excel_result_df(result_df, mode, backend, consistent_check=False, chunk_size=1000):
    """
    仅适用于excel比对场景，得到表格每行数据的精度比对指标（pass/warning/error）

    Args:
        result_df: DataFrame数据结构，即转换成excel前的表单结构
        mode: 比对模式，分为 tensor 模式、统计量模式和 md5 模式
        consistent_check: 是否使用verl训推一致指定比对算法
        chunk_size: 分块赋值参数，默认1000，把 result 分成小块，逐块赋值给 result_df，这样每次只占用小块内存，避免内存峰值过高
    """
    result_dict = divide_result_df(result_df)
    calculator = ApiIndicatorCalculator(mode, backend, consistent_check=consistent_check)
    calculated_result_lists = []
    for data_lists in result_dict.values():
        calculator.calculate(data_lists)
        calculated_result_lists.extend(data_lists)

    head = CompareConst.HEAD_OF_COMPARE_MODE.get(mode)
    if not head:
        logger.error(f'Unable to obtain header based on compare mode: {mode}')
        raise RuntimeError()
    # 配置列映射关系：[(result_df的目标列名, result子列表的列索引)]
    try:
        cols_mapping = [
            (CompareConst.RESULT, head.index(CompareConst.RESULT)),
            (CompareConst.ERROR_MESSAGE, head.index(CompareConst.ERROR_MESSAGE))
        ]
    except ValueError as e:
        logger.error(f'The {CompareConst.RESULT} or {CompareConst.ERROR_MESSAGE} does not exist in the header: {e}')
        raise e

    total_rows = len(calculated_result_lists)

    # 分块逐批赋值，降低内存瞬时峰值
    for i in range(0, total_rows, chunk_size):
        end_idx = min(i + chunk_size, total_rows)
        current_result_chunk = calculated_result_lists[i:end_idx]

        for df_col_name, result_col_idx in cols_mapping:
            col_data = [sublist[result_col_idx] for sublist in current_result_chunk]
            df_col_idx = result_df.columns.get_loc(df_col_name)
            result_df.iloc[i:end_idx, df_col_idx] = col_data


def calculate_result(result, mode, parallel_merge=False):
    """
    得到一个api或模块的指标和异常信息

    Args:
        result: List[List]数据结构，每个list元素代表api或模块参数的具体信息
        mode: 比对模式，分为 tensor 模式、统计量模式和 md5 模式
        parallel_merge: 是否为不同切分策略图合并比对场景，默认False

    Return:
        精度比对指标（pass/warning/error）
    """
    calculator = ApiIndicatorCalculator(mode, '', parallel_merge=parallel_merge)
    return calculator.calculate(result)
