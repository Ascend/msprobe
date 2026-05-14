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
import json
import re
import os
from typing import List

from msprobe.core.compare.acc_compare import ModeConfig
from msprobe.core.compare.multiprocessing_compute import CompareRealData, CompareInfo
from msprobe.core.compare.utils import read_op, merge_tensor, get_accuracy, make_result_table, \
    get_rela_diff_summary_mode
from msprobe.core.common.utils import set_dump_path, get_dump_mode
from msprobe.visualization.utils import GraphConst
from msprobe.core.common.const import Const, CompareConst
from msprobe.core.compare.indicator_analysis.calculator import calculate_result
from msprobe.core.common.file_utils import FileChecker, FileCheckConst
from msprobe.core.common.log import BaseLogger, logger

# 用于将节点名字解析成对应的NodeOp的规则
op_patterns = [
    # NodeOp.module
    r'^(Module.|Cell.|optimizer|clip_grad|DefaultModel)',
    # NodeOp.function_api
    r'^(Tensor.|Torch.|Functional.|NPU.|VF.|Distributed.|Aten.|Mint.|Primitive.|Jit.|MintFunctional.|MindSpeed.)'
]


def get_compare_mode(dump_path_param):
    """
    获得比较模式，包括summary、MD5和真实数据三种模式
    Args:
        dump_path_param: 调用acc_compare接口所依赖的参数
    Returns: 0 summary mode, 1 md5 mode, 2 true data mode
    """
    set_dump_path(dump_path_param)
    dump_mode = get_dump_mode(dump_path_param)
    compare_mode = GraphConst.DUMP_MODE_TO_GRAPHCOMPARE_MODE_MAPPING.get(dump_mode)
    return compare_mode


def run_real_data(dump_path_param, csv_path, framework, is_cross_frame=False):
    """
    多进程运行生成真实数据
    Args:
        dump_path_param: 调用acc_compare接口所依赖的参数
        csv_path: 生成文件路径
        framework: 框架类型, pytorch或mindspore
        is_cross_frame: 是否进行跨框架比对，仅支持mindspore比pytorch, 其中pytorch为标杆
    """
    config_dict = {
        'stack_mode': False,
        'auto_analyze': True,
        'fuzzy_match': False,
        'dump_mode': Const.ALL
    }
    mode_config = ModeConfig(**config_dict)

    if framework == Const.PT_FRAMEWORK:
        from msprobe.pytorch.compare.pt_compare import read_real_data
        return CompareRealData(read_real_data, mode_config, is_cross_frame).do_multi_process(dump_path_param, csv_path)
    else:
        from msprobe.mindspore.compare.ms_compare import read_real_data
        return CompareRealData(read_real_data, mode_config, is_cross_frame).do_multi_process(dump_path_param, csv_path)


def get_input_output(node_data, node_id):
    """
    将dump的原始数据进行拆解，分解为output和input两个数据
    Args:
        node_data: 属于单个节点的dump数据
        node_id: 节点名字
    """
    input_data = {}
    output_data = {}
    op_parsed_list = read_op(node_data, node_id)
    for item in op_parsed_list:
        full_op_name = item.get('full_op_name', '')
        if not full_op_name:
            continue
        if GraphConst.OUTPUT in full_op_name and GraphConst.INPUT not in full_op_name:
            output_data[full_op_name] = item
        else:
            name = item.get('data_name')
            # 节点参数名称尽量使用落盘数据的名称
            if isinstance(name, str) and name != '-1':
                input_data[name.rsplit(Const.SEP, 1)[0]] = item
            else:
                input_data[full_op_name] = item
    return input_data, output_data


def format_node_data(data_dict, node_id=None, compare_mode=None):
    """
    删除节点数据中不需要展示的字段
    """
    del_list = ['state', 'full_op_name']
    if GraphConst.MD5_COMPARE != compare_mode:
        del_list.append(Const.MD5)
    if node_id and GraphConst.BATCH_P2P in node_id:
        del_list.extend(['op', 'peer', 'tag', 'group_id'])
    for _, value in data_dict.items():
        if not isinstance(value, dict):
            continue
        for item in del_list:
            if item in value:
                del value[item]
        _format_data(value)
    return data_dict


def compare_node(node_n, node_b, compare_mode):
    """
    调用acc_compare.py中的get_accuracy获得精度对比指标
    真实数据对比模式无法获得精度对比指标，需要调用多进程比对接口
    Returns: 包含参数信息和对比指标（真实数据对比模式除外）的list
    """
    dump_mode = GraphConst.GRAPHCOMPARE_MODE_TO_DUMP_MODE_TO_MAPPING.get(compare_mode)
    merge_n = _parse_node(node_n, dump_mode)
    merge_b = _parse_node(node_b, dump_mode)
    result = []
    get_accuracy(result, merge_n, merge_b, dump_mode)
    return result


def get_api_indicator_info(compare_mode, result, parallel_merge=False):
    """
    得到一个api或模块的指标和异常信息
    """
    dump_mode = GraphConst.GRAPHCOMPARE_MODE_TO_DUMP_MODE_TO_MAPPING.get(compare_mode)
    return calculate_result(result, dump_mode, parallel_merge)


def get_real_api_data_list(node, compare_data_dict: dict):
    """
    真实数据模式下，得到的比对结果包含了所有api参数，需要转换为其中某一个api参数列表
    """
    api_data_list = []
    keys = list(node.input_data.keys()) + list(node.output_data.keys())
    for key in keys:
        if key in compare_data_dict:
            api_data_list.append(compare_data_dict.get(key))
    return api_data_list


def _parse_node(node, dump_mode):
    """
    转换节点，使其能够作为acc_compare.py中的get_accuracy的入参
    """
    op_parsed_list = []
    op_parsed_list.extend(node.input_data.values())
    op_parsed_list.extend(node.output_data.values())
    result = merge_tensor(op_parsed_list, dump_mode)
    if not result:
        result['op_name'] = []
    return result


def _format_decimal_string(s):
    """
    使用正则表达式匹配包含数字、小数点和可选的百分号的字符串
    """
    pattern = re.compile(r'^\d{1,20}\.\d{1,20}%?$')
    matches = pattern.findall(s)
    for match in matches:
        is_percent = match.endswith('%')
        number_str = match.rstrip('%')
        decimal_part = number_str.split('.')[1]
        # 如果小数位数大于6，进行处理
        if len(decimal_part) > GraphConst.ROUND_TH:
            number_float = float(number_str)
            formatted_number = f"{number_float:.{GraphConst.ROUND_TH}f}"
            # 如果原来是百分数，加回百分号
            if is_percent:
                formatted_number += '%'
            # 替换原字符串中的数值部分
            s = s.replace(match, formatted_number)
    return s


def _format_data(data_dict):
    """
    格式化数据，处理一些异常值
    """
    all_null = False

    keys_to_keep = ['type', 'group_ranks', 'group_id', 'data_name']
    if data_dict.get('type') == 'torch.ProcessGroup':
        keys_to_remove = [key for key in data_dict if key not in keys_to_keep]
        for key in keys_to_remove:
            del data_dict[key]

    for key, value in data_dict.items():
        if isinstance(value, str):
            # 将单引号删掉，None换成null避免前端解析错误
            value = value.replace("'", "").replace(GraphConst.NONE, GraphConst.NULL)
            value = _format_decimal_string(value)
        elif value is None or value == ' ':
            value = GraphConst.NULL
        # Inf会走入这里，确保转成Inf。另外给其他不符合预期的类型做兜底方案
        if key != GraphConst.ERROR_KEY:
            # 除了error_key不转str，其他都转str, 避免前端解析错误
            value = str(value)
        # max为null, 意味着这个参数值为null
        if key == Const.MAX and value == GraphConst.NULL:
            all_null = True
        data_dict[key] = value
    # 字典里的value全null，只保留一个null
    if all_null:
        data_dict.clear()
        data_dict[GraphConst.VALUE] = GraphConst.NULL


def get_csv_df(stack_mode, csv_data, compare_mode):
    """
    调用acc接口写入csv
    """

    dump_mode = GraphConst.GRAPHCOMPARE_MODE_TO_DUMP_MODE_TO_MAPPING.get(compare_mode)
    return make_result_table(csv_data, dump_mode, stack_mode)


class MatchedNodeCalculator:
    """
    功能：用户在前端选择NPU和Bench节点匹配，前端会查询db数据打包发送到此类，此类进行精度指标的计算，最终返回结果给前端
    """
    TENSOR_COMPARE_INDEX = CompareConst.ALL_COMPARE_INDEX + CompareConst.EXTRACT_INDEX
    STATISTICS_COMPARE_INDEX = CompareConst.SUMMARY_COMPARE_INDEX + CompareConst.EXTRACT_INDEX

    _RESP_SUCCESS = "success"
    _RESP_ERROR = "error"
    _RESP_DATA = "data"
    _INPUT_INFO = "input_info"
    _OUTPUT_INFO = "output_info"

    def __init__(self, npu_db_data, bench_db_data):
        """
        npu_db_data / bench_db_data: 基于分级可视化构图结果db文件，查询input_data、output_data和dump_data_dir得到的数据
        """
        self.npu_db_data = npu_db_data
        self.bench_db_data = bench_db_data
        self.framework = None
        self.is_cross_frame = False
        self.public_indicators_mapping = {
            Const.DTYPE: (CompareConst.NPU_DTYPE, CompareConst.BENCH_DTYPE),
            Const.SHAPE: (CompareConst.NPU_SHAPE, CompareConst.BENCH_SHAPE),
            Const.REQ_GRAD: (CompareConst.NPU_REQ_GRAD, CompareConst.BENCH_REQ_GRAD),
            Const.MAX: (CompareConst.NPU_MAX, CompareConst.BENCH_MAX),
            Const.MIN: (CompareConst.NPU_MIN, CompareConst.BENCH_MIN),
            Const.MEAN: (CompareConst.NPU_MEAN, CompareConst.BENCH_MEAN),
            Const.NORM: (CompareConst.NPU_NORM, CompareConst.BENCH_NORM)
        }
        self.md5_public_indicators_mapping = {
            Const.DTYPE: (CompareConst.NPU_DTYPE, CompareConst.BENCH_DTYPE),
            Const.SHAPE: (CompareConst.NPU_SHAPE, CompareConst.BENCH_SHAPE),
            Const.REQ_GRAD: (CompareConst.NPU_REQ_GRAD, CompareConst.BENCH_REQ_GRAD),
            Const.MD5: (CompareConst.NPU_MD5, CompareConst.BENCH_MD5)
        }
        self.tensor_indicators_index = {
            CompareConst.COSINE: CompareConst.COMPARE_RESULT_HEADER.index(CompareConst.COSINE),
            CompareConst.EUC_DIST: CompareConst.COMPARE_RESULT_HEADER.index(CompareConst.EUC_DIST),
            CompareConst.MAX_ABS_ERR: CompareConst.COMPARE_RESULT_HEADER.index(CompareConst.MAX_ABS_ERR),
            CompareConst.MAX_RELATIVE_ERR: CompareConst.COMPARE_RESULT_HEADER.index(CompareConst.MAX_RELATIVE_ERR),
            CompareConst.ONE_THOUSANDTH_ERR_RATIO: CompareConst.COMPARE_RESULT_HEADER.index(
                CompareConst.ONE_THOUSANDTH_ERR_RATIO),
            CompareConst.FIVE_THOUSANDTHS_ERR_RATIO: CompareConst.COMPARE_RESULT_HEADER.index(
                CompareConst.FIVE_THOUSANDTHS_ERR_RATIO)
        }
        self.extra_tensor_indicators_index = {
            CompareConst.REQ_GRAD_CONSIST: CompareConst.COMPARE_RESULT_HEADER.index(CompareConst.REQ_GRAD_CONSIST),
            CompareConst.RESULT: CompareConst.COMPARE_RESULT_HEADER.index(CompareConst.RESULT),
            CompareConst.ERROR_MESSAGE: CompareConst.COMPARE_RESULT_HEADER.index(CompareConst.ERROR_MESSAGE)
        }
        self.all_tensor_indicators_index = {**self.tensor_indicators_index, **self.extra_tensor_indicators_index}

        self.npu_summary_index = [CompareConst.SUMMARY_COMPARE_RESULT_HEADER.index(CompareConst.NPU_MAX),
                                  CompareConst.SUMMARY_COMPARE_RESULT_HEADER.index(CompareConst.NPU_MIN),
                                  CompareConst.SUMMARY_COMPARE_RESULT_HEADER.index(CompareConst.NPU_MEAN),
                                  CompareConst.SUMMARY_COMPARE_RESULT_HEADER.index(CompareConst.NPU_NORM)]
        self.bench_summary_index = [CompareConst.SUMMARY_COMPARE_RESULT_HEADER.index(CompareConst.BENCH_MAX),
                                    CompareConst.SUMMARY_COMPARE_RESULT_HEADER.index(CompareConst.BENCH_MIN),
                                    CompareConst.SUMMARY_COMPARE_RESULT_HEADER.index(CompareConst.BENCH_MEAN),
                                    CompareConst.SUMMARY_COMPARE_RESULT_HEADER.index(CompareConst.BENCH_NORM)]
        self.statistics_indicators_index = {
            CompareConst.MAX_DIFF: CompareConst.SUMMARY_COMPARE_RESULT_HEADER.index(CompareConst.MAX_DIFF),
            CompareConst.MIN_DIFF: CompareConst.SUMMARY_COMPARE_RESULT_HEADER.index(CompareConst.MIN_DIFF),
            CompareConst.MEAN_DIFF: CompareConst.SUMMARY_COMPARE_RESULT_HEADER.index(CompareConst.MEAN_DIFF),
            CompareConst.NORM_DIFF: CompareConst.SUMMARY_COMPARE_RESULT_HEADER.index(CompareConst.NORM_DIFF),
            CompareConst.MAX_RELATIVE_ERR: CompareConst.SUMMARY_COMPARE_RESULT_HEADER.index(
                CompareConst.MAX_RELATIVE_ERR),
            CompareConst.MIN_RELATIVE_ERR: CompareConst.SUMMARY_COMPARE_RESULT_HEADER.index(
                CompareConst.MIN_RELATIVE_ERR),
            CompareConst.MEAN_RELATIVE_ERR: CompareConst.SUMMARY_COMPARE_RESULT_HEADER.index(
                CompareConst.MEAN_RELATIVE_ERR),
            CompareConst.NORM_RELATIVE_ERR: CompareConst.SUMMARY_COMPARE_RESULT_HEADER.index(
                CompareConst.NORM_RELATIVE_ERR)
        }
        self.extra_statistics_indicators_index = {
            CompareConst.REQ_GRAD_CONSIST: CompareConst.SUMMARY_COMPARE_RESULT_HEADER.index(
                CompareConst.REQ_GRAD_CONSIST),
            CompareConst.RESULT: CompareConst.SUMMARY_COMPARE_RESULT_HEADER.index(CompareConst.RESULT),
            CompareConst.ERROR_MESSAGE: CompareConst.SUMMARY_COMPARE_RESULT_HEADER.index(CompareConst.ERROR_MESSAGE)
        }
        self.all_statistics_indicators_index = {**self.statistics_indicators_index,
                                                **self.extra_statistics_indicators_index}

        self.extra_md5_indicators_index = {
            CompareConst.REQ_GRAD_CONSIST: CompareConst.MD5_COMPARE_RESULT_HEADER.index(CompareConst.REQ_GRAD_CONSIST),
            CompareConst.RESULT: CompareConst.MD5_COMPARE_RESULT_HEADER.index(CompareConst.RESULT),
            CompareConst.ERROR_MESSAGE: CompareConst.MD5_COMPARE_RESULT_HEADER.index(CompareConst.ERROR_MESSAGE)
        }
        self.error_list = []
        # 记录error日志，用于前端展示
        BaseLogger.error = self._get_log_msg_wrapper(BaseLogger.error, self.error_list)

    @staticmethod
    def _safe_convert_shape(shape_str: str):
        """将字符串shape "[10,3,64,64]" 转换为列表 [10,3,64,64]"""
        if not shape_str or shape_str == "null":
            return ""
        try:
            return json.loads(shape_str)
        except json.JSONDecodeError:
            return ""

    @staticmethod
    def _safe_convert_float(num_str: str):
        """将字符串数字转换为float，异常返回空字符串"""
        if not num_str or num_str == "null":
            return ""
        try:
            return float(num_str)
        except (ValueError, TypeError):
            return ""

    @staticmethod
    def _get_log_msg_wrapper(fn, error_list: list):
        def decorated(self, msg):
            error_list.append(msg)
            return fn(self, msg)

        return decorated

    def get_db_tensor_compare_result(self):
        """
        供前端调用，解析前端传入的npu_db_data和bench_db_data，进行tensor比对
        """
        self.error_list.clear()
        try:
            if not self.framework:
                npu_framework, bench_framework = self._get_frameworks()
                self.framework = npu_framework
                self.is_cross_frame = npu_framework != bench_framework

            result_lists_in = self._compare_db_tensor_node(
                self.npu_db_data.get('input_data'),
                self.bench_db_data.get('input_data'),
                self.npu_db_data.get('dump_data_dir'),
                self.bench_db_data.get('dump_data_dir')
            )
            data_lists_in = self._convert_db_data2data_lists(
                self.npu_db_data.get('input_data'),
                self.bench_db_data.get('input_data'),
                result_lists_in
            )
            result_lists_out = self._compare_db_tensor_node(
                self.npu_db_data.get('output_data'),
                self.bench_db_data.get('output_data'),
                self.npu_db_data.get('dump_data_dir'),
                self.bench_db_data.get('dump_data_dir')
            )
            data_lists_out = self._convert_db_data2data_lists(
                self.npu_db_data.get('output_data'),
                self.bench_db_data.get('output_data'),
                result_lists_out
            )
            all_data_lists = data_lists_in + data_lists_out
            result = calculate_result(all_data_lists, Const.ALL)

            input_info = self._parse_data(data_lists_in)
            output_info = self._parse_data(data_lists_out)

            data = {
                GraphConst.JSON_INDEX_KEY: GraphConst.COMPARE_INDICATOR_TO_PRECISION_INDEX_MAPPING.get(result, 0),
                self._INPUT_INFO: input_info,
                self._OUTPUT_INFO: output_info
            }

            return self._build_response(success=True, data=data) if not self.error_list else self._build_response(
                success=False, error=self.error_list, data=data)

        except Exception as e:
            self.error_list.append(str(e))
            return self._build_response(success=False, error=self.error_list)

    def get_db_statistics_compare_result(self):
        """
        供前端调用，解析前端传入的npu_db_data和bench_db_data，进行statistics比对
        """
        self.error_list.clear()
        try:
            data_lists_in = self._convert_db_data2data_lists(
                self.npu_db_data.get('input_data'),
                self.bench_db_data.get('input_data'),
                compare_mode=Const.SUMMARY
            )
            data_lists_out = self._convert_db_data2data_lists(
                self.npu_db_data.get('output_data'),
                self.bench_db_data.get('output_data'),
                compare_mode=Const.SUMMARY
            )

            data_lists_in = self._compare_statistics(data_lists_in)
            data_lists_out = self._compare_statistics(data_lists_out)

            all_data_lists = data_lists_in + data_lists_out
            result = calculate_result(all_data_lists, Const.SUMMARY)

            input_info = self._parse_data(data_lists_in, Const.SUMMARY)
            output_info = self._parse_data(data_lists_out, Const.SUMMARY)

            data = {
                GraphConst.JSON_INDEX_KEY: GraphConst.COMPARE_INDICATOR_TO_PRECISION_INDEX_MAPPING.get(result, 0),
                self._INPUT_INFO: input_info,
                self._OUTPUT_INFO: output_info
            }

            return self._build_response(success=True, data=data) if not self.error_list else self._build_response(
                success=False, error=self.error_list, data=data)

        except Exception as e:
            self.error_list.append(str(e))
            return self._build_response(success=False, error=self.error_list)

    def get_db_md5_compare_result(self):
        """
        供前端调用，解析前端传入的npu_db_data和bench_db_data，进行md5比对
        """
        self.error_list.clear()
        try:
            data_lists_in = self._convert_db_data2data_lists(
                self.npu_db_data.get('input_data'),
                self.bench_db_data.get('input_data'),
                compare_mode=Const.MD5
            )
            data_lists_out = self._convert_db_data2data_lists(
                self.npu_db_data.get('output_data'),
                self.bench_db_data.get('output_data'),
                compare_mode=Const.MD5
            )

            all_data_lists = data_lists_in + data_lists_out
            result = calculate_result(all_data_lists, Const.MD5)

            input_info = self._parse_data(data_lists_in, Const.MD5)
            output_info = self._parse_data(data_lists_out, Const.MD5)

            data = {
                GraphConst.JSON_INDEX_KEY: GraphConst.COMPARE_INDICATOR_TO_PRECISION_INDEX_MAPPING.get(result, 0),
                self._INPUT_INFO: input_info,
                self._OUTPUT_INFO: output_info
            }

            return self._build_response(success=True, data=data) if not self.error_list else self._build_response(
                success=False, error=self.error_list, data=data)

        except Exception as e:
            self.error_list.append(str(e))
            return self._build_response(success=False, error=self.error_list)

    def _get_summary_data(self, data_list):
        """
        统计值模式，获取npu和bench侧max/min/mean/norm在统计值表头list中的索引
        """
        try:
            return [data_list[i] for i in self.npu_summary_index], [data_list[i] for i in self.bench_summary_index]
        except IndexError as e:
            logger.error(f'MatchedNodeCalculator._get_summary_data encountered an IndexError: {e}')
            raise

    def _compare_statistics(self, data_lists):
        new_data_lists = []
        for data_list in data_lists:
            npu_summary_data, bench_summary_data = self._get_summary_data(data_list)
            results = get_rela_diff_summary_mode(data_list, npu_summary_data, bench_summary_data, "")
            new_data_lists.append(results[0])
        return new_data_lists

    def _get_frameworks(self):
        def get_framework(target_path):
            for filename in os.listdir(target_path):
                suffix = os.path.splitext(filename)[1].lower()
                if suffix == ".pt":
                    return Const.PT_FRAMEWORK
                elif suffix == ".npy":
                    return Const.MS_FRAMEWORK
                return Const.PT_FRAMEWORK

        npu_dump_data_dir = self.npu_db_data.get('dump_data_dir')
        bench_dump_data_dir = self.bench_db_data.get('dump_data_dir')
        npu_dump_data_dir = FileChecker(npu_dump_data_dir, FileCheckConst.DIR).common_check()
        bench_dump_data_dir = FileChecker(bench_dump_data_dir, FileCheckConst.DIR).common_check()

        return get_framework(npu_dump_data_dir), get_framework(bench_dump_data_dir)

    def _compare_tensor(self, npu_op_name, bench_op_name, op_name_mapping_dict, input_param):
        """
        进行一对tensor数据的比对，得到Cosine、EucDist等精度比对指标，区分框架
        """
        config_dict = {
            'stack_mode': False,
            'auto_analyze': True,
            'fuzzy_match': False,
            'dump_mode': Const.ALL
        }
        mode_config = ModeConfig(**config_dict)
        data_name_pair = op_name_mapping_dict.get(str(npu_op_name) + str(bench_op_name))
        npu_data_name, bench_data_name = data_name_pair
        npu_info = CompareInfo(npu_op_name, npu_data_name, -1)
        bench_info = CompareInfo(bench_op_name, bench_data_name, -1)

        if self.framework == Const.PT_FRAMEWORK:
            from msprobe.pytorch.compare.pt_compare import read_real_data
            return CompareRealData(read_real_data, mode_config, self.is_cross_frame).compare_by_op(npu_info,
                                                                                                   bench_info,
                                                                                                   input_param)
        else:
            from msprobe.mindspore.compare.ms_compare import read_real_data
            return CompareRealData(read_real_data, mode_config, self.is_cross_frame).compare_by_op(npu_info,
                                                                                                   bench_info,
                                                                                                   input_param)

    def _compare_db_tensor_node(self, npu_data: dict, bench_data: dict, npu_data_dir, bench_data_dir):
        """
        解析db node节点，将两个匹配的node节点的输入和输出参数按顺序一一对应，进行tensor比对
        """
        result_lists = []
        for (npu_op_name, npu_value), (bench_op_name, bench_value) in zip(npu_data.items(), bench_data.items()):
            npu_data_name = npu_value.get('data_name')
            bench_data_name = bench_value.get('data_name')
            if not npu_data_name or not bench_data_name:
                result_lists.append([])
            else:
                input_param = {
                    CompareConst.NPU_DUMP_DATA_DIR: npu_data_dir,
                    CompareConst.BENCH_DUMP_DATA_DIR: bench_data_dir
                }
                op_name_mapping_dict = {
                    npu_op_name + bench_op_name: [npu_value.get('data_name'), bench_value.get('data_name')]
                }

                result = self._compare_tensor(npu_op_name, bench_op_name, op_name_mapping_dict, input_param)
                result_lists.append(result)
        return result_lists

    def _convert_db_data2data_lists(self, npu_dict: dict, bench_dict: dict, compare_results: list = None,
                                    compare_mode: str = Const.ALL) -> List[List]:
        """
        将db里的node数据转换为指标计算需要的List[List]格式
        并且插入比对结果
        :return: 最终结果 [ [行1数据], [行2数据], ... ]
        """
        header = CompareConst.HEAD_OF_COMPARE_MODE.get(compare_mode)
        indicators_mapping = self.md5_public_indicators_mapping if compare_mode == Const.MD5 \
            else self.public_indicators_mapping
        data_lists = []
        npu_keys = list(npu_dict.keys())
        bench_keys = list(bench_dict.keys())

        max_length = max(len(npu_keys), len(bench_keys))

        for idx in range(max_length):
            row = [""] * len(header)
            # 按顺序取对应键
            npu_key = npu_keys[idx] if idx < len(npu_keys) else CompareConst.NAN
            bench_key = bench_keys[idx] if idx < len(bench_keys) else CompareConst.NAN
            npu_data = npu_dict.get(npu_key, {})
            bench_data = bench_dict.get(bench_key, {})

            # 填充名称（顺序匹配后，NPU和Bench名称可以不同）
            row[header.index(CompareConst.NPU_NAME)] = npu_key
            row[header.index(CompareConst.BENCH_NAME)] = bench_key

            # 按映射表填充数据
            for raw_key, (npu_header, bench_header) in indicators_mapping.items():
                npu_val_raw = npu_data.get(raw_key, "")
                bench_val_raw = bench_data.get(raw_key, "")

                # ============== 核心类型转换 ==============
                if raw_key == Const.SHAPE:
                    npu_val = self._safe_convert_shape(npu_val_raw)
                    bench_val = self._safe_convert_shape(bench_val_raw)
                elif raw_key in Const.SUMMARY_METRICS_LIST:
                    npu_val = self._safe_convert_float(npu_val_raw)
                    bench_val = self._safe_convert_float(bench_val_raw)
                else:
                    npu_val = npu_val_raw
                    bench_val = bench_val_raw

                row[header.index(npu_header)] = npu_val
                row[header.index(bench_header)] = bench_val

            # 填充requires_grad一致性
            npu_grad = npu_data.get(Const.REQ_GRAD, "")
            bench_grad = bench_data.get(Const.REQ_GRAD, "")
            grad_consistent = str(npu_grad == bench_grad) if (npu_grad and bench_grad) else ""
            row[header.index(CompareConst.REQ_GRAD_CONSIST)] = grad_consistent

            # 基于预设索引，插入比对结果到指定位置
            if compare_results and idx < len(compare_results):
                calc_result = compare_results[idx]  # 取出当前行的比对结果
                for res_idx, col_idx in enumerate(list(self.tensor_indicators_index.values())):
                    if res_idx < len(calc_result):
                        row[col_idx] = calc_result[res_idx]
                    else:
                        row[col_idx] = ""

            data_lists.append(row)

        return data_lists

    def _parse_data(self, data_lists, compare_mode: str = Const.ALL):
        """
        封装前端需要的格式
        data_lists: 最终结果 [ [行1数据], [行2数据], ... ]

        return example:
        {
          "Functional.conv2d.0.forward.input.0": {"Cosine": 1.0, "EucDist": 0.0, ...},
          "Functional.conv2d.0.forward.input.1": {"Cosine": 1.0, "EucDist": 0.0, ...},
          ...
        }
        """
        indicators_index = self.all_tensor_indicators_index
        if compare_mode == Const.SUMMARY:
            indicators_index = self.all_statistics_indicators_index
        elif compare_mode == Const.MD5:
            indicators_index = self.extra_md5_indicators_index
        indicators_info = {}
        for data_list in data_lists:
            if not data_list:
                continue
            name = data_list[0]
            info = {}
            for indicator, index in indicators_index.items():
                info[indicator] = data_list[index]
            indicators_info[name] = info

        return indicators_info

    def _build_response(self, success: bool, error: list = None, data: dict = None):
        data = data or {}
        return {
            self._RESP_SUCCESS: success,
            self._RESP_ERROR: error,
            self._RESP_DATA: data
        }
