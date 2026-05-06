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
import math
import zlib
from dataclasses import dataclass
import multiprocessing

import numpy as np
import pandas as pd

from msprobe.core.common.const import Const, CompareConst, FileCheckConst
from msprobe.core.common.utils import CompareException, check_regex_prefix_format_valid, logger, \
    safe_get_value, is_module_available
from msprobe.core.common.file_utils import check_file_or_directory_path, load_json, find_proc_dir

json_file_mapping = {
    Const.DUMP_JSON_FILE: "dump.json",
    Const.DEBUG_JSON_FILE: "debug.json",
    Const.STACK_JSON_FILE: "stack.json"
}


def extract_json(dirname, json_file_type):
    json_path = ''
    for filename in os.listdir(dirname):
        target_file_name = json_file_mapping.get(json_file_type)
        if target_file_name is None:
            logger.error(f'extract_json failed, invalid json_file_type: {json_file_type}.')
            raise CompareException(CompareException.INVALID_KEY_ERROR)
        if filename == target_file_name:
            json_path = os.path.join(dirname, filename)
            break

    # Provide robustness on invalid directory inputs
    if not json_path:
        if json_file_type == Const.STACK_JSON_FILE:
            logger.warning(f'stack.json is not found in dump dir {dirname}.')
        elif json_file_type == Const.DUMP_JSON_FILE:
            logger.error(f'dump.json is not found in dump dir {dirname}.')
        elif json_file_type == Const.DEBUG_JSON_FILE:
            logger.warning(f'debug.json is not found in dump dir {dirname}.')

    return json_path


def set_stack_json_path(input_param):
    npu_data_dir = os.path.dirname(input_param.get("npu_path"))
    stack_path = extract_json(npu_data_dir, json_file_type=Const.STACK_JSON_FILE)
    input_param["stack_path"] = stack_path if stack_path else None
    return bool(stack_path), input_param


def check_and_return_dir_contents(dump_dir, prefix, skip_wrong_dir=False):
    """
    check the given dump dir and validate files in dump dir by using the given prefix patterns to build a
    pattern: ^{prefix}(?:0|[1-9][0-9]*)?$

    Args:
        dump_dir (str): dump dir
        prefix (str): prefix for the patterns, prefix should be less than 20 characters and alphanumeric/-/_ only
        skip_wrong_dir (bool): whether skip found dirs
    Returns:
        content [list]: dir contents
    Raises:
        CompareException: invalid path
        ValueError: prefix not match the patterns

    """
    check_regex_prefix_format_valid(prefix)
    check_file_or_directory_path(dump_dir, True)
    contents = os.listdir(dump_dir)
    pattern = re.compile(rf'^{prefix}(?:0|[1-9][0-9]*)?$')
    final_contents = []
    for name in contents:
        if not pattern.match(name):
            if skip_wrong_dir:
                pass
            else:
                logger.error(
                    f"dump_dir contains '{name}'. Expected '{prefix}'. This name is not in the format of dump "
                    f"output. Please check and delete irrelevant files in {dump_dir} and try again."
                )
                raise CompareException(CompareException.INVALID_PATH_ERROR)
        else:
            final_contents.append(name)
    return final_contents


def read_op(op_data, op_name):
    if not isinstance(op_name, str):
        logger.error(f"api name error: {op_name} is not a string, please check.")
        raise CompareException(CompareException.INVALID_API_NAME_ERROR)
    split_name = op_name.split(Const.SEP)
    if split_name[-1] == Const.DEBUG:
        op_parsed_list = op_item_parse(op_data, op_name, Const.DEBUG)
    # 倒序校验parameters_grad，先倒数1，再倒数2，排除因为parameters_grad在模块名中而误判的风险。-1适配老版，-2适配新版。
    elif split_name[-1] == Const.PARAMS_GRAD or (len(split_name) >= 2 and split_name[-2] == Const.PARAMS_GRAD):
        op_parsed_list = op_item_parse(op_data, op_name, Const.PARAMS_GRAD)
    else:
        op_parsed_list = []
        for name in CompareConst.IO_NAME_MAPPING:
            if name in op_data:
                op_parsed_list.extend(op_item_parse(op_data[name], op_name + CompareConst.IO_NAME_MAPPING[name], name))
    return op_parsed_list


def op_item_parse(op_data, op_name: str, state: str, depth: int = 0) -> list:
    if state == Const.INPUT_ARGS or state == Const.INPUT_KWARGS:
        state = Const.INPUT
    default_item = {
        'full_op_name': op_name,
        Const.TYPE: None,
        Const.MAX: None,
        Const.MIN: None,
        Const.MEAN: None,
        Const.NORM: None,
        Const.DTYPE: None,
        Const.SHAPE: None,
        Const.MD5: None,
        Const.VALUE: None,
        Const.DATA_NAME: '-1',
        Const.STATE: state,
        Const.REQ_GRAD: None
    }

    if depth > Const.MAX_DEPTH:
        logger.error(f'parse of api/module of {op_name} exceeds the recursion limit.')
        raise CompareException(CompareException.RECURSION_LIMIT_ERROR)

    if op_data is None:
        return [default_item]
    elif not op_data:
        return []

    item_list = []
    if isinstance(op_data, list):
        for i, data in enumerate(op_data):
            if Const.PARAMS_GRAD not in op_name.split(Const.SEP):
                item_list.extend(op_item_parse(data, op_name + Const.SEP + str(i), state, depth + 1))
            else:
                item_list.extend(op_item_parse(data, op_name, state, depth + 1))
    elif isinstance(op_data, dict):
        if is_p2pop_leaf_data(op_data):
            p2pop_item = {}
            for key in ['class_type', 'op', 'peer', 'tag', 'group_id']:
                p2pop_item[key] = op_data.get(key)
            op_data = op_data.get('tensor')
            if isinstance(op_data, dict):
                op_item = gen_op_item(op_data, op_name, state)
            else:
                op_item = default_item
            op_item.update(p2pop_item)
            return [op_item]
        if is_leaf_data(op_data):
            return [gen_op_item(op_data, op_name, state)]
        for sub_name, sub_data in op_data.items():
            item_list.extend(op_item_parse(sub_data, op_name + Const.SEP + str(sub_name), state, depth + 1))
    return item_list


def is_p2pop_leaf_data(op_data):
    return op_data.get('class_type') == 'torch.distributed.P2POp'


def is_leaf_data(op_data):
    return 'type' in op_data and isinstance(op_data['type'], str)


def gen_op_item(op_data, op_name, state):
    op_item = {}
    op_item.update({key: str(value) if isinstance(value, bool) else value for key, value in op_data.items()})
    data_name = op_data.get(Const.DATA_NAME) if op_data.get(Const.DATA_NAME) else '-1'  # 如果是""也返回-1
    op_item[Const.DATA_NAME] = data_name
    op_item['full_op_name'] = data_name.rsplit(Const.SEP, 1)[0] if data_name != '-1' else op_name
    op_item[Const.STATE] = state
    if Const.REQ_GRAD not in op_item:
        op_item[Const.REQ_GRAD] = None

    # 补齐统计量字段
    params = [Const.MAX, Const.MIN, Const.MEAN, Const.NORM]
    for i in params:
        if i not in op_item:
            op_item[i] = None

    # special cases
    if not op_item.get('dtype'):
        if op_item.get('type') == 'torch.Size':
            op_item['dtype'] = op_data.get('type')
            op_item['shape'] = str(op_data.get('value'))
        elif op_item.get('type') == 'slice':
            op_item['dtype'] = op_data.get('type')
            op_item['shape'] = str(np.shape(np.array(op_data.get('value'))))
        elif op_item.get('type') == 'ellipsis':
            op_item['dtype'] = op_data.get('type')
            op_item['shape'] = '[]'
            for i in params:
                op_item[i] = op_data.get('value')
        elif op_name.split(Const.SEP)[-1] in ['src', 'dst', 'group_src', 'group_dst']:
            op_item['dtype'] = op_data.get('type')
            op_item['shape'] = '[]'
            for i in params:
                op_item[i] = str(op_data.get('value'))
            op_item['md5'] = str(op_data.get('value'))
        elif op_item.get('type') == 'torch.ProcessGroup':
            op_item['dtype'] = op_data.get('type')
            op_item['shape'] = '[]'
            for i in params:
                op_item[i] = str(op_data.get('group_ranks'))
            op_item['md5'] = str(op_data.get('group_ranks'))
        else:
            op_item['dtype'] = str(type(op_data.get('value')))
            op_item['shape'] = '[]'
            for i in params:
                op_item[i] = op_data.get('value')
    if not op_item.get('md5'):
        op_item['md5'] = f"{zlib.crc32(str(op_data.get('value', '')).encode()):08x}"

    return op_item


@dataclass
class ApiItemInfo:
    name: str
    struct: tuple
    stack_info: list


def merge_tensor(tensor_list, dump_mode):
    keys = [
        CompareConst.OP_NAME,
        CompareConst.INPUT_STRUCT,
        CompareConst.KWARGS_STRUCT,
        CompareConst.OUTPUT_STRUCT,
        CompareConst.PARAMS_STRUCT,
        CompareConst.PARAMS_GRAD_STRUCT,
        CompareConst.DEBUG_STRUCT,
        Const.SUMMARY,
        Const.STACK_INFO,
        Const.STATE,
        Const.REQ_GRAD
    ]
    op_dict = {key: [] for key in keys}

    if dump_mode == Const.ALL:
        op_dict[Const.DATA_NAME] = []

    for tensor in tensor_list:
        # A dict(len=2) with 'full_op_name' and 'full_info' is added to the tensor only if self.stack_mode is True
        if len(tensor) == 2:
            op_dict[Const.STACK_INFO].append(tensor.get('full_info'))
            break

        op_dict[CompareConst.OP_NAME].append(tensor.get('full_op_name'))
        state = tensor.get(Const.STATE)
        op_dict[Const.STATE].append(state)
        op_dict[Const.REQ_GRAD].append(tensor.get(Const.REQ_GRAD))

        struct_key = CompareConst.STATE_TO_STRUCT_MAPPING.get(state)
        if not struct_key:
            continue
        if dump_mode == Const.MD5:
            op_dict.get(struct_key).append((tensor[Const.DTYPE], tensor[Const.SHAPE], tensor[Const.MD5]))
        else:
            op_dict.get(struct_key).append((tensor[Const.DTYPE], tensor[Const.SHAPE]))

        # 当统计量为None时，转成字符串None，避免后续操作list放到pd中时None被默认转成NaN
        op_dict[Const.SUMMARY].append(
            [str(tensor[key]) if tensor[key] is None else tensor[key] for key in Const.SUMMARY_METRICS_LIST])

        if dump_mode == Const.ALL:
            op_dict[Const.DATA_NAME].append(tensor.get(Const.DATA_NAME))

    if not op_dict[CompareConst.KWARGS_STRUCT]:
        del op_dict[CompareConst.KWARGS_STRUCT]
    return op_dict if op_dict[CompareConst.OP_NAME] else {}


def print_compare_ends_info():
    total_len = len(CompareConst.COMPARE_ENDS_SUCCESSFULLY) + Const.FILL_CHAR_NUMS
    logger.info('*' * total_len)
    logger.info(f"*{CompareConst.COMPARE_ENDS_SUCCESSFULLY.center(total_len - 2)}*")
    logger.info('*' * total_len)


def table_value_is_valid(value: str) -> bool:
    if not isinstance(value, str):
        return True
    try:
        # -1.00 or +1.00 should be considered as digit numbers
        float(value)
    except ValueError:
        # otherwise, they will be considered as formular injections
        return not bool(re.compile(FileCheckConst.CSV_BLACK_LIST).search(value))
    return True


class ApiBatch:
    def __init__(self, api_name: str, start: int):
        self.api_name = api_name
        self.start = start
        self.input_len = 1  # input的数量
        self.params_end_index = start + 1  # params的结束index
        self.output_end_index = start + 1  # output的结束index
        self.params_grad_end_index = start + 1  # params_grad的结束index
        # 内部state的标志("input", "output", "parameters", "parameters_grad"),
        # 用于控制计算input_len, output_end_index, params_end_index, self.params_grad_end_index
        self._state = Const.INPUT  # api_batch初始化为input

    def set_state(self, state: str):
        """设置当前状态"""
        if state in {Const.INPUT, Const.OUTPUT, Const.KWARGS, Const.PARAMS, Const.PARAMS_GRAD}:
            self._state = state
        else:
            raise ValueError(f"Invalid state: {state}")

    def increment(self, state: str):
        self.set_state(state)
        if self._state == Const.INPUT or self._state == Const.KWARGS:
            self.input_len += 1
            self.params_end_index += 1
            self.output_end_index += 1
        if self._state == Const.PARAMS:
            self.params_end_index += 1
            self.output_end_index += 1
        if self._state == Const.OUTPUT:
            self.output_end_index += 1
        self.params_grad_end_index += 1


def api_batches_update(api_batches, api_name, state, index):
    """
    当一个api的所有item更新完后，input, output的索引范围：
    input: [start: start+input_len]
    output: [start+input_len: output_end_index]
    params: [output_end_index: params_end_index]
    """
    if not api_batches:
        api_batches.append(ApiBatch(api_name, index))
    else:
        api_batch = api_batches[-1]
        if api_batch.api_name == api_name or (
                not re.search(Const.REGEX_FORWARD_BACKWARD, api_name) and api_name in api_batch.api_name):
            try:
                api_batch.increment(state)
            except ValueError as e:
                logger.error(f"api_batch: {api_batch} with invalid state, please check! {e}")
                raise CompareException(CompareException.INVALID_STATE_ERROR) from e
        else:
            api_batches.append(ApiBatch(api_name, index))


def reorder_index(op_parsed_list):
    """
    对单个api解析的op_items的index进行重排，将parameter的index放到output前面，返回新的重排后的index列表，op_parsed_list不变
    """
    index_param = []
    index_output = []
    index_param_grad = []
    index_other = []
    for i, op_item in enumerate(op_parsed_list[:-1]):
        state = op_item.get(Const.STATE)
        if state == Const.PARAMS:
            index_param.append(i)
        elif state == Const.OUTPUT:
            index_output.append(i)
        elif state == Const.PARAMS_GRAD:
            index_param_grad.append(i)
        else:
            index_other.append(i)
    # 合并others, parameters, 和output，确保parameters排在output前面
    reordered_index_list = index_other + index_param + index_output + index_param_grad
    return reordered_index_list


def reorder_op_name_list(op_name_list, state_list):
    if not op_name_list:
        return op_name_list, state_list

    parameters = []
    output = []
    parameters_grad = []
    others = []
    parameters_s = []
    output_s = []
    parameters_grad_s = []
    others_s = []
    for op_name, state in zip(op_name_list, state_list):
        if state == Const.PARAMS:
            parameters.append(op_name)
            parameters_s.append(state)
        elif state == Const.OUTPUT:
            output.append(op_name)
            output_s.append(state)
        elif state == Const.PARAMS_GRAD:
            parameters_grad.append(op_name)
            parameters_grad_s.append(state)
        else:
            others.append(op_name)
            others_s.append(state)
    # 合并others, parameters, 和output，确保parameters排在output前面
    op_name_reorder = others + parameters + output + parameters_grad
    state_reorder = others_s + parameters_s + output_s + parameters_grad_s
    return op_name_reorder, state_reorder


def process_summary_data(summary_data):
    """处理summary_data中的nan值，返回处理后的列表"""
    return [CompareConst.NAN if isinstance(x, float) and math.isnan(x) else x for x in summary_data]


def get_rela_diff_summary_mode(result_item, npu_summary_data, bench_summary_data, err_msg):
    start_idx = CompareConst.SUMMARY_COMPARE_RESULT_HEADER.index(CompareConst.MAX_DIFF)
    warning_flag = False
    for i, (npu_val, bench_val) in enumerate(zip(npu_summary_data, bench_summary_data)):
        if all(isinstance(val, (float, int)) and not isinstance(val, bool) for val in [npu_val, bench_val]):
            diff = npu_val - bench_val
            if math.isnan(diff):
                diff = CompareConst.NAN
                relative = CompareConst.NAN
            else:
                if bench_val != 0:
                    relative = str(abs((diff / bench_val) * 100)) + '%'
                else:
                    relative = CompareConst.N_A
                magnitude_diff = abs(diff) / (max(abs(npu_val), abs(bench_val)) + CompareConst.EPSILON)
                if magnitude_diff > CompareConst.MAGNITUDE:
                    warning_flag = True
            result_item[start_idx + i] = diff
            result_item[start_idx + i + CompareConst.STATISTICS_INDICATOR_NUM] = relative
        else:
            result_item[start_idx + i] = CompareConst.N_A
            result_item[start_idx + i + CompareConst.STATISTICS_INDICATOR_NUM] = CompareConst.N_A

    accuracy_check = CompareConst.WARNING if warning_flag else ""
    err_msg += "Need double check api accuracy." if warning_flag else ""
    for i in range(start_idx, len(result_item)):
        if str(result_item[i]) in ('inf', '-inf', 'nan'):
            result_item[i] = f'{result_item[i]}\t'
    return result_item, accuracy_check, err_msg


@dataclass
class ApiItemInfo:
    name: str
    struct: tuple
    stack_info: list


def stack_column_process(result_item, has_stack, index, key, npu_stack_info):
    if has_stack and index == 0 and key == CompareConst.INPUT_STRUCT:
        result_item.extend(npu_stack_info)
    else:
        result_item.append(CompareConst.NONE)
    return result_item


def result_item_init(n_info, b_info, requires_grad_pair, dump_mode):
    n_len = len(n_info.struct)
    b_len = len(b_info.struct)
    # requires_grad_pair内部创建，固定两个元素
    n_requires_grad = requires_grad_pair[0]
    b_requires_grad = requires_grad_pair[1]
    req_grad_consist = n_requires_grad == b_requires_grad
    struct_long_enough = (n_len > 2 and b_len > 2) if dump_mode == Const.MD5 else (n_len > 1 and b_len > 1)
    if struct_long_enough:
        result_item = [
            n_info.name, b_info.name, n_info.struct[0], b_info.struct[0], n_info.struct[1], b_info.struct[1],
            n_requires_grad, b_requires_grad
        ]
        if dump_mode == Const.MD5:
            md5_compare_result = CompareConst.PASS if n_info.struct[2] == b_info.struct[2] else CompareConst.DIFF
            result_item.extend([n_info.struct[2], b_info.struct[2], req_grad_consist, md5_compare_result])
        elif dump_mode == Const.SUMMARY:
            result_item.extend([" "] * 8)  # 8个统计量数据情况的比对指标
        else:
            result_item.extend([" "] * 6)  # 6个真实数据情况的比对指标
    else:
        err_msg = "index out of bounds error will occur in result_item_init, please check!\n" \
                  f"npu_info_struct is {n_info.struct}\n" \
                  f"bench_info_struct is {b_info.struct}"
        logger.error(err_msg)
        raise CompareException(CompareException.INDEX_OUT_OF_BOUNDS_ERROR)
    return result_item


def count_struct(op_dict):
    parts = [
        CompareConst.OP_NAME,
        CompareConst.INPUT_STRUCT,
        CompareConst.OUTPUT_STRUCT,
        CompareConst.PARAMS_STRUCT,
        CompareConst.PARAMS_GRAD_STRUCT
    ]
    lengths = [len(op_dict.get(part, [])) for part in parts]
    num = lengths[0]
    if num != sum(lengths[1:]):
        logger.error(f"Length of names and structs of op_dict not match. Please check! op_dict: {op_dict}")
        raise CompareException(CompareException.NAMES_STRUCTS_MATCH_ERROR)
    return tuple(lengths)


def get_accuracy(result, n_dict, b_dict, dump_mode):
    def get_accuracy_core(n_start, n_len, b_start, b_len, key):
        min_len = min(n_len, b_len)
        npu_stack_info = n_dict.get("stack_info", None)
        bench_stack_info = b_dict.get("stack_info", None)
        has_stack = npu_stack_info and bench_stack_info

        if dump_mode == Const.ALL:
            npu_data_name_list = n_dict.get("data_name", None)
            bench_data_name_list = b_dict.get("data_name", None)

        for index in range(min_len):
            n_name = safe_get_value(n_dict, n_start + index, "n_dict", key="op_name")
            b_name = safe_get_value(b_dict, b_start + index, "b_dict", key="op_name")
            n_struct = safe_get_value(n_dict, index, "n_dict", key=key)
            b_struct = safe_get_value(b_dict, index, "b_dict", key=key)
            n_requires_grad = safe_get_value(n_dict, n_start + index, "n_dict", key='requires_grad')
            b_requires_grad = safe_get_value(b_dict, b_start + index, "b_dict", key='requires_grad')
            requires_grad_pair = [n_requires_grad, b_requires_grad]
            req_grad_consist = n_requires_grad == b_requires_grad
            err_msg = ""

            npu_info = ApiItemInfo(n_name, n_struct, npu_stack_info)
            bench_info = ApiItemInfo(b_name, b_struct, bench_stack_info)
            result_item = result_item_init(npu_info, bench_info, requires_grad_pair, dump_mode)

            if dump_mode == Const.MD5:
                result_item = stack_column_process(result_item, has_stack, index, key, npu_stack_info)
                result.append(result_item)
                continue

            npu_summary_data = safe_get_value(n_dict, n_start + index, "n_dict", key=CompareConst.SUMMARY)
            bench_summary_data = safe_get_value(b_dict, b_start + index, "b_dict", key=CompareConst.SUMMARY)
            result_item.extend(process_summary_data(npu_summary_data))
            result_item.extend(process_summary_data(bench_summary_data))

            if dump_mode == Const.SUMMARY:
                result_item, accuracy_check, err_msg = get_rela_diff_summary_mode(result_item, npu_summary_data,
                                                                                  bench_summary_data, err_msg)

            result_item.append(req_grad_consist)
            err_msg += "Requires_grad inconsistent." if not req_grad_consist else ""
            result_item.append(accuracy_check if dump_mode == Const.SUMMARY else CompareConst.ACCURACY_CHECK_YES)
            result_item.append(err_msg)
            result_item = stack_column_process(result_item, has_stack, index, key, npu_stack_info)
            if dump_mode == Const.ALL:
                npu_data_name = safe_get_value(npu_data_name_list, n_start + index, "npu_data_name_list")
                bench_data_name = safe_get_value(bench_data_name_list, b_start + index, "bench_data_name_list")
                result_item.append([npu_data_name, bench_data_name])

            result.append(result_item)

        if n_len > b_len:
            for index in range(b_len, n_len):
                try:
                    n_name = safe_get_value(n_dict, n_start + index, "n_dict", key="op_name")
                    n_struct = safe_get_value(n_dict, index, "n_dict", key=key)
                    n_requires_grad = safe_get_value(n_dict, n_start + index, "n_dict", key='requires_grad')

                    if dump_mode == Const.MD5:
                        result_item = [
                            n_name, CompareConst.NAN, n_struct[0], CompareConst.NAN, n_struct[1], CompareConst.NAN,
                            n_requires_grad, CompareConst.NAN,
                            n_struct[2], CompareConst.NAN,
                            False,
                            CompareConst.NAN,
                            None
                        ]
                        result.append(result_item)
                        continue
                    if dump_mode == Const.ALL:
                        result_item = [
                            n_name, CompareConst.NAN, n_struct[0], CompareConst.NAN, n_struct[1], CompareConst.NAN,
                            n_requires_grad, CompareConst.NAN,
                            " ", " ", " ", " ", " ", " "
                        ]
                    else:
                        result_item = [
                            n_name, CompareConst.NAN, n_struct[0], CompareConst.NAN, n_struct[1], CompareConst.NAN,
                            n_requires_grad, CompareConst.NAN,
                            " ", " ", " ", " ", " ", " ", " ", " "
                        ]
                    summary_data = n_dict.get(CompareConst.SUMMARY)[n_start + index]
                    result_item.extend(summary_data)
                    summary_data = [CompareConst.NAN for _ in range(len(n_dict.get(CompareConst.SUMMARY)[0]))]
                    result_item.extend(summary_data)
                    result_item.append(False)
                except IndexError as e:
                    err_msg = "index out of bounds error occurs, please check!\n" \
                              f"n_dict is {n_dict}"
                    logger.error(err_msg)
                    raise CompareException(CompareException.INDEX_OUT_OF_BOUNDS_ERROR) from e

                err_msg = ""
                result_item.append(CompareConst.PASS)
                result_item.append(err_msg)
                result_item = stack_column_process(result_item, has_stack, index, key, npu_stack_info)
                if dump_mode == Const.ALL:
                    npu_data_name = safe_get_value(npu_data_name_list, n_start + index, "npu_data_name_list")
                    result_item.append([npu_data_name, "-1"])

                result.append(result_item)

    _, n_num_input, n_num_output, n_num_params, n_num_params_grad = count_struct(n_dict)
    _, b_num_input, b_num_output, b_num_params, b_num_params_grad = count_struct(b_dict)

    get_accuracy_core(0, n_num_input, 0, b_num_input, CompareConst.INPUT_STRUCT)
    get_accuracy_core(n_num_input, n_num_params, b_num_input, b_num_params,
                      CompareConst.PARAMS_STRUCT)
    get_accuracy_core(n_num_input + n_num_params, n_num_output, b_num_input + b_num_params, b_num_output,
                      CompareConst.OUTPUT_STRUCT)
    get_accuracy_core(n_num_input + n_num_params + n_num_output, n_num_params_grad,
                      b_num_input + b_num_params + b_num_output, b_num_params_grad,
                      CompareConst.PARAMS_GRAD_STRUCT)


def make_result_table(result, dump_mode, stack_mode):
    header = CompareConst.HEAD_OF_COMPARE_MODE[dump_mode][:]

    if stack_mode:
        header.append(CompareConst.STACK)
        if dump_mode == Const.ALL:
            header.append(CompareConst.DATA_NAME)
    else:
        if dump_mode == Const.ALL:
            for row in result:
                del row[-2]  # 输出结果不要堆栈信息时，删除中间结果result中的stack info，真实数据时为倒数第2列
            header.append(CompareConst.DATA_NAME)
        else:
            for row in result:
                del row[-1]  # 输出结果不要堆栈信息时，删除中间结果result中的stack info，非真实数据时为倒数第1列
    result_df = pd.DataFrame(result, columns=header, dtype='object')
    return result_df


def gen_api_batches(result: np.ndarray, header: list):
    api_name_index = header.index(Const.API_ORIGIN_NAME)
    state_name_index = header.index(Const.STATE)
    api_batches = []
    for i, res_i in enumerate(result):
        api_name = safe_get_value(res_i, api_name_index, "res_i")
        state = safe_get_value(res_i, state_name_index, "res_i")
        api_batches_update(api_batches, api_name, state, i)
    return api_batches


def get_paired_dirs(npu_path, bench_path):
    npu_dirs = set(os.listdir(npu_path))
    bench_dirs = set(os.listdir(bench_path))
    return list(npu_dirs & bench_dirs)


def _compare_parser(parser):
    parser.add_argument("-m", "--mode", dest="mode", type=str, default="auto",
                        help="<optional> Comparison mode: 'auto' (default) or 'torchair' for torchair accuracy compare",
                        required=False)
    parser.add_argument("-tp", "--target_path", dest="target_path", type=str,
                        help="<Required> The compare target device path", required=True)
    parser.add_argument("-gp", "--golden_path", dest="golden_path", type=str,
                        help="<Required> The compare golden device path", required=True)
    parser.add_argument("-o", "--output_path", dest="output_path", type=str,
                        help="<Required> The compare task result out path. Default path: ./output",
                        required=False, default="./output", nargs="?", const="./output")
    parser.add_argument("--xlsx", dest="xlsx", action="store_true",
                        help="<optional> Save compare results in xlsx format. Default output format is csv.",
                        required=False)
    parser.add_argument("-fm", "--fuzzy_match", dest="fuzzy_match", action="store_true",
                        help="<optional> Whether to perform a fuzzy match on the api name.", required=False)
    parser.add_argument("-cm", "--cell_mapping", dest="cell_mapping", type=str, nargs='?', const=True,
                        help="<optional> The cell mapping file path.", required=False)
    parser.add_argument("-am", "--api_mapping", dest="api_mapping", type=str, nargs='?', const=True,
                        help="<optional> The api mapping file path.", required=False)
    parser.add_argument("-dm", "--data_mapping", dest="data_mapping", type=str,
                        help="<optional> The data mapping file path.", required=False)
    parser.add_argument("-lm", "--layer_mapping", dest="layer_mapping", type=str, nargs='?', const=True,
                        help="<optional> The layer mapping file path.", required=False)
    parser.add_argument("-da", "--diff_analyze", dest="diff_analyze", action="store_true",
                        help="<optional> Whether to perform a diff analyze on the api name.", required=False)
    parser.add_argument("-tensor_log", "--is_print_compare_log", dest="is_print_compare_log", action="store_true",
                        help="<Optional> whether print compare log for compare auto mode task.", required=False)
    parser.add_argument("-fr", "--fusion_rule_file", dest="fusion_rule_file", type=str, default="",
                        help="<Optional> The fusion rule file path.", required=False)
    parser.add_argument("-qfr", "--quant_fusion_rule_file", dest="quant_fusion_rule_file", type=str, default="",
                        help="<Optional> The quant fusion rule file path.", required=False)
    parser.add_argument("-cfr", "--close_fusion_rule_file", dest="close_fusion_rule_file", type=str, default="",
                        help="<Optional> The close fusion rule file path.", required=False)
    # rank：ms静态图比对、推理离线模型一键式比对
    parser.add_argument("--rank", dest="rank", type=str, required=False,
                        help="<optional> Ranks to compare when compare kernel of MindSpore for <compare auto>. "
                             "Ranks to compare for <torchair dumps>. "
                             "Input rank ID [0, 255] for <compare offline_model>.")
    parser.add_argument("--step", dest="step", type=str, required=False,
                        help="<optional> Steps to compare when compare kernel of MindSpore.")
    parser.add_argument('--input_data', dest="input_data", default='',
                        help='The input data path of the model. Separate multiple inputs with commas(,).'
                             ' E.g: input_0.bin,input_1.bin')
    parser.add_argument('--input_shape', dest="input_shape", default='',
                        help="Shape of input shape. Separate multiple nodes with semicolons(;)."
                             " E.g: \"input_name1:1,224,224,3;input_name2:3,300\"")
    parser.add_argument('--output_size', dest="output_size", default='',
                        help='The size of output. Separate multiple sizes with commas(,). E.g: 10200,34000')
    parser.add_argument('--dym_shape_range', dest="dym_shape_range", default='',
                        help="Dynamic shape range using in dynamic model, using this means ignore input_shape."
                             " E.g: \"input_name1:1,3,200\~224,224-230;input_name2:1,300\"")
    parser.add_argument('-ofs', '--onnx_fusion_switch', dest="onnx_fusion_switch", default=True,
                        help='Onnxruntime fusion switch, set False for dump complete onnx data when necessary. '
                             'Usage: -ofs False')
    parser.add_argument("--consistent_check", dest="consistent_check", action="store_true",
                        help="<optional> Whether to compare train and infer data.", required=False)
    parser.add_argument("--backend", dest="backend", type=str, choices=["fsdp", "megatron"], default="",
                        help="<optional> Backend when comparing train and infer data.", required=False)


def get_sorted_ranks(npu_dump_dir, bench_dump_dir):
    """
    get the ranks and match by order
    """
    unsorted_npu_ranks = check_and_return_dir_contents(npu_dump_dir, 'rank', skip_wrong_dir=True)
    unsorted_bench_ranks = check_and_return_dir_contents(bench_dump_dir, 'rank', skip_wrong_dir=True)
    # 正则匹配已经校验rank后面必是数字，或者无数字的rank
    npu_ranks = sorted(unsorted_npu_ranks, key=lambda x: int(x[4:]) if len(x) > 4 else -1)  # 前四个字符都是rank，后面是卡号
    bench_ranks = sorted(unsorted_bench_ranks, key=lambda x: int(x[4:]) if len(x) > 4 else -1)
    if len(npu_ranks) != len(bench_ranks):
        logger.error('The number of ranks in the two runs are different. '
                     'Unable to match the ranks. Please use another folder to compare '
                     'or use compare() api and manually match the ranks.')
        raise CompareException(CompareException.INVALID_PATH_ERROR)
    return npu_ranks, bench_ranks


def multi_statistics_compare(func, func_args):
    def err_call(args):
        logger.error(f'Multiprocess statistics compare failed! Reason: {args}')

    compare_func, input_param_nr_list, output_path, kwargs = func_args

    param_num = len(input_param_nr_list)
    process_num = max(int((multiprocessing.cpu_count() + 1) // 4), 1)
    if param_num <= process_num:
        process_num = param_num
        chunks = [[input_param_nr] for input_param_nr in input_param_nr_list]
    else:
        chunk_size = param_num // process_num
        remainder = param_num % process_num
        chunks = [input_param_nr_list[i:i + chunk_size] for i in range(0, param_num - remainder, chunk_size)]
        for i in range(remainder):
            chunks[i].append(input_param_nr_list[param_num - remainder + i])

    pool = multiprocessing.Pool(process_num)

    async_results = []
    for chunk in chunks:
        result = pool.apply_async(func, args=(compare_func, chunk, output_path, kwargs), error_callback=err_call)
        async_results.append(result)

    pool.close()

    for ar in async_results:
        try:
            ar.get(timeout=3600)
        except Exception as e:
            logger.error(f"Task failed with exception: {e}")
            pool.terminate()
            raise CompareException(CompareException.MULTIPROCESS_ERROR) from e

    pool.join()


def mp_logger_init(ranks_str):
    """
    多进程比对需要对logger进行wrap和patch，在日志前加上卡号信息，从而实现不同进程日志的隔离
    """

    def wrap_logger(fn):
        def inner(msg, *args, **kwargs):
            return fn(ranks_str + msg, *args, **kwargs)

        return inner

    logger.info = wrap_logger(logger.info)
    logger.warning = wrap_logger(logger.warning)
    logger.error = wrap_logger(logger.error)


def multi_ranks_compare(compare_func, input_param_nr_list, output_path, kwargs):
    """
    将多卡数据分成多进程后，单进程内可能还有多张卡的数据，因此还需要多次比对
    """
    rank_list = [input_param_nr[1] for input_param_nr in input_param_nr_list]  # input_param_nr内部数据结构，2元素tuple
    ranks_str = f"[{' '.join(rank_list)}]"
    mp_logger_init(ranks_str)
    for input_param_nr in input_param_nr_list:
        input_param, nr = input_param_nr
        compare_entry(compare_func, input_param, output_path, nr, kwargs)


def compare_entry(compare_func, input_param, output_path, nr, kwargs):
    try:
        compare_func(input_param=input_param, output_path=output_path, suffix=f'_{nr}', **kwargs)
    except CompareException as e:
        if e.code == CompareException.INVALID_DATA_ERROR:
            logger.error(f"Invalid or missing 'data' in dump.json. Skipping {nr} comparison.")
        if e.code == CompareException.INVALID_TASK_ERROR:
            logger.error(f"Invalid or missing 'task' in dump.json. Skipping {nr} comparison.")


def compare_distributed_inner(npu_dump_dir, bench_dump_dir, output_path, **kwargs):
    def extract_compare_param(_file_type):
        npu_data_dir = os.path.join(npu_dump_dir, nr)
        bench_data_dir = os.path.join(bench_dump_dir, br)
        npu_path = extract_json(npu_data_dir, _file_type)
        bench_path = extract_json(bench_data_dir, _file_type)
        if npu_path == "" or bench_path == "":
            logger.debug(f'Did not find paired {_file_type} in {nr} and {br}, skip comparing.')
            return {}, True
        _input_param = {
            'npu_path': npu_path,
            'bench_path': bench_path,
            'is_print_compare_log': kwargs.get('is_print_compare_log', False)
        }
        return _input_param, False

    if kwargs.get('suffix'):
        logger.error("Argument 'suffix' is not supported for compare_distributed.")
        raise CompareException(CompareException.INVALID_PARAM_ERROR)

    npu_ranks, bench_ranks = get_sorted_ranks(npu_dump_dir, bench_dump_dir)
    if not kwargs.get('first_diff_analyze', False) and not npu_ranks:
        npu_proc_dir = find_proc_dir(npu_dump_dir)
        bench_proc_dir = find_proc_dir(bench_dump_dir)
        if npu_proc_dir and bench_proc_dir:
            npu_ranks = [os.path.basename(npu_proc_dir)]
            bench_ranks = [os.path.basename(bench_proc_dir)]
        else:
            logger.error("No dir found to compare, please check!")
            raise CompareException(CompareException.INVALID_PATH_ERROR)

    # ------------------预载rank0的json用于判断是什么类型dump数据------------------
    # 判断是否存在dump.json或debug.json
    if npu_ranks:
        dir_path = os.path.join(npu_dump_dir, npu_ranks[0])
        dump_file = os.path.join(dir_path, 'dump.json')
        debug_file = os.path.join(dir_path, 'debug.json')

        # 确定pre_check_dump_path
        if os.path.exists(dump_file):
            pre_check_dump_path = dump_file
        elif os.path.exists(debug_file):
            pre_check_dump_path = debug_file
        else:
            pre_check_dump_path = ''
    else:
        pre_check_dump_path = ''

    # 如果pre_check_dump_path为空，直接返回
    if not pre_check_dump_path:
        return

    dump_data = load_json(pre_check_dump_path)

    compare_framework = dump_data.get("framework", None)
    if not compare_framework:
        logger.warning(f"Unable to get framework information from dump.json, default setting is {Const.PT_FRAMEWORK}.")
        compare_framework = Const.PT_FRAMEWORK
    if compare_framework == Const.PT_FRAMEWORK:
        from msprobe.pytorch.compare.pt_compare import pt_compare
        compare_func = pt_compare
    elif compare_framework == Const.MS_FRAMEWORK:
        from msprobe.mindspore.compare.ms_compare import ms_compare
        compare_func = ms_compare
    else:
        logger.error(f"Unrecognized framework, now is {compare_framework}, please check dump.json.")
        raise CompareException(CompareException.INVALID_TASK_ERROR)

    compare_json_file_type = Const.DUMP_JSON_FILE
    if dump_data.get('level') == Const.LEVEL_DEBUG:
        compare_json_file_type = Const.DEBUG_JSON_FILE
    # ------------------统计量、md5比对------------------
    if dump_data.get('task') == Const.STATISTICS:
        # dump数据为统计量或md5时，多进程加速比对
        input_param_nr_list = []
        for nr, br in zip(npu_ranks, bench_ranks):
            input_param, skip = extract_compare_param(compare_json_file_type)
            if not skip:
                input_param_nr_list.append((input_param, nr))
        func_args = (compare_func, input_param_nr_list, output_path, kwargs)
        multi_statistics_compare(multi_ranks_compare, func_args)
        return

    # ------------------真实数据比对------------------
    for nr, br in zip(npu_ranks, bench_ranks):
        input_param, skip = extract_compare_param(compare_json_file_type)
        if not skip:
            compare_entry(compare_func, input_param, output_path, nr, kwargs)


def check_input_param_path(input_param):
    npu_path = input_param.get("npu_path", None)
    bench_path = input_param.get("bench_path", None)
    check_file_or_directory_path(npu_path)
    check_file_or_directory_path(bench_path)


def get_compare_framework(target_path, golden_path):
    target_dump_json_content = load_json(target_path)
    golden_dump_json_content = load_json(golden_path)
    target_framework = target_dump_json_content.get("framework", None)
    if not target_framework:
        logger.warning(f"Unable to get framework information from target dump.json, "
                       f"default setting is {Const.PT_FRAMEWORK}.")
        target_framework = Const.PT_FRAMEWORK
    golden_framework = golden_dump_json_content.get("framework", None)
    if not golden_framework:
        logger.warning(f"Unable to get framework information from golden dump.json, "
                       f"default setting is {Const.PT_FRAMEWORK}.")
        golden_framework = Const.PT_FRAMEWORK
    if target_framework == Const.PT_FRAMEWORK and golden_framework == Const.PT_FRAMEWORK:
        frame_name = Const.PT_FRAMEWORK
    elif (target_framework in [Const.MS_FRAMEWORK, Const.MT_FRAMEWORK] and
          golden_framework in [Const.PT_FRAMEWORK, Const.MS_FRAMEWORK]):
        frame_name = Const.MS_FRAMEWORK
    else:
        logger.error(f"Unrecognized framework, target now is {target_framework}, golden now is {golden_framework},"
                     f" please check dump.json.")
        raise CompareException(CompareException.INVALID_TASK_ERROR)
    return frame_name


def check_input_param_path_and_framework(args, target_framework):
    compare_framework = get_compare_framework(args.target_path, args.golden_path)
    if compare_framework != target_framework:
        logger.error(f"Expected frame to be {target_framework}, actual is {compare_framework}, please check dump.json.")
        raise CompareException(CompareException.INVALID_TASK_ERROR)

    if target_framework == Const.PT_FRAMEWORK and not is_module_available("torch"):
        logger.error("PyTorch does not exist, please install PyTorch library")
        raise Exception("PyTorch does not exist, please install PyTorch library")
    if target_framework == Const.MS_FRAMEWORK and not is_module_available("mindspore"):
        logger.error("MindSpore does not exist, please install MindSpore library")
        raise Exception("MindSpore does not exist, please install MindSpore library")


def split_tensors(data_name):
    tensor_list = data_name.split(';')
    return tensor_list
