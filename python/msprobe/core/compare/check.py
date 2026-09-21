# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

from msprobe.core.common.log import logger
from msprobe.core.common.utils import check_op_str_pattern_valid, CompareException
from msprobe.core.common.const import Const

cross_dtype_mapping = {
    "Int8": "int",
    "torch.int8": "int",
    "UInt8": "int",
    "torch.uint8": "int",
    "Int16": "int",
    "torch.int16": "int",
    "UInt16": "int",
    "torch.uint16": "int",
    "Int32": "int",
    "torch.int32": "int",
    "UInt32": "int",
    "torch.uint32": "int",
    "Int64": "int",
    "torch.int64": "int",
    "UInt64": "int",
    "torch.uint64": "int",
    "Float16": "float",
    "torch.float16": "float",
    "Float32": "float",
    "torch.float32": "float",
    "Float64": "float",
    "torch.float64": "float",
    "BFloat16": "float",
    "torch.bfloat16": "float",
    "Bool": "bool",
    "torch.bool": "bool",
    "Complex64": "complex",
    "torch.complex64": "complex",
    "Complex128": "complex",
    "torch.complex128": "complex",
}


def check_dump_json_str(op_data, op_name):
    input_list = (
        op_data.get(Const.INPUT_ARGS, None) if op_data.get(Const.INPUT_ARGS, None) else op_data.get(Const.INPUT, None)
    )
    input_kwargs = op_data.get(Const.INPUT_KWARGS, None)
    output_list = op_data.get(Const.OUTPUT, None)

    args = [input_list, input_kwargs, output_list]
    for arg in args:
        if not arg:
            continue
        if isinstance(arg, dict):
            check_json_key_value(arg, op_name)
        else:
            for ele in arg:
                if not ele:
                    continue
                check_json_key_value(ele, op_name)


def check_json_key_value(input_output, op_name, depth=0):
    if depth > Const.MAX_DEPTH:
        logger.error(f"string check of data info of {op_name} exceeds the recursion limit.")
        return
    if isinstance(input_output, list):
        for item in input_output:
            check_json_key_value(item, op_name, depth + 1)
    elif isinstance(input_output, dict):
        for key, value in input_output.items():
            if isinstance(value, dict):
                check_json_key_value(value, op_name, depth + 1)
            else:
                valid_key_value(key, value, op_name)


def valid_key_value(key, value, op_name):
    if key == "shape" and not isinstance(value, (list, tuple)):
        logger.error(f"shape of input or output of {op_name} is not list or tuple, please check!")
        raise CompareException(CompareException.INVALID_OBJECT_TYPE_ERROR)
    elif key == "requires_grad" and not isinstance(value, bool):
        logger.error(f"requires_grad of input or output of {op_name} is not bool, please check!")
        raise CompareException(CompareException.INVALID_OBJECT_TYPE_ERROR)
    else:
        check_op_str_pattern_valid(value, op_name)


def check_stack_json_str(stack_info, op_name):
    if isinstance(stack_info, list):
        for item in stack_info:
            check_op_str_pattern_valid(item, op_name, stack=True)
    else:
        logger.error(f"Expected stack_info to be a list, but got {type(stack_info).__name__} for '{op_name}'")
        raise CompareException(CompareException.INVALID_OBJECT_TYPE_ERROR)


def check_configuration_param(config):
    arg_list = [config.stack_mode, config.fuzzy_match, config.first_diff_analyze, config.is_print_compare_log]
    arg_names = ['stack_mode', 'fuzzy_match', 'first_diff_analyze', 'is_print_compare_log']
    for arg, name in zip(arg_list, arg_names):
        if not isinstance(arg, bool):
            logger.error(f"Invalid input parameter, {name} which should be only bool type.")
            raise CompareException(CompareException.INVALID_PARAM_ERROR)


def check_consistent_param(consistent_check: bool, backend: str):
    if consistent_check:
        if backend not in [Const.FSDP, Const.MEGATRON]:
            logger.error(
                f"If the 'consistent_check' parameter is specified, "
                f"the 'backend' parameter must be set to '{Const.FSDP}' or '{Const.MEGATRON}'. "
                f"Current 'backend' parameter is {backend}."
            )
            raise CompareException(CompareException.INVALID_PARAM_ERROR)
