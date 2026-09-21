#!/usr/bin/env python3
# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
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
# pylint: disable=duplicate-code

import os
from collections import namedtuple
import importlib

import torch

try:
    import torch_npu
except ImportError:
    IS_GPU = True
else:
    IS_GPU = False

from msprobe.pytorch.common.log import logger
from msprobe.pytorch.common.utils import save_pt
from msprobe.core.common.file_utils import create_directory
from msprobe.core.common.const import Const
from msprobe.core.common.utils import CompareException

ApiData = namedtuple(
    'ApiData', ['name', 'args', 'kwargs', 'result', 'step', 'rank'], defaults=['unknown', None, None, None, 0, 0]
)


class DumpException(CompareException):
    pass


def check_object_type(check_object, allow_type):
    """
    Function Description:
        Check if the object belongs to a certain data type
    Parameter:
        check_object: the object to be checked
        allow_type: legal data type
    Exception Description:
        when invalid data throw exception
    """
    if not isinstance(check_object, allow_type):
        logger.error(f"{check_object} not of {allow_type} type")
        raise CompareException(CompareException.INVALID_DATA_ERROR)


class SoftlinkCheckException(Exception):
    pass


def check_need_convert(api_name):
    convert_type = None
    for key, value in Const.CONVERT_API.items():
        if api_name not in value:  # pylint: disable=no-else-continue
            continue
        else:
            convert_type = key
    return convert_type


def cross_entropy_process(api_info_dict):
    """
    Function Description:
        Preprocesses the cross_entropy API information.
    Parameter:
        api_info_dict: argument of the API.
    Return api_info_dict:
        api_info_dict: Processed argument of the API.
    """
    if (
        'input_args' in api_info_dict
        and len(api_info_dict['input_args']) > 1
        and 'Min' in api_info_dict['input_args'][1]
    ):
        if api_info_dict['input_args'][1]['Min'] <= 0:
            # The second argument in cross_entropy should be -100 or not less than 0
            api_info_dict['input_args'][1]['Min'] = 0
    return api_info_dict


def histc_process(api_info_dict):
    input_args = api_info_dict['input_args']
    if input_args and input_args[0].get('dtype'):
        dtype = input_args[0]['dtype']
        if dtype in Const.TORCH_INT_DTYPE:
            api_info_dict['input_args'][0]['dtype'] = Const.TORCH_FLOAT32
    return api_info_dict


API_PROCESS_MAP = {'cross_entropy': cross_entropy_process, 'histc': histc_process}


def api_info_preprocess(api_name, api_info_dict):
    """
    Function Description:
        Preprocesses the API information.
    Parameter:
        api_name: Name of the API.
        api_info_dict: argument of the API.
    Return api_info_dict:
        convert_type: Type of conversion.
        api_info_dict: Processed argument of the API.
    """
    convert_type = check_need_convert(api_name)
    if api_name in API_PROCESS_MAP:
        api_info_dict = API_PROCESS_MAP[api_name](api_info_dict)
    return convert_type, api_info_dict


def initialize_save_path(save_path, dir_name):
    data_path = os.path.join(save_path, dir_name)
    create_directory(data_path)
    return data_path


def get_full_data_path(data_path, real_data_path):
    if not data_path:
        return data_path
    full_data_path = os.path.join(real_data_path, data_path)
    return os.path.realpath(full_data_path)


class UtDataProcessor:
    def __init__(self, save_path):
        self.save_path = save_path
        self.index = 0

    def save_tensors_in_element(self, api_name, element):
        self.index = 0
        self._save_recursive(api_name, element)

    def _save_recursive(self, api_name, element, depth=0):
        if depth > Const.MAX_DEPTH:
            logger.error(f"Maximum depth of {Const.MAX_DEPTH} exceeded for {api_name}")
            raise DumpException(DumpException.RECURSION_LIMIT_ERROR)
        if isinstance(element, torch.Tensor):
            api_args = api_name + Const.SEP + str(self.index)
            create_directory(self.save_path)
            file_path = os.path.join(self.save_path, f'{api_args}.pt')
            try:
                tensor = element.contiguous().detach().cpu()
            except Exception as err:
                logger.error(f"Failed to transfer tensor to cpu for {api_args}")
                raise DumpException(DumpException.INVALID_DATA_ERROR) from err
            save_pt(tensor, file_path)
            self.index += 1
        elif element is None or isinstance(element, (bool, int, float, str, slice)):
            self.index += 1
        elif isinstance(element, (list, tuple)):
            for item in element:
                self._save_recursive(api_name, item, depth=depth + 1)
        elif isinstance(element, dict):
            for value in element.values():
                self._save_recursive(api_name, value, depth=depth + 1)
        else:
            self.index += 1


def extract_basic_api_segments(api_full_name):
    """
    Function Description:
        Extract the name of the API.
    Parameter:
        api_full_name: Full name of the API. Example: torch.matmul.0, torch.linalg.inv.0
    Return:
        api_type: Type of api. Example: torch, tensor, etc.
        api_name: Name of api. Example: matmul, linalg.inv, etc.
    """
    api_type = None
    api_parts = api_full_name.split(Const.SEP)
    api_parts_length = len(api_parts)
    if api_parts_length == Const.THREE_SEGMENT:
        api_type, api_name, _ = api_parts
    elif api_parts_length == Const.FOUR_SEGMENT:
        api_type, prefix, api_name, _ = api_parts
        api_name = Const.SEP.join([prefix, api_name])
    else:
        api_name = None
    return api_type, api_name


def extract_detailed_api_segments(full_api_name_with_direction_status):
    """
    Function Description:
        Extract the name of the API.
    Parameter:
        full_api_name_with_direction_status: Full name of the API. Example: torch.matmul.0.forward.output.0
    Return:
        api_name: Name of api. Example: matmul, mul, etc.
        full_api_name: Full name of api. Example: torch.matmul.0
        direction_status: Direction status of api. Example: forward, backward, etc.
    """
    api_type = None
    prefix = None
    api_name = None
    direction_status = None
    api_parts = full_api_name_with_direction_status.split(Const.SEP)
    api_parts_length = len(api_parts)
    if api_parts_length == Const.SIX_SEGMENT:
        api_type, api_name, api_order, direction_status, _, _ = api_parts
        full_api_name = Const.SEP.join([api_type, api_name, api_order])
    elif api_parts_length == Const.SEVEN_SEGMENT:
        api_type, prefix, api_name, api_order, direction_status, _, _ = api_parts
        full_api_name = Const.SEP.join([api_type, prefix, api_name, api_order])
        api_name = Const.SEP.join([prefix, api_name])
    else:
        full_api_name = None
    return api_name, full_api_name, direction_status


def get_module_and_atttribute_name(attribute):
    '''
    Function Description:
        Get the module and attribute name.
    Parameter:
        name: Attribute of a module. Example: torch.float16
    Return:
        module_name: Name of the module. Example: torch.
        attribute_name: Name of the attribute. Example: float16.
    '''
    try:
        module_name, attribute_name = attribute.split(Const.SEP)
    except ValueError as e:
        logger.error(f"Failed to get module and attribute name from {attribute}")
        raise CompareException(CompareException.INVALID_DATA_ERROR) from e
    return module_name, attribute_name


def get_attribute(module_name, attribute_name):
    '''
    Function Description:
        Get the attribute of the module.
    Parameter:
        module_name: Name of the module.
        attribute_name: Name of the attribute.
    '''
    attribute = None
    if module_name not in Const.MODULE_WHITE_LIST:
        logger.error(f"Module {module_name} is not in white list")
        raise CompareException(CompareException.INVALID_DATA_ERROR)
    try:
        module = importlib.import_module(module_name)
        attribute = getattr(module, attribute_name)
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to get attribute {attribute_name} from module {module_name}: {e}")
        raise CompareException(CompareException.INVALID_ATTRIBUTE_ERROR) from e
    return attribute


def is_dtype_fp8(dtype):
    """
    Function Description:
        Check if the data type is float8.
    Parameter:
        dtype: Data type (torch.dtype or string).
    Return:
        True or False.
    """
    # 处理字符串类型的 dtype
    if isinstance(dtype, str):
        return dtype in ["torch.float8_e4m3fn", "torch.float8_e5m2"]

    # 处理 torch.dtype 类型
    return dtype in [torch.float8_e4m3fn, torch.float8_e5m2]


def is_dtype_hif8(dtype):
    """
    Function Description:
        Check if the data type is HiFloat8Tensor.
    Parameter:
        dtype: Data type (string).
    Return:
        True or False.
    """
    # 处理字符串类型的 dtype
    if isinstance(dtype, str):
        dtype_str = dtype
    # 处理类对象或 dtype 对象
    else:
        dtype_str = str(dtype)

    # 检查是否匹配 HiFloat8Tensor 的字符串表示
    return (
        dtype_str == "<class 'torch_npu.utils.hif8_tensor.HiFloat8Tensor'>" or dtype_str == "torch_npu.HiFloat8Tensor"  # pylint: disable=consider-using-in
    )


def is_dtype_fp8_or_hif8(dtype):
    """
    Function Description:
        Check if the data type is FP8 (native or HiFloat8 variant).
    Parameters:
        dtype: Data type (torch.dtype, class, or string).
    Returns:
        True if dtype is FP8 or HiFloat8, False otherwise.
    """
    return is_dtype_fp8(dtype) or is_dtype_hif8(dtype)


def is_hifloat8_tensor(tensor):
    if not IS_GPU and hasattr(torch_npu, "HiFloat8Tensor") and isinstance(tensor, torch_npu.HiFloat8Tensor):
        return True
    return False
