# coding=utf-8
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

"""
Function:
This file mainly involves the dump data proto function.
"""

import numpy as np

from cmp_utils import log
from cmp_utils.constant.const_manager import ConstManager, DD
from cmp_utils.constant.compare_error import CompareError


def contain_depth_dimension(tensor_format: any) -> bool:
    """
    Contain depth dimension
    :param tensor_format: the tensor format
    :return: bool, if true contain depth dimension
    """
    return tensor_format in [DD.FORMAT_NDHWC, DD.FORMAT_NCDHW, DD.FORMAT_DHWCN, DD.FORMAT_NDC1HWC0]


def get_format_string(dump_format: any) -> str:
    """
    Get format string by dump_format
    :param dump_format: the dump format
    :return: the string
    """

    for key, value in ConstManager.STRING_TO_FORMAT_MAP.items():
        if dump_format == value:
            return key
    message = "The format({}) does not support.".format(str(dump_format))
    log.print_error_log(message)
    raise CompareError(CompareError.MSACCUCMP_INVALID_FORMAT_ERROR, message)


def get_data_type_by_dtype(dtype: any) -> any:
    """
    Get data type by dtype
    :param dtype: the numpy dtype
    :return:the OutputDataType
    """
    if dtype == np.double:
        return DD.DT_DOUBLE
    for (key, value) in list(ConstManager.DATA_TYPE_TO_DTYPE_MAP.items()):
        if dtype == value.get(ConstManager.DTYPE_KEY):
            return key
    message = "The dtype({}) does not support.".format(str(dtype))
    log.print_error_log(message)
    raise CompareError(CompareError.MSACCUCMP_INVALID_DATA_TYPE_ERROR, message)


def get_dtype_by_data_type(data_type: any) -> any:
    """
    Get dtype by output data type
    :param data_type: OutputDataType
    :return: the dtype
    """
    if data_type not in ConstManager.DATA_TYPE_TO_DTYPE_MAP:
        message = "The output data type ({}) does not support." .format(str(data_type))
        log.print_error_log(message)
        raise CompareError(CompareError.MSACCUCMP_INVALID_DATA_TYPE_ERROR, message)
    dtype = ConstManager.DATA_TYPE_TO_DTYPE_MAP.get(data_type).get(ConstManager.DTYPE_KEY)
    if dtype == 'bfloat16':
        try:
            from ml_dtypes import bfloat16
        except ModuleNotFoundError as ee:
            raise TypeError('bfloat16 is not supported in numpy, run `pip install ml_dtypes` for support.') from ee
        dtype = bfloat16
    return dtype


def get_struct_format_by_data_type(data_type: any) -> any:
    """
    Get the struct format for data type
    :param data_type: the data type
    :return: struct format
    """
    if data_type not in ConstManager.DATA_TYPE_TO_DTYPE_MAP:
        message = "The output data type({}) does not support.".format(str(data_type))
        log.print_error_log(message)
        raise CompareError(CompareError.MSACCUCMP_INVALID_DATA_TYPE_ERROR, message)
    return ConstManager.DATA_TYPE_TO_DTYPE_MAP.get(data_type).get(ConstManager.STRUCT_FORMAT_KEY)


def get_sub_format(tensor: any, default_value: int = 0) -> int:
    """
    Adapt group conv sub_format, to get group
    :param tensor: the tensor
    :param default_value: default value for sub format
    :return: the value of sub format
    """
    sub_format = default_value
    if hasattr(tensor, 'sub_format'):
        sub_format = tensor.sub_format
    return sub_format
