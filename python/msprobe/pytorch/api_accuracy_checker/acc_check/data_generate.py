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
import math
import torch
import numpy

try:
    import torch_npu
except ImportError:
    IS_GPU = True
else:
    IS_GPU = False

from msprobe.pytorch.api_accuracy_checker.acc_check.acc_check_utils import hf_32_standard_api
from msprobe.pytorch.api_accuracy_checker.common.utils import (
    check_object_type,
    get_full_data_path,
    CompareException,
    get_module_and_atttribute_name,
    get_attribute,
    is_dtype_fp8,
    is_dtype_hif8,
    is_hifloat8_tensor,
    is_dtype_fp8_or_hif8,
)
from msprobe.core.common.file_utils import FileChecker, load_npy
from msprobe.pytorch.common.log import logger
from msprobe.pytorch.common.utils import load_pt
from msprobe.core.common.const import Const, FileCheckConst, CompareConst


TORCH_TYPE = ["torch.device", "torch.dtype"]
TENSOR_DATA_LIST = ["torch.Tensor", "torch.nn.parameter.Parameter"]
FLOAT_TYPE = [
    'torch.float32',
    'torch.float',
    'torch.float64',
    'torch.double',
    'torch.float16',
    'torch.half',
    'torch.bfloat16',
    'torch.float8_e4m3fn',
    'torch.float8_e5m2',
    'torch_npu.HiFloat8Tensor',
]
NUMPY_TYPE = [
    "numpy.int8",
    "numpy.int16",
    "numpy.int32",
    "numpy.int64",
    "numpy.uint8",
    "numpy.uint16",
    "numpy.uint32",
    "numpy.uint64",
    "numpy.float16",
    "numpy.float32",
    "numpy.float64",
    "numpy.float128",
    "numpy.complex64",
    "numpy.complex128",
    "numpy.complex256",
    "numpy.bool_",
    "numpy.string_",
    "numpy.bytes_",
    "numpy.unicode_",
]


def gen_data(info, api_name, need_grad, convert_type, real_data_path=None):
    """
    Function Description:
        Based on arg basic information, generate arg data
    Parameter:
        info: arg basic information. Dict
        api_name: API name
        need_grad: set Tensor grad for backward
        convert_type: convert ori_type to dist_type flag.
    """
    check_object_type(info, dict)
    data_type = info.get('type')
    data_path = info.get('datapath', info.get('data_name'))
    data_path = get_full_data_path(data_path, real_data_path)
    dtype = info.get('dtype')
    if is_dtype_fp8_or_hif8(dtype) and IS_GPU:
        raise CompareException("GPU does not need to support float8 data type")
    if data_type in TENSOR_DATA_LIST:
        if data_path:
            data = gen_real_tensor(data_path, convert_type)
        else:
            data = gen_random_tensor(info, convert_type)
        if api_name in hf_32_standard_api and data.dtype == torch.float32:
            data = fp32_to_hf32_to_fp32(data)
        if info.get('requires_grad') and need_grad:
            if is_hifloat8_tensor(data):
                origin_dtype = info.get('dtype')
            else:
                origin_dtype = data.dtype
            if is_dtype_fp8(origin_dtype):
                data = data.to(torch.float32)
            data.requires_grad_(True)
            temp_data = data * 1
            data = temp_data.type_as(data)
            data.retain_grad()
            if is_dtype_fp8(origin_dtype):
                data = data.to(origin_dtype)
            if is_dtype_hif8(origin_dtype):
                data = torch_npu.HiFloat8Tensor.to_hifloat8(data)
    elif data_type.startswith("numpy"):
        if data_type not in NUMPY_TYPE:
            raise Exception("{} is not supported now".format(data_type))  # pylint: disable=broad-exception-raised
        data = info.get("value")
        try:
            module_name, attribute_name = get_module_and_atttribute_name(data_type)
            data = get_attribute(module_name, attribute_name)(data)
        except Exception as err:
            logger.error("Failed to convert the type to numpy: %s" % str(err))
    elif data_type == "torch.Size":
        data = torch.Size(info.get("value"))
    else:
        data = info.get('value')
        if info.get("type") == "slice":
            data = slice(*data)
        if info.get("type") == "ellipsis":
            data = ...
    return data


def gen_real_tensor(data_path, convert_type):
    """
    Function Description:
        Based on API data path, generate input parameters real data
    Parameter:
        data_path: API data path
        convert_type: convert ori_type to dist_type flag.
    """
    data_path = os.path.realpath(data_path)
    data_path_checker = FileChecker(data_path, FileCheckConst.FILE, ability=FileCheckConst.READ_ABLE)
    data_path = data_path_checker.common_check()
    if not data_path.endswith('.pt') and not data_path.endswith('.npy'):
        error_info = f"The file: {data_path} is not a pt or numpy file."
        raise CompareException(CompareException.INVALID_FILE_ERROR, error_info)
    if data_path.endswith('.pt'):
        data = load_pt(data_path, to_cpu=True)
    else:
        data_np = load_npy(data_path)
        data = torch.from_numpy(data_np)
    if convert_type:
        ori_dtype = Const.CONVERT.get(convert_type)[0]
        dist_dtype = Const.CONVERT.get(convert_type)[1]
        module_name, attribute_name = get_module_and_atttribute_name(dist_dtype)
        if str(data.dtype) == ori_dtype:
            data = data.type(get_attribute(module_name, attribute_name))
    return data


def gen_random_tensor(info, convert_type):
    """
    Function Description:
        Based on API MAX and MIN, generate input parameters random data
    Parameter:
        info: API data info
        convert_type: convert ori_type to dist_type flag.
    """
    check_object_type(info, dict)

    low_origin = info.get('Min')
    low = info.get('Min_except_inf_nan', low_origin)
    high_origin = info.get('Max')
    high = info.get('Max_except_inf_nan', high_origin)

    low_info = [low, low_origin]
    high_info = [high, high_origin]
    data_dtype = info.get('dtype')
    shape = tuple(info.get('shape'))
    if 0 in shape:
        low, low_origin = 0, 0
        high, high_origin = 0, 0
        low_info = [low, low_origin]
        high_info = [high, high_origin]
    elif not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
        error_info = f'Data info Min: {low} , Max: {high}, info type must be int or float.'
        raise CompareException(CompareException.INVALID_PARAM_ERROR, error_info)
    if data_dtype == "torch.bool":
        data = gen_bool_tensor(low, high, shape)
    else:
        data = gen_common_tensor(low_info, high_info, shape, data_dtype, convert_type)
    return data


def fp32_to_hf32_to_fp32(input_tensor):
    # 将输入的float32 tensor转为hf32 tensor，再转为float32 tensor
    input_np = input_tensor.detach().numpy()
    input_int = input_np.view(numpy.int32)
    input_int = numpy.right_shift(numpy.right_shift(input_int, 11) + 1, 1)
    input_int = numpy.left_shift(input_int, 12)
    input_fp32 = input_int.view(numpy.float32)
    input_hf32 = torch.from_numpy(input_fp32)
    return input_hf32


def gen_common_tensor(low_info, high_info, shape, data_dtype, convert_type):
    """
    Function Description:
        Based on API basic information, generate int or float tensor
    Parameter:
        low_info: [low, low_origin], low is the minimum value in the tensor removed inf and nan,
        low_origin is the original minimum value in the tensor
        high_info: [high, high_origin], high is the maximum value in the tensor removed inf and nan,
        high_origin is the original maximum value in the tensor
        shape:The shape of Tensor
        data_dtype: The data type of Tensor
        convert_type: convert ori_type to dist_type flag.
    """
    if convert_type:
        ori_dtype = Const.CONVERT.get(convert_type)[0]
        if ori_dtype == data_dtype:
            data_dtype = Const.CONVERT.get(convert_type)[1]
    low, low_origin = low_info[0], low_info[1]
    high, high_origin = high_info[0], high_info[1]
    module_name, attribute_name = get_module_and_atttribute_name(data_dtype)
    dtype = get_attribute(module_name, attribute_name)
    if data_dtype in FLOAT_TYPE:
        if math.isnan(high):
            tensor = torch.full(shape, float('nan'), dtype=dtype)
            return tensor
        # high_origin为新版json中的属性，只有当high_origin不为None,且high为inf或-inf时，原tensor全为inf或-inf
        if high_origin and high in [float(CompareConst.INF), float(CompareConst.NEG_INF)]:
            tensor = torch.full(shape, high, dtype=dtype)
            tensor[-1] = low
            return tensor

        low_scale, high_scale = low, high
        if is_dtype_hif8(dtype):
            finfo_dtype = torch.float32
        else:
            finfo_dtype = dtype
        dtype_finfo = torch.finfo(finfo_dtype)
        # 适配老版json high和low为inf或-inf的情况，取dtype的最大值或最小值进行放缩
        if high == float(CompareConst.INF):
            high_scale = dtype_finfo.max
        elif high == float(CompareConst.NEG_INF):
            high_scale = dtype_finfo.min
        if low == float(CompareConst.INF):
            low_scale = dtype_finfo.max
        elif low == float(CompareConst.NEG_INF):
            low_scale = dtype_finfo.min

        scale = high_scale - low_scale
        if is_dtype_fp8(dtype) or is_dtype_hif8(dtype):
            generate_dtype = torch.float32
        else:
            generate_dtype = dtype
        rand01 = torch.rand(shape, dtype=generate_dtype)
        tensor = rand01 * scale + low_scale
    elif 'int' in data_dtype or 'long' in data_dtype:
        low, high = int(low), int(high)
        tensor = torch.randint(low, high + 1, shape, dtype=dtype)
    else:
        logger.error('Dtype is not supported: ' + data_dtype)
        raise NotImplementedError()
    if tensor.nelement() == 0:
        return tensor
    tmp_tensor = tensor.reshape(-1)
    if high_origin and math.isnan(high_origin):
        if tmp_tensor.numel() <= 2:
            tmp_tensor[0] = float('nan')
            tmp_tensor[-1] = high
        else:
            tmp_tensor[0] = low
            tmp_tensor[1] = float('nan')
            tmp_tensor[-1] = high
    else:
        tmp_tensor[0] = low
        tmp_tensor[-1] = high
        if high_origin in [float(CompareConst.INF), float(CompareConst.NEG_INF)]:
            tmp_tensor[-1] = high_origin
        if low_origin in [float(CompareConst.INF), float(CompareConst.NEG_INF)]:
            tmp_tensor[0] = low_origin
    data = tmp_tensor.reshape(shape)
    if is_dtype_fp8(dtype):
        data = data.to(dtype)
    if is_dtype_hif8(dtype):
        if IS_GPU:
            raise CompareException("GPU does not support torch_npu.HiFloat8Tensor")
        data = torch_npu.HiFloat8Tensor.to_hifloat8(data)
    return data


def gen_bool_tensor(low, high, shape):
    """
    Function Description:
        Based on API basic information, generate bool tensor
    Parameter:
        low: The minimum value in Tensor
        high: The max value in Tensor
        shape:The shape of Tensor
    """
    low, high = int(low), int(high)
    if low > high:
        low, high = high, low
    tensor = torch.randint(low, high + 1, shape)
    data = torch.gt(tensor, 0)
    return data


def gen_args(args_info, api_name, func_options):
    """
    Function Description:
        Based on API basic information, generate input parameters: args, for API forward running
    Parameter:
        api_info: API basic information. List
        api_name: API name
        need_grad: set Tensor grad for backward
        convert_type: convert ori_type to dist_type flag.
        real_data_path: the root directory for storing real data.
    """
    check_object_type(args_info, list)
    args_result = []

    need_grad = func_options.get('need_grad', True)
    convert_type = func_options.get('convert_type', None)
    real_data_path = func_options.get('real_data_path', None)
    depth = func_options.get('depth', 0)

    if depth > Const.MAX_DEPTH:
        logger.error("The depth of args is too large, please check the input args.")
        raise CompareException(CompareException.RECURSION_LIMIT_ERROR)

    for arg in args_info:
        if isinstance(arg, (list, tuple)):
            func_options['depth'] = depth + 1
            data = gen_args(arg, api_name, func_options)
        elif isinstance(arg, dict):
            data = gen_data(arg, api_name, need_grad, convert_type, real_data_path)
        elif arg is None:
            data = None
        else:
            logger.warning(f'Warning: {arg} is not supported')
            raise NotImplementedError()
        args_result.append(data)
    return args_result


def gen_kwargs(api_info, api_name, convert_type=None, real_data_path=None):
    """
    Function Description:
        Based on API basic information, generate input parameters: kwargs, for API forward running
    Parameter:
        api_info: API basic information. Dict
        api_name: API name
        convert_type: convert ori_type to dist_type flag.
        real_data_path: the root directory for storing real data.
    """
    check_object_type(api_info, dict)
    kwargs_params = api_info.get("input_kwargs")
    for key, value in kwargs_params.items():
        if isinstance(value, (list, tuple)):
            kwargs_params[key] = gen_list_kwargs(value, api_name, convert_type, real_data_path)
        elif value is None:
            kwargs_params[key] = None
        elif key == 'atten_mask' and api_name == 'npu_fusion_attention':
            sparse_mode = kwargs_params.get('sparse_mode', {})
            if isinstance(sparse_mode, dict):
                sparse_mode_value = sparse_mode.get('value', 0)
            elif isinstance(sparse_mode, int):
                sparse_mode_value = sparse_mode
            else:
                msg = f'The sparse_mode value is not int or dict, but {type(sparse_mode)}'
                raise CompareException(CompareException.INVALID_PARAM_ERROR, msg)
            if sparse_mode_value in Const.FA_SPECIAL_SPARSE_MODE:
                kwargs_params[key] = gen_atten_mask(value, convert_type, real_data_path)
            else:
                kwargs_params[key] = gen_data(value, api_name, True, convert_type, real_data_path)
        elif value.get('type') in TENSOR_DATA_LIST or (
            value.get('type') is not None and value.get('type').startswith("numpy")
        ):
            kwargs_params[key] = gen_data(value, api_name, True, convert_type, real_data_path)
        elif value.get('type') in TORCH_TYPE:
            gen_torch_kwargs(kwargs_params, key, value)
        else:
            kwargs_params[key] = value.get('value')
    return kwargs_params


def gen_atten_mask(info, convert_type, real_data_path):
    """
    Function Description:
        Based on API basic information, generate input parameters: atten_mask, for API forward running
    Parameter:
        info: API basic information. Dict
        convert_type: convert ori_type to dist_type flag.
        real_data_path: the root directory for storing real data.
    """
    check_object_type(info, dict)
    data_type = info.get('type')
    data_path = info.get('datapath', info.get('data_name'))
    data_path = get_full_data_path(data_path, real_data_path)
    data = None
    if data_type in TENSOR_DATA_LIST:
        if data_path:
            data = gen_real_tensor(data_path, convert_type)
        else:
            # 生成一个2048x2048的三角矩阵，对角线为1，其余为0
            # 这是npu_fusion_attention的sparse_mode为[2, 3, 4]时，atten_mask的shape
            data = torch.triu(torch.ones([2048, 2048]), diagonal=1).to(torch.bool)
    return data


def gen_torch_kwargs(kwargs_params, key, value):
    if value.get('type') != "torch.device":
        module_name, attribute_name = get_module_and_atttribute_name(value.get('value'))
        kwargs_params[key] = get_attribute(module_name, attribute_name)


def gen_list_kwargs(kwargs_item_value, api_name, convert_type, real_data_path=None):
    """
    Function Description:
        When kwargs value is list, generate the list of kwargs result
    Parameter:
        kwargs_item_value: kwargs value before to generate. List
        api_name: API name
        convert_type: convert ori_type to dist_type flag.
    """
    kwargs_item_result = []
    for item in kwargs_item_value:
        if item.get('type') in TENSOR_DATA_LIST:
            item_value = gen_data(item, api_name, False, convert_type, real_data_path)
        elif item.get('type') == "torch.Size":
            item_value = torch.Size(item.get('value'))
        else:
            item_value = item.get('value')
        kwargs_item_result.append(item_value)
    return kwargs_item_result


def get_output_dtype(api_info):
    """
    Function Description:
        Based on API basic information, get the output data dtype
    Parameter:
        api_info: API basic information. Dict
    """
    output_dtype = None
    output_info = api_info.get(Const.OUTPUT)
    if output_info and isinstance(output_info[0], dict):
        output_str_dtype = output_info[0].get(Const.DTYPE)
        if output_str_dtype in Const.TORCH_FLOAT_DTYPE:
            module_name, attribute_name = get_module_and_atttribute_name(output_str_dtype)
            output_dtype = get_attribute(module_name, attribute_name)
    return output_dtype


def gen_api_params(api_info, api_name, need_grad=True, convert_type=None, real_data_path=None):
    """
    Function Description:
        Based on API basic information, generate input parameters: args, kwargs, for API forward running
    Parameter:
        api_info: API basic information. Dict
        api_name: API name
        need_grad: set grad for backward
        convert_type: convert ori_type to dist_type flag.
    """
    check_object_type(api_info, dict)
    if convert_type and convert_type not in Const.CONVERT:
        error_info = f"convert_type params not support {convert_type}."
        raise CompareException(CompareException.INVALID_PARAM_ERROR, error_info)
    kwargs_params = gen_kwargs(api_info, api_name, convert_type, real_data_path)
    func_options = {'need_grad': need_grad, 'convert_type': convert_type, 'real_data_path': real_data_path, 'depth': 0}
    if api_info.get("input_args"):
        args_params = gen_args(api_info.get("input_args"), api_name, func_options)
    else:
        logger.warning(f'Warning: No args in {api_info} ')
        args_params = []
    output_dtype = get_output_dtype(api_info)
    return args_params, kwargs_params, output_dtype
