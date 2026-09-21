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

import time
import os
import math

import torch

from msprobe.core.common.utils import CompareException
from msprobe.core.common.file_utils import load_yaml
from msprobe.core.common.const import Const
from msprobe.pytorch.common.log import logger


current_time = time.strftime("%Y%m%d%H%M%S")
API_PRECISION_COMPARE_RESULT_FILE_NAME = "api_precision_compare_result_" + current_time + ".csv"
API_PRECISION_COMPARE_DETAILS_FILE_NAME = "api_precision_compare_details_" + current_time + ".csv"
BENCHMARK_COMPARE_SUPPORT_LIST = [
    'torch.float16',
    'torch.bfloat16',
    'torch.float32',
    "torch.float8_e4m3fn",
    "torch.float8_e5m2",
]
API_PRECISION_COMPARE_UNSUPPORT_LIST = ['torch.float64', 'torch.complex64', 'torch.complex128']
ULP_COMPARE_SUPPORT_LIST = ['torch.float16', 'torch.bfloat16', 'torch.float32']
BINARY_COMPARE_UNSUPPORT_LIST = BENCHMARK_COMPARE_SUPPORT_LIST + API_PRECISION_COMPARE_UNSUPPORT_LIST


cur_path = os.path.dirname(os.path.realpath(__file__))
standard_yaml_path = os.path.join(cur_path, "api_precision_standard.yaml")
apis = load_yaml(standard_yaml_path)
absolute_standard_api = apis.get('AbsoluteThreshStandard')
binary_standard_api = apis.get('BinaryCompareStandard')
ulp_standard_api = apis.get('ULPStandard')
thousandth_standard_api = apis.get('ThousandthStandard')
accumulative_error_standard_api = apis.get('AccumulativeErrorStandard')


DETAIL_TEST_ROWS = [
    [
        "API Name",
        "Bench Dtype",
        "DEVICE Dtype",
        "Shape",
        "余弦相似度",
        "最大绝对误差",
        "双百指标",
        "双千指标",
        "双万指标",
        "二进制一致错误率",
        "误差均衡性",
        "均方根误差",
        "小值域错误占比",
        "相对误差最大值",
        "相对误差平均值",
        "inf/nan错误率",
        "相对误差错误率",
        "绝对误差错误率",
        "ULP误差最大值",
        "ULP误差平均值",
        "ULP误差大于阈值占比",
        "Status",
        "Message",
    ]
]


precision_configs = {
    torch.float16: {'small_value': [1e-3], 'small_value_atol': [1e-5]},
    torch.bfloat16: {'small_value': [1e-3], 'small_value_atol': [1e-5]},
    torch.float32: {'small_value': [1e-6], 'small_value_atol': [1e-9]},
}


ULP_PARAMETERS = {
    torch.float16: {'min_eb': [-14], 'exponent_num': [10]},
    torch.bfloat16: {'min_eb': [-126], 'exponent_num': [7]},
    torch.float32: {'min_eb': [-126], 'exponent_num': [23]},
}


class ApiPrecisionCompareColumn:
    API_NAME = 'API Name'
    DEVICE_DTYPE = 'DEVICE Dtype'
    SHAPE = 'Shape'
    SMALL_VALUE_ERROR_RATE = '小值域错误占比'
    RMSE = '均方根误差'
    MAX_REL_ERR = '相对误差最大值'
    MEAN_REL_ERR = '相对误差平均值'
    EB = '误差均衡性'
    SMALL_VALUE_ERROR_RATIO = '小值域错误比值'
    SMALL_VALUE_ERROR_STATUS = '小值域判定结果'
    RMSE_RATIO = '均方根误差比值'
    RMSE_STATUS = '均方根误差判定结果'
    MAX_REL_ERR_RATIO = '相对误差最大值比值'
    MAX_REL_ERR_STATUS = '相对误差最大值判定结果'
    MEAN_REL_ERR_RATIO = '相对误差平均值比值'
    MEAN_REL_ERR_STATUS = '相对误差平均值判定结果'
    EB_RATIO = '误差均衡性比值'
    EB_STATUS = '误差均衡性判定结果'
    ERROR_RATE = '二进制一致错误率'
    ERROR_RATE_STATUS = '二进制一致错误率判定结果'
    INF_NAN_ERROR_RATIO = 'inf/nan错误率'
    INF_NAN_ERROR_RATIO_STATUS = 'inf/nan判定结果'
    REL_ERR_RATIO = '相对误差错误率'
    REL_ERR_RATIO_STATUS = '相对误差判定结果'
    ABS_ERR_RATIO = '绝对误差错误率'
    ABS_ERR_RATIO_STATUS = '绝对误差判定结果'
    MEAN_ULP_ERR = 'ULP误差平均值'
    ULP_ERR_PROPORTION = 'ULP误差大于阈值占比'
    ULP_ERR_PROPORTION_RATIO = 'ULP误差大于阈值占比比值'
    ULP_ERR_STATUS = 'ULP误差判定结果'
    REL_ERR_THOUSANDTH = '双千指标'
    REL_ERR_THOUSANDTH_STATUS = '双千指标判定结果'
    FINAL_RESULT = '比对结果'
    ALGORITHM = '比对算法'
    FORWWARD_STATUS = 'Forward Test Success'
    BACKWARD_STATUS = 'Backward Test Success'
    MESSAGE = 'Message'

    @staticmethod
    def to_required_columns():
        return [
            ApiPrecisionCompareColumn.API_NAME,
            ApiPrecisionCompareColumn.DEVICE_DTYPE,
            ApiPrecisionCompareColumn.SMALL_VALUE_ERROR_RATE,
            ApiPrecisionCompareColumn.RMSE,
            ApiPrecisionCompareColumn.MAX_REL_ERR,
            ApiPrecisionCompareColumn.MEAN_REL_ERR,
            ApiPrecisionCompareColumn.EB,
            ApiPrecisionCompareColumn.ERROR_RATE,
            ApiPrecisionCompareColumn.INF_NAN_ERROR_RATIO,
            ApiPrecisionCompareColumn.REL_ERR_RATIO,
            ApiPrecisionCompareColumn.ABS_ERR_RATIO,
            ApiPrecisionCompareColumn.MEAN_ULP_ERR,
            ApiPrecisionCompareColumn.ULP_ERR_PROPORTION,
            ApiPrecisionCompareColumn.REL_ERR_THOUSANDTH,
        ]

    @staticmethod
    def get_detail_csv_title():
        return [
            ApiPrecisionCompareColumn.API_NAME,
            ApiPrecisionCompareColumn.SMALL_VALUE_ERROR_RATIO,
            ApiPrecisionCompareColumn.SMALL_VALUE_ERROR_STATUS,
            ApiPrecisionCompareColumn.RMSE_RATIO,
            ApiPrecisionCompareColumn.RMSE_STATUS,
            ApiPrecisionCompareColumn.MAX_REL_ERR_RATIO,
            ApiPrecisionCompareColumn.MAX_REL_ERR_STATUS,
            ApiPrecisionCompareColumn.MEAN_REL_ERR_RATIO,
            ApiPrecisionCompareColumn.MEAN_REL_ERR_STATUS,
            ApiPrecisionCompareColumn.EB_RATIO,
            ApiPrecisionCompareColumn.EB_STATUS,
            ApiPrecisionCompareColumn.INF_NAN_ERROR_RATIO,
            ApiPrecisionCompareColumn.INF_NAN_ERROR_RATIO_STATUS,
            ApiPrecisionCompareColumn.REL_ERR_RATIO,
            ApiPrecisionCompareColumn.REL_ERR_RATIO_STATUS,
            ApiPrecisionCompareColumn.ABS_ERR_RATIO,
            ApiPrecisionCompareColumn.ABS_ERR_RATIO_STATUS,
            ApiPrecisionCompareColumn.ERROR_RATE,
            ApiPrecisionCompareColumn.ERROR_RATE_STATUS,
            ApiPrecisionCompareColumn.MEAN_ULP_ERR,
            ApiPrecisionCompareColumn.ULP_ERR_PROPORTION,
            ApiPrecisionCompareColumn.ULP_ERR_PROPORTION_RATIO,
            ApiPrecisionCompareColumn.ULP_ERR_STATUS,
            ApiPrecisionCompareColumn.REL_ERR_THOUSANDTH,
            ApiPrecisionCompareColumn.REL_ERR_THOUSANDTH_STATUS,
            ApiPrecisionCompareColumn.FINAL_RESULT,
            ApiPrecisionCompareColumn.ALGORITHM,
            ApiPrecisionCompareColumn.MESSAGE,
        ]

    @staticmethod
    def get_result_csv_title():
        return [
            ApiPrecisionCompareColumn.API_NAME,
            ApiPrecisionCompareColumn.FORWWARD_STATUS,
            ApiPrecisionCompareColumn.BACKWARD_STATUS,
            ApiPrecisionCompareColumn.MESSAGE,
        ]


CompareMessage = {
    "topk": "在npu上，topk的入参sorted=False时不生效，会返回有序tensor，而cpu上会返回无序tensor。 如果topk精度不达标，请检查是否是该原因导致的。"
}


def check_dtype_comparable(x, y):
    if x.dtype in Const.FLOAT_TYPE:
        if y.dtype in Const.FLOAT_TYPE:
            return True
        return False
    if x.dtype in Const.BOOL_TYPE:
        if y.dtype in Const.BOOL_TYPE:
            return True
        return False
    if x.dtype in Const.INT_TYPE:
        if y.dtype in Const.INT_TYPE:
            return True
        return False
    logger.warning(f"Compare: Unexpected dtype {x.dtype}, {y.dtype}")
    return False


def convert_str_to_float(input_data):
    if isinstance(input_data, str) and input_data.strip() == "":
        msg = 'ERROR: Input data is an empty string'
        raise CompareException(CompareException.INVALID_DATA_ERROR, msg)
    try:
        float_data = float(input_data)
        return float_data
    except ValueError as e:
        msg = 'ERROR: Input data cannot be converted to float'
        raise CompareException(CompareException.INVALID_DATA_ERROR, msg) from e


def is_inf_or_nan(x):
    return math.isnan(x) or math.isinf(x)


def handle_infinity(x, y, column_name):
    if math.isinf(x) and math.isinf(y):
        if x == y:
            return float("nan"), True, f"{column_name}同为同号inf或nan\n"
        else:
            return float("nan"), False, f"{column_name}inf或nan不一致\n"
    else:
        return float("nan"), False, f"{column_name}inf或nan不一致\n"


def handle_nan(x, y, column_name):
    if math.isnan(x) and math.isnan(y):
        return float("nan"), True, f"{column_name}同为同号inf或nan\n"
    else:
        return float("nan"), False, f"{column_name}inf或nan不一致\n"


def check_inf_or_nan(x, y, column_name):
    if math.isinf(x) or math.isinf(y):
        return handle_infinity(x, y, column_name)
    else:
        return handle_nan(x, y, column_name)
