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

# 定义比对算法及比对标准
import torch
import numpy as np

from msprobe.pytorch.api_accuracy_checker.compare.compare_utils import ULP_PARAMETERS
from msprobe.pytorch.api_accuracy_checker.precision_standard.standard_config import StandardConfig
from msprobe.pytorch.api_accuracy_checker.common.utils import is_dtype_fp8
from msprobe.core.common.const import CompareConst


# cos
def cosine_sim(bench_output, device_output):
    msg = ""
    n_value = device_output.reshape(-1)
    b_value = bench_output.reshape(-1)
    cos = CompareConst.SPACE
    np.seterr(divide="ignore", invalid="ignore")
    if n_value.shape != b_value.shape:
        msg = f"Shape of device and bench outputs don't match. device: {n_value.shape}, bench: {b_value.shape}."
        return -1, False, msg
    if len(n_value) == 1:
        msg = "All the data in device dump data is scalar. Please refer to other compare algorithms."
        return cos, True, msg
    n_value_max = np.max(np.abs(n_value))
    b_value_max = np.max(np.abs(b_value))
    if n_value_max <= np.finfo(float).eps and b_value_max <= np.finfo(float).eps:
        msg = "All the data in device and bench outputs are zero."
        return cos, True, msg
    elif n_value_max <= np.finfo(float).eps:
        msg = "All the data is zero in device dump data."
        return CompareConst.SPACE, False, msg
    elif b_value_max <= np.finfo(float).eps:
        msg = "All the data is zero in bench dump data."
        return CompareConst.SPACE, False, msg
    else:
        n_value = n_value.astype(float) / n_value_max
        b_value = b_value.astype(float) / b_value_max
        cos = np.dot(n_value, b_value) / (np.linalg.norm(n_value) * np.linalg.norm(b_value))
        if np.isnan(cos):
            msg = "Dump data has NaN when comparing with Cosine Similarity."
            return np.nan, True, msg
        cos = np.clip(cos, -1, 1)
        return cos, cos > 0.99, msg


# rmse
def get_rmse(abs_err, inf_nan_mask):
    masked_ae = np.where(inf_nan_mask, 0, abs_err)
    mse = np.mean(np.square(masked_ae))
    inf_nan_cnt = np.sum(inf_nan_mask)
    mse = mse * (abs_err.size / (abs_err.size - inf_nan_cnt + 0.0001) + 0.0001)
    rmse = np.sqrt(mse)
    return rmse


# 误差均衡性
def get_error_balance(bench_data, device_data):
    larger_count = np.sum(np.greater(device_data - bench_data.astype(device_data.dtype), 0))
    smaller_count = np.sum(np.less(device_data - bench_data.astype(device_data.dtype), 0))
    total_count = bench_data.size
    error_balance = abs(larger_count - smaller_count) / total_count if total_count > 0 else 0
    return error_balance


# 小值域错误占比
def get_small_value_err_ratio(small_value_mask, abs_err_greater_mask):
    err_mask = np.logical_and(small_value_mask, abs_err_greater_mask)
    small_value_err_num = np.sum(err_mask)
    small_value_num = np.sum(small_value_mask)
    return 0 if small_value_num == 0 else small_value_err_num / small_value_num


def get_rel_err(abs_err, abs_bench_with_eps, small_value_mask, inf_nan_mask):
    rel_err_tmp = abs_err / abs_bench_with_eps
    rel_err_mask = np.logical_or(small_value_mask, inf_nan_mask)
    rel_err = np.where(rel_err_mask, -1, rel_err_tmp)
    return rel_err


def get_abs_err(bench_data, device_data):
    abs_err = np.abs(device_data - bench_data)
    return abs_err


def get_rel_err_origin(abs_err, b_value):
    rel_err_origin = np.abs(abs_err / b_value)
    return rel_err_origin


def get_max_abs_err(abs_err):
    max_abs_err = abs_err.max()
    bool_result = max_abs_err < 0.001
    return max_abs_err, bool_result


# 相对误差最大值
def get_max_rel_err(rel_err):
    return np.max(rel_err) if np.max(rel_err) >= 0 else 0


# 相对误差均值
def get_mean_rel_err(rel_err):
    non_negative_rel_err = rel_err[rel_err >= 0]
    return np.mean(non_negative_rel_err) if non_negative_rel_err.size > 0 else 0


def get_rel_err_ratio(rel_err, thresholding):
    if np.size(rel_err) == 0:
        ratio = 1
    else:
        ratio = np.divide(np.sum(rel_err < thresholding), np.size(rel_err))
    bool_result = ratio > (1 - thresholding)
    return ratio, bool_result


def get_finite_and_infinite_mask(bench_output, device_output):
    device_finite_mask = np.isfinite(device_output)
    bench_finite_mask = np.isfinite(bench_output.astype(device_output.dtype))
    both_finite_mask = np.logical_and(device_finite_mask, bench_finite_mask)
    inf_nan_mask = np.logical_not(both_finite_mask)
    return both_finite_mask, inf_nan_mask


def get_small_value_mask(abs_bench, both_finite_mask, small_value_threshold):
    small_value_mask = np.less_equal(abs_bench, small_value_threshold)
    small_value_mask = np.logical_and(small_value_mask, both_finite_mask)
    return small_value_mask


def get_abs_bench_with_eps(bench, dtype):
    abs_bench = np.abs(bench)
    eps = np.finfo(bench.dtype).eps if dtype != torch.bfloat16 else CompareConst.BFLOAT16_EPS
    abs_bench_with_eps = abs_bench + eps
    return abs_bench, abs_bench_with_eps


def check_inf_nan_value(inf_nan_mask, bench_output, device_output, dtype, rtol):
    '''
    新精度标准的绝对阈值法中，检查npu和golden输出的inf、nan是否一致
    输入：
        inf_nan_mask：npu输出和golden输出的inf、nan的mask
        bench_output：golden输出
        device_output：npu输出
        dtype：npu输出的dtype
    输出：
        inf_nan_err_ratio：npu输出和golden输出的inf、nan不一致的比例
    '''
    _, abs_gpu_with_eps = get_abs_bench_with_eps(bench_output, dtype)
    golden_same_dtype = bench_output.astype(device_output.dtype)
    a_min = np.finfo(device_output.dtype).min if dtype != torch.bfloat16 else CompareConst.BFLOAT16_MIN
    a_max = np.finfo(device_output.dtype).max if dtype != torch.bfloat16 else CompareConst.BFLOAT16_MAX
    golden_clip = np.clip(golden_same_dtype, a_min, a_max)
    npu_clip = np.clip(device_output, a_min, a_max)
    clipped_abs_ae = np.abs(npu_clip - golden_clip)
    clipped_re = clipped_abs_ae / abs_gpu_with_eps
    pass_mask = np.less_equal(clipped_re, rtol)
    both_nan_mask = np.logical_and(np.isnan(device_output), np.isnan(golden_clip))
    pass_mask = np.logical_or(pass_mask, both_nan_mask)
    not_pass_mask = np.logical_not(pass_mask)
    not_pass_mask = np.logical_and(not_pass_mask, inf_nan_mask)

    inf_nan_err_cnt = np.sum(not_pass_mask)
    return 0 if np.sum(inf_nan_mask) == 0 else inf_nan_err_cnt / np.sum(inf_nan_mask)


def check_small_value(abs_err, small_value_mask, small_value_atol):
    '''
    新精度标准的绝对阈值法中，检查npu和golden正常值输出的绝对误差是否满足阈值
    输入：
        abs_err：npu输出和golden输出的绝对误差
        normal_value_mask：npu输出和golden输出的正常值mask
        atol：绝对误差的阈值
    输出：
        abs_err_ratio：npu输出和golden输出的绝对误差不满足阈值的比例
    '''
    greater_mask = np.greater(abs_err, small_value_atol)
    err_mask = np.logical_and(greater_mask, small_value_mask)
    err_cnt = np.sum(err_mask)
    return 0 if np.sum(small_value_mask) == 0 else err_cnt / np.sum(small_value_mask)


def check_norm_value(normal_value_mask, rel_err, rtol):
    '''
    新精度标准的相对阈值法中，检查npu和golden小值域输出的相对误差是否满足阈值
    输入：
        rel_err：npu输出和golden输出的相对误差
        normal_value_mask：npu输出和golden输出的正常值mask
        rtol：相对误差的阈值
    输出：
        rel_err_ratio：npu输出和golden输出的相对误差不满足阈值的比例
    '''
    err_mask = np.greater(rel_err, rtol)
    err_mask = np.logical_and(err_mask, normal_value_mask)
    err_cnt = np.sum(err_mask)
    return 0 if np.sum(normal_value_mask) == 0 else err_cnt / np.sum(normal_value_mask)


def get_ulp_err(bench_output, device_output, dtype):
    if is_dtype_fp8(dtype):
        return calc_ulp_err_fp8(bench_output, device_output)
    parameters = ULP_PARAMETERS.get(dtype)
    min_eb = parameters.get('min_eb')[0]
    exponent_num = parameters.get('exponent_num')[0]
    abs_bench = np.abs(bench_output)
    eb = np.where(abs_bench == 0, 0, np.floor(np.log2(abs_bench)))
    eb = np.maximum(eb, min_eb)

    if dtype == torch.float32:
        ulp_err = calc_ulp_err(bench_output, device_output, eb, exponent_num, np.float64)
    else:
        ulp_err = calc_ulp_err(bench_output, device_output, eb, exponent_num, np.float32)
    ulp_err = np.abs(ulp_err)
    return ulp_err


def calc_ulp_err_fp8(bench_output, device_output):
    # compute ulp error of FP8
    x = np.float64(bench_output)
    hi_fp8 = np.float64(device_output)

    ex = np.log2(abs(x) + 2 ** (-1000))
    ex[ex < -22] = -22
    exponent = np.floor(ex)  # Exponent

    eabs = np.abs(exponent)
    wm = np.zeros_like(x)  # Mantissa width Init
    wm[eabs <= 15] = 1
    wm[eabs <= 7] = 2
    wm[eabs <= 3] = 3
    ulp_err = (hi_fp8 - x) * 2 ** (-exponent + wm)  # for wm = 1~3

    s_ex = ex * np.where(x >= 0, 1, -1)
    eh = np.log2(abs(hi_fp8) + 2 ** (-1000))

    s_eh = eh * np.where(hi_fp8 >= 0, 1, -1)
    ulp_err1 = s_eh - s_ex  # for wm = 0

    ulp_err[wm == 0] = ulp_err1[wm == 0]  # Merge 2 cases
    return ulp_err


def calc_ulp_err(bench_output, device_output, eb, exponent_num, data_type):
    return (device_output.astype(data_type) - bench_output).astype(data_type) * np.exp2(-eb + exponent_num).astype(
        data_type
    )


def calc_ratio(x, y, dtype):
    """
    Calculate the ratio between NPU and GPU statistical values.

    Args:
        x (float): Statistical value from the NPU side
        y (float): Statistical value from the GPU side
        dtype: Data type used to determine the minimum error value

    Returns:
        float: The ratio of NPU to GPU statistical values

    Notes:
        - Takes absolute values of both x and y for calculation
        - Uses StandardConfig.get_minmum_err(dtype) to get minimum error for the specified dtype
        - Prevents division by zero by ensuring denominator is not less than minimum error
        - Returns |x| / max(|y|, minimum_error)
    """
    x, y = abs(x), abs(y)
    minmum_err = StandardConfig.get_minmum_err(dtype)
    err_y = max(y, minmum_err)
    return x / err_y


def compare_bool_tensor(bench_output, device_output):
    error_nums = (bench_output != device_output).sum()
    error_rate = float(error_nums / bench_output.size)
    result = CompareConst.PASS if error_rate == 0 else CompareConst.ERROR
    return error_rate, result, ""
