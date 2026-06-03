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


import abc
from dataclasses import dataclass
from typing import Any

import numpy as np

from msprobe.core.common.const import Const, CompareConst
from msprobe.core.common.log import logger
from msprobe.core.common.utils import CompareException, format_value


@dataclass
class CompareResult:
    """
    对比流程中的中间结果对象
    用于在不同阶段之间传递数据与状态。
    """

    n_value: Any
    b_value: Any
    error_flag: bool = False
    err_msg: str = ""


def handle_inf_nan(n_value, b_value):
    """处理inf和nan的数据"""

    def convert_to_float(value):
        try:
            if isinstance(value, np.ndarray):
                return value.astype(float)
            else:
                return float(value)
        except ValueError as e:
            logger.error('\n'.join(e.args))
            raise CompareException(CompareException.INVALID_DATA_ERROR) from e

    n_value_convert, b_value_convert = convert_to_float(n_value), convert_to_float(b_value)

    n_inf = np.isinf(n_value_convert)
    b_inf = np.isinf(b_value_convert)
    n_nan = np.isnan(n_value_convert)
    b_nan = np.isnan(b_value_convert)
    n_invalid = np.any(n_inf) or np.any(n_nan)
    b_invalid = np.any(b_inf) or np.any(b_nan)
    if n_invalid or b_invalid:
        if np.array_equal(n_inf, b_inf) and np.array_equal(n_nan, b_nan):
            n_value[n_inf] = 0
            b_value[b_inf] = 0
            n_value[n_nan] = 0
            b_value[b_nan] = 0
        else:
            return CompareConst.NAN, CompareConst.NAN
    return n_value, b_value


def reshape_value(n_value, b_value):
    """返回reshape后的数据"""
    if not n_value.shape:  # 判断数据是否为0维tensor， 如果0维tensor，不会转成1维tensor，直接返回
        if n_value.dtype == bool:
            n_value = n_value.astype(float)
            b_value = b_value.astype(float)
        return n_value, b_value

    n_value = n_value.reshape(-1).astype(float)  # 32转64为了防止某些数转dataframe时出现误差
    b_value = b_value.reshape(-1).astype(float)
    return n_value, b_value


def npy_data_check(n_value, b_value):
    error_message = ""
    if not isinstance(n_value, np.ndarray) or not isinstance(b_value, np.ndarray):
        error_message += "Dump file is not ndarray.\n"

    # 检查 n_value 和 b_value 是否为空
    if not error_message and (n_value.size == 0 or b_value.size == 0):
        error_message += "This is empty data, can not compare.\n"

    if not error_message:
        if not n_value.shape or not b_value.shape:
            error_message += "This is type of scalar data, can not compare.\n"
        if n_value.shape != b_value.shape:
            error_message += "Shape of NPU and bench Tensor do not match.\n"
        if n_value.dtype != b_value.dtype:
            error_message += "Dtype of NPU and bench Tensor do not match. Skipped.\n"

    if not error_message:
        try:
            n_value, b_value = handle_inf_nan(n_value, b_value)  # 判断是否有nan/inf数据
        except CompareException:
            logger.error('Numpy data is unreadable, please check!')
            return True, 'Numpy data is unreadable, please check!'
        # handle_inf_nan 会返回'Nan'或ndarray类型，使用类型判断是否存在无法处理的nan/inf数据
        if not isinstance(n_value, np.ndarray) or not isinstance(b_value, np.ndarray):
            error_message += "The position of inf or nan in NPU and bench Tensor do not match.\n"
    if error_message == "":
        error_flag = False
    else:
        error_flag = True
    return error_flag, error_message


def statistics_data_check(result_dict):
    error_message = ""

    if result_dict.get(CompareConst.NPU_NAME) is None or result_dict.get(CompareConst.BENCH_NAME) is None:
        error_message += "Dump file not found.\n"

    if not result_dict.get(CompareConst.NPU_SHAPE) or not result_dict.get(CompareConst.BENCH_SHAPE):
        error_message += "This is type of scalar data, can not compare.\n"
    elif result_dict.get(CompareConst.NPU_SHAPE) != result_dict.get(CompareConst.BENCH_SHAPE):
        error_message += "Tensor shapes do not match.\n"

    if result_dict.get(CompareConst.NPU_DTYPE) != result_dict.get(CompareConst.BENCH_DTYPE):
        error_message += "Dtype of NPU and bench Tensor do not match. Skipped.\n"

    if error_message == "":
        error_flag = False
    else:
        error_flag = True
    return error_flag, error_message


class TensorComparisonBasic(abc.ABC):
    """NPU和bench中npy数据的比较模板"""

    @abc.abstractmethod
    def apply(self, n_value, b_value, relative_err, err_msg):
        raise NotImplementedError


def get_relative_err(n_value, b_value):
    """计算相对误差"""
    with np.errstate(divide='ignore', invalid='ignore'):
        if n_value.dtype not in CompareConst.FLOAT_TYPE:
            n_value = n_value.astype(float)
        if b_value.dtype not in CompareConst.FLOAT_TYPE:
            b_value = b_value.astype(float)

        n_value_copy = n_value.copy()
        b_value_copy = b_value.copy()
        zero_mask = b_value_copy == 0
        b_value_copy[zero_mask] += Const.FLOAT_EPSILON
        n_value_copy[zero_mask] += Const.FLOAT_EPSILON
        relative_err = np.divide((n_value_copy - b_value_copy), b_value_copy)
    return np.abs(relative_err)


class GetCosineSimilarity(TensorComparisonBasic):
    """计算cosine相似度"""

    @staticmethod
    def correct_data(result):
        if result == CompareConst.NAN:
            return result
        if float(result) > CompareConst.COSINE_THRESHOLD:
            return round(float(result), 6)
        return result

    def apply(self, n_value, b_value, relative_err, err_msg):
        if "This is type of 0-d tensor" in err_msg:
            return CompareConst.UNSUPPORTED, err_msg

        with np.errstate(divide="ignore", invalid="ignore"):
            if len(n_value) == 1:
                return CompareConst.UNSUPPORTED, "This is a 1-d tensor of length 1."
            num = n_value.dot(b_value)
            a_norm = np.linalg.norm(n_value)
            b_norm = np.linalg.norm(b_value)

            if a_norm <= Const.FLOAT_EPSILON and b_norm <= Const.FLOAT_EPSILON:
                return 1.0, ""
            if a_norm <= Const.FLOAT_EPSILON:
                return CompareConst.NAN, "Cannot compare by Cosine Similarity, All the data is Zero in npu dump data."
            if b_norm <= Const.FLOAT_EPSILON:
                return CompareConst.NAN, "Cannot compare by Cosine Similarity, All the data is Zero in Bench dump data."

            cos = num / (a_norm * b_norm)
            if np.isnan(cos):
                return CompareConst.NAN, "Cannot compare by Cosine Similarity, the dump data has NaN."
            result = format_value(cos)
            result = self.correct_data(result)
        return result, ""


class GetEuclideanDistance(TensorComparisonBasic):
    """计算欧式距离"""

    def apply(self, n_value, b_value, relative_err, err_msg):
        if "This is type of 0-d tensor" in err_msg:
            return CompareConst.UNSUPPORTED, err_msg

        distance = np.linalg.norm(n_value - b_value, ord=2)

        return distance, ""


class GetMaxAbsErr(TensorComparisonBasic):
    """计算最大绝对误差"""

    def apply(self, n_value, b_value, relative_err, err_msg):
        temp_res = n_value - b_value
        max_value = np.max(np.abs(temp_res))
        if np.isnan(max_value):
            msg = "Cannot compare by MaxAbsError, the data contains nan/inf/-inf in dump data."
            return CompareConst.NAN, msg
        return format_value(max_value), ""


class GetMaxRelativeErr(TensorComparisonBasic):
    """计算最大相对误差"""

    def apply(self, n_value, b_value, relative_err, err_msg):
        max_relative_err = np.max(np.abs(relative_err))
        if np.isnan(max_relative_err):
            msg = "Cannot compare by MaxRelativeError, the data contains nan/inf/-inf in dump data."
            return CompareConst.NAN, msg
        return format_value(max_relative_err), ""


class GetErrRatio(TensorComparisonBasic):
    """计算相对误差小于指定阈值(千分之一、千分之五)的比例"""

    def __init__(self, threshold):
        self.threshold = threshold

    def apply(self, n_value, b_value, relative_err, err_msg):
        if "This is type of 0-d tensor" in err_msg:
            return CompareConst.UNSUPPORTED, err_msg

        if not np.size(relative_err):
            return CompareConst.NAN, ""

        ratio = np.sum(relative_err < self.threshold) / np.size(relative_err)
        return format_value(ratio), ""


class CompareOps:
    compare_ops = {
        "cosine_similarity": GetCosineSimilarity(),
        "euclidean_distance": GetEuclideanDistance(),
        "max_abs_error": GetMaxAbsErr(),
        "max_relative_error": GetMaxRelativeErr(),
        "one_thousand_err_ratio": GetErrRatio(CompareConst.THOUSAND_RATIO_THRESHOLD),
        "five_thousand_err_ratio": GetErrRatio(CompareConst.FIVE_THOUSAND_RATIO_THRESHOLD),
    }


def error_value_process(n_value):
    if n_value in [
        CompareConst.READ_NONE,
        CompareConst.UNREADABLE,
        CompareConst.NONE,
        CompareConst.NO_REAL_DATA,
        CompareConst.API_UNMATCH,
    ]:
        return CompareConst.UNSUPPORTED, ""
    if n_value == CompareConst.SHAPE_UNMATCH:
        return CompareConst.SHAPE_UNMATCH, ""
    if n_value == CompareConst.NAN:
        return CompareConst.N_A, ""
    return CompareConst.N_A, ""


def compare_ops_apply(n_value, b_value, error_flag, err_msg):
    result_list = []
    if error_flag:
        result, msg = error_value_process(n_value)
        result_list = [result] * len(CompareOps.compare_ops)
        err_msg += msg
        return result_list, err_msg

    relative_err = get_relative_err(n_value, b_value)
    n_value, b_value = reshape_value(n_value, b_value)

    for op in CompareOps.compare_ops.values():
        result, msg = op.apply(n_value, b_value, relative_err, err_msg)
        result_list.append(result)
        err_msg += msg
    return result_list, err_msg


class ValidateTensor:
    def __init__(self):
        pass

    @staticmethod
    def _check_empty(result):
        """
        检查 tensor 是否为空
        """
        n_value = result.n_value

        if n_value.size == 0:
            return CompareResult(CompareConst.NONE, CompareConst.NONE, True, "This is empty data, can not compare.")

        return result

    @staticmethod
    def _check_scalar(result):
        """
        检查是否为 0 维 tensor
        """
        n_value = result.n_value
        b_value = result.b_value

        if not n_value.shape:
            msg = (
                f"This is type of 0-d tensor, can not calculate "
                f"'{CompareConst.COSINE}', '{CompareConst.EUC_DIST}', "
                f"'{CompareConst.ONE_THOUSANDTH_ERR_RATIO}' and "
                f"'{CompareConst.FIVE_THOUSANDTHS_ERR_RATIO}'. "
            )
            # 0-d tensor 最大绝对误差、最大相对误差仍然支持计算，因此error_flag设置为False，不做统一处理
            return CompareResult(n_value, b_value, False, msg)

        return result

    @staticmethod
    def _check_shape(result):
        """
        检查 NPU 与 Bench tensor 的 shape 是否一致
        """
        n_value = result.n_value
        b_value = result.b_value

        if n_value.shape != b_value.shape:
            return CompareResult(
                CompareConst.SHAPE_UNMATCH,
                CompareConst.SHAPE_UNMATCH,
                True,
                "Shape of NPU and bench tensor do not match. Skipped.",
            )

        return result

    @staticmethod
    def _check_nan_inf(result):
        """
        检查 tensor 中的 nan / inf
        """
        n_value = result.n_value
        b_value = result.b_value

        try:
            n_value, b_value = handle_inf_nan(n_value, b_value)
        except CompareException:
            logger.error("Numpy data is unreadable.")

            return CompareResult(CompareConst.UNREADABLE, CompareConst.UNREADABLE, True, "Data is unreadable.")

        if n_value is CompareConst.NAN or b_value is CompareConst.NAN:
            return CompareResult(
                CompareConst.NAN,
                CompareConst.NAN,
                True,
                "The position of inf or nan in NPU and bench Tensor do not match.",
            )

        return result

    @staticmethod
    def _check_dtype(result):
        """
        检查 tensor 的 dtype 是否一致
        """
        n_value = result.n_value
        b_value = result.b_value

        if n_value.dtype != b_value.dtype:
            return CompareResult(n_value, b_value, False, "Dtype of NPU and bench tensor do not match.")

        return result

    def check_tensor(self, result):
        """
        对 tensor 进行合法性校验
        通过规则链依次执行各个校验规则，
        一旦某个规则返回错误结果，则立即返回。

        参数:
            result (CompareResult): 包含 NPU 与 Bench tensor 数据
        返回:
            CompareResult: 校验后的结果
        """
        validators = [self._check_empty, self._check_shape, self._check_scalar, self._check_nan_inf, self._check_dtype]

        for validator in validators:
            result = validator(result)
            if result.err_msg:
                return result

        return result
