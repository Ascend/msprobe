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


from dataclasses import dataclass
from typing import Any

import numpy as np

from msprobe.core.common.const import CompareConst
from msprobe.core.common.log import logger
from msprobe.core.common.utils import CompareException
from msprobe.core.compare.algorithm.algorithm_scheduler import AlgorithmScheduler


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


def compare_ops_apply(n_value, b_value, error_flag, err_msg, column_names=None):
    if not column_names:
        column_names = CompareConst.BUILTIN_COMPARE_COLUMNS
    if error_flag:
        result, msg = error_value_process(n_value)
        result_list = [result] * len(column_names)
        err_msg += msg
        return result_list, err_msg
    results, err_msgs = AlgorithmScheduler.get_instance().compare(n_value, b_value, column_names=column_names)
    return results, err_msg + "".join(err_msgs)


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
