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

from collections import namedtuple
import numpy as np
import torch

from msprobe.pytorch.api_accuracy_checker.precision_standard.standard_config import StandardConfig
from msprobe.pytorch.api_accuracy_checker.precision_standard.base_standard import BaseCompare, BasePrecisionCompare
from msprobe.core.common.const import Const, CompareConst
from msprobe.pytorch.api_accuracy_checker.compare.algorithm import calc_ratio, get_ulp_err
from msprobe.pytorch.api_accuracy_checker.compare.compare_utils import (
    ApiPrecisionCompareColumn,
    check_inf_or_nan,
    is_inf_or_nan,
)
from msprobe.pytorch.api_accuracy_checker.common.utils import is_dtype_fp8_or_hif8


UlpInfNanConsistency = namedtuple(
    'UlpInfNanConsistency', ['mean_ulp_err_inf_nan_consistency', 'ulp_err_proportion_ratio_inf_nan_consistency']
)


class UlpCompare(BaseCompare):
    """
    Ulp compare comparison class for calculating accuracy metrics.

    Attributes:
        bench_output (array-like): The benchmark output values.
        device_output (array-like): The device output values.
        dtype (torch.dtype): The data type of the outputs (e.g., torch.float32 or torch.float16).
        ulp_err (array-like): The ULP errors calculated from the benchmark and device outputs.

    Methods:
        _stat_max_ulp_err(ulp_err): Calculates the maximum ULP error.
        _stat_mean_ulp_err(ulp_err): Calculates the mean ULP error.
        _stat_ulp_error_proportion(ulp_err): Calculates the proportion of ULP errors exceeding a threshold.
        _pre_compare(): Prepares for comparison by calculating ULP errors.
        _compute_metrics(): Computes the ULP error metrics.
    """

    def __init__(self, input_data):  # pylint: disable=useless-parent-delegation
        super(UlpCompare, self).__init__(input_data)  # pylint: disable=super-with-arguments

    @staticmethod
    def _stat_max_ulp_err(ulp_err):
        return np.max(ulp_err)

    @staticmethod
    def _stat_mean_ulp_err(ulp_err):
        return np.mean(ulp_err)

    def _stat_ulp_error_proportion(self, ulp_err):
        if self.dtype == torch.float32:
            return np.sum(ulp_err > CompareConst.ULP_FLOAT32_THRESHOLD) / self.bench_output.size
        else:
            return np.sum(ulp_err > CompareConst.ULP_FLOAT16_THRESHOLD) / self.bench_output.size

    def _pre_compare(self):
        self.ulp_err = get_ulp_err(self.bench_output, self.device_output, self.dtype)  # pylint: disable=attribute-defined-outside-init

    def _compute_metrics(self):
        """
        Computes the ULP error metrics for the comparison.

        This method calculates three key metrics:
        1. Maximum ULP error: The maximum difference in ULPs between the benchmark and device outputs.
        2. Mean ULP error: The average difference in ULPs between the benchmark and device outputs.
        3. ULP error proportion: The proportion of ULP errors that exceed a certain threshold.

        Args:
        None (this method uses instance variables)

        Returns:
        dict: A dictionary containing the computed ULP error metrics.
            The dictionary has the following keys:
            - "max_ulp_error": The maximum ULP error.
            - "mean_ulp_error": The mean ULP error.
            - "ulp_error_proportion": The proportion of ULP errors exceeding the threshold.
        """
        max_ulp_error = self._stat_max_ulp_err(self.ulp_err)
        mean_ulp_error = self._stat_mean_ulp_err(self.ulp_err)

        ulp_error_proportion = self._stat_ulp_error_proportion(self.ulp_err)

        return {
            "max_ulp_error": max_ulp_error,
            "mean_ulp_error": mean_ulp_error,
            "ulp_error_proportion": ulp_error_proportion,
        }


class UlpPrecisionCompare(BasePrecisionCompare):
    def __init__(self, input_data):
        super().__init__(input_data)
        self.compare_algorithm = CompareConst.ULP_COMPARE_ALGORITHM_NAME

    @staticmethod
    def _compute_ulp_err_proportion_ratio(npu_value, gpu_value, dtype):
        column_name = ApiPrecisionCompareColumn.ULP_ERR_PROPORTION
        if is_inf_or_nan(npu_value) or is_inf_or_nan(gpu_value):
            return check_inf_or_nan(npu_value, gpu_value, column_name)
        else:
            return calc_ratio(npu_value, gpu_value, dtype), True, ""

    def _compute_mean_ulp_err(self):
        column_name = ApiPrecisionCompareColumn.MEAN_ULP_ERR
        npu_value, gpu_value = self._get_and_convert_values(column_name)
        if is_inf_or_nan(npu_value) or is_inf_or_nan(gpu_value):
            _, mean_ulp_err_inf_nan_consistency, message = check_inf_or_nan(npu_value, gpu_value, column_name)
            return npu_value, mean_ulp_err_inf_nan_consistency, message
        else:
            return npu_value, True, ""

    def _compute_ulp_err_proportion(self):
        column_name = ApiPrecisionCompareColumn.ULP_ERR_PROPORTION
        npu_value, gpu_value = self._get_and_convert_values(column_name)
        return npu_value, gpu_value

    def _get_status(self, metrics, inf_nan_consistency):
        ulp_inf_nan_consistency = (
            inf_nan_consistency.mean_ulp_err_inf_nan_consistency
            and inf_nan_consistency.ulp_err_proportion_ratio_inf_nan_consistency
        )

        if not ulp_inf_nan_consistency:
            status_dict = {CompareConst.ULP_ERR_STATUS: CompareConst.ERROR}
            compare_result = CompareConst.ERROR
            metrics[CompareConst.COMPARE_MESSAGE] = (
                metrics.get(CompareConst.COMPARE_MESSAGE, "") + "ERROR: ULP误差不满足标准\n"
            )
            metrics.update({CompareConst.COMPARE_RESULT: compare_result})
            return metrics

        dtype = self.row_npu.get(ApiPrecisionCompareColumn.DEVICE_DTYPE)
        mean_ulp_err = metrics.get(CompareConst.MEAN_ULP_ERR)
        ulp_err_proportion = metrics.get(CompareConst.ULP_ERR_PROPORTION)
        ulp_err_proportion_ratio = metrics.get(CompareConst.ULP_ERR_PROPORTION_RATIO)
        if is_dtype_fp8_or_hif8(dtype):
            status, final_message = self._get_fp8_ulp_err_status(ulp_err_proportion)
        elif dtype == Const.TORCH_FLOAT32:
            status, final_message = self._get_fp32_ulp_err_status(
                mean_ulp_err, ulp_err_proportion, ulp_err_proportion_ratio
            )
        else:
            status, final_message = self._get_fp16_ulp_err_status(ulp_err_proportion, ulp_err_proportion_ratio)
        metrics[CompareConst.COMPARE_MESSAGE] = metrics.get(CompareConst.COMPARE_MESSAGE, "") + final_message

        status_dict = {CompareConst.ULP_ERR_STATUS: status}
        compare_result = status
        metrics.update(status_dict)
        metrics.update({CompareConst.COMPARE_RESULT: compare_result})
        return metrics

    def _get_fp32_ulp_err_status(self, mean_ulp_err, ulp_err_proportion, ulp_err_proportion_ratio):
        mean_ulp_err_threshold, ulp_err_proportion_threshold, ulp_err_proportion_ratio_threshold = (
            StandardConfig.get_ulp_threshold(torch.float32)
        )
        if mean_ulp_err < mean_ulp_err_threshold:
            return CompareConst.PASS, ""
        elif ulp_err_proportion < ulp_err_proportion_threshold:
            return CompareConst.PASS, ""
        elif ulp_err_proportion_ratio < ulp_err_proportion_ratio_threshold:
            return CompareConst.PASS, ""
        compare_message = "ERROR: ULP误差不满足标准\n"
        return CompareConst.ERROR, compare_message

    def _get_fp16_ulp_err_status(self, ulp_err_proportion, ulp_err_proportion_ratio):
        _, ulp_err_proportion_threshold, ulp_err_proportion_ratio_threshold = StandardConfig.get_ulp_threshold(
            torch.float16
        )
        if ulp_err_proportion < ulp_err_proportion_threshold:
            return CompareConst.PASS, ""
        elif ulp_err_proportion_ratio < ulp_err_proportion_ratio_threshold:
            return CompareConst.PASS, ""
        compare_message = "ERROR: ULP误差不满足标准\n"
        return CompareConst.ERROR, compare_message

    def _get_fp8_ulp_err_status(self, ulp_err_proportion):
        _, ulp_err_proportion_threshold, _ = StandardConfig.get_ulp_threshold(torch.float16)
        if ulp_err_proportion < ulp_err_proportion_threshold:
            return CompareConst.PASS, ""
        compare_message = "ERROR: ULP误差不满足标准\n"
        return CompareConst.ERROR, compare_message

    def _compute_ratio(self):
        compare_message = ""
        dtype = self.row_npu.get(ApiPrecisionCompareColumn.DEVICE_DTYPE)
        if is_dtype_fp8_or_hif8(dtype):
            mean_ulp_err = CompareConst.SPACE
            ulp_err_proportion_ratio = CompareConst.SPACE
            npu_ulp_err_proportion = self._get_and_convert_value(
                self.row_npu, ApiPrecisionCompareColumn.ULP_ERR_PROPORTION, "NPU"
            )
            mean_ulp_err_inf_nan_consistency = True
            ulp_err_proportion_ratio_inf_nan_consistency = True
        else:
            mean_ulp_err, mean_ulp_err_inf_nan_consistency, mean_ulp_err_message = self._compute_mean_ulp_err()
            compare_message += mean_ulp_err_message
            npu_ulp_err_proportion, gpu_ulp_err_proportion = self._compute_ulp_err_proportion()
            ulp_err_proportion_ratio, ulp_err_proportion_ratio_inf_nan_consistency, ulp_err_proportion_ratio_message = (
                self._compute_ulp_err_proportion_ratio(npu_ulp_err_proportion, gpu_ulp_err_proportion, str(self.dtype))
            )
            compare_message += ulp_err_proportion_ratio_message
        metrics = {
            CompareConst.MEAN_ULP_ERR: mean_ulp_err,
            CompareConst.ULP_ERR_PROPORTION: npu_ulp_err_proportion,
            CompareConst.ULP_ERR_PROPORTION_RATIO: ulp_err_proportion_ratio,
            CompareConst.COMPARE_MESSAGE: compare_message,
        }
        return metrics, UlpInfNanConsistency(
            mean_ulp_err_inf_nan_consistency, ulp_err_proportion_ratio_inf_nan_consistency
        )
