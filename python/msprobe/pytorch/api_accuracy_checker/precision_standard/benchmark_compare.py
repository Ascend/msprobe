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
# pylint: disable=attribute-defined-outside-init
# pylint: disable=duplicate-code


from collections import namedtuple
import numpy as np

from msprobe.pytorch.api_accuracy_checker.precision_standard.standard_config import StandardConfig
from msprobe.pytorch.api_accuracy_checker.precision_standard.base_standard import BaseCompare, BasePrecisionCompare
from msprobe.pytorch.api_accuracy_checker.compare.algorithm import (
    calc_ratio,
    get_small_value_err_ratio,
    get_rel_err,
    get_rmse,
    get_error_balance,
    get_max_rel_err,
    get_mean_rel_err,
)
from msprobe.pytorch.api_accuracy_checker.compare.compare_utils import (
    ApiPrecisionCompareColumn,
    check_inf_or_nan,
    is_inf_or_nan,
)
from msprobe.core.common.const import CompareConst


BenchmarkInfNanConsistency = namedtuple(
    'BenchmarkInfNanConsistency',
    [
        'small_value_inf_nan_consistency',
        'rmse_inf_nan_consistency',
        'max_rel_inf_nan_consistency',
        'mean_rel_inf_nan_consistency',
        'eb_inf_nan_consistency',
    ],
)


class BenchmarkCompare(BaseCompare):
    """
    Benchmark comparison class for calculating accuracy metrics.

    This class is designed to compare the output of a benchmark test with the output of a device.
    It calculates various metrics such as small value error ratio, RMSE, error balance, max relative error,
    and mean relative error to assess the accuracy of the device output against the benchmark output.

    Attributes:
        bench_output (np.ndarray): The output from the benchmark.
        device_output (np.ndarray): The output from the device.
        dtype (torch.dtype): The data type of the outputs.
        abs_bench (np.ndarray): The absolute value of the benchmark output.
        abs_bench_with_eps (np.ndarray): The absolute value of the benchmark output with epsilon.
        both_finite_mask (np.ndarray): A mask indicating where both outputs are finite.
        inf_nan_mask (np.ndarray): A mask indicating where either output is infinite or NaN.
        abs_err (np.ndarray): The absolute error between the benchmark and device outputs.
        small_value (float): The small value threshold for comparison.
        small_value_atol (float): The absolute tolerance for small values.
        small_value_mask (np.ndarray): A mask indicating where values are small.
        rel_err (np.ndarray): The relative error between the benchmark and device outputs.
        abs_err_greater_mask (np.ndarray): A mask indicating where absolute error is greater than the small value
        tolerance.

    Methods:
        _get_abs_err_greater_mask(small_value_atol): Calculates a mask where absolute error is greater than the small
        value tolerance.
        _compute_rel_err(): Computes the relative error between the benchmark and device outputs.
        _pre_compare(): Prepares the comparison by calculating various metrics.
        _compute_metrics(): Computes the accuracy metrics.

    Note:
        This class assumes that the input data is a dictionary containing 'bench_output', 'device_output',
        'compare_column' and 'dtype'.
        The data type should be a PyTorch data type.

    See Also:
        BaseCompare: The base class for comparison classes.
        InputData: The class containing input data for comparison.
    """

    def __init__(self, input_data):
        super(BenchmarkCompare, self).__init__(input_data)  # pylint: disable=super-with-arguments
        self.compare_algorithm = CompareConst.BENCHMARK

    def _get_abs_err_greater_mask(self, small_value_atol):
        abs_err_greater_mask = np.greater(self.abs_err, small_value_atol)
        return abs_err_greater_mask

    def _compute_rel_err(self):
        rel_err = get_rel_err(self.abs_err, self.abs_bench_with_eps, self.small_value_mask, self.inf_nan_mask)
        return rel_err

    def _pre_compare(self):
        self.abs_bench, self.abs_bench_with_eps = self.stat_abs_bench_with_eps()
        self.both_finite_mask, self.inf_nan_mask = self.stat_finite_and_infinite_mask()
        self.abs_err = self.stat_abs_error()
        self.small_value, self.small_value_atol = self.get_small_value_threshold()
        self.small_value_mask = self.stat_small_value_mask(self.abs_bench, self.both_finite_mask, self.small_value)
        self.rel_err = self._compute_rel_err()
        self.abs_err_greater_mask = self._get_abs_err_greater_mask(self.small_value_atol)

    def _compute_metrics(self):
        """
        Computes a comprehensive set of error metrics for the comparison between benchmark and device outputs.

        This method calculates five key metrics:
        1. Small Value Error Ratio: The proportion of errors associated with small values.
        2. Root Mean Square Error (RMSE): The square root of the mean of the squared errors.
        3. Error Balance (EB): A measure of the balance between the errors in the benchmark and device outputs.
        4. Maximum Relative Error: The maximum relative error between the benchmark and device outputs.
        5. Mean Relative Error: The mean relative error between the benchmark and device outputs.

        Returns:
        dict: A dictionary containing the computed error metrics.
            The dictionary has the following keys:
            - "small_value_err_ratio": The proportion of errors associated with small values.
            - "max_rel_error": The maximum relative error.
            - "mean_rel_error": The mean relative error.
            - "rmse": The root mean square error.
            - "eb": The error balance.
        """
        small_value_err_ratio = get_small_value_err_ratio(self.small_value_mask, self.abs_err_greater_mask)
        rmse = get_rmse(self.abs_err, np.logical_or(self.inf_nan_mask, self.small_value_mask))
        eb = get_error_balance(self.bench_output, self.device_output)
        max_rel_error = get_max_rel_err(self.rel_err)
        mean_rel_error = get_mean_rel_err(self.rel_err)

        return {
            "small_value_err_ratio": small_value_err_ratio,
            "max_rel_error": max_rel_error,
            "mean_rel_error": mean_rel_error,
            "rmse": rmse,
            "eb": eb,
        }


class BenchmarkPrecisionCompare(BasePrecisionCompare):
    def __init__(self, input_data):
        super().__init__(input_data)
        self.compare_algorithm = CompareConst.BENCHMARK_COMPARE_ALGORITHM_NAME
        self.api_name = getattr(input_data, "api_name", None)

    @staticmethod
    def get_final_status(status_list):
        compare_result = CompareConst.PASS
        if CompareConst.ERROR in status_list:
            compare_result = CompareConst.ERROR
        elif CompareConst.WARNING in status_list:
            compare_result = CompareConst.WARNING
        return compare_result

    def _calc_ratio(self, column_name):
        npu_value, gpu_value = self._get_and_convert_values(column_name)
        if is_inf_or_nan(npu_value) or is_inf_or_nan(gpu_value):
            return check_inf_or_nan(npu_value, gpu_value, column_name)
        else:
            return calc_ratio(npu_value, gpu_value, str(self.dtype)), True, ""

    def _compute_ratio(self):
        compare_message = ""
        small_value_err_ratio, small_value_inf_nan_consistency, small_value_message = self._calc_ratio(
            ApiPrecisionCompareColumn.SMALL_VALUE_ERROR_RATE
        )
        compare_message += small_value_message
        rmse_ratio, rmse_inf_nan_consistency, rmse_message = self._calc_ratio(ApiPrecisionCompareColumn.RMSE)
        compare_message += rmse_message
        max_rel_err_ratio, max_rel_inf_nan_consistency, max_rel_message = self._calc_ratio(
            ApiPrecisionCompareColumn.MAX_REL_ERR
        )
        compare_message += max_rel_message
        mean_rel_err_ratio, mean_rel_inf_nan_consistency, mean_rel_message = self._calc_ratio(
            ApiPrecisionCompareColumn.MEAN_REL_ERR
        )
        compare_message += mean_rel_message
        eb_ratio, eb_inf_nan_consistency, eb_message = self._calc_ratio(ApiPrecisionCompareColumn.EB)
        compare_message += eb_message

        metrics = {
            CompareConst.SMALL_VALUE_ERR_RATIO: small_value_err_ratio,
            CompareConst.RMSE_RATIO: rmse_ratio,
            CompareConst.MAX_REL_ERR_RATIO: max_rel_err_ratio,
            CompareConst.MEAN_REL_ERR_RATIO: mean_rel_err_ratio,
            CompareConst.EB_RATIO: eb_ratio,
            CompareConst.COMPARE_MESSAGE: compare_message,
        }

        return metrics, BenchmarkInfNanConsistency(
            small_value_inf_nan_consistency,
            rmse_inf_nan_consistency,
            max_rel_inf_nan_consistency,
            mean_rel_inf_nan_consistency,
            eb_inf_nan_consistency,
        )

    def _get_threshold(self, metric):
        return StandardConfig.get_benchmark_ratio_threshold(metric, api_name=self.api_name)

    def _get_single_metric_status(self, ratio, metric):
        if is_inf_or_nan(ratio):
            return CompareConst.PASS
        error_threshold = self._get_threshold(metric)
        if ratio > error_threshold:
            return CompareConst.ERROR
        return CompareConst.PASS

    def _get_status(self, metrics, inf_nan_consistency):
        small_value_err_ratio = metrics.get(CompareConst.SMALL_VALUE_ERR_RATIO)
        rmse_ratio = metrics.get(CompareConst.RMSE_RATIO)
        max_rel_err_ratio = metrics.get(CompareConst.MAX_REL_ERR_RATIO)
        mean_rel_err_ratio = metrics.get(CompareConst.MEAN_REL_ERR_RATIO)
        eb_ratio = metrics.get(CompareConst.EB_RATIO)

        small_value_err_status = (
            self._get_single_metric_status(small_value_err_ratio, CompareConst.SMALL_VALUE)
            if inf_nan_consistency.small_value_inf_nan_consistency
            else CompareConst.ERROR
        )
        rmse_status = (
            self._get_single_metric_status(rmse_ratio, CompareConst.RMSE)
            if inf_nan_consistency.rmse_inf_nan_consistency
            else CompareConst.ERROR
        )
        max_rel_err_status = (
            self._get_single_metric_status(max_rel_err_ratio, CompareConst.MAX_REL_ERR)
            if inf_nan_consistency.max_rel_inf_nan_consistency
            else CompareConst.ERROR
        )
        mean_rel_err_status = (
            self._get_single_metric_status(mean_rel_err_ratio, CompareConst.MEAN_REL_ERR)
            if inf_nan_consistency.mean_rel_inf_nan_consistency
            else CompareConst.ERROR
        )
        eb_status = (
            self._get_single_metric_status(eb_ratio, CompareConst.EB)
            if inf_nan_consistency.eb_inf_nan_consistency
            else CompareConst.ERROR
        )
        status_list = [small_value_err_status, rmse_status, max_rel_err_status, mean_rel_err_status]
        compare_result = self.get_final_status(status_list)
        status_dict = {
            CompareConst.SMALL_VALUE_ERR_STATUS: small_value_err_status,
            CompareConst.RMSE_STATUS: rmse_status,
            CompareConst.MAX_REL_ERR_STATUS: max_rel_err_status,
            CompareConst.MEAN_REL_ERR_STATUS: mean_rel_err_status,
            CompareConst.EB_STATUS: eb_status,
        }
        metrics.update(status_dict)
        metrics.update({CompareConst.COMPARE_RESULT: compare_result})
        return metrics
