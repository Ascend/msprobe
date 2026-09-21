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

from msprobe.core.common.const import CompareConst
from msprobe.pytorch.common.log import logger


class CompareColumn:
    __slots__ = [
        'bench_type',
        'npu_type',
        'shape',
        'cosine_sim',
        'max_abs_err',
        'rel_err_hundredth',
        'rel_err_ten_thousandth',
        'inf_nan_error_ratio',
        'rel_err_ratio',
        'abs_err_ratio',
        'small_value_err_ratio',
        'max_rel_error',
        'mean_rel_error',
        'rmse',
        'eb',
        'max_ulp_error',
        'mean_ulp_error',
        'ulp_error_proportion',
        'error_rate',
        'rel_err_thousandth',
    ]

    def __init__(self):
        self.bench_type = CompareConst.SPACE
        self.npu_type = CompareConst.SPACE
        self.shape = CompareConst.SPACE
        self.cosine_sim = CompareConst.SPACE
        self.max_abs_err = CompareConst.SPACE
        self.rel_err_hundredth = CompareConst.SPACE
        self.rel_err_thousandth = CompareConst.SPACE
        self.rel_err_ten_thousandth = CompareConst.SPACE
        self.error_rate = CompareConst.SPACE
        self.eb = CompareConst.SPACE
        self.rmse = CompareConst.SPACE
        self.small_value_err_ratio = CompareConst.SPACE
        self.max_rel_error = CompareConst.SPACE
        self.mean_rel_error = CompareConst.SPACE
        self.inf_nan_error_ratio = CompareConst.SPACE
        self.rel_err_ratio = CompareConst.SPACE
        self.abs_err_ratio = CompareConst.SPACE
        self.max_ulp_error = CompareConst.SPACE
        self.mean_ulp_error = CompareConst.SPACE
        self.ulp_error_proportion = CompareConst.SPACE

    def update(self, metrics):
        """
        Updates the object's attributes with the provided metrics.

        Args:
            metrics (dict): A dictionary containing attribute names and their corresponding values.

        Raises:
            AttributeError: If the metric key is not a valid attribute of CompareColumn.
        """
        for key, value in metrics.items():
            if value is None:
                continue
            if key not in self.__slots__:
                logger.error(f"The key '{key}' is not a valid attribute of CompareColumn.")
                continue
            setattr(self, key, value)

    def to_column_value(self, is_pass, message):
        return [
            self.bench_type,
            self.npu_type,
            self.shape,
            self.cosine_sim,
            self.max_abs_err,
            self.rel_err_hundredth,
            self.rel_err_thousandth,
            self.rel_err_ten_thousandth,
            self.error_rate,
            self.eb,
            self.rmse,
            self.small_value_err_ratio,
            self.max_rel_error,
            self.mean_rel_error,
            self.inf_nan_error_ratio,
            self.rel_err_ratio,
            self.abs_err_ratio,
            self.max_ulp_error,
            self.mean_ulp_error,
            self.ulp_error_proportion,
            is_pass,
            message,
        ]


class ApiPrecisionOutputColumn:
    __slots__ = [
        'api_name',
        'small_value_err_ratio',
        'small_value_err_status',
        'rmse_ratio',
        'rmse_status',
        'max_rel_err_ratio',
        'max_rel_err_status',
        'mean_rel_err_ratio',
        'mean_rel_err_status',
        'eb_ratio',
        'eb_status',
        'inf_nan_error_ratio',
        'inf_nan_error_ratio_status',
        'rel_err_ratio',
        'rel_err_ratio_status',
        'abs_err_ratio',
        'abs_err_ratio_status',
        'error_rate',
        'error_rate_status',
        'mean_ulp_err',
        'ulp_err_proportion',
        'ulp_err_proportion_ratio',
        'ulp_err_status',
        'rel_err_thousandth',
        'rel_err_thousandth_status',
        'compare_result',
        'compare_algorithm',
        'compare_message',
    ]

    def __init__(self):
        self.api_name = CompareConst.SPACE
        self.small_value_err_ratio = CompareConst.SPACE
        self.small_value_err_status = CompareConst.SPACE
        self.rmse_ratio = CompareConst.SPACE
        self.rmse_status = CompareConst.SPACE
        self.max_rel_err_ratio = CompareConst.SPACE
        self.max_rel_err_status = CompareConst.SPACE
        self.mean_rel_err_ratio = CompareConst.SPACE
        self.mean_rel_err_status = CompareConst.SPACE
        self.eb_ratio = CompareConst.SPACE
        self.eb_status = CompareConst.SPACE
        self.inf_nan_error_ratio = CompareConst.SPACE
        self.inf_nan_error_ratio_status = CompareConst.SPACE
        self.rel_err_ratio = CompareConst.SPACE
        self.rel_err_ratio_status = CompareConst.SPACE
        self.abs_err_ratio = CompareConst.SPACE
        self.abs_err_ratio_status = CompareConst.SPACE
        self.error_rate = CompareConst.SPACE
        self.error_rate_status = CompareConst.SPACE
        self.mean_ulp_err = CompareConst.SPACE
        self.ulp_err_proportion = CompareConst.SPACE
        self.ulp_err_proportion_ratio = CompareConst.SPACE
        self.ulp_err_status = CompareConst.SPACE
        self.rel_err_thousandth = CompareConst.SPACE
        self.rel_err_thousandth_status = CompareConst.SPACE
        self.compare_result = CompareConst.SPACE
        self.compare_algorithm = CompareConst.SPACE
        self.compare_message = CompareConst.SPACE

    def update(self, metrics):
        """
        Updates the object's attributes with the provided metrics.

        Args:
            metrics (dict): A dictionary containing attribute names and their corresponding values.

        Raises:
            AttributeError: If the metric key is not a valid attribute of CompareColumn.
        """
        for key, value in metrics.items():
            if value is None:
                continue
            if key not in self.__slots__:
                logger.error("The key '%s' is not a valid attribute of CompareColumn.", key)
                continue
            setattr(self, key, value)

    def to_column_value(self):
        return [
            self.api_name,
            self.small_value_err_ratio,
            self.small_value_err_status,
            self.rmse_ratio,
            self.rmse_status,
            self.max_rel_err_ratio,
            self.max_rel_err_status,
            self.mean_rel_err_ratio,
            self.mean_rel_err_status,
            self.eb_ratio,
            self.eb_status,
            self.inf_nan_error_ratio,
            self.inf_nan_error_ratio_status,
            self.rel_err_ratio,
            self.rel_err_ratio_status,
            self.abs_err_ratio,
            self.abs_err_ratio_status,
            self.error_rate,
            self.error_rate_status,
            self.mean_ulp_err,
            self.ulp_err_proportion,
            self.ulp_err_proportion_ratio,
            self.ulp_err_status,
            self.rel_err_thousandth,
            self.rel_err_thousandth_status,
            self.compare_result,
            self.compare_algorithm,
            self.compare_message,
        ]
