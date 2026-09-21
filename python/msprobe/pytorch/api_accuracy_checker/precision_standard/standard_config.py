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

import torch

from msprobe.core.common.const import CompareConst


class StandardConfig:
    """
    Standard configuration class for managing precision and comparison thresholds.

    This class provides a centralized way to manage the small value thresholds, absolute tolerances,
    and relative tolerances (rtol) used in precision comparisons. It allows for different thresholds
    based on the data type, with default values provided for common data types.

    Attributes:
        _small_value (dict): A dictionary mapping data types to their corresponding small value thresholds.
        _small_value_atol (dict): A dictionary mapping data types to their corresponding absolute tolerances.
        _rtol (dict): A dictionary mapping data types to their corresponding relative tolerances.

    Methods:
        get_small_value(dtype): Retrieves the small value threshold for the given data type.
        get_small_value_atol(dtype): Retrieves the absolute tolerance for the given data type.
        get_rtol(dtype): Retrieves the relative tolerance for the given data type.

    Example:
        >>> small_value = StandardConfig.get_small_value(torch.float32)
        >>> atol = StandardConfig.get_small_value_atol(torch.float32)
        >>> rtol = StandardConfig.get_rtol(torch.float32)
        >>> print(small_value, atol, rtol)
        1e-6 1e-9 1e-6

    Note:
        The data type is expected to be a PyTorch data type. If the data type is not found in the dictionary,
        the default value is returned.

    See Also:
        torch.dtype: PyTorch data types.
    """

    # 维护三份列表（示例）
    L1_API_LIST = {"embedding", "npu_fusion_attention_grad", "npu_fusion_attention", "scatter_", "scatter_add_"}
    L2_API_LIST = {
        "addmm",
        "npu_cross_entropy_loss",
        "npu_cross_entropy_loss_backward",
        "GELU",
        "layer_norm",
        "matmul",
        "mm",
        "index",
        "one_hot",
        "sum",
        "npu_rms_norm",
        "npu_rms_norm_backward",
        "nonzero",
    }
    _small_value = {torch.float16: 2**-10, torch.bfloat16: 2**-10, torch.float32: 2**-20, "default": 2**-20}
    _threshold_small_value_atol = {
        torch.float16: 2**-16,
        torch.bfloat16: 1e-16,
        torch.float32: 2**-30,
        "default": 2**-30,
    }
    _benchmark_small_value_atol = {
        torch.float16: 1e-16,
        torch.bfloat16: 1e-16,
        torch.float32: 2**-30,
        "default": 2**-30,
    }
    _rtol = {torch.float16: 2**-10, torch.bfloat16: 2**-8, torch.float32: 2**-20, "default": 2**-20}
    _accumulative_error_bound = {torch.float16: 2**-8, torch.bfloat16: 2**-7, torch.float32: 2**-11, "default": 2**-11}
    _small_value_threshold = {'error_threshold': 2, 'warning_threshold': 1, "default": 1}
    _rmse_threshold = {'error_threshold': 2, 'warning_threshold': 1, "default": 1}
    _max_rel_err_threshold = {'error_threshold': 10, 'warning_threshold': 1, "default": 1}
    _mean_rel_err_threshold = {'error_threshold': 2, 'warning_threshold': 1, "default": 1}
    _eb_threshold = {'error_threshold': 2, 'warning_threshold': 1, "default": 1}
    _minmum_err = {'torch.float16': 2**-11, 'torch.bfloat16': 2**-8, 'torch.float32': 2**-14, 'default': 2**-14}
    _accumulative_error_eb_threshold = {
        'torch.float16': 2**-20,
        'torch.bfloat16': 2**-7,
        'torch.float32': 2**-14,
        'default': 2**-14,
    }

    _fp32_mean_ulp_err_threshold = 64
    ulp_err_proportion_ratio = 1
    _fp32_ulp_err_proportion = 0.05
    _fp16_ulp_err_proportion = 0.001
    _special_samll_value = 1

    @classmethod
    def get_api_level(cls, api_name: str) -> str:
        if api_name in cls.L2_API_LIST:
            return "L2"
        if api_name in cls.L1_API_LIST:
            return "L1"
        return "L0"

    @classmethod
    def get_benchmark_ratio_threshold(cls, metric, api_name: str = None, level: str = None):
        # 三档阈值
        thresholds = {
            "L0": {
                CompareConst.MAX_REL_ERR: 10.0,
                CompareConst.MEAN_REL_ERR: 2.0,
                CompareConst.RMSE: 2.0,
            },
            "L1": {
                CompareConst.MAX_REL_ERR: 5.0,
                CompareConst.MEAN_REL_ERR: 1.5,
                CompareConst.RMSE: 1.5,
            },
            "L2": {
                CompareConst.MAX_REL_ERR: 2.0,
                CompareConst.MEAN_REL_ERR: 1.2,
                CompareConst.RMSE: 1.2,
            },
        }

        if level is None:
            level = cls.get_api_level(api_name) if api_name else "L0"

        # 其他指标不参与判定：inf
        if metric in (CompareConst.MAX_REL_ERR, CompareConst.MEAN_REL_ERR, CompareConst.RMSE):
            return thresholds.get(level, thresholds["L0"]).get(metric, float("inf"))
        else:
            return StandardConfig.get_benchmark_threshold(metric)

    @classmethod
    def get_small_value(cls, dtype, standard):
        if standard == CompareConst.ACCUMULATIVE_ERROR_COMPARE:
            return cls._special_samll_value
        return cls._small_value.get(dtype, cls._small_value["default"])

    @classmethod
    def get_small_value_atol(cls, dtype, standard):
        standard_dict = {
            CompareConst.ABSOLUTE_THRESHOLD: cls._threshold_small_value_atol,
            CompareConst.BENCHMARK: cls._benchmark_small_value_atol,
        }
        small_value_atol_standard = standard_dict.get(standard, cls._benchmark_small_value_atol)
        return small_value_atol_standard.get(dtype, small_value_atol_standard["default"])

    @classmethod
    def get_rtol(cls, dtype):
        return cls._rtol.get(dtype, cls._rtol["default"])

    @classmethod
    def get_small_value_threshold(cls, threshold_type):
        return cls._small_value_threshold.get(threshold_type, cls._small_value_threshold["default"])

    @classmethod
    def get_rmse_threshold(cls, threshold_type):
        return cls._rmse_threshold.get(threshold_type, cls._rmse_threshold["default"])

    @classmethod
    def get_max_rel_err_threshold(cls, threshold_type):
        return cls._max_rel_err_threshold.get(threshold_type, cls._max_rel_err_threshold["default"])

    @classmethod
    def get_mean_rel_err_threshold(cls, threshold_type):
        return cls._mean_rel_err_threshold.get(threshold_type, cls._mean_rel_err_threshold["default"])

    @classmethod
    def get_eb_threshold(cls, threshold_type):
        return cls._eb_threshold.get(threshold_type, cls._eb_threshold["default"])

    @classmethod
    def get_benchmark_threshold(cls, metric):
        metric_threshold_functions = {
            'small_value': StandardConfig.get_small_value_threshold,
            'rmse': StandardConfig.get_rmse_threshold,
            'max_rel_err': StandardConfig.get_max_rel_err_threshold,
            'mean_rel_err': StandardConfig.get_mean_rel_err_threshold,
            'eb': StandardConfig.get_eb_threshold,
        }

        threshold_func = metric_threshold_functions.get(metric)
        return threshold_func('error_threshold')

    @classmethod
    def get_fp32_mean_ulp_err_threshold(cls):
        return cls._fp32_mean_ulp_err_threshold

    @classmethod
    def get_ulp_err_proportion_ratio_threshold(cls):
        return cls.ulp_err_proportion_ratio

    @classmethod
    def get_fp32_ulp_err_proportion_threshold(cls):
        return cls._fp32_ulp_err_proportion

    @classmethod
    def get_fp16_ulp_err_proportion_threshold(cls):
        return cls._fp16_ulp_err_proportion

    @classmethod
    def get_ulp_threshold(cls, dtype):
        ulp_err_proportion_ratio_threshold = StandardConfig.get_ulp_err_proportion_ratio_threshold()
        if dtype == torch.float32:
            mean_ulp_err_threshold = StandardConfig.get_fp32_mean_ulp_err_threshold()
            ulp_err_proportion_threshold = StandardConfig.get_fp32_ulp_err_proportion_threshold()
            return mean_ulp_err_threshold, ulp_err_proportion_threshold, ulp_err_proportion_ratio_threshold
        else:
            ulp_err_proportion_threshold = StandardConfig.get_fp16_ulp_err_proportion_threshold()
            return None, ulp_err_proportion_threshold, ulp_err_proportion_ratio_threshold

    @classmethod
    def get_minmum_err(cls, dtype):
        return cls._minmum_err.get(dtype, cls._minmum_err["default"])

    @classmethod
    def get_accumulative_error_bound(cls, dtype):
        return cls._accumulative_error_bound.get(dtype, cls._accumulative_error_bound["default"])

    @classmethod
    def get_accumulative_error_eb_threshold(cls, dtype):
        return cls._accumulative_error_eb_threshold.get(dtype, cls._accumulative_error_eb_threshold["default"])
