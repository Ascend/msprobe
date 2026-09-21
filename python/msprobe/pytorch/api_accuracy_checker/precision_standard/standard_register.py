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
from typing import Callable
from msprobe.pytorch.api_accuracy_checker.common.utils import is_dtype_fp8_or_hif8
from msprobe.pytorch.api_accuracy_checker.compare.compare_utils import (
    absolute_standard_api,
    binary_standard_api,
    ulp_standard_api,
    thousandth_standard_api,
    accumulative_error_standard_api,
    BINARY_COMPARE_UNSUPPORT_LIST,
)
from msprobe.core.common.const import CompareConst


class StandardRegistry:
    """
    Registry class for managing comparison standards and functions.

    This class provides a centralized registry for different comparison standards and their corresponding functions.
    It allows for dynamic registration of comparison functions based on the standard category.

    Attributes:
        comparison_functions (dict): A dictionary mapping standard categories to their corresponding comparison
        functions.
        standard_categories (dict): A dictionary mapping standard names to their corresponding API categories.

    Methods:
        _get_standard_category(api_name, dtype): Determines the standard category for a given API name and data type.
        register(standard, func): Registers a comparison function for a given standard category.
        get_comparison_function(api_name, dtype): Retrieves the comparison function for a given API name and data type.

    Note:
        The data type is used to determine the standard category if it is not supported by binary comparison.
        If the API name is not found in any standard category, it defaults to the 'benchmark' category.

    See Also:
        BaseCompare: The base class for comparison classes.
    """

    def __init__(self):
        self.comparison_functions = {}
        self.api_standard_function_map = {
            CompareConst.ABSOLUTE_THRESHOLD: absolute_standard_api,
            CompareConst.BINARY_CONSISTENCY: binary_standard_api,
            CompareConst.ULP_COMPARE: ulp_standard_api,
            CompareConst.THOUSANDTH_STANDARD: thousandth_standard_api,
            CompareConst.ACCUMULATIVE_ERROR_COMPARE: accumulative_error_standard_api,
        }

    def register(self, standard: str, func: Callable) -> None:
        """
        Registers a comparison function for a given standard category.

        Args:
            standard (str): The name of the standard category.
            func (Callable): The comparison function to be registered.

        Raises:
            ValueError: If the standard category is not supported.
        """
        if not callable(func):
            raise ValueError("The function to be registered must be callable.")
        self.comparison_functions[standard] = func

    def get_comparison_function(self, api_name, dtype, in_dtype):
        standard = self._get_standard_category(api_name, dtype, in_dtype)
        return self.comparison_functions.get(standard)

    def _get_standard_category(self, api_name, out_dtype, in_dtype):
        """
        Determines the standard category for a given API name and data type.

        This method checks if the provided data type is supported for binary comparison.
        If it is, the method returns 'binary_consistency'. Otherwise, it iterates over the
        api_standard_function_map to find a matching category for the API name.

        Args:
            api_name (str): The name of the API for which to determine the standard category.
            out_dtype (type): The data type of the output tensor.
            in_dtype (type): The data type of the input tensor.

        Returns:
            str: The name of the standard category that matches the API name and data type, or 'benchmark' if no match
            is found.

        Note:
        This method assumes that the api_standard_function_map is properly populated with standard categories and
        their corresponding API functions.
        The BINARY_COMPARE_UNSUPPORT_LIST should be defined and contain all data types that are not supported for
        binary comparison.
        """
        if is_dtype_fp8_or_hif8(out_dtype):
            return CompareConst.ULP_COMPARE
        if is_dtype_fp8_or_hif8(in_dtype):
            return CompareConst.ABSOLUTE_THRESHOLD
        if out_dtype in (
            "torch.float16",
            "torch.bfloat16",
            "torch.float32",
            "torch.float64",
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ):
            return CompareConst.BENCHMARK
        if str(out_dtype) not in BINARY_COMPARE_UNSUPPORT_LIST:
            return CompareConst.BINARY_CONSISTENCY
        for name, category in self.api_standard_function_map.items():
            if api_name in category:
                return name
        return CompareConst.BENCHMARK
