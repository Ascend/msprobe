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

import numpy as np


class CompareInput:
    """
    A class to encapsulate the input data required for comparison operations.

    Attributes:
        bench_output (np.ndarray): The benchmark output values.
        device_output (np.ndarray): The device output values.
        compare_column (class): A clasee to store and update comparison metrics.
        dtype (type, optional): The data type of the outputs. Defaults to None.
        rel_err_orign (float or array-like, optional): The original relative error values. Defaults to None.

    Methods:
        __init__(bench_output, device_output, compare_column, dtype, rel_err_orign):
        Initializes an instance of CompareInput.
    """

    def __init__(self, bench_output, device_output, compare_column, dtype=None, rel_err_orign=None):
        self.bench_output = bench_output
        self.device_output = device_output
        if not isinstance(bench_output, np.ndarray) or not isinstance(device_output, np.ndarray):
            raise TypeError("The input should be numpy array")
        self.compare_column = compare_column
        self.dtype = dtype
        self.rel_err_orign = rel_err_orign


class PrecisionCompareInput:
    def __init__(self, row_npu, row_gpu, dtype, compare_column, api_name=None):
        self.row_npu = row_npu
        self.row_gpu = row_gpu
        self.dtype = dtype
        self.compare_column = compare_column
        self.api_name = api_name
