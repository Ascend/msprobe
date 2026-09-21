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

from msprobe.pytorch.api_accuracy_checker.compare.algorithm import compare_bool_tensor
from msprobe.pytorch.api_accuracy_checker.precision_standard.base_standard import BaseCompare


class BinaryCompare(BaseCompare):
    """
    Binary comparison class for comparing boolean tensors.

    This class is designed to compare the output of a binary operation between a benchmark and a device.
    It calculates the error rate of the comparison and provides a simple metric for assessing the accuracy.

    Attributes:
        bench_output (np.ndarray): The output from the benchmark.
        device_output (np.ndarray): The output from the device.
        compare_column (object): The column object to store comparison results.
        dtype (torch.dtype): The data type of the outputs.

    Methods:
        _compute_metrics(): Computes the comparison metrics, specifically the error rate.

    Note:
        This class assumes that the input data is an instance of InputData containing the benchmark output,
        device output, comparison column, and data type. The outputs are expected to be boolean tensors.

    See Also:
        BaseCompare: The base class for comparison classes.
        compare_bool_tensor: The function used to compare boolean tensors.
    """

    def __init__(self, input_data):  # pylint: disable=useless-parent-delegation
        super(BinaryCompare, self).__init__(input_data)  # pylint: disable=super-with-arguments

    def _pre_compare(self):
        pass

    def _compute_metrics(self):
        """
        Computes the error rate metric for the comparison between benchmark and device outputs.

        This method calculates the proportion of mismatches between the benchmark output and the device output.
        It uses the `compare_bool_tensor` function to compare the two tensors and extract the error rate.

        Returns:
        dict: A dictionary containing the computed error rate metric.
            The dictionary has the following key:
            - "error_rate": The proportion of mismatches between the benchmark and device outputs.
        """
        error_rate, _, _ = compare_bool_tensor(self.bench_output, self.device_output)

        return {"error_rate": error_rate}
