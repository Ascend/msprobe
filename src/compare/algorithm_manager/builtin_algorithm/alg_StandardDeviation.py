# coding=utf-8
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

"""
Function:
StandardDeviation algorithm. This file mainly involves the compare function.
"""

import numpy as np

from algorithm_manager.algorithm_parameter import AlgorithmParameter
from cmp_utils.constant.const_manager import ConstManager


def compare(my_output_dump_data: any, ground_truth_dump_data: any, args: AlgorithmParameter) -> (str, str):
    """
    compare the my output dump data and the ground truth dump data
    by standard deviation
    :param my_output_dump_data: the my output dump data to compare
    :param ground_truth_dump_data: the ground truth dump data to compare
    :param args: the algorithm parameter
    :return: the result of the standard deviation value and error message (the default is "")
    """
    _ = args  # Bypassing parameter is not used
    left_std = np.std(my_output_dump_data, dtype=np.float64)
    right_std = np.std(ground_truth_dump_data, dtype=np.float64)
    left_mean = np.mean(my_output_dump_data, dtype=np.float64)
    right_mean = np.mean(ground_truth_dump_data, dtype=np.float64)
    left_std = 0.0 if abs(left_std) < ConstManager.MINIMUM_VALUE else left_std
    left_mean = 0.0 if abs(left_mean) < ConstManager.MINIMUM_VALUE else left_mean
    right_std = 0.0 if abs(right_std) < ConstManager.MINIMUM_VALUE else right_std
    right_mean = 0.0 if abs(right_mean) < ConstManager.MINIMUM_VALUE else right_mean
    return "(%.3f;%.3f),(%.3f;%.3f)" % (left_mean, left_std, right_mean, right_std), ""
