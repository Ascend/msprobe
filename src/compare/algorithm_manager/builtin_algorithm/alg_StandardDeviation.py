# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
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
