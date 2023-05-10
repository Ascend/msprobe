
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
"""
Function:
AccumulatedRelativeError algorithm. This file mainly involves the compare function.
"""

import numpy as np

from algorithm_manager.algorithm_parameter import AlgorithmParameter
from cmp_utils import utils
from cmp_utils.constant.const_manager import ConstManager


def compare(my_output_dump_data: any, ground_truth_dump_data: any, args: AlgorithmParameter) -> (str, str):
    """
    compare the my output dump data and the ground truth dump data
    by accumulated relative error
    formula is (|x[i]-y[i]|) / |y[i]| + (x[i+1]-y[i+1]) / y[i+1]
    :param my_output_dump_data: the my output dump data to compare
    :param ground_truth_dump_data: the ground truth dump data to compare
    :param args: the algorithm parameter
    :return: the result of accumulated relative error value and error message (the default is "")
    """
    np.seterr(divide='ignore', invalid='ignore')
    _ = args
    result = 0
    for my_output_data, ground_truth_data in zip(my_output_dump_data, ground_truth_dump_data):
        temp_x = float(my_output_data)
        temp_y = float(ground_truth_data)
        if abs(temp_y) > ConstManager.FLOAT_EPSILON and abs(temp_y) != 0:
            result += abs(temp_x - temp_y) / abs(temp_y)

    return utils.format_value(result), ""
