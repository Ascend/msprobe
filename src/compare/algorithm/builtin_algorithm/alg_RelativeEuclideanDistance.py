#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
"""
Function:
RelativeEuclideanDistance algorithm. This file mainly involves the compare function.
"""

import numpy as np

from src.compare.algorithm.algorithm_parameter import AlgorithmParameter
import utils

from cmp_utils.constant.const_manager import ConstManager


def compare(my_output_dump_data: any, ground_truth_dump_data: any, args: AlgorithmParameter) -> (str, str):
    """
    compare the my output dump data and the ground truth dump data
    by relative euclidean distance
    formula is sqrt(sum((x[i]-y[i])*(x[i]-y[i]))) / sqrt(sum(y[i]*y[i]))
    :param my_output_dump_data: the my output dump data to compare
    :param ground_truth_dump_data: the ground truth dump data to compare
    :param args: the algorithm parameter
    :return: the result of relative euclidean distance value and error message (the default is "")
    """
    np.seterr(divide='ignore', invalid='ignore')
    _ = args
    sum_numerator = 0.0
    sum_denominator = 0.0
    for my_output_data, ground_truth_data in zip(my_output_dump_data, ground_truth_dump_data):
        temp_x = float(my_output_data)
        temp_y = float(ground_truth_data)
        sum_numerator += (temp_x - temp_y) * (temp_x - temp_y)
        sum_denominator += temp_y * temp_y

    result = '0.0'
    if abs(sum_denominator ** 0.5) > ConstManager.FLOAT_EPSILON and (sum_denominator ** 0.5) != 0:
        result = utils.format_value((sum_numerator ** 0.5) / (sum_denominator ** 0.5))
    return result, ""
