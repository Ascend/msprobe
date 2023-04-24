
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
"""
Function:
MaxAbsoluteError algorithm. This file mainly involves the compare function.
"""

import numpy as np

from src.compare.algorithm.algorithm_parameter import AlgorithmParameter
from src.compare.cmp_utils import utils


def compare(my_output_dump_data: any, ground_truth_dump_data: any, args: AlgorithmParameter) -> (str, str):
    """
    compare the my output dump data and the ground truth dump data
    by max absolute error
    formula is max(|x[i]-y[i]|)
    :param my_output_dump_data: the my output dump data to compare
    :param ground_truth_dump_data: the ground truth dump data to compare
    :param args: the algorithm parameter
    :return: the result of max absolute error value and error message (the default is "")
    """
    np.seterr(divide='ignore', invalid='ignore')
    _ = args
    max_value = 0.0
    for my_output_data, ground_truth_data in zip(my_output_dump_data, ground_truth_dump_data):
        temp_x = float(my_output_data)
        temp_y = float(ground_truth_data)
        abs_error = abs(temp_x - temp_y)
        if abs_error > max_value:
            max_value = abs_error

    return utils.format_value(max_value), ""
