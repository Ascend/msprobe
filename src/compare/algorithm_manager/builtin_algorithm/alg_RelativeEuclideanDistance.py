
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
"""
Function:
RelativeEuclideanDistance algorithm. This file mainly involves the compare function.
"""

import numpy as np

from algorithm_manager.algorithm_parameter import AlgorithmParameter
from cmp_utils.constant.const_manager import ConstManager
from cmp_utils import utils


def compare(my_output_dump_data: any, ground_truth_dump_data: any, args: AlgorithmParameter) -> (str, str):
    """
    compare my output dump data and the ground truth dump data
    by relative euclidean distance
    formula is sqrt(sum((x[i]-y[i])*(x[i]-y[i]))) / sqrt(sum(y[i]*y[i]))
    :param my_output_dump_data: output dump data to compare
    :param ground_truth_dump_data: the ground truth dump data to compare
    :param args: the algorithm parameter
    :return: the result of relative euclidean distance value and error message (the default is "")
    """
    _ = args  # Bypassing parameter is not used
    ground_truth_square_num = (ground_truth_dump_data ** 2).sum()
    if ground_truth_square_num ** 0.5 <= ConstManager.FLOAT_EPSILON:
        result = 0.0
    else:
        result = ((my_output_dump_data - ground_truth_dump_data) ** 2).sum() / ground_truth_square_num
    return utils.format_value(result ** 0.5), ""
