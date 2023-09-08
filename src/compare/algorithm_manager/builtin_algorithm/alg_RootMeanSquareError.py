
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
"""
Function:
Root Mean Square Error algorithm. This file mainly involves the compare function.
"""
import numpy as np

from algorithm_manager.algorithm_parameter import AlgorithmParameter
from cmp_utils import utils


def compare(my_output_dump_data: any, ground_truth_dump_data: any, args: AlgorithmParameter) -> (str, str):
    """
    compare my output dump data and the ground truth dump data
    by mean absolute error
    formula is: RMSE = sqrt(((x[0]-y[0])^2 + (x[1]-y[1])^2 + ... + (x[i]-y[i])^2) / n)
    :param my_output_dump_data: my output dump data to compare
    :param ground_truth_dump_data: the ground truth dump data to compare
    :param args: the algorithm parameter
    :return: the result of root_mean_square error value and error message (the default is "")
    """
    _ = args  # Bypassing parameter is not used
    result = np.sqrt(np.average(((my_output_dump_data - ground_truth_dump_data) ** 2)))
    return utils.format_value(result), ""
