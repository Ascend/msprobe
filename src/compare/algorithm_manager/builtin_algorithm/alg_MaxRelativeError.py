
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
"""
Function:
MaxRelativeError algorithm. This file mainly involves the compare function.
"""

import numpy as np

from algorithm_manager.algorithm_parameter import AlgorithmParameter
from cmp_utils.constant.const_manager import ConstManager
from cmp_utils import utils


def compare(my_output_dump_data: any, ground_truth_dump_data: any, args: AlgorithmParameter) -> (str, str):
    """
    compare my output dump data and the ground truth dump data
    by max absolute error
    formula is MaxRE = max|(x[i]-y[i]) / y[i]|
    :param my_output_dump_data: my output dump data to compare
    :param ground_truth_dump_data: the ground truth dump data to compare
    :param args: the algorithm parameter
    :return: the result of max Relative error value and error message (the default is "")
    """
    _ = args  # Bypassing parameter is not used
    result = np.where(
        np.abs(ground_truth_dump_data) > ConstManager.FLOAT_EPSILON,
        np.abs(my_output_dump_data / ground_truth_dump_data - 1),  # abs(aa - bb) / abs(bb) -> abs(aa / bb - 1)
        0,
    ).max()
    return utils.format_value(result), ""
