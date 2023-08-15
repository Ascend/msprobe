
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
"""
Function:
Mean AbsoluteError algorithm. This file mainly involves the compare function.
"""

import numpy as np

from algorithm_manager.algorithm_parameter import AlgorithmParameter
from cmp_utils import utils


def compare(my_output_dump_data: any, ground_truth_dump_data: any, args: AlgorithmParameter) -> (str, str):
    """
    compare my output dump data and the ground truth dump data
    by mean absolute error
    formula is: MeanAE = 1/n(|x[1]-y[1]| + |x[2]-y[2]| + ... + |x[i]-y[i]|)
    :param my_output_dump_data: my output dump data to compare
    :param ground_truth_dump_data: the ground truth dump data to compare
    :param args: the algorithm parameter
    :return: the result of mean absolute error value and error message (the default is "")
    """
    _ = args  # Bypassing parameter is not used
    result = np.abs(my_output_dump_data - ground_truth_dump_data).mean()
    return utils.format_value(result), ""
