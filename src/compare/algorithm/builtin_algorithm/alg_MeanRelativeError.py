
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
"""
Function:
Mean Relative Error algorithm. This file mainly involves the compare function.
"""

import numpy as np

from src.compare.algorithm.algorithm_parameter import AlgorithmParameter
from src.compare.cmp_utils.constant.const_manager import ConstManager
from src.compare.cmp_utils import utils
from src.compare.cmp_utils import log


def compare(my_output_dump_data: any, ground_truth_dump_data: any, args: AlgorithmParameter) -> (str, str):
    """
    compare the my output dump data and the ground truth dump data
    by mean absolute error
    formula is: MeanRE = 1/n * (|(x[1]-y[1])/y[1]| + |(x[2]-y[2])/y[2]| + ... + |(x[i]-y[i]|)/y[i]|)
    :param my_output_dump_data: the my output dump data to compare
    :param ground_truth_dump_data: the ground truth dump data to compare
    :param args: the algorithm parameter
    :return: the result of mean relative error value and error message (the default is "")
    """
    np.seterr(divide='ignore', invalid='ignore')

    relative_error = np.divide((my_output_dump_data - ground_truth_dump_data), ground_truth_dump_data)
    mean_relative_error = np.average(np.abs(relative_error))
    if np.isnan(mean_relative_error):
        message = 'Cannot compare by MeanRelativeError, The data contains 0 or nan in %s '\
                  'or %s.' % (args.my_output_dump_file, args.ground_truth_dump_file)
        log.print_warn_log(message)
        return ConstManager.NAN, message
    return utils.format_value(mean_relative_error), ""
