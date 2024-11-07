# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
"""
Function:
KullbackLeiblerDivergence algorithm. This file mainly involves the compare function.
"""

import numpy as np

from algorithm_manager.algorithm_parameter import AlgorithmParameter
from cmp_utils import utils, log
from cmp_utils.constant.const_manager import ConstManager


FLOAT_EPSILON = np.finfo(np.float32).eps


def _normalized(dump_data: any) -> any:
    max_value = dump_data.max()
    min_value = dump_data.min()
    range_value = max_value - min_value
    if range_value != 0:
        dump_data_to_1 = (dump_data - min_value) / range_value
    else:
        dump_data_to_1 = dump_data
    # normalized, the sum of dump data is not equal with zero
    return np.maximum(dump_data_to_1, FLOAT_EPSILON)


def compare(my_output_dump_data: any, ground_truth_dump_data: any, args: AlgorithmParameter) -> (str, str):
    """
    compare my output dump data and the ground truth dump data
    by kullback leibler divergence
    1. P(x)=(x-Min)/(Max-Min)
    2. pdf = x/sum(x)
    3. sum(P(x)log(P(x)/Q(x)))
    :param my_output_dump_data: my output dump data to compare
    :param ground_truth_dump_data: the ground truth dump data to compare
    :param args: the algorithm parameter
    :return: the result of relative euclidean distance value and error message (the default is "")
    """
    my_output_dump_data_pdf = _normalized(my_output_dump_data)
    ground_true_dump_data_pdf = _normalized(ground_truth_dump_data)
    is_my_output_all_zero = np.all(my_output_dump_data_pdf == 0)
    is_ground_truth_all_zero = np.all(ground_true_dump_data_pdf == 0)
    if is_my_output_all_zero and is_ground_truth_all_zero:
        message = 'Cannot compare by KL Divergence. All the data is zero in %r and %r.' \
                  % (args.my_output_dump_file, args.ground_truth_dump_file)
        log.print_warn_log(message)
        return ConstManager.NAN, message
    if is_my_output_all_zero:
        message = 'Cannot compare by KL Divergence. All the data is zero in ' + args.my_output_dump_file + '.'
        log.print_warn_log(message)
        return ConstManager.NAN, message
    if is_ground_truth_all_zero:
        message = 'Cannot compare by KL Divergence. All the data is zero in ' + args.ground_truth_dump_file + '.'
        log.print_warn_log(message)
        return ConstManager.NAN, message

    norm_xx = my_output_dump_data_pdf / my_output_dump_data_pdf.sum()  # cannot be all 0
    norm_yy = ground_true_dump_data_pdf / ground_true_dump_data_pdf.sum()  # cannot be all 0
    result = (norm_xx * np.log(norm_xx / norm_yy)).sum()

    inf_message = ''
    if abs(result) < ConstManager.FLOAT_EPSILON:
        result = 0.0
    if str(result) == "inf":
        inf_message = 'Cannot compare by KL Divergence. The data contains 0 in %r.' % args.ground_truth_dump_file
    return utils.format_value(result), inf_message
