
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
"""
Function:
KullbackLeiblerDivergence algorithm. This file mainly involves the compare function.
"""

import numpy as np
import scipy.stats

from algorithm.algorithm_parameter import AlgorithmParameter
from cmp_utils import utils
from cmp_utils import log
from cmp_utils.constant.const_manager import ConstManager


def _normalized(dump_data: any) -> any:
    dump_data = dump_data.astype(np.float64)
    max_value = dump_data.max()
    min_value = dump_data.min()
    range_value = max_value - min_value
    if range_value != 0:
        dump_data_to_1 = (dump_data - min_value) / range_value
    else:
        dump_data_to_1 = dump_data
    # normalized, the sum of dump data is not equal with zero
    return dump_data_to_1


def compare(my_output_dump_data: any, ground_truth_dump_data: any, args: AlgorithmParameter) -> (str, str):
    """
    compare the my output dump data and the ground truth dump data
    by kullback leibler divergence
    1. P(x)=（x-Min）/(Max-Min)
    2. pdf = x/sum(x)
    3. sum(P(x)log(P(x)/Q(x)))
    :param my_output_dump_data: the my output dump data to compare
    :param ground_truth_dump_data: the ground truth dump data to compare
    :param args: the algorithm parameter
    :return: the result of relative euclidean distance value and error message (the default is "")
    """
    np.seterr(divide='ignore', invalid='ignore')
    my_output_dump_data_pdf = _normalized(my_output_dump_data)
    ground_true_dump_data_pdf = _normalized(ground_truth_dump_data)
    if np.all(my_output_dump_data_pdf == 0) and np.all(ground_true_dump_data_pdf == 0):
        message = 'Cannot compare by KL Divergence. All the data is zero in %s and %s.' \
                  % (args.my_output_dump_file, args.ground_truth_dump_file)
        log.print_warn_log(message)
        return ConstManager.NAN, message
    if np.all(my_output_dump_data_pdf == 0):
        message = 'Cannot compare by KL Divergence. All the data is zero in ' + args.my_output_dump_file + '.'
        log.print_warn_log(message)
        return ConstManager.NAN, message
    if np.all(ground_true_dump_data_pdf == 0):
        message = 'Cannot compare by KL Divergence. All the data is zero in ' + args.ground_truth_dump_file + '.'
        log.print_warn_log(message)
        return ConstManager.NAN, message
    result = scipy.stats.entropy(my_output_dump_data_pdf, ground_true_dump_data_pdf)
    inf_message = ''
    if str(result) == "inf":
        inf_message = 'Cannot compare by KL Divergence. The data contains 0 in %s.' % args.ground_truth_dump_file
    if abs(result) < ConstManager.FLOAT_EPSILON:
        result = 0.0
    return utils.format_value(result), inf_message
