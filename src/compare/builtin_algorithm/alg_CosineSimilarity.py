#!/usr/bin/env python
# coding=utf-8
"""
Function:
CosineSimilarity algorithm. This file mainly involves the compare function.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2019-2021
"""

import numpy as np

from algorithm_parameter import AlgorithmParameter
import utils
import log

from const_manager import ConstManager


def compare(my_output_dump_data: any, ground_truth_dump_data: any, args: AlgorithmParameter) -> (str, str):
    """
    compare the my output dump data and the ground truth dump data
    by cosine similarity
    cos(sitar) = sum(x[i] * y[i]) /
    (sqrt(sum(x[i] * x[i])) * sqrt(sum(y[i] * y[i])))
    :param my_output_dump_data: the my output dump data to compare
    :param ground_truth_dump_data: the ground truth dump data to compare
    :param args: the algorithm parameter
    :return: the result of cosine similarity value and error message (the default is "")
    """
    np.seterr(divide='ignore', invalid='ignore')
    if args.shape_type == utils.ShapeType.Scalar:
        return utils.format_value(1), "This tensor is scalar."
    numerator = 0.0
    source_denominator = 0.0
    compare_denominator = 0.0
    for my_output_data, ground_truth_data in zip(my_output_dump_data, ground_truth_dump_data):
        temp_x = float(my_output_data)
        temp_y = float(ground_truth_data)
        numerator += temp_x * temp_y
        source_denominator += temp_x * temp_x
        compare_denominator += temp_y * temp_y

    if abs(source_denominator) <= ConstManager.FLOAT_EPSILON and \
            abs(compare_denominator) <= ConstManager.FLOAT_EPSILON:
        result = '1.0'
    elif abs(source_denominator ** 0.5) <= ConstManager.FLOAT_EPSILON:
        message = 'Cannot compare by Cosine Similarity. All the data is zero in ' + args.my_output_dump_file + '.'
        log.print_warn_log(message)
        return ConstManager.NAN, message
    elif abs(compare_denominator ** 0.5) <= ConstManager.FLOAT_EPSILON:
        message = 'Cannot compare by Cosine Similarity. All the data is zero in ' + args.ground_truth_dump_file + '.'
        log.print_warn_log(message)
        return ConstManager.NAN, message
    else:
        denominator = ((source_denominator ** 0.5) * (compare_denominator ** 0.5))
        result = 0
        if denominator != 0:
            result = utils.format_value(numerator / denominator)
    return result, ""
