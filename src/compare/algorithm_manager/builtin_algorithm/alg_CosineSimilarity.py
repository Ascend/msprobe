# coding=utf-8
# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""
Function:
CosineSimilarity algorithm. This file mainly involves the compare function.
"""

import numpy as np

from algorithm_manager.algorithm_parameter import AlgorithmParameter
from cmp_utils import utils, utils_type, log
from cmp_utils.constant.const_manager import ConstManager


def compare(my_output_dump_data: any, ground_truth_dump_data: any, args: AlgorithmParameter) -> (str, str):
    """
    compare my output dump data and the ground truth dump data
    by cosine similarity
    cos(sitar) = sum(x[i] * y[i]) /
    (sqrt(sum(x[i] * x[i])) * sqrt(sum(y[i] * y[i])))
    :param my_output_dump_data: my output dump data to compare
    :param ground_truth_dump_data: the ground truth dump data to compare
    :param args: the algorithm parameter
    :return: the result of cosine similarity value and error message (the default is "")
    """
    if args.shape_type == utils_type.ShapeType.Scalar:
        return utils.format_value(1), "This tensor is scalar."

    my_output_norm = np.linalg.norm(my_output_dump_data, axis=-1, keepdims=True)
    ground_truth_norm = np.linalg.norm(ground_truth_dump_data, axis=-1, keepdims=True)
    if my_output_norm <= ConstManager.FLOAT_EPSILON and ground_truth_norm < ConstManager.FLOAT_EPSILON:
        return "1.0", ""
    elif my_output_norm ** 0.5 <= ConstManager.FLOAT_EPSILON:
        message = 'Cannot compare by Cosine Similarity. All the data is zero in ' + args.my_output_dump_file + '.'
        log.print_warn_log(message)
        return ConstManager.NAN, message
    elif ground_truth_norm ** 0.5 <= ConstManager.FLOAT_EPSILON:
        message = 'Cannot compare by Cosine Similarity. All the data is zero in ' + args.ground_truth_dump_file + '.'
        log.print_warn_log(message)
        return ConstManager.NAN, message

    result = ((my_output_dump_data / my_output_norm) * (ground_truth_dump_data / ground_truth_norm)).sum()
    return utils.format_value(min(result, 1.0)), ""
