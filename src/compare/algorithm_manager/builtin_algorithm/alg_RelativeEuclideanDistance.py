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
