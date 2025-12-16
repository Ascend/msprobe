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
convert format from NC1HWC0 to NCHW.
"""
import numpy as np
from cmp_utils import log
from cmp_utils.constant.compare_error import CompareError


def convert(shape_from: list, shape_to: list, array: any) -> any:
    """
    Convert the data format from NC1HWC0 to NCHW
    :param shape_from: the shape before convert
    :param shape_to: the shape after convert
    :param array: the one-dimensional array
    :return: the data array of NCHW shape
    """
    _ = shape_to
    if len(shape_from) < 5:
        log.print_error_log("length of shape of NC1HWC0 is less than 5, please check.")
        raise CompareError(CompareError.MSACCUCMP_INDEX_OUT_OF_BOUNDS_ERROR)
    n_from = shape_from[0]
    c1_from = shape_from[1]
    h_from = shape_from[2]
    w_from = shape_from[3]
    c0_from = shape_from[4]

    array_shape = array.reshape(n_from, c1_from, h_from, w_from, c0_from)
    tmp_input_tensor = np.transpose(array_shape, axes=(0, 1, 4, 2, 3))
    tmp_input_tensor = tmp_input_tensor.reshape((n_from, c1_from * c0_from, h_from, w_from))
    return tmp_input_tensor
