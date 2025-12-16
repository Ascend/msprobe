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
convert format from NDC1HWC0 to NCDHW.
"""
import numpy as np
from cmp_utils import log
from cmp_utils.constant.compare_error import CompareError


def convert(shape_from: list, shape_to: list, array: any) -> any:
    """
    Convert the data format from NDC1HWC0 to NCDHW
    :param shape_from: the shape before convert
    :param shape_to: the shape after convert
    :param array: the one-dimensional array
    :return:the data array of NCDHW shape
    """
    if len(shape_from) < 6:
        log.print_error_log("length of shape of NDC1HWC0 is less than 6, please check.")
        raise CompareError(CompareError.MSACCUCMP_INDEX_OUT_OF_BOUNDS_ERROR)
    axis_n = shape_from[0]
    axis_d = shape_from[1]
    axis_c1 = shape_from[2]
    axis_h = shape_from[3]
    axis_w = shape_from[4]
    axis_c0 = shape_from[5]
    c_pad = None if axis_c1 * axis_c0 == shape_to[1] else shape_to[1] - axis_c1 * axis_c0
    tmp_array = array.reshape(shape_from)
    tmp_array = np.transpose(tmp_array, axes=(0, 2, 5, 1, 3, 4))
    tmp_array = tmp_array.reshape((axis_n, axis_c1 * axis_c0, axis_d, axis_h, axis_w))
    return tmp_array[:, :c_pad, :, :, :]
