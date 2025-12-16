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
convert format from NDC1HWC0 to ND.
"""
import numpy as np


def convert(shape_from: list, shape_to: list, array: any) -> any:
    """
    Convert the data format from NDC1HWC0 to ND. Current only support 5 dimensions ND
    :param shape_from: the shape before convert
    :param shape_to: the shape after convert
    :param array: the one-dimensional array
    :return:the data array of ND shape
    """
    axis_n, axis_d, axis_c1, axis_h, axis_w, axis_c0 = shape_from

    tmp_array = array.reshape(shape_from)
    tmp_array = np.transpose(tmp_array, axes=(0, 2, 5, 1, 3, 4))
    # first convert to NCDHW
    tmp_array = tmp_array.reshape((axis_n, axis_c1 * axis_c0, axis_d, axis_h, axis_w))

    if len(shape_to) == 5:
        c_pad = None if axis_c1 * axis_c0 == shape_to[1] else shape_to[1] - axis_c1 * axis_c0
        return tmp_array[:, :c_pad, ...]
    else:
        return tmp_array
