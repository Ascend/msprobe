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
convert format from FRACTAL_NZ to NCHW.
"""
from functools import reduce

import numpy as np


def convert(shape_from: list, shape_to: list, array: any) -> any:
    """
    Convert the data format from FRACTAL_NZ to NCHW
    :param shape_from: the shape before convert
    :param shape_to: the shape after convert
    :param array: the one-dimensional array
    :return: the data array of NCHW shape
    """
    if len(shape_to) == 1:
        axis_h, axis_n, axis_c = 1, 1, shape_to[0]
    elif len(shape_to) == 2:
        axis_h, axis_n, axis_c = 1, shape_to[0], shape_to[1]
    else:
        axis_h, axis_n, axis_c = reduce(lambda x, y: x * y, shape_to[:-2]), shape_to[-2], shape_to[-1]
    axis_c0 = shape_from[-1]
    axis_ni = shape_from[-2]
    axis_no = shape_from[-3]
    axis_c1 = shape_from[-4]
    c_pad = None if axis_c1 * axis_c0 == axis_c else axis_c - axis_c1 * axis_c0
    n_pad = None if axis_no * axis_ni == axis_n else axis_n - axis_no * axis_ni
    tmp_input_tensor = array.reshape(axis_h, axis_c1, axis_no, axis_ni, axis_c0)
    tmp_input_tensor = np.transpose(tmp_input_tensor, axes=(0, 2, 3, 1, 4))
    tmp_input_tensor = tmp_input_tensor.reshape((axis_h, axis_no * axis_ni, axis_c1 * axis_c0))
    data_y = tmp_input_tensor[:, :n_pad, :c_pad]
    if len(shape_to) <= 2:
        data_y = data_y.reshape([data_y.shape[1], data_y.shape[2]])
    return data_y
