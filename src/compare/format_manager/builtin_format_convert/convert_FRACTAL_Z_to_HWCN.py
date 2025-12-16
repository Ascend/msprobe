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
convert format from FRACTAL_Z to HWCN.
"""
import numpy as np


def convert(shape_from: list, shape_to: list, array: any) -> any:
    """
    Convert the data format from FRACTAL_Z to HWCN
    :param shape_from: the shape before convert
    :param shape_to: the shape after convert
    :param array: the one-dimensional array
    :return: the data array of HWCN shape
    """
    axis_c = shape_to[2]
    axis_n = shape_to[3]
    axis_no = shape_from[1]
    axis_ni = shape_from[2]
    axis_h = shape_to[0]
    axis_w = shape_to[1]
    axis_c1 = shape_from[0] // (axis_h * axis_w)
    axis_c0 = shape_from[3]
    c_pad = None if axis_c1 * axis_c0 == axis_c else axis_c - axis_c1 * axis_c0
    n_pad = None if axis_no * axis_ni == axis_n else axis_n - axis_no * axis_ni
    tmp_input_tensor = array.reshape(axis_c1, axis_h, axis_w, axis_no, axis_ni, axis_c0)
    # transpose the shape from (c1,h,w,no,ni,c0) to (h,w,c1,c0,no,ni)
    tmp_input_tensor = np.transpose(tmp_input_tensor, (1, 2, 0, 5, 3, 4))
    tmp_input_tensor = tmp_input_tensor.reshape((axis_h, axis_w, axis_c1 * axis_c0, axis_no * axis_ni))
    return tmp_input_tensor[:, :, :c_pad, :n_pad]
