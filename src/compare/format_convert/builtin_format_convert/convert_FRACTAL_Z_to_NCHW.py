
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
"""
Function:
convert format from FRACTAL_Z to NCHW.
"""
import numpy as np


def convert(shape_from: list, shape_to: list, array: any) -> any:
    """
    Convert the data format from FRACTAL_Z to NCHW
    :param shape_from: the shape before convert
    :param shape_to: the shape after convert
    :param array: the one-dimensional array
    :return: the data array of NCHW shape
    """
    axis_c = shape_to[1]
    axis_n = shape_to[0]
    axis_no = shape_from[1]
    axis_ni = shape_from[2]
    axis_h = shape_to[2]
    axis_w = shape_to[3]
    axis_c1 = shape_from[0] // (axis_h * axis_w)
    axis_c0 = shape_from[3]
    c_pad = None if axis_c1 * axis_c0 == axis_c else axis_c - axis_c1 * axis_c0
    n_pad = None if axis_no * axis_ni == axis_n else axis_n - axis_no * axis_ni
    tmp_input_tensor = array.reshape(axis_c1, axis_h, axis_w, axis_no, axis_ni, axis_c0)
    # transpose the shape from (c1,h,w,no,ni,c0) to (no,ni,c1,c0,h,w)
    tmp_input_tensor = np.transpose(tmp_input_tensor, (3, 4, 0, 5, 1, 2))
    tmp_input_tensor = tmp_input_tensor.reshape((axis_no * axis_ni, axis_c1 * axis_c0, axis_h, axis_w))
    return tmp_input_tensor[:n_pad, :c_pad, :, :, ]
