
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
"""
Function:
convert format from NDC1HWC0 to NCDHW.
"""
import numpy as np


def convert(shape_from: list, shape_to: list, array: any) -> any:
    """
    Convert the data format from NDC1HWC0 to NCDHW
    :param shape_from: the shape before convert
    :param shape_to: the shape after convert
    :param array: the one-dimensional array
    :return:the data array of NCDHW shape
    """
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
