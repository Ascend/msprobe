
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
"""
Function:
convert format from FRACTAL_NZ to ND.
"""
from functools import reduce
import numpy as np


def convert(shape_from: list, shape_to: list, array: any) -> any:
    """
    Convert the data format from FRACTAL_NZ to ND
    :param shape_from: the shape before convert
    :param shape_to: the shape after convert
    :param array: the one-dimensional array
    :return: the data array of ND shape
    """
    if len(shape_to) == 1:
        axis_h, axis_n, axis_c = 1, 1, shape_to[0]
    elif len(shape_to) == 2:
        axis_h, axis_n, axis_c = 1, shape_to[0], shape_to[1]
    else:
        axis_h, axis_n, axis_c = reduce(lambda x, y: x * y, shape_to[:-2]), shape_to[-2], shape_to[-1]
    axis_c1 = shape_from[-4]
    axis_no = shape_from[-3]
    axis_ni = shape_from[-2]
    axis_c0 = shape_from[-1]
    c_pad = None if axis_c1 * axis_c0 == axis_c else axis_c - axis_c1 * axis_c0
    n_pad = None if axis_no * axis_ni == axis_n else axis_n - axis_no * axis_ni
    tmp_input_tensor = array.reshape(axis_h, axis_c1, axis_no, axis_ni, axis_c0)
    tmp_input_tensor = np.transpose(tmp_input_tensor, axes=(0, 2, 3, 1, 4))
    tmp_input_tensor = tmp_input_tensor.reshape((axis_h, axis_no * axis_ni, axis_c1 * axis_c0))
    data_y = tmp_input_tensor[:, :n_pad, :c_pad]
    if len(shape_to) <= 2:
        data_y = data_y.reshape([data_y.shape[1], data_y.shape[2]])
    return data_y
