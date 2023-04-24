
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
"""
Function:
convert format from HWCN to NHWC.
"""


def convert(shape_from: list, shape_to: list, array: any) -> any:
    """
    Convert the data format from HWCN to NHWC
    :param shape_from: the shape before convert
    :param shape_to: the shape after convert
    :param array: the one-dimensional array
    :return: the data array of NHWC shape
    """
    _ = shape_to
    h_from = shape_from[0]
    w_from = shape_from[1]
    c_from = shape_from[2]
    n_from = shape_from[3]
    return array.reshape(h_from, w_from, c_from, n_from).transpose(3, 0, 1, 2)
