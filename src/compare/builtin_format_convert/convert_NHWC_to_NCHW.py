#!/usr/bin/env python
# coding=utf-8
"""
Function:
convert format from NHWC to NCHW.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2019-2021
"""


def convert(shape_from: list, shape_to: list, array: any) -> any:
    """
    Convert the data format from NHWC to NCHW
    :param shape_from: the shape before convert
    :param shape_to: the shape after convert
    :param array: the one-dimensional array
    :return: the data array of NCHW shape
    """
    _ = shape_to
    n_from = shape_from[0]
    h_from = shape_from[1]
    w_from = shape_from[2]
    c_from = shape_from[3]
    return array.reshape(n_from, h_from, w_from, c_from).transpose(0, 3, 1, 2)
