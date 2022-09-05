#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
"""
Function:
convert format from NHWC to FRACTAL_Z.
"""
import math
import numpy as np

from const_manager import ConstManager


def lcm(left: int, right: int) -> int:
    """
    Least common multiple, in this file, n could not zero
    :param left: One of the calculation parameters
    :param right: One of the calculation parameters
    :return: left, right Least common multiple
    """
    return (left * right) // math.gcd(left, right)


def ceil(left: int, right: int) -> int:
    """
    Ceiling divide, in this file, n could not zero
    :param left: One of the calculation parameters
    :param right: One of the calculation parameters
    :return: left, right Ceiling divide
    """
    return (left + right - 1) // right


def _get_axis(gnh_axis: list, value_map: dict, dst_c: int, w_axis: int) -> int:
    g_axis = gnh_axis[0]
    h_axis = gnh_axis[2]
    kh_axis = value_map.get('kh_axis')
    kw_axis = value_map.get('kw_axis')
    e_multi = value_map.get('e_multi')
    tmp_value = (g_axis // e_multi) * (dst_c // ConstManager.C0_AXIS) * kh_axis * kw_axis
    return tmp_value + (dst_c // ConstManager.C0_AXIS) * kh_axis * kw_axis + h_axis * kw_axis + w_axis


def _convert_for_w_and_c(gnh_axis: list, value_map: dict, array_to: any, array_shape: any) -> None:
    g_axis = gnh_axis[0]
    n_axis = gnh_axis[1]
    h_axis = gnh_axis[2]
    c_ori = value_map.get('c_ori')
    n_ori = value_map.get('n_ori')
    for w_axis in range(value_map.get('kw_axis')):
        for c_axis in range(c_ori):
            e_val = g_axis % value_map.get('e_multi')
            dst_c = e_val * c_ori + c_axis
            dst_n = e_val * n_ori + n_axis
            src_n = g_axis * n_ori + n_axis
            array_to[_get_axis(gnh_axis, value_map, dst_c, w_axis)][dst_n // ConstManager.N0_AXIS][
                dst_n % ConstManager.N0_AXIS][dst_c % ConstManager.C0_AXIS] = \
                array_shape[src_n][h_axis][w_axis][c_axis]


def _get_count_for_axis(shape_from: list, g_num: int, e_multi: int) -> int:
    c_ori = shape_from[3]
    kh_axis = shape_from[1]
    kw_axis = shape_from[2]
    c_opt = ceil(e_multi * c_ori, ConstManager.C0_AXIS) * ConstManager.C0_AXIS
    c1_axis = ceil(c_opt, ConstManager.C0_AXIS)
    return g_num * c1_axis * kh_axis * kw_axis


def convert(shape_from: list, shape_to: list, array: any, group: int = 1) -> any:
    """
    Convert the data format from NHWC to FRACTAL_Z
    :param shape_from: the shape before convert
    :param shape_to: the shape after convert
    :param array: the one-dimensional array
    :param group: group for group_conv, default value is 1
    :return: the data array of FRACTAL_Z shape
    """
    _ = shape_to
    # get before convert shape
    kh_axis = shape_from[1]
    kw_axis = shape_from[2]
    c_ori = shape_from[3]
    array_shape = array.reshape(shape_from[0], kh_axis, kw_axis, c_ori)

    # CUBE Unit K and N
    n_ori = shape_from[0] // group

    # Specific multiplying algorithm to get e_multi
    e_multi = min(lcm(lcm(c_ori, ConstManager.C0_AXIS) // c_ori, lcm(n_ori, ConstManager.N0_AXIS) // n_ori), group)
    array_to = np.zeros(
        (_get_count_for_axis(shape_from, ceil(group, e_multi), e_multi),
         ceil(ceil(e_multi * n_ori, ConstManager.N0_AXIS) * ConstManager.N0_AXIS, ConstManager.N0_AXIS),
         ConstManager.N0_AXIS, ConstManager.C0_AXIS), dtype=array_shape.dtype)
    # convert nhwc to gc1hwn1n0c0
    for g_axis in range(group):
        for n_axis in range(n_ori):
            for h_axis in range(kh_axis):
                _convert_for_w_and_c([g_axis, n_axis, h_axis],
                                     {'c_ori': c_ori, 'n_ori': n_ori, 'e_multi': e_multi,
                                      'kh_axis': kh_axis, 'kw_axis': kw_axis},
                                     array_to, array_shape)
    return array_to
