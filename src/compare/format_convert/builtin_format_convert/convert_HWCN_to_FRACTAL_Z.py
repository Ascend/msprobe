
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
"""
Function:
convert format from HWCN to FRACTAL_Z.
"""
from itertools import product
import numpy as np

from cmp_utils.constant.const_manager import ConstManager
from cmp_utils.utils import least_common_multiple as lcm
from cmp_utils.utils import ceiling_divide as ceil


def _get_axis(ghw_axis: list, value_map: dict, dst_c: int, w_axis: int) -> int:
    g_axis = ghw_axis[0]
    h_axis = ghw_axis[1]
    kh_axis = value_map.get('kh_axis')
    kw_axis = value_map.get('kw_axis')
    e_multi = value_map.get('e_multi')
    tmp_value = (g_axis // e_multi) * (dst_c // ConstManager.C0_AXIS) * kh_axis * kw_axis
    return tmp_value + (dst_c // ConstManager.C0_AXIS) * kh_axis * kw_axis + h_axis * kw_axis + w_axis


def _get_count_for_axis(shape_from: list, g_num: int, e_multi: int) -> int:
    kh_axis = shape_from[0]
    kw_axis = shape_from[1]
    c_ori = shape_from[2]
    c_opt = ceil(e_multi * c_ori, ConstManager.C0_AXIS) * ConstManager.C0_AXIS
    c1_axis = ceil(c_opt, ConstManager.C0_AXIS)
    return g_num * c1_axis * kh_axis * kw_axis


def convert(shape_from: list, shape_to: list, array: any, group: int = 1) -> any:
    """
    Convert the data format from HWCN to FRACTAL_Z
    :param shape_from: the shape before convert
    :param shape_to: the shape after convert
    :param array: the one-dimensional array
    :param group: group for group_conv, default value is 1
    :return: the data array of FRACTAL_Z shape
    """
    _ = shape_to
    # get before convert shape
    kh_axis = shape_from[0]
    kw_axis = shape_from[1]
    c_ori = shape_from[2]
    array_shape = array.reshape(kh_axis, kw_axis, c_ori, shape_from[3])

    # CUBE Unit K and N
    n_ori = shape_from[3] // group

    # Specific multiplying algorithm to get e_multi
    e_multi = min(lcm(lcm(c_ori, ConstManager.C0_AXIS) // c_ori, lcm(n_ori, ConstManager.N0_AXIS) // n_ori), group)
    array_to = np.zeros(
        (_get_count_for_axis(shape_from, ceil(group, e_multi), e_multi),
         ceil(ceil(e_multi * n_ori, ConstManager.N0_AXIS) * ConstManager.N0_AXIS, ConstManager.N0_AXIS),
         ConstManager.N0_AXIS,
         ConstManager.C0_AXIS), dtype=array_shape.dtype)
    # convert hwcn to gc1hwn1n0c0
    for g_axis, h_axis, w_axis, c_axis, n_axis in \
            product(range(group), range(kh_axis), range(kw_axis), range(c_ori), range(n_ori)):
        e_val = g_axis % e_multi
        dst_c = e_val * c_ori + c_axis
        dst_n = e_val * n_ori + n_axis
        src_n = g_axis * n_ori + n_axis
        array_to[_get_axis([g_axis, h_axis, w_axis],
                            {'e_multi': e_multi, 'kh_axis': kh_axis, 'kw_axis': kw_axis},
                            dst_c, w_axis)][dst_n // ConstManager.N0_AXIS][
            dst_n % ConstManager.N0_AXIS][dst_c % ConstManager.C0_AXIS] = \
            array_shape[h_axis][w_axis][c_axis][src_n]
    return array_to


