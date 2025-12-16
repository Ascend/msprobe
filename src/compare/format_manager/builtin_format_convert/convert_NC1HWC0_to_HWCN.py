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
convert format from NC1HWC0 to HWCN.
"""
import numpy as np


def convert(shape_from: list, shape_to: list, array: any) -> any:
    """
    Convert the data format from NC1HWC0 to HWCN
    :param shape_from: the shape before convert
    :param shape_to: the shape after convert
    :param array: the one-dimensional array
    :return: the data array of HWCN shape
    """
    _ = shape_to
    n_from = shape_from[0]
    c1_from = shape_from[1]
    h_from = shape_from[2]
    w_from = shape_from[3]
    c0_from = shape_from[4]

    array_shape = array.reshape(n_from, c1_from, h_from, w_from, c0_from)
    tmp_input_tensor = np.transpose(array_shape, axes=(2, 3, 1, 4, 0))
    tmp_input_tensor = tmp_input_tensor.reshape((h_from, w_from, c1_from * c0_from, n_from))
    return tmp_input_tensor
