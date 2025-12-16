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
This file mainly involves the common function definition.
"""
from enum import Enum


class ShapeType(Enum):
    """
    The enum for shape type
    """
    Scalar = 0
    Vector = 1
    Matrix = 2
    Tensor = 3


class FusionRelation(Enum):
    """
    The enum for fusion relation
    """
    OneToOne = 0
    MultiToOne = 1
    OneToMulti = 2
    MultiToMulti = 3
    L1Fusion = 4


class DatasetAttr(Enum):
    """
    The enum for pytorch dump data attribute
    """
    DataType = 0
    DeviceType = 1
    FormatType = 2
    Type = 3
    Stride = 4


class DeviceType(Enum):
    """
    The enum for device type
    """
    GPU = 1
    NPU = 10
    CPU = 0
