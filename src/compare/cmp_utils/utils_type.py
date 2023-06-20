
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
