#!/usr/bin/env python
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


class PathType(Enum):
    """
    The enum for path type
    """
    All = 0
    File = 1
    Directory = 2


class FusionRelation(Enum):
    """
    The enum for fusion relation
    """
    OneToOne = 0
    MultiToOne = 1
    OneToMulti = 2
    MultiToMulti = 3
    L1Fusion = 4