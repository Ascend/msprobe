#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
"""
Function:
This file mainly involves the common function definition.
"""

class FusionRelation(Enum):
    """
    The enum for fusion relation
    """
    OneToOne = 0
    MultiToOne = 1
    OneToMulti = 2
    MultiToMulti = 3
    L1Fusion = 4