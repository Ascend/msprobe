#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

import unittest
import libcst as cst


class TestDynamicShapeAnalysisConvert(unittest.TestCase):
    def test_dynamic_shape_analysis_converter(self):
        from analysis.dynamic_shape_analysis.dynamic_shape_converter import DynamicShapeTransformer
        code = '''
import torch

a = torch.tensor([1,2])
b = a.mean() + a.mean()
for i in range(5):
    pass

@torch.jit.script
def jit_script():
    a = torch.tensor([1,2])

@torch.jit.script
class jit_class:
    def __call__(self):
        a = torch.tensor([1,2])
'''
        output_code = '''
import torch
from msft_dynamic_analysis.hook import DETECTOR

a = DETECTOR.hook_func(torch.tensor, 'torch.tensor', 0, [1,2])
b = DETECTOR.hook_func(a.mean, 'a.mean', 0) + DETECTOR.hook_func(a.mean, 'a.mean', 1)
for i in range(5):
    pass

@torch.jit.script
def jit_script():
    a = torch.tensor([1,2])

@torch.jit.script
class jit_class:
    def __call__(self):
        a = torch.tensor([1,2])
'''
        wrapped_module = cst.MetadataWrapper(cst.parse_module(code))
        new_module = wrapped_module.visit(DynamicShapeTransformer())

        assert output_code == new_module.code
