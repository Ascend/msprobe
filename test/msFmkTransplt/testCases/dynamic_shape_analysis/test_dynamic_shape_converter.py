#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

import libcst as cst
import unittest


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
