#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2023-2023. All rights reserved
import sys
import os
import unittest

try:
    import torch
except ImportError:
    torch = None
import analysis.dynamic_shape_analysis.msft_dynamic_analysis.hook
from analysis.dynamic_shape_analysis.msft_dynamic_analysis.hook import DETECTOR


@unittest.skipIf(torch is None, reason='Environment is not satisfied')
class TestDynamicShapeAnalysisHook(unittest.TestCase):
    saved_csv_path = os.path.join(os.path.dirname(
        os.path.dirname(analysis.dynamic_shape_analysis.msft_dynamic_analysis.hook.__file__)),
        "x2ms_dynamic_shape_analysis_report.csv")

    def setup(self):
        if os.path.exists(self.saved_csv_path):
            os.chmod(self.saved_csv_path, 0o750)
            os.remove(self.saved_csv_path)

    def test_hook(self):
        for _ in DETECTOR.start(range(5)):
            tensor = DETECTOR.hook_func(torch.randn, 'torch.randn', 0, 4, 3)
            mean_value = DETECTOR.hook_func(tensor.mean, 'tensor.mean', 0)
            nonzero = DETECTOR.hook_func(torch.nonzero, 'torch.nonzero', 0, (tensor > mean_value))

        assert len(DETECTOR.dynamic_api_set) == 1
        assert DETECTOR.dynamic_api_set.pop().api_name == 'torch.nonzero'
