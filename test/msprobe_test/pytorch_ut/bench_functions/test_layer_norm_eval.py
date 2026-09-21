# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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
import torch

from msprobe.pytorch.bench_functions.layer_norm_eval import npu_layer_norm_eval


class TestLayerNormEval(unittest.TestCase):
    def setUp(self):
        # 张量shape(2,3,4)
        self.data = torch.tensor(
            [[[-1.67685795, 0.05698672, 1.30941069, -0.35421315],
              [-1.50275874, 0.61931103, 0.55866784, 0.55921876],
              [-0.49006873, 0.89775485, -0.40745175, -0.26377496]],

             [[-1.98273647, -0.49512714, 0.47868022, 0.87898421],
              [1.37767696, 2.46151638, -2.13115454, 0.28628591],
              [1.33543849, 0.23067756, 1.04813659, 1.25032032]]]
        )
        self.normalized_shape = (4,)  # 归一化形状

    def test_npu_layer_norm_eval(self):
        result = npu_layer_norm_eval(self.data, self.normalized_shape)
        expected_result = torch.tensor(
            [[[-1.41726744, 0.20935509, 1.38432837, -0.17641592],
              [-1.73139334, 0.62175864, 0.55451173, 0.55512267],
              [-0.75446880, 1.71396613, -0.60752314, -0.35197425]],

             [[-1.54399860, -0.19503248, 0.68801737, 1.05101359],
              [0.51652253, 1.15334439, -1.54513049, -0.12473646],
              [0.84455395, -1.68196177, 0.18751335, 0.64989424]]]
        )
        self.assertTrue(torch.allclose(result, expected_result))
