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

from msprobe.pytorch.bench_functions.scaled_mask_softmax import npu_scaled_masked_softmax, \
    npu_scaled_masked_softmax_backward


class TestNPUScaledMaskedSoftmax(unittest.TestCase):
    def setUp(self):
        self.x = torch.randn(2, 3, 4, requires_grad=True)
        self.mask = torch.tensor([[False, True, False, True],
                                  [True, False, True, False],
                                  [False, True, False, True]], dtype=torch.bool).unsqueeze(0).repeat(2, 1, 1)
        self.scale = 0.5
        self.fixed_triu_mask = False

    def test_scaled_masked_softmax(self):
        result = npu_scaled_masked_softmax(self.x, self.mask, self.scale, self.fixed_triu_mask)
        expected_result = (self.x * self.scale).masked_fill(self.mask, -10000)
        expected_result = expected_result - torch.max(expected_result, dim=-1, keepdims=True)[0]
        expected_result = torch.exp(expected_result)
        expected_result = torch.div(expected_result, torch.sum(expected_result, dim=-1, keepdims=True))
        self.assertTrue(torch.allclose(result, expected_result, atol=1e-6))

    def test_scaled_masked_softmax_backward(self):
        y = npu_scaled_masked_softmax(self.x, self.mask, self.scale, self.fixed_triu_mask)
        y_grad = torch.randn_like(y)
        x_grad = npu_scaled_masked_softmax_backward(y_grad, y, self.mask, self.scale, self.fixed_triu_mask)

        # 计算预期的梯度
        y = y.float()
        y_grad = y_grad.float()
        expected_x_grad = y_grad * y
        expected_x_grad = y_grad - torch.sum(expected_x_grad, dim=-1, keepdims=True)
        expected_x_grad = expected_x_grad * y
        expected_x_grad = expected_x_grad * self.scale
        expected_x_grad = expected_x_grad.masked_fill(self.mask, 0)

        self.assertTrue(torch.allclose(x_grad, expected_x_grad, atol=1e-6))
