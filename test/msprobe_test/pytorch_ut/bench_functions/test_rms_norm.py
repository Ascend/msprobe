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

import torch
import unittest

from msprobe.pytorch.bench_functions.rms_norm import npu_rms_norm_backward, npu_rms_norm


class TestRMSNorm(unittest.TestCase):
    def setUp(self):
        self.x = torch.randn(2, 3, 4, requires_grad=True)
        self.gamma = torch.ones(4, requires_grad=True)
        self.epsilon = 1e-5

    def test_basic_forward(self):
        # 基本前向传播测试
        out, _ = npu_rms_norm(self.x, self.gamma, self.epsilon)
        self.assertEqual(out.shape, self.x.shape)

    def test_backward(self):
        # 反向传播测试
        grad = torch.randn(2, 3, 4)
        out, rstd = npu_rms_norm(self.x, self.gamma, self.epsilon)
        out.backward(grad)
        grad_x_expected, grad_gamma_expected = npu_rms_norm_backward(grad, self.x, self.gamma, rstd)
        self.assertEqual(self.x.grad.shape, grad_x_expected.shape)
        self.assertEqual(self.x.grad.shape, grad_gamma_expected.shape)

    def test_zero_epsilon(self):
        # 测试 epsilon 为 0 的情况
        out, _ = npu_rms_norm(self.x, self.gamma, 0.0)
        self.assertIsNotNone(out)

    def test_large_input(self):
        # 测试大输入值
        large_x = 1000 * torch.randn(2, 3, 4)
        out, _ = npu_rms_norm(large_x, self.gamma, self.epsilon)
        self.assertIsNotNone(out)

    def test_small_input(self):
        # 测试小输入值
        small_x = 1e-10 * torch.randn(2, 3, 4)
        out, _ = npu_rms_norm(small_x, self.gamma, self.epsilon)
        self.assertIsNotNone(out)
