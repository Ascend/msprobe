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

import pytest
import torch
import unittest

from msprobe.pytorch.bench_functions.rotary_mul import npu_rotary_mul, npu_rotary_mul_backward


class TestRotaryMul(unittest.TestCase):

    def test_npu_rotary_mul(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        r1 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        r2 = torch.tensor([[9.0, 10.0], [11.0, 12.0]])
        out = npu_rotary_mul(x, r1, r2)
        expected_out = torch.tensor([[5.0 * 1.0 + 9.0 * (-2.0), 6.0 * 2.0 + 10.0 * 1.0],
                                     [7.0 * 3.0 + 11.0 * (-4.0), 8.0 * 4.0 + 12.0 * 3.0]])
        self.assertTrue(torch.allclose(out, expected_out, atol=1e-6))

    def test_npu_rotary_mul_backward(self):
        dy_tensor = torch.randn(2, 2, 2, 2)
        x = torch.randn(2, 2, 2, 2)

        # test condition_1
        r1 = torch.randn(1, 2, 1, 2)
        r2 = torch.randn(1, 2, 1, 2)
        x_grad, r1_grad, r2_grad = npu_rotary_mul_backward(dy_tensor, x, r1, r2)
        self.assertTrue(x_grad.shape, x.shape)
        self.assertTrue(r1_grad.shape, r1.shape)
        self.assertTrue(r2_grad.shape, r2.shape)

        # test condition_2
        r1 = torch.randn(1, 1, 2, 2)
        r2 = torch.randn(1, 1, 2, 2)
        x_grad, r1_grad, r2_grad = npu_rotary_mul_backward(dy_tensor, x, r1, r2)
        self.assertTrue(x_grad.shape, x.shape)
        self.assertTrue(r1_grad.shape, r1.shape)
        self.assertTrue(r2_grad.shape, r2.shape)

        # test condition_3
        r1 = torch.randn(2, 1, 1, 2)
        r2 = torch.randn(2, 1, 1, 2)
        x_grad, r1_grad, r2_grad = npu_rotary_mul_backward(dy_tensor, x, r1, r2)
        self.assertTrue(x_grad.shape, x.shape)
        self.assertTrue(r1_grad.shape, r1.shape)
        self.assertTrue(r2_grad.shape, r2.shape)

        # test error condition
        with pytest.raises(RuntimeError):
            npu_rotary_mul_backward(dy_tensor, x, r1, torch.randn(3, 3, 3, 3))
