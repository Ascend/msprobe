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

from msprobe.pytorch.bench_functions.swiglu import npu_swiglu, npu_swiglu_backward, swish_grad, swish


class TestNpuSwiglu(unittest.TestCase):
    def setUp(self):
        self.x_float32 = torch.randn(2, 6, requires_grad=True)
        self.x_float16 = torch.randn(2, 6, dtype=torch.float16, requires_grad=True)
        self.x_bfloat16 = torch.randn(2, 6, dtype=torch.bfloat16, requires_grad=True)

    def test_swiglu_float32(self):
        result = npu_swiglu(self.x_float32)
        in_tensors = torch.chunk(self.x_float32, 2, dim=-1)
        tensor_scalar = torch.sigmoid(in_tensors[0])
        expected_result = torch.mul(torch.mul(tensor_scalar, in_tensors[0]), in_tensors[1])
        self.assertTrue(torch.allclose(result, expected_result, atol=1e-6))

    def test_swiglu_float16(self):
        result = npu_swiglu(self.x_float16)
        in_tensors = torch.chunk(self.x_float16, 2, dim=-1)
        tensor_self_float = in_tensors[0].type(torch.float)
        tensor_other_float = in_tensors[1].type(torch.float)
        expected_result = (torch.nn.functional.silu(tensor_self_float) * tensor_other_float).type(torch.float16)
        self.assertTrue(torch.allclose(result, expected_result, atol=1e-3))

    def test_swiglu_backward_float32(self):
        y = npu_swiglu(self.x_float32)
        grad = torch.randn_like(y)
        x_grad = npu_swiglu_backward(grad, self.x_float32)

        in_tensors = torch.chunk(self.x_float32, 2, dim=-1)
        expected_out1 = torch.mul(torch.mul(in_tensors[1], swish_grad(1.0, in_tensors[0])), grad)
        expected_out2 = torch.mul(grad, swish(1.0, in_tensors[0]))
        expected_x_grad = torch.cat((expected_out1, expected_out2), dim=-1)

        self.assertTrue(torch.allclose(x_grad, expected_x_grad, atol=1e-6))

    def test_swiglu_backward_float16(self):
        y = npu_swiglu(self.x_float16)
        grad = torch.randn_like(y)
        x_grad = npu_swiglu_backward(grad, self.x_float16)

        in_tensors = torch.chunk(self.x_float16, 2, dim=-1)
        tensor_self_float = in_tensors[0].type(torch.float)
        tensor_other_float = in_tensors[1].type(torch.float)
        tensor_gradout_float = grad.type(torch.float)

        expected_out1 = torch.mul(
            torch.mul(tensor_other_float, swish_grad(1.0, tensor_self_float)),
            tensor_gradout_float).type(torch.float16)
        expected_out2 = torch.mul(
            tensor_gradout_float,
            swish(1.0, tensor_self_float)).type(torch.float16)
        expected_x_grad = torch.cat((expected_out1, expected_out2), dim=-1)

        self.assertTrue(torch.allclose(x_grad, expected_x_grad, atol=1e-3))

    def test_swiglu_backward_bfloat16(self):
        y = npu_swiglu(self.x_bfloat16)
        grad = torch.randn_like(y)
        x_grad = npu_swiglu_backward(grad, self.x_bfloat16)

        in_tensors = torch.chunk(self.x_bfloat16, 2, dim=-1)
        tensor_self_float = in_tensors[0].type(torch.float)
        tensor_other_float = in_tensors[1].type(torch.float)
        tensor_gradout_float = grad.type(torch.float)

        expected_out1 = torch.mul(
            tensor_gradout_float,
            swish_grad(1.0, tensor_self_float)).type(torch.bfloat16).type(torch.float32) * tensor_other_float
        expected_out2 = swish(1.0, tensor_self_float).type(torch.bfloat16).type(torch.float32) * tensor_gradout_float
        expected_x_grad = torch.cat((expected_out1, expected_out2), dim=-1).type(torch.bfloat16)

        self.assertTrue(torch.allclose(x_grad, expected_x_grad, atol=1e-3))
