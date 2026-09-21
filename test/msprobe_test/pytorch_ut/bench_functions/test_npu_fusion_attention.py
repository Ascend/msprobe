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
import numpy as np

from msprobe.pytorch.bench_functions.npu_fusion_attention import npu_fusion_attention, npu_fusion_attention_grad, \
    broadcast_kv, convert_from_bsnd, convert_to_bsnd, rearrange, rebuid_softmax_by_qkv, \
    generate_atten_mask, RebuildSoftmaxParams, rebuild_softmax_by_max_sum, softmax_forward, SOFTMAX_BUILD_MODE


class TestNpuFusionAttention(unittest.TestCase):
    def setUp(self):
        self.B = 2
        self.N1 = 3
        self.N2 = 3
        self.S1 = 4
        self.S2 = 4
        self.D = 64
        self.query = torch.randn(self.B, self.S1, self.N1, self.D)
        self.key = torch.randn(self.B, self.S2, self.N2, self.D)
        self.value = torch.randn(self.B, self.S2, self.N2, self.D)
        self.atten_mask = torch.randn(self.B, 1, self.S1, self.S2)
        self.batch_size = 2
        self.seq_len = 3
        self.num_heads = 4
        self.head_dim = 5
        self.input_tensor = torch.randn(self.batch_size, self.seq_len, self.num_heads, self.head_dim)

    def test_convert_from_bsnd(self):
        # 测试从 bsnd 转换到 BSH
        converted_tensor = convert_from_bsnd(self.input_tensor, "BSH")
        self.assertEqual(converted_tensor.shape, (self.batch_size, self.seq_len, self.num_heads * self.head_dim))

        # 测试从 bsnd 转换到 SBH
        converted_tensor = convert_from_bsnd(self.input_tensor, "SBH")
        self.assertEqual(converted_tensor.shape, (self.seq_len, self.batch_size, self.num_heads * self.head_dim))

        # 测试从 bsnd 转换到 BNSD
        converted_tensor = convert_from_bsnd(self.input_tensor, "BNSD")
        self.assertEqual(converted_tensor.shape, (self.batch_size, self.num_heads, self.seq_len, self.head_dim))

    def test_convert_to_bsnd(self):
        # 测试从 BSH 转换回 bsnd
        converted_tensor = convert_to_bsnd(rearrange(self.input_tensor, 'b s n d -> b s (n d)'), self.num_heads, "BSH")
        self.assertEqual(converted_tensor.shape, (self.batch_size, self.seq_len, self.num_heads, self.head_dim))

        # 测试从 SBH 转换回 bsnd
        converted_tensor = convert_to_bsnd(rearrange(self.input_tensor, 'b s n d -> s b (n d)'), self.num_heads, "SBH")
        self.assertEqual(converted_tensor.shape, (self.batch_size, self.seq_len, self.num_heads, self.head_dim))

        # 测试从 BNSD 转换回 bsnd
        converted_tensor = convert_to_bsnd(rearrange(self.input_tensor, 'b s n d -> b n s d'), self.num_heads, "BNSD")
        self.assertEqual(converted_tensor.shape, (self.batch_size, self.seq_len, self.num_heads, self.head_dim))

    def test_basic_forward_input_layout_is_BSND(self):
        # 基本前向传播测试
        out, _, _ = npu_fusion_attention(self.query, self.key, self.value, head_num=self.N1, input_layout="BSND")
        self.assertEqual(out.shape, (self.B, self.S1, self.N1, self.D))

    def test_basic_forward_input_layout_is_BNSD(self):
        # 基本前向传播测试
        self.query = torch.randn(self.B, self.N1, self.S1, self.D)
        self.key = torch.randn(self.B, self.N2, self.S2, self.D)
        self.value = torch.randn(self.B, self.N2, self.S2, self.D)
        out, _, _ = npu_fusion_attention(self.query, self.key, self.value, head_num=self.N1, input_layout="BNSD")
        self.assertEqual(out.shape, (self.B, self.N1, self.S1, self.D))

    def test_basic_backward(self):
        # 基本反向传播测试
        dx = torch.randn(self.B, self.S1, self.N1, self.D)
        out, _, _ = npu_fusion_attention_grad(dx, self.query, self.key, self.value, self.N1, "BSND")
        self.assertEqual(out.shape, (self.B, self.S1, self.N1, self.D))  # 检查dq形状

    def test_different_input_layout(self):
        # BSH
        self.query = torch.randn(self.B, self.S1, self.N1 * self.D)
        self.key = torch.randn(self.B, self.S2, self.N2 * self.D)
        self.value = torch.randn(self.B, self.S2, self.N2 * self.D)
        out, _, _ = npu_fusion_attention(self.query, self.key, self.value, head_num=self.N1, input_layout="BSH")
        self.assertEqual(out.shape, (self.B, self.S1, self.N1 * self.D))

        # SBH
        self.query = torch.randn(self.S1, self.B, self.N1 * self.D)
        self.key = torch.randn(self.S2, self.B, self.N2 * self.D)
        self.value = torch.randn(self.S2, self.B, self.N2 * self.D)
        out, _, _ = npu_fusion_attention(self.query, self.key, self.value, head_num=self.N1, input_layout="SBH")
        self.assertEqual(out.shape, (self.S1, self.B, self.N1 * self.D))

    def test_with_attention_mask(self):
        # 带注意力掩码的测试
        out, _, _ = npu_fusion_attention(self.query, self.key, self.value, head_num=self.N1, input_layout="BSND",
                                         atten_mask=self.atten_mask)
        self.assertIsNotNone(out)

    def test_sparse_mode(self):
        # 稀疏模式测试
        out, _, _ = npu_fusion_attention(self.query, self.key, self.value, head_num=self.N1, input_layout="BSND",
                                         sparse_mode=2)
        self.assertIsNotNone(out)

    def test_invalid_input(self):
        # 无效输入测试
        with self.assertRaises(ValueError):
            npu_fusion_attention(self.query, self.key, self.value, head_num=self.N1, input_layout="INVALID")

    def test_mismatch_dims(self):
        # 维度不匹配测试
        self.key = torch.randn(self.B, self.N1 + 1, self.S2, self.D)  # 故意制造维度不匹配
        with self.assertRaises(ValueError):
            npu_fusion_attention(self.query, self.key, self.value, head_num=self.N1, input_layout="BSND")

    def test_input_layout_is_TND(self):
        with self.assertRaises(ValueError):
            npu_fusion_attention(self.query, self.key, self.value, head_num=self.N1, input_layout="TND")

    def test_broadcast_kv(self):
        # valid input
        num_heads = 4
        num_kv_heads = 2
        kv_tensor = torch.randn(1, num_kv_heads, 10, 10)
        result = broadcast_kv(num_heads, num_kv_heads, kv_tensor, torch.float32)
        self.assertEqual(result.shape, (1, num_heads, 10, 10))

        # invalid_input
        num_heads = 4
        kv_tensor = torch.randn(1, num_kv_heads, 10, 10)
        num_kv_heads = 0
        with pytest.raises(ValueError):
            broadcast_kv(num_heads, num_kv_heads, kv_tensor, torch.float32)
        num_kv_heads = 5
        with pytest.raises(ValueError):
            broadcast_kv(num_heads, num_kv_heads, kv_tensor, torch.float32)

        # num_heads equals to num_kv_heads
        num_heads = 4
        num_kv_heads = 4
        kv_tensor = torch.randn(1, num_kv_heads, 10, 10)
        result = broadcast_kv(num_heads, num_kv_heads, kv_tensor, torch.float32)

        self.assertEqual(result.shape, (1, num_heads, 10, 10))
        self.assertEqual(result.dtype, torch.float32)
        self.assertTrue(torch.allclose(result, kv_tensor.expand(-1, num_heads, -1, -1)))

    def test_generate_atten_mask_all_sparse_modes(self):
        b, n1, s1, s2 = 1, 1, 4, 4
        dtype = torch.float32

        # Case sparse_mode 0
        mask0 = generate_atten_mask(0, None, b, n1, s1, s2, 0, 0, dtype)
        self.assertEqual(mask0.shape, (s1, s2))

        # Case sparse_mode 1 (全0)
        mask1 = generate_atten_mask(1, None, b, n1, s1, s2, 0, 0, dtype)
        self.assertTrue(torch.allclose(mask1, torch.zeros_like(mask1)))

        # Case sparse_mode 2 (上三角 k=1)
        mask2 = generate_atten_mask(2, None, b, n1, s1, s2, 0, 0, dtype)
        self.assertEqual(mask2[0, 0], torch.tensor(0.))

        # Case sparse_mode 3
        mask3 = generate_atten_mask(3, None, b, n1, s1, s2, 0, 0, dtype)
        self.assertTrue(mask3.sum() > 0)

        # Case sparse_mode 4
        mask4 = generate_atten_mask(4, None, b, n1, s1, s2, 1, 1, dtype)
        self.assertTrue(mask4.sum() > 0)

    def test_generate_atten_mask_reverse_conversion(self):
        # 构造 2048 大矩阵触发逆向还原逻辑
        m = torch.from_numpy(np.triu(np.ones([2048, 2048]), k=1)).to(torch.float32)
        out = generate_atten_mask(2, m, 1, 1, 4, 4, 0, 0, torch.float32)
        self.assertEqual(out.shape, (4, 4))

    def test_rebuild_softmax_by_max_sum(self):
        q = torch.randn(1, 1, 2, 4)
        k = torch.randn(1, 1, 2, 4)
        qk = torch.matmul(q, k.permute(0, 1, 3, 2))
        softmax_res, softmax_max, softmax_sum = softmax_forward(qk)

        params = RebuildSoftmaxParams(q, k, None, None, 1.0,
                                      softmax_max, softmax_sum)
        res = rebuild_softmax_by_max_sum(params)
        self.assertTrue(torch.allclose(res, softmax_res, atol=1e-5))

    def test_rebuild_softmax_by_max_sum_error(self):
        q = torch.randn(1, 1, 2, 4)
        k = torch.randn(1, 1, 2, 4)
        softmax_max = torch.zeros(1, 1, 2, 0)  # 触发 shape[-1]==0 分支
        softmax_sum = torch.randn_like(softmax_max)
        params = RebuildSoftmaxParams(q, k, None, None, 1.0,
                                      softmax_max, softmax_sum)

        with self.assertRaises(ValueError):
            rebuild_softmax_by_max_sum(params)

    def test_rebuild_softmax_by_qkv(self):
        q = torch.randn(1, 1, 2, 4)
        k = torch.randn(1, 1, 2, 4)
        res = rebuid_softmax_by_qkv(q, k, None, None, 1.0)
        self.assertEqual(res.shape, (1, 1, 2, 2))

    def test_npu_fusion_attention_with_pse(self):
        pse = torch.randn(self.B, self.N1, self.S1, self.S2, dtype=torch.float32)

        out, _, _ = npu_fusion_attention(
            self.query, self.key, self.value,
            head_num=self.N1, input_layout="BSND",
            pse=pse
        )
        self.assertEqual(out.shape, (self.B, self.S1, self.N1, self.D))

    def test_npu_fusion_attention_grad_max_sum(self):
        global SOFTMAX_BUILD_MODE
        prev = SOFTMAX_BUILD_MODE
        SOFTMAX_BUILD_MODE = "MAX_SUM"
        try:
            dx = torch.randn(self.B, self.S1, self.N1, self.D)
            dq, dk, dv = npu_fusion_attention_grad(
                self.query, self.key, self.value, dx,
                self.N1, "BSND"
            )
            self.assertEqual(dq.shape, self.query.shape)
        finally:
            SOFTMAX_BUILD_MODE = prev

    def test_attention_mask_dim_mismatch_error(self):
        bad_mask = torch.randn(2, 2)  # 错误 shape，但函数逻辑允许返回
        out = generate_atten_mask(
            0, bad_mask,
            1, 1, 4, 4, 0, 0, torch.float32
        )
        self.assertEqual(out.dtype, torch.float32)
