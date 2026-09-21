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
import numpy as np

from msprobe.mindspore.api_accuracy_checker.bench_functions.flash_attention_score import (
    softmax_forward, softmax_grad, broadcast_kv, calculate_qk,
    fusion_attention_forward, fusion_attention_backward,
    parse_bsnd_args, convert_from_bnsd, convert_to_bnsd,
    generate_attn_mask, generate_kv, rebuid_softmax_by_qkv,
    rebuild_softmax_by_max_sum, FlashAttentionScore,
    npu_fusion_attention_forward_patch, npu_fusion_attention_backward_patch,
    FaForwardParams, FaBackwardParams, RebuildSoftmaxParams, GTYPE,
    get_head_num, get_input_layout, SOFTMAX_BUILD_MODE
)


class TestBenchFunctions(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        np.random.seed(0)

    def test_softmax_forward(self):
        x = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 4.0]], dtype=torch.float64)
        res, x_max, x_sum = softmax_forward(x)
        expected = torch.softmax(x, dim=-1)
        self.assertTrue(torch.allclose(res, expected, atol=1e-6))
        self.assertTrue(torch.allclose(x_max, x.max(dim=-1, keepdim=True)[0], atol=1e-6))
        manual_sum = torch.exp(x - x_max).sum(dim=-1, keepdim=True)
        self.assertTrue(torch.allclose(x_sum, manual_sum, atol=1e-6))

    def test_softmax_grad(self):
        x = torch.randn(4, 5, dtype=torch.float64)
        y, _, _ = softmax_forward(x)
        dp = torch.randn_like(y)
        grad = softmax_grad(dp, y)
        sum_grad = grad.sum(dim=-1)
        self.assertTrue(torch.allclose(sum_grad, torch.zeros_like(sum_grad), atol=1e-6))

    def test_broadcast_kv(self):
        B, N_kv, S, D = 1, 2, 3, 4
        num_heads = 4
        kv = torch.arange(B*N_kv*S*D, dtype=torch.float32).reshape(B, N_kv, S, D)
        out = broadcast_kv(num_heads, N_kv, kv, kv.dtype)
        self.assertEqual(out.shape, (B, num_heads, S, D))
        self.assertTrue(torch.equal(out[:, :2, :, :], kv[:, 0:1, :, :].expand(B,2,S,D)))
        self.assertTrue(torch.equal(out[:, 2:, :, :], kv[:, 1:2, :, :].expand(B,2,S,D)))

    def test_broadcast_kv_invalid(self):
        kv = torch.randn(1,3,4)  # 3D tensor
        with self.assertRaises(ValueError):
            broadcast_kv(4, 2, kv, kv.dtype)
        kv4d = torch.randn(1,2,3,4)
        with self.assertRaises(ValueError):
            broadcast_kv(0, 2, kv4d, kv4d.dtype)
        with self.assertRaises(ValueError):
            broadcast_kv(4, 3, kv4d, kv4d.dtype)
        with self.assertRaises(ValueError):
            broadcast_kv(3, 2, kv4d, kv4d.dtype)

    def test_calculate_qk_basic(self):
        q = torch.randn(2,2,3,4)
        k = torch.randn(2,2,3,4)
        scalar = 2.0
        qk = calculate_qk(q, k, None, None, scalar)
        expected = torch.matmul(q, k.permute(0,1,3,2)) * scalar
        self.assertTrue(torch.allclose(qk, expected, atol=1e-6))

    def test_calculate_qk_errors(self):
        q = torch.randn(2,2,3,4)
        # head_dim mismatch
        k = torch.randn(2,2,3,5)
        with self.assertRaises(ValueError):
            calculate_qk(q, k, None, None, 1.0)
        # too few dims
        q3 = torch.randn(2,3,4)
        k3 = torch.randn(2,3,4)
        with self.assertRaises(ValueError):
            calculate_qk(q3, k3, None, None, 1.0)

    def test_calculate_qk_with_pse_and_mask(self):
        q = torch.ones(1,1,2,2)
        k = torch.ones(1,1,2,2)
        pse = torch.ones(1,1,2,2)
        mask = torch.zeros(1,1,2,2)
        scalar = 1.0
        qk = calculate_qk(q, k, mask, pse, scalar)
        expected = (torch.matmul(q, k.permute(0,1,3,2)) + pse) * scalar + mask.bool() * (-40000.0)
        self.assertTrue(torch.allclose(qk, expected, atol=1e-6))

    def test_parse_bsnd_args_bsh(self):
        q = torch.randn(2,3,4)
        k = torch.randn(2,5,4)
        head_num = 2
        args = parse_bsnd_args(q, k, head_num, "BSH")
        b, s1, s2, n1, n2, d, h1, h2, dtype = args
        self.assertEqual((b, s1, s2, n1, n2, d, h1, h2), (2,3,5,2,2,2,4,4))

    def test_parse_bsnd_args_errors(self):
        q = torch.randn(2,3,0)  # leads to d=0
        k = torch.randn(2,5,0)
        with self.assertRaises(ValueError):
            parse_bsnd_args(q, k, 2, "BSH")
        with self.assertRaises(ValueError):
            parse_bsnd_args(q, k, 2, "TND")
        with self.assertRaises(ValueError):
            parse_bsnd_args(q, k, 0, "BSH")

    def test_convert_from_and_to_bnsd(self):
        B, N, S, D = 1, 2, 3, 4
        x = torch.arange(B*N*S*D).reshape(B, N, S, D)
        for layout in ["BSH","SBH","BSND","BNSD"]:
            out = convert_from_bnsd(x, layout)
            back = convert_to_bnsd(out, N, layout)
            self.assertTrue(torch.equal(back, x.to(GTYPE)))
        with self.assertRaises(ValueError):
            convert_to_bnsd(torch.randn(2,2), N, "TND")

    def test_generate_attn_mask_shapes_and_reverse(self):
        b,n1,s1,s2 = 1,1,3,3
        dtype = torch.float32
        for mode in range(5):
            mask = generate_attn_mask(mode, None, b, n1, s1, s2, 0, 0, dtype)
            self.assertEqual(mask.shape, (s1, s2))
        # reverse from full 2048 mask
        orig = torch.from_numpy(np.triu(np.ones([2048,2048]), k=1)).to(dtype)
        for mode in [2,3,4]:
            rev = generate_attn_mask(mode, orig, b, n1, s1, s2, 0, 0, dtype)
            self.assertEqual(rev.shape, (s1, s2))

    def test_generate_kv(self):
        key = torch.randn(1,2,3,4)
        value = torch.randn(1,2,3,4)
        k_new, v_new = generate_kv(key, value, 2, 2)
        self.assertTrue(torch.equal(k_new, key))
        k2, _ = generate_kv(key, value, 4, 2)
        self.assertEqual(k2.shape[1], 4)


    def test_rebuild_softmax_by_max_sum_and_errors(self):
        # 正常路径
        B, N, S, D = 1, 1, 3, 4
        q = torch.randn(B, N, S, D)
        k = torch.randn(B, N, S, D)
        attn_mask = None
        pse = None
        scalar = 1.0
        # 手动构造 softmax_max、softmax_sum
        qk, softmax_max, softmax_sum = softmax_forward(torch.matmul(q, k.permute(0,1,3,2)))
        params = RebuildSoftmaxParams(q=q, k=k, attn_mask=attn_mask, pse=pse,
                                     scalar_value=scalar,
                                     softmax_max=softmax_max, softmax_sum=softmax_sum)
        res = rebuild_softmax_by_max_sum(params)
        self.assertTrue(torch.allclose(res, torch.softmax(torch.matmul(q,k.permute(0,1,3,2)), dim=-1)))

        # softmax_max 最后一维为 0 时抛错
        bad_max = torch.empty(B, N, S, 0)
        bad_params = params._replace(softmax_max=bad_max)
        with self.assertRaises(ValueError):
            rebuild_softmax_by_max_sum(bad_params)

    def test_npu_patch_forward_and_backward_patch(self):
        # forward_patch 长度不足报错
        with self.assertRaises(RuntimeError):
            npu_fusion_attention_forward_patch(1)
        # backward_patch 长度不等于6 报错
        with self.assertRaises(ValueError):
            npu_fusion_attention_backward_patch(1,2,3)

        # 正常调用，检查返回结构
        B, S1, S2, N1, D = 1, 2, 2, 2, 4
        head_num = 1
        layout = "BSH"
        q = torch.randn(B, S1, N1 * D)
        k = torch.randn(B, S2, N1 * D)
        # forward_patch 返回 args, dims_kwargs, new_kwargs
        args, dims, new_kwargs = npu_fusion_attention_forward_patch(q, k, None, head_num, layout)
        self.assertIn("b", dims)
        # backward_patch
        dx = torch.randn_like(q)
        args2, dims2, new_kwargs2 = npu_fusion_attention_backward_patch(q, k, None, dx, head_num, layout)
        self.assertIn("s1", dims2)


    def test_fusion_attention_forward_with_drop_mask(self):
        B, N, S, D = 1, 2, 3, 4
        q = torch.randn(B, N, S, D, dtype=torch.float64)
        k = torch.randn(B, N, S, D, dtype=torch.float64)
        v = torch.randn(B, N, S, D, dtype=torch.float64)
        # 制造一个 drop_mask
        drop_mask = torch.randint(0, 2, (B, N, S, S), dtype=torch.float64)
        # 注意 drop_mask 需要能广播到 softmax_res 形状 (B,N,S,S)
        params = FaForwardParams(
            q=q, k=k, v=v,
            drop_mask=drop_mask,
            attn_mask=None,
            pse=None,
            scalar_value=1.0,
            keep_prob=0.5
        )
        y1, _, _ = fusion_attention_forward(params)
        # 手动计算：先 softmax，再 mask，再 matmul
        qk = calculate_qk(q, k, None, None, 1.0)
        sm, _, _ = softmax_forward(qk)
        masked = sm * drop_mask * (1.0 / 0.5)
        y2 = torch.matmul(masked, v)
        self.assertTrue(torch.allclose(y1, y2))

    def test_fusion_attention_backward_with_drop_mask(self):
        B, N, S, D = 1, 2, 3, 4
        # 构造前向结果
        dx = torch.randn(B, N, S, D, dtype=torch.float64)
        q = torch.randn(B, N, S, D, dtype=torch.float64)
        k = torch.randn(B, N, S, D, dtype=torch.float64)
        v = torch.randn(B, N, S, D, dtype=torch.float64)
        # 构造 softmax_res (B,N,S,S) 和 drop_mask
        sm = torch.softmax(torch.randn(B, N, S, S, dtype=torch.float64), dim=-1)
        drop_mask = torch.randint(0, 2, (B, N, S, S), dtype=torch.float64)
        params = FaBackwardParams(
            dx=dx, q=q, k=k, v=v,
            softmax_res=sm,
            drop_mask=drop_mask,
            pse=None,
            scalar_value=1.0,
            keep_prob=0.8
        )
        dq1, dk1, dv1 = fusion_attention_backward(params)
        # 直接对比形状和 dtype
        self.assertEqual(dq1.shape, q.shape)
        self.assertEqual(dk1.shape, k.shape)
        self.assertEqual(dv1.shape, v.shape)
        self.assertEqual(dq1.dtype, torch.float64)

    def test_get_head_num_and_input_layout_errors(self):
        # 既无 kwargs, 也无足够 args
        with self.assertRaises(ValueError):
            get_head_num(1, 2, 3)
        with self.assertRaises(ValueError):
            get_input_layout(1, 2, 3, 4)

    def test_npu_forward_patch_sanity(self):
        # 测试 sparse_mode 非零路径
        B, S1, S2, N1, D = 1, 3, 5, 2, 4
        head_num = 2
        layout = "BSH"
        q = torch.randn(B, S1, N1 * D)
        k = torch.randn(B, S2, N1 * D)
        # 传入 sparse_mode, pre/next token，pse
        args, dims, new_kwargs = npu_fusion_attention_forward_patch(
            q, k, None,
            head_num, layout,
            sparse_mode=3,
            pre_tockens=1,
            next_tockens=2,
            pse=torch.ones(1),
        )
        # dims 检查
        self.assertEqual(dims["b"], B)
        self.assertEqual(dims["s1"], S1)
        self.assertEqual(dims["s2"], S2)
        self.assertEqual(new_kwargs["sparse_mode"], 3)
        self.assertTrue("pse" in new_kwargs)

    def test_npu_backward_patch_sanity(self):
        B, S1, S2, N1, D = 1, 4, 4, 2, 4
        head_num = 2
        layout = "BSH"
        q = torch.randn(B, S1, N1 * D)
        k = torch.randn(B, S2, N1 * D)
        dx = torch.randn(B, S1, N1 * D)
        # 正确长度
        args, dims, new_kwargs = npu_fusion_attention_backward_patch(
            q, k, None, dx, head_num, layout
        )
        self.assertEqual(dims["n1"], N1)
        self.assertEqual(dims["n2"], N1)
        # 传入 n2 不整除 n1 抛错
        with self.assertRaises(ValueError):
            npu_fusion_attention_backward_patch(
                q, k, None, dx, 3, layout
            )

    def test_gtype_constant(self):
        # Ensure GTYPE matches expected torch dtype
        self.assertEqual(GTYPE, torch.float64)

    def test_softmax_forward_and_sum(self):
        x = torch.tensor([[0.5, -0.5], [2.0, 3.0]], dtype=GTYPE)
        res, x_max, x_sum = softmax_forward(x)
        expected = torch.softmax(x, dim=-1)
        self.assertTrue(torch.allclose(res, expected, atol=1e-6))
        self.assertTrue(torch.allclose(x_sum, torch.exp(x - x_max).sum(dim=-1, keepdim=True), atol=1e-6))

    def test_softmax_grad_zero_sum(self):
        x = torch.randn(3, 4, dtype=GTYPE)
        y, _, _ = softmax_forward(x)
        dp = torch.randn_like(y)
        grad = softmax_grad(dp, y)
        self.assertTrue(torch.allclose(grad.sum(dim=-1), torch.zeros_like(grad.sum(dim=-1)), atol=1e-6))

    def test_broadcast_kv_and_errors(self):
        B, N_kv, S, D = 2, 1, 3, 4
        num_heads = 2
        kv = torch.arange(B*N_kv*S*D, dtype=torch.float32).reshape(B, N_kv, S, D)
        out = broadcast_kv(num_heads, N_kv, kv, kv.dtype)
        self.assertEqual(out.shape, (B, num_heads, S, D))
        # invalid dims
        with self.assertRaises(ValueError):
            broadcast_kv(2, 0, kv, kv.dtype)
        with self.assertRaises(ValueError):
            broadcast_kv(3, 2, kv, kv.dtype)

    def test_calculate_qk_basic_and_errors(self):
        q = torch.randn(1,1,2,3)
        k = torch.randn(1,1,2,3)
        scalar = 0.5
        out = calculate_qk(q, k, None, None, scalar)
        expected = torch.matmul(q, k.permute(0,1,3,2)) * scalar
        self.assertTrue(torch.allclose(out, expected))
        # shape mismatch
        k_bad = torch.randn(1,1,2,4)
        with self.assertRaises(ValueError):
            calculate_qk(q, k_bad, None, None, scalar)
        # low dims
        q3 = torch.randn(2,3,4)
        with self.assertRaises(ValueError):
            calculate_qk(q3, q3, None, None, scalar)

    def test_fusion_attention_forward_backward_no_mask(self):
        B, N, S, D = 1, 2, 3, 4
        q = torch.randn(B,N,S,D,dtype=GTYPE)
        k = torch.randn(B,N,S,D,dtype=GTYPE)
        v = torch.randn(B,N,S,D,dtype=GTYPE)
        params = FaForwardParams(q=q, k=k, v=v, drop_mask=None, attn_mask=None,
                                  pse=None, scalar_value=1.0, keep_prob=1.0)
        y, m, s = fusion_attention_forward(params)
        # gradient
        dx = torch.randn_like(y)
        backward = FaBackwardParams(dx=dx, q=q, k=k, v=v,
                                     softmax_res=torch.softmax(calculate_qk(q, k, None, None, 1.0), dim=-1),
                                     drop_mask=None, pse=None, scalar_value=1.0, keep_prob=1.0)
        dq, dk, dv = fusion_attention_backward(backward)
        self.assertEqual(dq.shape, q.shape)
        self.assertEqual(dk.shape, k.shape)
        self.assertEqual(dv.shape, v.shape)

    def test_parse_and_convert_layouts(self):
        q = torch.randn(2,3,4)
        k = torch.randn(2,5,4)
        head = 2
        args = parse_bsnd_args(q, k, head, "BSH")
        self.assertEqual(args[0], 2)
        B,N,S,D = 1,2,3,4
        x = torch.arange(B*N*S*D).reshape(B,N,S,D)
        for layout in ["BSH","SBH","BSND","BNSD"]:
            out = convert_from_bnsd(x, layout)
            back = convert_to_bnsd(out, N, layout)
            self.assertTrue(torch.equal(back, x.to(GTYPE)))

    def test_generate_attn_mask(self):
        for mode in range(5):
            mask = generate_attn_mask(mode, None, 1,1,3,3,0,0,torch.float32)
            self.assertEqual(mask.shape, (3,3))
        # reverse large mask
        orig = torch.from_numpy(np.triu(np.ones([2048,2048]),1)).to(torch.float32)
        rev = generate_attn_mask(2, orig, 1,1,3,3,0,0,torch.float32)
        self.assertEqual(rev.shape, (3,3))

    def test_generate_kv(self):
        k = torch.randn(1,2,3,4)
        v = torch.randn(1,2,3,4)
        k2, v2 = generate_kv(k, v, 4, 2)
        self.assertEqual(k2.shape[1], 4)

    def test_get_head_and_layout(self):
        with self.assertRaises(ValueError):
            get_head_num(1)
        with self.assertRaises(ValueError):
            get_input_layout(1,2,3)

    def test_npu_patches(self):
        # forward patch
        q = torch.randn(1,2,2*4)
        k = torch.randn(1,3,2*4)
        with self.assertRaises(RuntimeError):
            npu_fusion_attention_forward_patch(1)
        args, dims, new_kwargs = npu_fusion_attention_forward_patch(q, k, None, 2, "BSH")
        self.assertIn("b", dims)
        # backward patch
        dx = torch.randn_like(q)
        with self.assertRaises(ValueError):
            npu_fusion_attention_backward_patch(q,k,None,dx,3)
        args2, dims2, new_kwargs2 = npu_fusion_attention_backward_patch(q,k,None,dx,2, "BSH")
        self.assertIn("s1", dims2)


class TestBenchFunctionsExtra(unittest.TestCase):

    def test_calculate_qk_fallback(self):
        q = torch.randn(1, 1, 2, 4)
        k = torch.randn(1, 2, 4)  # dim=3
        with self.assertRaises(ValueError):
            calculate_qk(q, k, None, None, 1.0)

    def test_fusion_forward_keep_prob_zero(self):
        B,N,S,D = 1,1,2,4
        fwd = FaForwardParams(
            q=torch.randn(B,N,S,D),
            k=torch.randn(B,N,S,D),
            v=torch.randn(B,N,S,D),
            drop_mask=None,
            attn_mask=None,
            pse=None,
            scalar_value=1.0,
            keep_prob=0.0)
        with self.assertRaises(ValueError):
            fusion_attention_forward(fwd)

    def test_fusion_backward_keep_prob_zero(self):
        B,N,S,D = 1,1,2,4
        bwd = FaBackwardParams(
            dx=torch.randn(B,N,S,D),
            q=torch.randn(B,N,S,D),
            k=torch.randn(B,N,S,D),
            v=torch.randn(B,N,S,D),
            softmax_res=torch.softmax(torch.randn(B,N,S,S),dim=-1),
            drop_mask=None,
            pse=None,
            scalar_value=1.0,
            keep_prob=0.0)
        with self.assertRaises(ValueError):
            fusion_attention_backward(bwd)

    def test_convert_from_bnsd_tnd_error(self):
        x = torch.randn(1,2,3,4)
        with self.assertRaises(ValueError):
            convert_from_bnsd(x, "TND")

    def test_npu_patch_error_head_mismatch(self):
        q = torch.randn(1,2,8)
        k = torch.randn(1,2,8)
        with self.assertRaises(ValueError):
            npu_fusion_attention_forward_patch(q,k,None,3,"BSH")  # 3不可整除

    def test_attn_mask_dim4(self):
        mask = torch.ones(1,1,3,3)
        out = generate_attn_mask(1, mask, 1,1,3,3,0,0,torch.float32)
        self.assertEqual(out.shape, (1,1,3,3))


if __name__ == '__main__':
    unittest.main()
