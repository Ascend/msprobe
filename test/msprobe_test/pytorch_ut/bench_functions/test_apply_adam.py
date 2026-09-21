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

from msprobe.pytorch.bench_functions.apply_adam import npu_apply_adam


class TestNPUApplyAdam(unittest.TestCase):
    def setUp(self):
        # 初始化测试数据
        self.beta1_power = 0.9
        self.beta2_power = 0.999
        self.lr = 0.001
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.grad = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        self.use_locking = False
        self.var = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        self.m = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        self.v = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        self.out = (self.var, self.m, self.v)

    def test_npu_apply_adam_without_nesterov(self):
        # 测试不使用 Nesterov 动量的情况
        use_nesterov = False
        var_t, m_t, v_t = npu_apply_adam(
            self.beta1_power, self.beta2_power, self.lr, self.beta1, self.beta2,
            self.epsilon, self.grad, self.use_locking, use_nesterov, self.out
        )

        # 验证 var_t 的结果
        expected_var_t = torch.tensor([-0.0010, -0.0010, -0.0010], dtype=torch.float32)
        self.assertTrue(torch.allclose(var_t, expected_var_t, atol=1e-4))

        # 验证 m_t 的结果
        expected_m_t = torch.tensor([0.1000, 0.2000, 0.3000], dtype=torch.float32)
        self.assertTrue(torch.allclose(m_t, expected_m_t, atol=1e-4))

        # 验证 v_t 的结果
        expected_v_t = torch.tensor([0.0010, 0.0040, 0.0090], dtype=torch.float32)
        self.assertTrue(torch.allclose(v_t, expected_v_t, atol=1e-4))

    def test_npu_apply_adam_with_nesterov(self):
        # 测试使用 Nesterov 动量的情况
        use_nesterov = True
        var_t, m_t, v_t = npu_apply_adam(
            self.beta1_power, self.beta2_power, self.lr, self.beta1, self.beta2,
            self.epsilon, self.grad, self.use_locking, use_nesterov, self.out
        )

        # 验证 var_t 的结果
        expected_var_t = torch.tensor([-0.0019, -0.0019, -0.0019], dtype=torch.float32)
        self.assertTrue(torch.allclose(var_t, expected_var_t, atol=1e-4))

        # 验证 m_t 的结果
        expected_m_t = torch.tensor([0.1000, 0.2000, 0.3000], dtype=torch.float32)
        self.assertTrue(torch.allclose(m_t, expected_m_t, atol=1e-4))

        # 验证 v_t 的结果
        expected_v_t = torch.tensor([0.0010, 0.0040, 0.0090], dtype=torch.float32)
        self.assertTrue(torch.allclose(v_t, expected_v_t, atol=1e-4))

    def test_npu_apply_adam_with_non_zero_initial_values(self):
        # 测试非零初始值的情况
        self.m = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        self.v = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        self.out = (self.var, self.m, self.v)

        use_nesterov = False
        var_t, m_t, v_t = npu_apply_adam(
            self.beta1_power, self.beta2_power, self.lr, self.beta1, self.beta2,
            self.epsilon, self.grad, self.use_locking, use_nesterov, self.out
        )

        # 验证 var_t 的结果
        expected_var_t = torch.tensor([-0.0003, -0.0004, -0.0005], dtype=torch.float32)
        self.assertTrue(torch.allclose(var_t, expected_var_t, atol=1e-4))

        # 验证 m_t 的结果
        expected_m_t = torch.tensor([1., 2., 3.], dtype=torch.float32)
        self.assertTrue(torch.allclose(m_t, expected_m_t, atol=1e-4))

        # 验证 v_t 的结果
        expected_v_t = torch.tensor([1.0000, 2.0020, 3.0060], dtype=torch.float32)
        self.assertTrue(torch.allclose(v_t, expected_v_t, atol=1e-4))
