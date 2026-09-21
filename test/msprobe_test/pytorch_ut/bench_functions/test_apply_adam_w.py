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
from unittest import TestCase

from msprobe.pytorch.bench_functions.apply_adam_w import npu_apply_adam_w


class TestNpuApplyAdamW(TestCase):
    def test_npu_apply_adam_w(self):
        beta1_power = torch.tensor(0.9)
        beta2_power = torch.tensor(0.999)
        lr = 0.001
        weight_decay = 0.01
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        grad = torch.tensor([0.1, 0.2, 0.3])
        max_grad_norm = torch.tensor([1.0])
        amsgrad = False
        maximize = False
        out = [torch.tensor([0.1, 0.2, 0.3]), torch.tensor([0.1, 0.2, 0.3]), torch.tensor([0.1, 0.2, 0.3])]

        var_out, m_out, v_out = npu_apply_adam_w(beta1_power, beta2_power, lr, weight_decay, beta1, beta2, eps, grad,
                                                 max_grad_norm, amsgrad, maximize, out)
        self.assertTrue(torch.allclose(var_out, torch.tensor([0.09992456, 0.19989273, 0.29986808]), atol=1e-6))
        self.assertTrue(torch.allclose(m_out, torch.tensor([0.09999999, 0.19999999, 0.30000001]), atol=1e-6))
        self.assertTrue(torch.allclose(v_out, torch.tensor([0.09991000, 0.19983999, 0.29979002]), atol=1e-6))

        amsgrad = True
        var_out, m_out, v_out = npu_apply_adam_w(beta1_power, beta2_power, lr, weight_decay, beta1, beta2, eps, grad,
                                                 max_grad_norm, amsgrad, maximize, out)
        self.assertTrue(torch.allclose(m_out, torch.tensor([0.09999999, 0.19999999, 0.30000001]), atol=1e-6))
        self.assertTrue(torch.allclose(v_out, torch.tensor([0.09991000, 0.19983999, 0.29979002]), atol=1e-6))
