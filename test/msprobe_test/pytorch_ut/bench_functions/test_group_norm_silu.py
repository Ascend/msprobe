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

from msprobe.pytorch.bench_functions.group_norm_silu import npu_group_norm_silu


class TestNPUGroupNormSILU(unittest.TestCase):
    def setUp(self):
        self.input0 = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        self.gama = torch.tensor([1.0])
        self.beta = torch.tensor([0.0])
        self.group = 1
        self.eps = 1e-5

    def test_npu_group_norm_silu_positive(self):
        # 调用 npu_group_norm_silu 函数
        result = npu_group_norm_silu(self.input0, self.gama, self.beta, self.group, self.eps)
        
        # 预期的结果
        expected_result = torch.tensor([[[[-0.2780, -0.1744], [0.2728, 1.0636]]]])
        
        # 使用 torch.allclose 进行近似比较
        self.assertTrue(torch.allclose(result[0], expected_result, atol=1e-4))
