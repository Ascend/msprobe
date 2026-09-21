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

from msprobe.pytorch.bench_functions.mish import npu_mish


class TestNPUMish(unittest.TestCase):
    def setUp(self):
        self.input0 = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        self.eps = 1e-5

    def test_npu_mish_positive(self):
        # 调用 npu_mish 函数
        result = npu_mish(self.input0)
        
        # 预期的结果
        expected_result = torch.tensor([[[[0.8651, 1.9440], [2.9865, 3.9974]]]])
        
        # 使用 torch.allclose 进行近似比较
        self.assertTrue(torch.allclose(result[0], expected_result, atol=1e-4))
