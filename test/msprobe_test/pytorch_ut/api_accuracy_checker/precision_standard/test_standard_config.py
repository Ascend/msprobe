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
from msprobe.pytorch.api_accuracy_checker.precision_standard.standard_config import StandardConfig
from msprobe.core.common.const import CompareConst

class TestStandardConfig(unittest.TestCase):
    def test_get_small_value(self):
        # 测试已定义的数据类型
        self.assertEqual(StandardConfig.get_small_value(torch.float16, CompareConst.BENCHMARK), 2**-10)
        self.assertEqual(StandardConfig.get_small_value(torch.bfloat16, CompareConst.BENCHMARK), 2**-10)
        self.assertEqual(StandardConfig.get_small_value(torch.float32, CompareConst.BENCHMARK), 2**-20)
        
        # 测试未定义的数据类型（应返回默认值）
        self.assertEqual(StandardConfig.get_small_value(torch.int32, CompareConst.BENCHMARK), 2**-20)
        
        self.assertEqual(StandardConfig.get_small_value(torch.float16, CompareConst.ACCUMULATIVE_ERROR_COMPARE), 1)

    def test_get_small_value_atol(self):
        standard = 'absolute_threshold'
        # 测试已定义的数据类型
        self.assertEqual(StandardConfig.get_small_value_atol(torch.float16, standard), 2**-16)
        self.assertEqual(StandardConfig.get_small_value_atol(torch.bfloat16, standard), 1e-16)
        self.assertEqual(StandardConfig.get_small_value_atol(torch.float32, standard), 2**-30)
        
        # 测试未定义的数据类型（应返回默认值）
        self.assertEqual(StandardConfig.get_small_value_atol(torch.int32, standard), 2**-30)
        
        standard = 'benchmark'
        # 测试已定义的数据类型
        self.assertEqual(StandardConfig.get_small_value_atol(torch.float16, standard), 1e-16)
        self.assertEqual(StandardConfig.get_small_value_atol(torch.bfloat16, standard), 1e-16)
        self.assertEqual(StandardConfig.get_small_value_atol(torch.float32, standard), 2**-30)
        
        # 测试未定义的数据类型（应返回默认值）
        self.assertEqual(StandardConfig.get_small_value_atol(torch.int32, standard), 2**-30)

    def test_get_rtol(self):
        # 测试已定义的数据类型
        self.assertEqual(StandardConfig.get_rtol(torch.float16), 2**-10)
        self.assertEqual(StandardConfig.get_rtol(torch.bfloat16), 2**-8)
        self.assertEqual(StandardConfig.get_rtol(torch.float32), 2**-20)
        
        # 测试未定义的数据类型（应返回默认值）
        self.assertEqual(StandardConfig.get_rtol(torch.int32), 2**-20)

if __name__ == '__main__':
    unittest.main()