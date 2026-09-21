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
from unittest.mock import Mock
import numpy as np

from msprobe.pytorch.api_accuracy_checker.precision_standard.standard_register import StandardRegistry
from msprobe.pytorch.api_accuracy_checker.compare.compare_utils import (
    absolute_standard_api, binary_standard_api, BINARY_COMPARE_UNSUPPORT_LIST
)

class TestStandardRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = StandardRegistry()
        
    def test_register_valid_function(self):
        """测试正常注册比较函数"""
        mock_func = Mock()
        self.registry.register("test_standard", mock_func)
        self.assertEqual(self.registry.comparison_functions["test_standard"], mock_func)

    def test_register_invalid_function(self):
        """测试注册非callable对象时抛出异常"""
        with self.assertRaises(ValueError):
            self.registry.register("test_standard", "not_callable")

    def test_get_comparison_function_binary_consistency(self):
        """测试获取二进制一致性比较函数"""
        mock_func = Mock()
        self.registry.register("binary_consistency", mock_func)
        # 使用支持二进制比较的数据类型
        result = self.registry.get_comparison_function("abs", dtype='torch.int8', in_dtype='torch.float32')
        self.assertEqual(result, mock_func)

    def test_get_comparison_function_absolute_threshold(self):
        """测试获取绝对阈值比较函数"""
        mock_func = Mock()
        self.registry.register("benchmark", mock_func)
        # 假设'test_api'在absolute_standard_api列表中
        result = self.registry.get_comparison_function("mul", dtype='torch.float32', in_dtype='torch.float32')
        print(f"result:{result}")
        self.assertEqual(result, mock_func)

    def test_get_comparison_function_ulp(self):
        """测试获取ULP比较函数"""
        mock_func = Mock()
        self.registry.register("benchmark", mock_func)
        result = self.registry.get_comparison_function("matmul", dtype='torch.float32', in_dtype='torch.float32')
        self.assertEqual(result, mock_func)

    def test_get_comparison_function_thousandth(self):
        """测试获取双千比较函数"""
        mock_func = Mock()
        self.registry.register("benchmark", mock_func)
        result = self.registry.get_comparison_function("conv2d", dtype='torch.float32', in_dtype='torch.float32')
        self.assertEqual(result, mock_func)

    def test_get_comparison_function_benchmark(self):
        """测试获取默认benchmark比较函数"""
        mock_func = Mock()
        self.registry.register("benchmark", mock_func)
        result = self.registry.get_comparison_function("npu_fusion_attention", dtype='torch.float32', 
                                                       in_dtype='torch.float32')
        self.assertEqual(result, mock_func)

    def test_get_standard_category_binary(self):
        """测试获取二进制一致性标准类别"""
        dtype = 'torch.int8'
        self.assertNotIn(dtype, BINARY_COMPARE_UNSUPPORT_LIST)
        category = self.registry._get_standard_category("abs", out_dtype=dtype, in_dtype='torch.float32')
        self.assertEqual(category, "binary_consistency")

    def test_get_standard_category_absolute(self):
        """测试获取绝对阈值标准类别"""
        category = self.registry._get_standard_category("mul", out_dtype='torch.float32', in_dtype='torch.float32')
        self.assertEqual(category, "benchmark")

    def test_get_standard_category_default(self):
        """测试获取默认benchmark标准类别"""
        category = self.registry._get_standard_category("unknown_api", out_dtype='torch.float32', 
                                                        in_dtype='torch.float32')
        self.assertEqual(category, "benchmark")

    def test_get_standard_category_ulp(self):
        """测试获取ULP标准类别"""
        category = self.registry._get_standard_category("matmul", out_dtype='torch.float32', in_dtype='torch.float32')
        self.assertEqual(category, "benchmark")

    def test_get_standard_category_thousandth(self):
        """测试获取双千比对标准类别"""
        category = self.registry._get_standard_category("conv2d", out_dtype='torch.float32', in_dtype='torch.float32')
        self.assertEqual(category, "benchmark")


if __name__ == '__main__':
    unittest.main()