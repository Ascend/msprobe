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
import numpy as np
import torch

from msprobe.pytorch.api_accuracy_checker.precision_standard.benchmark_compare import BenchmarkCompare


class InputData:
    """测试数据类"""
    def __init__(self, bench_output, device_output, dtype, compare_column):
        self.bench_output = bench_output
        self.device_output = device_output
        self.dtype = dtype
        self.compare_column = compare_column


class TestBenchmarkCompare(unittest.TestCase):
    def setUp(self):
        """创建基础测试数据"""
        self.bench_output = np.array([1.0, 2.0, 3.0, float('inf'), float('nan')])
        self.device_output = np.array([1.1, 2.1, 3.1, float('inf'), float('nan')])
        self.compare_column = {}
        self.dtype = torch.float32
        
        self.input_data = InputData(
            bench_output=self.bench_output,
            device_output=self.device_output,
            compare_column=self.compare_column,
            dtype=self.dtype
        )
        
        self.compare = BenchmarkCompare(self.input_data)
        self.compare._pre_compare()

    def test_get_abs_err_greater_mask(self):
        """测试_get_abs_err_greater_mask函数"""
        # 测试不同的阈值
        small_value_atol = 0.05
        mask = self.compare._get_abs_err_greater_mask(small_value_atol)
        
        # 验证掩码类型和形状
        self.assertIsInstance(mask, np.ndarray)
        self.assertEqual(mask.dtype, bool)
        self.assertEqual(mask.shape, self.bench_output.shape)
        
        # 验证掩码值是否正确
        expected_mask = np.array([True, True, True, False, False])  # 前三个差值大于0.05，后两个是inf/nan
        np.testing.assert_array_equal(mask, expected_mask)

    def test_compute_rel_err(self):
        """测试_compute_rel_err函数"""
        rel_err = self.compare._compute_rel_err()
        
        # 验证相对误差的类型和形状
        self.assertIsInstance(rel_err, np.ndarray)
        self.assertEqual(rel_err.shape, self.bench_output.shape)
        
        # 验证前三个有效值的相对误差
        expected_rel_err = np.array([0.1, 0.05, 0.033333], dtype=np.float32)
        np.testing.assert_array_almost_equal(rel_err[:3], expected_rel_err, decimal=5)

    def test_pre_compare(self):
        """测试_pre_compare函数"""
        # 创建新的比较对象进行预处理
        compare = BenchmarkCompare(self.input_data)
        compare._pre_compare()
        
        # 验证预处理后的属性是否正确设置
        self.assertTrue(hasattr(compare, 'abs_bench'))
        self.assertTrue(hasattr(compare, 'abs_bench_with_eps'))
        self.assertTrue(hasattr(compare, 'both_finite_mask'))
        self.assertTrue(hasattr(compare, 'inf_nan_mask'))
        self.assertTrue(hasattr(compare, 'abs_err'))
        self.assertTrue(hasattr(compare, 'small_value'))
        self.assertTrue(hasattr(compare, 'small_value_atol'))
        self.assertTrue(hasattr(compare, 'small_value_mask'))
        self.assertTrue(hasattr(compare, 'rel_err'))
        self.assertTrue(hasattr(compare, 'abs_err_greater_mask'))
        
        # 验证有限值掩码
        expected_finite_mask = np.array([True, True, True, False, False])
        np.testing.assert_array_equal(compare.both_finite_mask, expected_finite_mask)
        
        # 验证绝对误差
        expected_abs_err = np.abs(self.device_output - self.bench_output)
        np.testing.assert_array_equal(compare.abs_err[:3], expected_abs_err[:3])

    def test_compute_metrics(self):
        """测试_compute_metrics函数"""
        metrics = self.compare._compute_metrics()
        
        # 验证返回的指标字典
        expected_metrics = {
            "small_value_err_ratio",
            "max_rel_error",
            "mean_rel_error",
            "rmse",
            "eb"
        }
        self.assertEqual(set(metrics.keys()), expected_metrics)


if __name__ == '__main__':
    unittest.main()