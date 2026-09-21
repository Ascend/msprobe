#!/usr/bin/env python3
# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
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

from msprobe.pytorch.api_accuracy_checker.precision_standard.base_standard import BaseCompare


class MockInputData:
    def __init__(self, bench_output, device_output, dtype):
        self.bench_output = bench_output
        self.device_output = device_output
        self.compare_column = {}
        self.dtype = dtype


class TestCompare(BaseCompare):
    """Test implementation of BaseCompare"""
    def _pre_compare(self):
        """实现抽象方法"""
        pass


class TestBaseStandard(unittest.TestCase):
    """Test base_standard.py"""
    def setUp(self):
        """Test environment setup"""
        self.bench_output = np.array([1.0, 2.0, 3.0, float('inf'), float('nan')])
        self.device_output = np.array([1.1, 2.1, 3.1, float('inf'), float('nan')])
        self.input_data = MockInputData(self.bench_output, self.device_output, torch.float32)
        self.compare = TestCompare(self.input_data)

    def test_init(self):
        """Test BaseCompare initialization"""
        np.testing.assert_array_equal(self.compare.bench_output, self.input_data.bench_output)
        np.testing.assert_array_equal(self.compare.device_output, self.input_data.device_output)
        self.assertEqual(self.compare.dtype, self.input_data.dtype)

    def test_stat_finite_and_infinite_mask(self):
        """Test finite and infinite mask generation"""
        both_finite_mask, inf_nan_mask = self.compare.stat_finite_and_infinite_mask()
        
        # Check first three values are finite
        self.assertTrue(np.array_equal(both_finite_mask[:3], [True, True, True]))
        # Check last two values are infinite or NaN
        self.assertTrue(np.array_equal(inf_nan_mask[3:], [True, True]))

    def test_stat_abs_error(self):
        """Test absolute error calculation"""
        abs_err = self.compare.stat_abs_error()
        
        expected_errors = np.array([0.1, 0.1, 0.1])
        np.testing.assert_array_almost_equal(abs_err[:3], expected_errors)

    def test_stat_small_value_mask(self):
        """Test small value mask generation"""
        abs_bench = np.array([1e-10, 1.0, 1e-8])
        both_finite_mask = np.array([True, True, True])
        small_value = 1e-9
        
        result = TestCompare.stat_small_value_mask(abs_bench, both_finite_mask, small_value)
        expected = np.array([True, False, False])
        self.assertTrue(np.array_equal(result, expected))

    def test_compare_workflow(self):
        """Test compare workflow execution"""
        self.compare.compare()
        self.assertEqual(self.compare.compare_column, {})

    def test_get_small_value_threshold(self):
        """Test small value threshold retrieval"""
        small_value, small_value_atol = self.compare.get_small_value_threshold()
        self.assertIsInstance(small_value, (int, float))
        self.assertIsInstance(small_value_atol, (int, float))

    def test_stat_abs_bench_with_eps(self):
        """Test absolute benchmark with epsilon calculation"""
        abs_bench, abs_bench_with_eps = self.compare.stat_abs_bench_with_eps()
        
        # Check finite values
        self.assertTrue(np.array_equal(abs_bench[:3], np.abs(self.input_data.bench_output[:3])))
        self.assertTrue(np.all(abs_bench_with_eps[:3] >= abs_bench[:3]))
        
        # 添加极小值测试
        small_bench_output = np.array([1e-10, 1e-8, 1e-6])
        small_input_data = MockInputData(small_bench_output, small_bench_output + 1e-10, torch.float32)
        small_compare = TestCompare(small_input_data)
        
        abs_bench_small, abs_bench_with_eps_small = small_compare.stat_abs_bench_with_eps()
        
        # 验证极小值的误差容限计算
        self.assertTrue(np.all(abs_bench_with_eps_small > 0))  # 确保误差容限不为0
        self.assertTrue(np.all(abs_bench_with_eps_small >= abs_bench_small))  # 验证误差容限始终大于等于原值
        
        # 验证相对误差在合理范围内
        relative_tolerance = (abs_bench_with_eps_small - abs_bench_small) / abs_bench_small
        self.assertTrue(np.all(relative_tolerance <= 1.0))  # 相对误差不应过大


if __name__ == '__main__':
    unittest.main()