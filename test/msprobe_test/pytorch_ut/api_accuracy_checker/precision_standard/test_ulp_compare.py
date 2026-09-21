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

from msprobe.pytorch.api_accuracy_checker.precision_standard.ulp_compare import UlpCompare
from msprobe.core.common.const import CompareConst


class InputData:
    """测试数据类"""
    def __init__(self):
        self.bench_output = None
        self.device_output = None
        self.dtype = None
        self.compare_column = None


class TestUlpCompare(unittest.TestCase):
    """UlpCompare类的单元测试"""
    def setUp(self):
        """测试前的准备工作"""
        self.input_data = InputData()
        self.input_data.bench_output = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        self.input_data.device_output = np.array([1.0, 2.0001, 3.0], dtype=np.float32)
        self.input_data.dtype = torch.float32
        self.input_data.compare_column = ['output']
        self.ulp_compare = UlpCompare(self.input_data)

    def test_init(self):
        """测试初始化"""
        self.assertIsInstance(self.ulp_compare, UlpCompare)
        self.assertEqual(self.ulp_compare.dtype, torch.float32)
        np.testing.assert_array_equal(self.ulp_compare.bench_output, self.input_data.bench_output)
        np.testing.assert_array_equal(self.ulp_compare.device_output, self.input_data.device_output)

    def test_stat_max_ulp_err(self):
        """测试最大ULP误差计算"""
        test_ulp_err = np.array([0, 2, 5], dtype=np.float32)
        max_err = self.ulp_compare._stat_max_ulp_err(test_ulp_err)
        self.assertEqual(max_err, 5)

    def test_stat_mean_ulp_err(self):
        """测试平均ULP误差计算"""
        test_ulp_err = np.array([1, 2, 3], dtype=np.float32)
        mean_err = self.ulp_compare._stat_mean_ulp_err(test_ulp_err)
        self.assertEqual(mean_err, 2)

    def test_stat_ulp_error_proportion_float32(self):
        """测试float32的ULP误差比例计算"""
        test_ulp_err = np.array([
            CompareConst.ULP_FLOAT32_THRESHOLD - 1,  # 低于阈值
            CompareConst.ULP_FLOAT32_THRESHOLD + 1,  # 高于阈值
            CompareConst.ULP_FLOAT32_THRESHOLD - 1   # 低于阈值
        ], dtype=np.float32)
        proportion = self.ulp_compare._stat_ulp_error_proportion(test_ulp_err)
        self.assertAlmostEqual(proportion, 1/3)

    def test_stat_ulp_error_proportion_float16(self):
        """测试float16的ULP误差比例计算"""
        self.ulp_compare.dtype = torch.float16
        test_ulp_err = np.array([
            CompareConst.ULP_FLOAT16_THRESHOLD - 1,  # 低于阈值
            CompareConst.ULP_FLOAT16_THRESHOLD + 1,  # 高于阈值
            CompareConst.ULP_FLOAT16_THRESHOLD - 1   # 低于阈值
        ], dtype=np.float32)
        proportion = self.ulp_compare._stat_ulp_error_proportion(test_ulp_err)
        self.assertAlmostEqual(proportion, 1/3)

    def test_pre_compare(self):
        """测试预处理比较步骤"""
        self.ulp_compare._pre_compare()
        self.assertTrue(hasattr(self.ulp_compare, 'ulp_err'))
        self.assertIsInstance(self.ulp_compare.ulp_err, np.ndarray)

    def test_compute_metrics(self):
        """测试完整的指标计算流程"""
        self.ulp_compare._pre_compare()
        metrics = self.ulp_compare._compute_metrics()
        
        # 验证返回的字典包含所有必要的键
        expected_keys = {'max_ulp_error', 'mean_ulp_error', 'ulp_error_proportion'}
        self.assertEqual(set(metrics.keys()), expected_keys)
        
        # 验证返回值类型
        for value in metrics.values():
            self.assertIsInstance(value, (np.float32, np.float64))

        # 验证数值范围
        self.assertGreaterEqual(metrics['max_ulp_error'], 0)
        self.assertGreaterEqual(metrics['mean_ulp_error'], 0)
        self.assertGreaterEqual(metrics['ulp_error_proportion'], 0)
        self.assertLessEqual(metrics['ulp_error_proportion'], 1)


if __name__ == '__main__':
    unittest.main()