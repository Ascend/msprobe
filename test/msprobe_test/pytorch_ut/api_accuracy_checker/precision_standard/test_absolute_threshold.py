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
from msprobe.pytorch.api_accuracy_checker.precision_standard.absolute_threshold import AbsolutethdCompare


class InputData:
    """测试数据类"""
    def __init__(self, bench_output, device_output, compare_column, dtype):
        self.bench_output = bench_output
        self.device_output = device_output
        self.dtype = dtype
        self.compare_column = compare_column


class TestAbsolutethdCompare(unittest.TestCase):

    def setUp(self):
        # 设置测试数据
        self.compare_column = {}
        self.input_data = InputData(
            bench_output=np.array([1.0, 2.0, 3.0, float('inf'), float('nan')]),
            device_output=np.array([1.1, 1.9, 3.1, float('inf'), 1.0]),
            compare_column={},
            dtype=torch.float32
        )
        self.compare = AbsolutethdCompare(self.input_data)

    def test_get_rtol(self):
        # 测试_get_rtol方法
        rtol = self.compare._get_rtol()
        self.assertEqual(rtol, 2**-20)

    def test_get_rel_err(self):
        # 测试_get_rel_err方法
        abs_err = np.abs(self.input_data.bench_output - self.input_data.device_output)
        abs_bench_with_eps = np.abs(self.input_data.bench_output) + np.finfo(np.float32).eps
        rel_err = self.compare._get_rel_err(abs_err, abs_bench_with_eps)
        self.assertTrue(np.all(np.isfinite(rel_err[~np.isnan(rel_err)])))

    def test_get_normal_value_mask(self):
        # 测试_get_normal_value_mask方法
        self.compare._pre_compare()
        
        # 创建一个示例small_value_mask
        small_value_mask = np.array([True, False, False, False, False])
        
        normal_mask = self.compare._get_normal_value_mask(self.compare.both_finite_mask, small_value_mask)
        
        # 验证返回值是布尔数组
        self.assertTrue(isinstance(normal_mask, np.ndarray))
        self.assertEqual(normal_mask.dtype, bool)
        
        # 验证normal_mask是both_finite_mask和not small_value_mask的逻辑与
        expected_mask = np.logical_and(self.compare.both_finite_mask, 
                                     np.logical_not(small_value_mask))
        np.testing.assert_array_equal(normal_mask, expected_mask)

    def test_pre_compare(self):
        # 测试_pre_compare方法
        self.compare._pre_compare()
        self.assertIsNotNone(self.compare.abs_bench)
        self.assertIsNotNone(self.compare.both_finite_mask)
        self.assertIsNotNone(self.compare.small_value_mask)
        self.assertIsNotNone(self.compare.normal_value_mask)

    def test_compute_metrics(self):
        # 测试_compute_metrics方法
        self.compare._pre_compare()
        metrics = self.compare._compute_metrics()
        self.assertIn("inf_nan_error_ratio", metrics)
        self.assertIn("rel_err_ratio", metrics)
        self.assertIn("abs_err_ratio", metrics)


if __name__ == '__main__':
    unittest.main()