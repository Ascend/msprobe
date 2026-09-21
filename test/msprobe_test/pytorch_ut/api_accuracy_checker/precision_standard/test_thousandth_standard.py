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
from unittest.mock import Mock

from msprobe.pytorch.api_accuracy_checker.precision_standard.thousandth_standard import ThousandthStdCompare
from msprobe.core.common.const import CompareConst

class TestThousandthStdCompare(unittest.TestCase):
    def setUp(self):
        # 创建模拟的input_data对象
        self.mock_input = Mock()
        
    def test_initialization(self):
        """测试ThousandthStdCompare类的初始化"""
        # 设置模拟数据
        self.mock_input.rel_err_orign = np.array([0.0001, 0.002, 0.0005])
        self.mock_input.compare_column = Mock()
        
        # 创建实例
        compare = ThousandthStdCompare(self.mock_input)
        
        # 验证属性是否正确设置
        np.testing.assert_array_equal(compare.rel_err_orign, self.mock_input.rel_err_orign)
        self.assertEqual(compare.compare_column, self.mock_input.compare_column)

    def test_compute_metrics_all_within_threshold(self):
        """测试所有值都在阈值内的情况"""
        # 设置模拟数据 - 所有值都小于阈值(0.001)
        self.mock_input.rel_err_orign = np.array([0.0001, 0.0005, 0.0008])
        compare = ThousandthStdCompare(self.mock_input)
        
        # 计算指标
        result = compare._compute_metrics()
        
        # 验证结果
        self.assertEqual(result['rel_err_thousandth'], 1.0)

    def test_compute_metrics_mixed_values(self):
        """测试混合值的情况"""
        # 设置模拟数据 - 部分值超过阈值
        self.mock_input.rel_err_orign = np.array([0.0005, 0.002, 0.003, 0.0008])
        compare = ThousandthStdCompare(self.mock_input)
        
        # 计算指标
        result = compare._compute_metrics()
        
        # 验证结果 - 2个值在阈值内，2个值超过阈值
        self.assertEqual(result['rel_err_thousandth'], 0.5)

    def test_compute_metrics_all_exceed_threshold(self):
        """测试所有值都超过阈值的情况"""
        # 设置模拟数据 - 所有值都大于阈值
        self.mock_input.rel_err_orign = np.array([0.002, 0.003, 0.005])
        compare = ThousandthStdCompare(self.mock_input)
        
        # 计算指标
        result = compare._compute_metrics()
        
        # 验证结果
        self.assertEqual(result['rel_err_thousandth'], 0.0)

if __name__ == '__main__':
    unittest.main()