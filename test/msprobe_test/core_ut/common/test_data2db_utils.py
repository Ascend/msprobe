import unittest
from msprobe.core.common.data2db_utils import process_tensor_value
from msprobe.core.common.const import Data2DBConst


class TestProcessTensorValue(unittest.TestCase):
    """测试process_tensor_value函数"""

    def test_process_valid_float_string(self):
        """处理合法的float字符串"""
        result = process_tensor_value("3.14")
        self.assertAlmostEqual(result, 3.14)

    def test_process_valid_int_string(self):
        """处理合法的int字符串"""
        result = process_tensor_value("42")
        self.assertAlmostEqual(result, 42.0)

    def test_process_negative_string(self):
        """处理负数字符串"""
        result = process_tensor_value("-1.5")
        self.assertAlmostEqual(result, -1.5)

    def test_process_list_value(self):
        """处理list类型（非int/float）"""
        result = process_tensor_value([1, 2, 3])
        self.assertIsNone(result)

    def test_process_positive_infinity(self):
        """处理正无穷：应返回 MAX_FLOAT_VALUE + 1"""
        result = process_tensor_value(float("inf"))
        self.assertEqual(result, Data2DBConst.MAX_FLOAT_VALUE + 1)

    def test_process_negative_infinity(self):
        """处理负无穷：应返回 MIN_FLOAT_VALUE - 1"""
        result = process_tensor_value(float("-inf"))
        self.assertEqual(result, Data2DBConst.MIN_FLOAT_VALUE - 1)

    def test_process_value_exceeds_max_float(self):
        """处理超过最大float的值：应截断为MAX_FLOAT_VALUE"""
        huge_value = Data2DBConst.MAX_FLOAT_VALUE + 1000
        result = process_tensor_value(huge_value)
        self.assertEqual(result, Data2DBConst.MAX_FLOAT_VALUE)

    def test_process_value_below_min_float(self):
        """处理低于最小float的值：应截断为MIN_FLOAT_VALUE"""
        tiny_value = Data2DBConst.MIN_FLOAT_VALUE - 1000
        result = process_tensor_value(tiny_value)
        self.assertEqual(result, Data2DBConst.MIN_FLOAT_VALUE)
