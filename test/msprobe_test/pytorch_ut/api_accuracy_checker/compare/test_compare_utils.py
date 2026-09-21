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
import numpy as np

from msprobe.pytorch.api_accuracy_checker.common.utils import CompareException
from msprobe.pytorch.api_accuracy_checker.compare.compare_utils import *


class TestCompareUtils(unittest.TestCase):
    
    def setUp(self):
        self.column = ApiPrecisionCompareColumn()
    
    def test_get_detail_csv_title(self):
        expect_detail_csv_title = [
                self.column.API_NAME, self.column.SMALL_VALUE_ERROR_RATIO, 
                self.column.SMALL_VALUE_ERROR_STATUS, self.column.RMSE_RATIO, 
                self.column.RMSE_STATUS, self.column.MAX_REL_ERR_RATIO, 
                self.column.MAX_REL_ERR_STATUS, self.column.MEAN_REL_ERR_RATIO, 
                self.column.MEAN_REL_ERR_STATUS, self.column.EB_RATIO, 
                self.column.EB_STATUS, self.column.INF_NAN_ERROR_RATIO, 
                self.column.INF_NAN_ERROR_RATIO_STATUS, self.column.REL_ERR_RATIO, 
                self.column.REL_ERR_RATIO_STATUS, self.column.ABS_ERR_RATIO, 
                self.column.ABS_ERR_RATIO_STATUS, self.column.ERROR_RATE, 
                self.column.ERROR_RATE_STATUS, self.column.MEAN_ULP_ERR, 
                self.column.ULP_ERR_PROPORTION, self.column.ULP_ERR_PROPORTION_RATIO, 
                self.column.ULP_ERR_STATUS, self.column.REL_ERR_THOUSANDTH, 
                self.column.REL_ERR_THOUSANDTH_STATUS, self.column.FINAL_RESULT, 
                self.column.ALGORITHM, self.column.MESSAGE
                ]
        self.assertEqual(self.column.get_detail_csv_title(), expect_detail_csv_title)
    
    def test_get_result_csv_title(self):
        expect_result_csv_title =[
                self.column.API_NAME, self.column.FORWWARD_STATUS, 
                self.column.BACKWARD_STATUS, self.column.MESSAGE
                ]
        self.assertEqual(self.column.get_result_csv_title(), expect_result_csv_title)

    def test_check_dtype_comparable(self):
        x = np.array([1, 2, 3], dtype=np.int32)
        y = np.array([4, 5, 6], dtype=np.int32)
        self.assertTrue(check_dtype_comparable(x, y))

        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        y = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        self.assertTrue(check_dtype_comparable(x, y))

        x = np.array([True, False, True], dtype=np.bool_)
        y = np.array([False, True, False], dtype=np.bool_)
        self.assertTrue(check_dtype_comparable(x, y))

        x = np.array([1, 2, 3], dtype=np.int32)
        y = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        self.assertFalse(check_dtype_comparable(x, y))
        
        x = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        y = np.array([1, 2, 3], dtype=np.int32)
        self.assertFalse(check_dtype_comparable(x, y))

        x = np.array([1, 2, 3], dtype=np.int32)
        y = np.array([True, False, True], dtype=np.bool_)
        self.assertFalse(check_dtype_comparable(x, y))
        
        x = np.array([True, False, True], dtype=np.bool_)
        y = np.array([1, 2, 3], dtype=np.int32)
        self.assertFalse(check_dtype_comparable(x, y))
        
        x = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32)
        y = np.array([1, 2, 3], dtype=np.int32)
        self.assertFalse(check_dtype_comparable(x, y))

    def test_convert_str_to_float_when_valid_float(self):
        self.assertEqual(convert_str_to_float("123.45"), 123.45)

    def test_convert_str_to_float_when_valid_int(self):
        self.assertEqual(convert_str_to_float("123.0"), 123.0)

    def test_convert_str_to_float_when_valid_int_with_spaces(self):
        self.assertEqual(convert_str_to_float("   123.0   "), 123.0)

    def test_convert_str_to_float_when_empty_string(self):
        with self.assertRaises(CompareException) as cm:
            convert_str_to_float('')
        self.assertEqual(cm.exception.code, CompareException.INVALID_DATA_ERROR)
        
    def test_convert_str_to_float_wiht_whitespace_string(self):
        with self.assertRaises(CompareException) as context:
            convert_str_to_float('    ')
        self.assertEqual(context.exception.code, CompareException.INVALID_DATA_ERROR)
        
    def test_handle_infinity_both_inf_same_sign(self):
        result, consistent, message = handle_infinity(float('inf'), float('inf'), 'test_column')
        self.assertTrue(math.isnan(result))
        self.assertTrue(consistent)
        self.assertEqual(message, "test_column同为同号inf或nan\n")

    def test_handle_infinity_both_inf_different_sign(self):
        result, consistent, message = handle_infinity(float('inf'), float('-inf'), 'test_column')
        self.assertTrue(math.isnan(result))
        self.assertFalse(consistent)
        self.assertEqual(message, "test_columninf或nan不一致\n")

    def test_handle_infinity_one_inf(self):
        result, consistent, message = handle_infinity(float('inf'), 1.0, 'test_column')
        self.assertTrue(math.isnan(result))
        self.assertFalse(consistent)
        self.assertEqual(message, "test_columninf或nan不一致\n")

    def test_handle_nan_both_nan(self):
        result, consistent, message = handle_nan(float('nan'), float('nan'), 'test_column')
        self.assertTrue(math.isnan(result))
        self.assertTrue(consistent)
        self.assertEqual(message, "test_column同为同号inf或nan\n")

    def test_handle_nan_one_nan(self):
        result, consistent, message = handle_nan(float('nan'), 1.0, 'test_column')
        self.assertTrue(math.isnan(result))
        self.assertFalse(consistent)
        self.assertEqual(message, "test_columninf或nan不一致\n")

    def test_check_inf_or_nan_both_inf(self):
        result, consistent, message = check_inf_or_nan(float('inf'), float('inf'), 'test_column')
        self.assertTrue(math.isnan(result))
        self.assertTrue(consistent)
        self.assertEqual(message, "test_column同为同号inf或nan\n")

    def test_check_inf_or_nan_both_nan(self):
        # 测试两个值都是 NaN
        result, consistent, message = check_inf_or_nan(float('nan'), float('nan'), 'test_column')
        self.assertTrue(math.isnan(result))
        self.assertTrue(consistent)
        self.assertEqual(message, "test_column同为同号inf或nan\n")
