# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
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
from conversion.dtype_conversion import hifloat8_to_float32, float8e4m3fn_to_float32, float8e5m2_to_float32
 
 
class TestFloatConversions(unittest.TestCase):
    # hifloat8_to_float32 tests
    def test_hifloat8_to_float32_given_zero_when_unsigned_then_return_zero(self):
        self.assertEqual(hifloat8_to_float32(0x00), 0.0)
    
    def test_hifloat8_to_float32_given_positive_inf_when_special_value_then_return_inf(self):
        self.assertTrue(np.isinf(hifloat8_to_float32(0x6F)))
    
    def test_hifloat8_to_float32_given_negative_inf_when_special_value_then_return_neg_inf(self):
        self.assertTrue(np.isinf(hifloat8_to_float32(0xEF)))
    
    def test_hifloat8_to_float32_given_nan_when_special_value_then_return_nan(self):
        self.assertTrue(np.isnan(hifloat8_to_float32(0x80)))
    
    def test_hifloat8_to_float32_given_dml_mode_when_dot_value_neg1_then_calculate_value(self):
        self.assertAlmostEqual(hifloat8_to_float32(0x01), 2**-22, delta=1e-6)
 
    # float8e4m3fn_to_float32 tests
    def test_float8e4m3fn_to_float32_given_zero_when_unsigned_then_return_zero(self):
        self.assertEqual(float8e4m3fn_to_float32(0x00), 0.0)
    
    def test_float8e4m3fn_to_float32_given_zero_when_signed_then_return_neg_zero(self):
        self.assertTrue(np.signbit(float8e4m3fn_to_float32(0x80)))
    
    def test_float8e4m3fn_to_float32_given_denormal_when_exp_zero_then_calculate_value(self):
        self.assertAlmostEqual(float8e4m3fn_to_float32(0x01), 2**-9, delta=1e-6)
    
    def test_float8e4m3fn_to_float32_given_normal_when_exp_7_then_calculate_value(self):
        self.assertAlmostEqual(float8e4m3fn_to_float32(0x3C), 1.5, delta=1e-6)
 
    # float8e5m2_to_float32 tests
    def test_float8e5m2_to_float32_given_zero_when_unsigned_then_return_zero(self):
        self.assertEqual(float8e5m2_to_float32(0x00), 0.0)
    
    def test_float8e5m2_to_float32_given_zero_when_signed_then_return_neg_zero(self):
        self.assertTrue(np.signbit(float8e5m2_to_float32(0x80)))
    
    def test_float8e5m2_to_float32_given_denormal_when_exp_zero_then_calculate_value(self):
        self.assertAlmostEqual(float8e5m2_to_float32(0x01), 2**-16, delta=1e-6)
    
    def test_float8e5m2_to_float32_given_nan_when_exp_max_and_mantissa_nonzero_then_return_nan(self):
        self.assertTrue(np.isnan(float8e5m2_to_float32(0x7D)))
    
    def test_float8e5m2_to_float32_given_normal_when_exp_15_then_calculate_value(self):
        self.assertAlmostEqual(float8e5m2_to_float32(0x3E), 1.5, delta=1e-6)
    
    def test_float8e5m2_to_float32_given_max_value_when_exp_30_then_calculate_value(self):
        self.assertAlmostEqual(float8e5m2_to_float32(0x7B), 57344.0, delta=1e-6)