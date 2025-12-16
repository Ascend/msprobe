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
from unittest.mock import patch
import numpy as np
from dump_parse.proto_dump_data import DumpData, OpInput, OpOutput
from dump_parse.dump_data_object import build_dump_tensor, _deserialize_dump_data_david_to_array


class TestUtilsMethods(unittest.TestCase):
    @staticmethod
    def make_dump_data_without_shape():
        input = OpInput()
        input.data = b"\000\312\232;"
        input.size = 4
        return input

    def test_build_dump_tensor(self):
        dump_data = [self.make_dump_data_without_shape()]
        try:
            build_dump_tensor(dump_data_object_data=dump_data, is_input=True, is_ffts=False)
        except Exception as e:
            self.fail(f"build dump tensor failed: {e}")
        self.assertEqual(dump_data[0].size, 4)
        self.assertEqual(dump_data[0].shape, [4])
    
    def setUp(self):
        self.mock_float8e4m3fn = np.array([0x3C], dtype=np.uint8)  # 1.5
        self.mock_hifloat8 = np.array([0x40], dtype=np.uint8)      # 16.0
        self.mock_float8e5m2 = np.array([0x45], dtype=np.uint8)    # 5.0
 
    def test_deserialize_dump_data_david_to_array_given_empty_shape_when_zero_in_shape_then_return_empty_array(self):
        result = _deserialize_dump_data_david_to_array(b'', 'float8_e4m3fn', [0])
        self.assertEqual(result.size, 0)
        self.assertEqual(result.shape, (0,))
 
    def test_deserialize_dump_data_david_to_array_given_shape_when_float8_e4m3fn_then_return_correct_array(self):
        with patch('cmp_utils.common.get_dtype_by_data_type', return_value="float8_e4m3fn"):
            result = _deserialize_dump_data_david_to_array(self.mock_float8e4m3fn.tobytes(), 'float8_e4m3fn', [1])
            self.assertEqual(result[0], 1.5)
            self.assertEqual(result.shape, (1,))
 
    def test_deserialize_dump_data_david_to_array_given_shape_when_hifloat8_then_return_correct_array(self):
        with patch('cmp_utils.common.get_dtype_by_data_type', return_value="float8_e5m2"):
            result = _deserialize_dump_data_david_to_array(self.mock_float8e5m2.tobytes(), 'float8_e5m2', [1])
            self.assertEqual(result[0], 5.0)
            self.assertEqual(result.shape, (1,))
 
    def test_deserialize_dump_data_david_to_array_given_shape_when_float8_e5m2_then_return_correct_array(self):
        with patch('cmp_utils.common.get_dtype_by_data_type', return_value="hifloat8"):
            result = _deserialize_dump_data_david_to_array(self.mock_hifloat8, 'hifloat8', [1])
            self.assertEqual(result[0], 16.0)
            self.assertEqual(result.shape, (1,))