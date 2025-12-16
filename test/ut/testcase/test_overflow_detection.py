#!/usr/bin/env python
# coding=utf-8
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

"""
Function:
This file mainly involves xxxx function.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2021
"""
import struct

import unittest
import numpy as np
from unittest import mock

from overflow.overflow_detection import OverflowDetection
from dump_parse import dump, mapping
from vector_cmp.compare_detail import detail
from dump_parse import dump_utils
from dump_parse.proto_dump_data import DumpData, OpInput, OpOutput
from cmp_utils.constant.const_manager import DD


class TestUtilsMethods(unittest.TestCase):
    def test_parse_dump_file(self):
        detail_info = mock.Mock()
        detail_info.tensor_id = detail.TensorId('MaxPool_3', 'output', '0')
        compare_data = dump.CompareData("Pooling.MaxPool_3.5.1612779097467502", "Null", 2)
        compare_data.left_dump_info.op_name_to_file_map = {"MaxPool_3": ["Pooling.MaxPool_3.5.1612779097467502"]}
        compare_data.left_dump_info.type = dump.DumpType.Quant
        dump_data = DumpData()
        dump_data.input.append(
            self._make_op_input(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data.output.append(
            self._make_op_output(DD.FORMAT_NCHW))
        dump_data = dump_utils.convert_dump_data(dump_data)
        with mock.patch('dump_parse.dump_utils.parse_dump_file', return_value=dump_data):
            overflow_detection = OverflowDetection(compare_data, detail_info.tensor_id.op_name)
            input_tensor_data_info, output_tensor_data_info = overflow_detection.parse_dump_file()
            self.assertEqual(len(input_tensor_data_info) == 0, True)
            self.assertEqual(len(output_tensor_data_info) == 0, False)

    def test_get_tensor_data_info(self):
        tensor_type = "output"
        tensor = mock.Mock
        tensor.data = np.array([1, 2, 3])
        tensor.data_type = DD.DT_FLOAT16
        tensor_list = [tensor]
        dump_file_path = "/home/demo"
        detail_info = mock.Mock()
        detail_info.tensor_id = detail.TensorId('MaxPool_3', 'output', '0')
        compare_data = dump.CompareData("Pooling.MaxPool_3.5.1612779097467502", "Null", 2)
        compare_data.left_dump_info.op_name_to_file_map = {"MaxPool_3": ["Pooling.MaxPool_3.5.1612779097467502"]}
        compare_data.left_dump_info.type = dump.DumpType.Quant
        overflow_detection = OverflowDetection(compare_data, detail_info.tensor_id.op_name)
        tensor_data_info = overflow_detection._get_tensor_data_info(tensor_type, tensor_list, dump_file_path)
        self.assertEqual(len(tensor_data_info) != 0, True)

    def test_check_overflow_tensor(self):
        detail_info = mock.Mock()
        detail_info.tensor_id = detail.TensorId('MaxPool_3', 'output', '0')
        compare_data = dump.CompareData("Pooling.MaxPool_3.5.1612779097467502", "Null", 2)
        compare_data.left_dump_info.op_name_to_file_map = {"MaxPool_3": ["Pooling.MaxPool_3.5.1612779097467502"]}
        compare_data.left_dump_info.type = dump.DumpType.Quant
        overflow_detection = OverflowDetection(compare_data, detail_info.tensor_id.op_name)
        tensor_type = "input"
        np_array = np.array([43529352, -1])
        tensor = {"tensor_type": tensor_type, "index": str(0),
                  "tensor_data": np_array, "tensor_info": ""}
        tensor_list = [tensor]
        overflow_detection._check_overflow_tensor(tensor_list, [])
        overflow_detection._print_overflow_info_to_console(overflow_detection.overflow_tensor_list)

    def test_process_op_overflow_detection(self):
        detail_info = mock.Mock()
        detail_info.tensor_id = detail.TensorId('MaxPool_3', 'output', '0')
        compare_data = dump.CompareData("Pooling.MaxPool_3.5.1612779097467502", "Null", 2)
        compare_data.left_dump_info.op_name_to_file_map = {"MaxPool_3": ["Pooling.MaxPool_3.5.1612779097467502"]}
        compare_data.left_dump_info.type = dump.DumpType.Quant
        dump_data = DumpData()
        dump_data.input.append(
            self._make_op_input(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data.output.append(
            self._make_op_output(DD.FORMAT_NCHW))
        dump_data = dump_utils.convert_dump_data(dump_data)
        with mock.patch('dump_parse.dump_utils.parse_dump_file', return_value=dump_data):
            with mock.patch("dump_parse.dump_data_object._deserialize_dump_data_to_array",
                            return_value=np.array([19345143, 2, 3])):
                overflow_detection = OverflowDetection(compare_data, detail_info.tensor_id.op_name)
                overflow_detection.process_op_overflow_detection()

    def test_process_model_overflow_detection1(self):
        op_name = 'temp'
        index = 0
        is_input = True
        tensor = mock.Mock()
        tensor.data = np.array([19345143, 2, 3])
        tensor.data_type = 2
        res = OverflowDetection.process_model_overflow_detection(op_name, index, is_input, tensor)
        self.assertEqual(res, 'YES')

    def test_process_model_overflow_detection2(self):
        op_name = 'temp'
        index = 0
        is_input = True
        tensor = mock.Mock()
        tensor.data = np.array([1, 2, 3])
        tensor.data_type = 2
        res = OverflowDetection.process_model_overflow_detection(op_name, index, is_input, tensor)
        self.assertEqual(res, 'NO')

    def test_process_model_overflow_detection3(self):
        op_name = 'temp'
        index = 0
        is_input = True
        array = [
            [1, 2.1, -3.3333],
            [-72039, 33.2, 00.2]
        ]
        array = np.array(array)
        tensor = mock.Mock()
        tensor.data = array
        tensor.data_type = 3
        res = OverflowDetection.process_model_overflow_detection(op_name, index, is_input, tensor)
        self.assertEqual(res, 'NaN')

    def test_process_model_overflow_detection4(self):
        op_name = 'temp'
        index = 0
        is_input = True
        tensor = None
        res = OverflowDetection.process_model_overflow_detection(op_name, index, is_input, tensor)
        self.assertEqual(res, 'NaN')

    @staticmethod
    def _make_op_output(dd_format):
        op_output = OpOutput()
        op_output.data_type = DD.DT_FLOAT16
        op_output.format = dd_format
        length = 3
        origin_numpy = np.array(np.array([19345143, 2, 3]), np.float16)
        op_output.data = struct.pack('f' * length, *origin_numpy)
        return op_output

    @staticmethod
    def _make_op_input(dd_format, shape):
        op_input = OpInput()
        op_input.data_type = DD.DT_FLOAT
        op_input.format = dd_format
        length = 1
        if shape is None:
            length = 20
        else:
            for dim in shape:
                op_input.shape.dim.append(dim)
                length *= dim
        data_list = np.arange(length)
        origin_numpy = np.array(data_list, np.float16)
        op_input.data = struct.pack('f' * length, *origin_numpy)
        return op_input

