#!/usr/bin/env python
# coding=utf-8
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

from overflow_detection import OverflowDetection
from fusion_op import OutputDesc, FusionOp, OpAttr

import detail
import dump
import dump_data_pb2 as DD


class TestUtilsMethods(unittest.TestCase):

    def test_process_op_overflow_detection(self):
        detail_info = mock.Mock()
        detail_info.tensor_id = detail.TensorId('MaxPool_3', 'output', '0')
        compare_data = dump.CompareData("Pooling.MaxPool_3.5.1612779097467502", "Null", 2)
        compare_data.left_dump_info.op_name_to_file_map = {"MaxPool_3": ["Pooling.MaxPool_3.5.1612779097467502"]}
        compare_data.left_dump_info.type = dump.DumpType.Quant
        dump_data = DD.DumpData()
        dump_data.input.append(
            self._make_op_input(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data.output.append(
            self._make_op_output(DD.FORMAT_NCHW))
        with mock.patch('utils.parse_dump_file', return_value=dump_data):
            with mock.patch("utils.deserialize_dump_data_to_array", return_value=np.array([19345143, 2, 3])):
                overflow_detection = OverflowDetection(compare_data, detail_info.tensor_id.op_name)
                overflow_detection.process_op_overflow_detection()

    def test_process_model_overflow_detection(self):
        attr = OpAttr(['conv1', 'conv1_relu'], '', False, 12)
        output_desc_list = []
        output_desc = OutputDesc('conv1_relu', 0, 'NCHW',
                                 [1, 3, 224, 224])
        output_desc_list.append(output_desc)
        fusion_op = FusionOp(0, 'conv1conv1_relu', ['a:0,b:0'], 'Relu', output_desc_list, attr)
        left_dump_data = DD.DumpData()
        left_dump_data.input.append(self._make_op_input(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        left_dump_data.output.append(self._make_op_output(DD.FORMAT_NCHW))
        for input_type in ['input', 'output']:
            tensor = left_dump_data.input if input_type == 'input' else left_dump_data.output
            with mock.patch("utils.deserialize_dump_data_to_array", return_value=np.array([19345143, 2, 3])):
                OverflowDetection.process_model_overflow_detection(
                    fusion_op.op_name, 0, input_type, tensor[0])

    @staticmethod
    def _make_op_output(dd_format):
        op_output = DD.OpOutput()
        op_output.data_type = DD.DT_FLOAT16
        op_output.format = dd_format
        length = 3
        origin_numpy: q = np.array(np.array([19345143, 2, 3]), np.float16)
        op_output.data = struct.pack('f' * length, *origin_numpy)
        return op_output

    @staticmethod
    def _make_op_input(dd_format, shape):
        op_input = DD.OpInput()
        op_input.data_type = DD.DT_FLOAT16
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

