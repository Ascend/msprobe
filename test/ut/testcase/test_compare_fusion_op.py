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
import unittest
from unittest import mock
import numpy as np
import struct
from dump_parse.proto_dump_data import DumpData, OpInput, OpOutput
from vector_cmp.fusion_manager.compare_fusion_op import FusionOpComparison
from algorithm_manager.algorithm_manager import AlgorithmManager
from vector_cmp.fusion_manager.fusion_op import FusionOp, OutputDesc, OpAttr
from dump_parse.dump import DumpType
from cmp_utils.constant.compare_error import CompareError
from cmp_utils.constant.const_manager import DD
from dump_parse import dump, dump_utils, mapping


class TestUtilsMethods(unittest.TestCase):
    def test_make_gpu_and_npu_mapping_table1(self):
        op_name = "output_ids"
        op_id = 0
        original_op_names = ["input_ids"]
        input_list = []
        op_type = "Data"
        l1_fusion_no = ""
        is_multi_op = False
        origin_name = "input_ids"
        origin_format = "NHWC"
        origin_shape = [1, 128]
        origin_output_index = 0
        attr = OpAttr(original_op_names, l1_fusion_no, is_multi_op, 0)
        output_desc = OutputDesc(origin_name, origin_output_index, origin_format, origin_shape)
        fusion_list = [FusionOp(op_id, op_name, input_list, op_type, output_desc, attr)]
        compare_rule = mock.Mock
        compare_data = mock.Mock
        compare_data.fusion_info = mock.Mock
        compare_data.fusion_info.fusion_op_name_to_op_map = {"demo": fusion_list}
        compare_data.left_dump_info = mock.Mock
        compare_data.left_dump_info.type = None
        compare_data.dump_version = 2
        compare_data.left_dump_info.get_op_dump_file = mock.Mock(return_value="/home/demo")
        format_manager = ""
        fusion_op_name = "demo"
        with mock.patch("dump_parse.dump_utils.parse_dump_file", side_effect=CompareError(1)):
            FusionOpComparison(fusion_op_name, compare_rule, compare_data,
                               format_manager,
                               {'algorithm_manager': AlgorithmManager('', 'all', '')}).make_gpu_and_npu_mapping_table()

    def test_make_gpu_and_npu_mapping_table2(self):
        op_name = "demo2"
        op_id = 1
        original_op_names = ["input_ids"]
        input_list = []
        op_type = "Data"
        l1_fusion_no = ""
        is_multi_op = False
        origin_name = "input_ids"
        origin_format = "NHWC"
        origin_shape = [1, 128]
        origin_output_index = 0
        attr = OpAttr(original_op_names, l1_fusion_no, is_multi_op, 0)
        output_desc = OutputDesc(origin_name, origin_output_index, origin_format, origin_shape)
        fusion_list = [FusionOp(op_id, op_name, input_list, op_type, output_desc, attr)]
        compare_rule = mock.Mock
        compare_data = mock.Mock
        compare_data.fusion_info = mock.Mock
        compare_data.fusion_info.fusion_op_name_to_op_map = {"demo": fusion_list}
        compare_data.left_dump_info = mock.Mock
        compare_data.left_dump_info.type = DumpType.Offline
        compare_data.dump_version = 2
        compare_data.left_dump_info.get_op_dump_file = mock.Mock(return_value="/home/demo")
        format_manager = ""
        fusion_op_name = "demo"
        dump_data = DumpData()
        dump_data.input.append(
            self._make_op_input(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data.output.append(
            self._make_op_output(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data = dump_utils.convert_dump_data(dump_data)
        with mock.patch("dump_parse.dump_utils.parse_dump_file", return_value=dump_data):
            FusionOpComparison(fusion_op_name, compare_rule, compare_data,
                               format_manager,
                               {'algorithm_manager': AlgorithmManager('', 'all', '')}).make_gpu_and_npu_mapping_table()

    def test_make_gpu_and_npu_mapping_table3(self):
        op_name = "demo2"
        op_id = 1
        original_op_names = ["input_ids"]
        input_list = []
        op_type = "Data"
        l1_fusion_no = ""
        is_multi_op = False
        origin_name = "input_ids"
        origin_format = "NHWC"
        origin_shape = [1, 128]
        origin_output_index = 0
        attr = OpAttr(original_op_names, l1_fusion_no, is_multi_op, 0)
        output_desc = OutputDesc(origin_name, origin_output_index, origin_format, origin_shape)
        fusion_list = [FusionOp(op_id, op_name, input_list, op_type, output_desc, attr)]
        compare_rule = mock.Mock
        compare_data = mock.Mock
        compare_data.fusion_info = mock.Mock
        compare_data.fusion_info.fusion_op_name_to_op_map = {"demo": fusion_list}
        compare_data.left_dump_info = mock.Mock
        compare_data.left_dump_info.type = DumpType.Quant
        compare_data.dump_version = 2
        compare_data.left_dump_info.get_op_dump_file = mock.Mock(return_value="/home/demo")
        format_manager = ""
        fusion_op_name = "demo"
        dump_data = DumpData()
        dump_data.input.append(
            self._make_op_input(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data.output.append(
            self._make_op_output(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data = dump_utils.convert_dump_data(dump_data)
        with mock.patch("dump_parse.dump_utils.parse_dump_file", return_value=dump_data):
            FusionOpComparison(fusion_op_name, compare_rule, compare_data,
                               format_manager,
                               {'algorithm_manager': AlgorithmManager('', 'all', '')}).make_gpu_and_npu_mapping_table()

    @staticmethod
    def _make_op_output(dd_format, shape):
        op_output = OpOutput()
        op_output.data_type = DD.DT_FLOAT
        op_output.format = dd_format
        length = 1
        if shape is None:
            length = 20
        else:
            for dim in shape:
                op_output.shape.dim.append(dim)
                length *= dim
        data_list = np.arange(length)
        origin_numpy = np.array(data_list, np.float16)
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

