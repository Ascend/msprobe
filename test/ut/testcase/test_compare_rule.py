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
import json
import unittest
import pytest
from unittest import mock

from vector_cmp.fusion_manager import compare_rule
from cmp_utils.constant.compare_error import CompareError


class TestUtilsMethods(unittest.TestCase):

    def test_check_arguments_valid1(self):
        fusion_json_file_path = "/home/demo/1.json"
        quant_fusion_rule_file_path = "/home/demo/2.json"
        compare_rule_object = compare_rule.CompareRule(fusion_json_file_path, quant_fusion_rule_file_path)
        with pytest.raises(CompareError) as error:
            with mock.patch("cmp_utils.path_check.check_path_valid", return_value=1):
                compare_rule_object.check_arguments_valid()
        self.assertEqual(error.value.args[0], 1)

    def test_check_arguments_valid2(self):
        fusion_json_file_path = "/home/demo/1.json"
        quant_fusion_rule_file_path = "/home/demo/2.json"
        compare_rule_object = compare_rule.CompareRule(fusion_json_file_path, quant_fusion_rule_file_path)
        with mock.patch("cmp_utils.path_check.check_path_valid", return_value=0):
            compare_rule_object.check_arguments_valid()

    def test_check_arguments_valid3(self):
        fusion_json_file_path = "/home/demo/1.json"
        quant_fusion_rule_file_path = "/home/demo/2.json"
        close_fusion_rule_file_path = "/home/demo/3.json"
        compare_rule_object = compare_rule.CompareRule(fusion_json_file_path, quant_fusion_rule_file_path,
                                                       close_fusion_rule_file_path)
        with pytest.raises(CompareError) as error:
            with mock.patch("cmp_utils.path_check.check_path_valid", side_effect=[0, 0, 1]):
                compare_rule_object.check_arguments_valid()
        self.assertEqual(error.value.args[0], 1)

    def test_sort_file_by_timestamp1(self):
        fusion_json_file_path = "/home/demo/1.json"
        quant_fusion_rule_file_path = "/home/demo/2.json"
        compare_rule_object = compare_rule.CompareRule(fusion_json_file_path, quant_fusion_rule_file_path)
        dump_info = mock.Mock()
        dump_info.op_name_to_file_map = {"Add": ["/home/demo/CON.aDD.1.23431252326"]}
        dump_info.op_name_to_task_mode_map = {"Add": 0}
        dump_info.path = "/home/demo"
        with mock.patch("cmp_utils.path_check.check_path_valid", return_value=0):
            compare_rule_object._sort_file_by_timestamp(dump_info)

    def test_sort_file_by_timestamp2(self):
        fusion_json_file_path = "/home/demo/1.json"
        quant_fusion_rule_file_path = "/home/demo/2.json"
        compare_rule_object = compare_rule.CompareRule(fusion_json_file_path, quant_fusion_rule_file_path)
        dump_info = mock.Mock()
        dump_info.op_name_to_file_map = {"Add": ["/home/demo/1223453545232"]}
        dump_info.op_name_to_task_mode_map = {"Add": 0}
        dump_info.path = "/home/demo"
        dump_info.hash_to_file_name_map = {"1223453545232": "CON.aDD.1.23431252326"}
        with mock.patch("cmp_utils.path_check.check_path_valid", return_value=0):
            compare_rule_object._sort_file_by_timestamp(dump_info)

    def test_make_npu_vs_npu_fusion_rule1(self):
        fusion_json_file_path = "/home/demo/1.json"
        quant_fusion_rule_file_path = "/home/demo/2.json"
        compare_rule_object = compare_rule.CompareRule(fusion_json_file_path, quant_fusion_rule_file_path)
        left_sort_dic = {("aDD", 23431252326): ["CON.aDD.1.23431252326"]}
        compare_rule_object._sort_file_by_timestamp = mock.Mock(return_value=left_sort_dic)
        compare_data = mock.Mock()
        compare_data.left_dump_info = mock.Mock()
        compare_data.left_dump_info.op_name_to_file_map = {"Add": ["CON.aDD.1.23431252326"]}
        compare_data.left_dump_info.op_name_to_task_mode_map = {"Add": 0}
        compare_rule_object._make_npu_vs_npu_fusion_rule(compare_data)

    def test_parse_fusion_rule1(self):
        fusion_json_file_path = "/home/demo/1.json"
        quant_fusion_rule_file_path = "/home/demo/2.json"
        compare_rule_object = compare_rule.CompareRule(fusion_json_file_path, quant_fusion_rule_file_path)
        compare_data = mock.Mock()
        with mock.patch("os.path.getsize", return_value=100):
            with mock.patch('builtins.open', mock.mock_open(read_data=self._make_json())):
                compare_rule_object.parse_fusion_rule(compare_data)

    def test_parse_fusion_rule2(self):
        fusion_json_file_path = "/home/demo/1.json"
        quant_fusion_rule_file_path = ""
        compare_rule_object = compare_rule.CompareRule(fusion_json_file_path, quant_fusion_rule_file_path)
        compare_data = mock.Mock()
        with mock.patch("os.path.getsize", return_value=100):
            with mock.patch('builtins.open', mock.mock_open(read_data=self._make_json())):
                compare_rule_object.parse_fusion_rule(compare_data)

    def test_parse_fusion_rule3(self):
        fusion_json_file_path = ""
        quant_fusion_rule_file_path = "/home/demo/2.json"
        compare_rule_object = compare_rule.CompareRule(fusion_json_file_path, quant_fusion_rule_file_path)
        compare_data = mock.Mock()
        with mock.patch("os.path.getsize", return_value=100):
            with mock.patch('builtins.open', mock.mock_open(read_data=self._make_json())):
                compare_rule_object.parse_fusion_rule(compare_data)

    def test_parse_fusion_rule4(self):
        fusion_json_file_path = "/home/demo/2.json"
        quant_fusion_rule_file_path = ""
        close_fusion_rule_file_path = "/home/demo/3.json"
        compare_rule_object = compare_rule.CompareRule(fusion_json_file_path, quant_fusion_rule_file_path,
                                                       close_fusion_rule_file_path)
        compare_data = mock.Mock()
        with mock.patch("os.path.getsize", return_value=100):
            with mock.patch('builtins.open', mock.mock_open(read_data=self._make_json())):
                compare_rule_object.parse_fusion_rule(compare_data)

    @staticmethod
    def _make_json():
        return json.dumps({'name': 'resnet50', 'graph': [
            {'name': 'merge1', 'op':
                [{'name': 'conv1conv1_relu',
                  'type': 'Relu',
                  "attr": [
                      {"key": "_datadump_original_op_names",
                       "value": {"list": {"val_type": 1,
                                          "s": ["scale_conv1", "conv1",
                                                "bn_conv1", "conv1_relu"]}}}
                  ],
                  "input": [
                      "data:0",
                      "dynamic_const_471:0",
                      "dynamic_const_387:0",
                      "xxxx:-1",
                      "gddddd:-1"
                  ],
                  'output_desc': [
                      {'attr': [
                          {'key': '_datadump_origin_name',
                           'value': {'s': 'conv1_relu'}},
                          {'key': '_datadump_origin_output_index',
                           'value': {'i': 1}},
                          {'key': '_datadump_origin_format',
                           'value': {'s': 'NHWC'}},
                      ]},
                  ]
                  },
                 ]
             }]})


if __name__ == '__main__':
    unittest.main()
