#!/usr/bin/env python
# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
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

    def setUp(self):
        self._check_path_valid_patcher = mock.patch(
            "cmp_utils.file_utils.check_path_valid",
            return_value=CompareError.MSACCUCMP_NONE_ERROR,
        )
        self._check_path_valid_patcher.start()

    def tearDown(self):
        self._check_path_valid_patcher.stop()

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

    def test_make_npu_vs_npu_fusion_rule_preserves_official_order(self):
        compare_rule_object = compare_rule.CompareRule("", "", "", "/home/demo/model.json")
        left_sort_dic = {
            ("late_op", 100): ["CON.late_op.1.100"],
            ("early_op", 200): ["CON.early_op.1.200"],
        }
        right_sort_dic = {
            ("late_op", 300): ["CON.late_op.1.300"],
            ("early_op", 400): ["CON.early_op.1.400"],
        }
        compare_rule_object._sort_file_by_timestamp = mock.Mock(side_effect=[left_sort_dic, right_sort_dic])
        compare_data = mock.Mock()

        compare_rule_object._make_npu_vs_npu_fusion_rule(compare_data)

        op_map = compare_rule_object.fusion_info.fusion_op_name_to_op_map
        self.assertEqual(list(op_map), ["late_op", "early_op"])
        self.assertEqual(op_map["late_op"][0].op_id, 0)
        self.assertEqual(op_map["late_op"][1].op_id, 0)
        self.assertEqual(op_map["early_op"][0].op_id, 1)
        self.assertEqual(op_map["early_op"][1].op_id, 1)

        with mock.patch(
            "vector_cmp.fusion_manager.compare_rule.ModelGraphOrderParser.parse",
            return_value={"early_op": 0, "late_op": 1},
        ):
            index_to_sequence = compare_rule_object.make_index_to_graph_sequence()

        self.assertEqual(index_to_sequence, {0: 1, 1: 0})
        self.assertEqual(list(op_map), ["late_op", "early_op"])
        self.assertEqual(op_map["late_op"][0].op_id, 0)
        self.assertEqual(op_map["early_op"][0].op_id, 1)

    def test_make_index_to_graph_sequence_requires_fusion_info(self):
        compare_rule_object = compare_rule.CompareRule("", "", "", "/home/demo/model.json")
        with mock.patch("vector_cmp.fusion_manager.compare_rule.ModelGraphOrderParser.parse") as mock_parse:
            with self.assertRaises(CompareError) as error:
                compare_rule_object.make_index_to_graph_sequence()
        self.assertEqual(error.exception.code, CompareError.MSACCUCMP_UNKNOWN_ERROR)
        self.assertIn("Call parse_fusion_rule() first.", error.exception.message)
        mock_parse.assert_not_called()

    def test_make_index_to_graph_sequence_without_graph_needs_no_fusion_info(self):
        compare_rule_object = compare_rule.CompareRule("", "", "")
        with mock.patch("vector_cmp.fusion_manager.compare_rule.ModelGraphOrderParser.parse") as mock_parse:
            self.assertIsNone(compare_rule_object.make_index_to_graph_sequence())
        mock_parse.assert_not_called()

    def test_make_index_to_graph_sequence_without_match_stops_comparison(self):
        compare_rule_object = compare_rule.CompareRule("", "", "", "/home/demo/model.json")
        left_op = mock.Mock(op_id=0, op_name="dump_op", op_type="Left")
        compare_rule_object.fusion_info = mock.Mock(fusion_op_name_to_op_map={"dump_op": [left_op]})
        with mock.patch(
            "vector_cmp.fusion_manager.compare_rule.ModelGraphOrderParser.parse",
            return_value={"graph_op": 1},
        ), mock.patch("vector_cmp.fusion_manager.compare_rule.log.print_error_log") as mock_error:
            with self.assertRaises(CompareError) as error:
                compare_rule_object.make_index_to_graph_sequence()
        self.assertEqual(error.exception.code, CompareError.MSACCUCMP_INVALID_PARAM_ERROR)
        self.assertIn('No target dump operator matches', error.exception.message)
        self.assertIn('dump and graph operator name formats', error.exception.message)
        self.assertIn('before result sorting and CSV renumbering', error.exception.message)
        mock_error.assert_called_once_with(error.exception.message)

    def test_make_index_to_graph_sequence_uses_target_dump_naming_mode(self):
        raw_names = ['partition0_sgt_graph_0_first_lxslice0', 'partition1_sgt_graph_1_second_lxslice1']
        graph = {'graph': [{'op': [{'name': name}]} for name in raw_names]}
        for ffts, dump_names in ((False, ['first', 'second']), (True, raw_names)):
            with self.subTest(ffts=ffts):
                rule = compare_rule.CompareRule('', '', '', '/home/demo/model.json')
                compare_data = mock.Mock(left_dump_info=mock.Mock(ffts=ffts))
                dump_order = {(dump_names[1], 100): ['second.dump'], (dump_names[0], 200): ['first.dump']}
                with mock.patch.object(rule, '_sort_file_by_timestamp', return_value=dump_order), mock.patch(
                    'vector_cmp.fusion_manager.model_graph_order_parser.FileUtils.load_json_file', return_value=graph
                ):
                    rule.parse_fusion_rule(compare_data)
                    self.assertEqual(rule.make_index_to_graph_sequence(), {0: 1, 1: 0})
                self.assertEqual(list(rule.fusion_info.fusion_op_name_to_op_map), list(reversed(dump_names)))

    def test_make_index_to_graph_sequence_partial_match_warns(self):
        compare_rule_object = compare_rule.CompareRule("", "", "", "/home/demo/model.json")
        matched_op = mock.Mock(op_id=0, op_name="graph_op", op_type="Left")
        unmatched_op = mock.Mock(op_id=1, op_name="orphan_op", op_type="Left")
        compare_rule_object.fusion_info = mock.Mock(
            fusion_op_name_to_op_map={"graph_op": [matched_op], "orphan_op": [unmatched_op]}
        )
        with mock.patch(
            "vector_cmp.fusion_manager.compare_rule.ModelGraphOrderParser.parse",
            return_value={"graph_op": 3},
        ), mock.patch("vector_cmp.fusion_manager.compare_rule.log.print_warn_log") as mock_warn:
            result = compare_rule_object.make_index_to_graph_sequence()
        self.assertEqual(result, {0: 3})
        mock_warn.assert_called_once_with(
            '1 target operator groups are not found in the model graph and will be placed last.'
        )

    def test_model_graph_file_cannot_be_used_with_fusion_rule(self):
        fusion_rule_paths = [
            ("/home/demo/open.json", "", ""),
            ("", "/home/demo/quant.json", ""),
            ("", "", "/home/demo/close.json"),
        ]
        for fusion_rule_file, quant_rule_file, close_rule_file in fusion_rule_paths:
            with self.subTest(
                fusion_rule_file=fusion_rule_file,
                quant_rule_file=quant_rule_file,
                close_rule_file=close_rule_file,
            ):
                compare_rule_object = compare_rule.CompareRule(
                    fusion_rule_file,
                    quant_rule_file,
                    close_rule_file,
                    "/home/demo/model.json",
                )
                with self.assertRaises(CompareError) as error:
                    compare_rule_object.check_arguments_valid()
                self.assertEqual(error.exception.code, CompareError.MSACCUCMP_INVALID_PARAM_ERROR)
                for option in ('-fr/--fusion_rule_file', '-qfr/--quant_fusion_rule_file',
                               '-cfr/--close_fusion_rule_file',
                               'inner msaccucmp --fusion_rule_file, --quant_fusion_rule_file, --close_fusion_rule_file',
                               'also applies in detail/mapping mode'):
                    self.assertIn(option, error.exception.message)

                # Ignoring an unused graph file does not relax CLI mutual exclusion.
                with self.assertRaises(CompareError) as error:
                    compare_rule_object.check_arguments_valid(validate_model_graph_file=False)
                self.assertEqual(error.exception.code, CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_parse_fusion_rule1(self):
        fusion_json_file_path = "/home/demo/1.json"
        quant_fusion_rule_file_path = "/home/demo/2.json"
        compare_rule_object = compare_rule.CompareRule(fusion_json_file_path, quant_fusion_rule_file_path)
        compare_data = mock.Mock(left_dump_info=mock.Mock(ffts=False))
        with mock.patch("os.path.getsize", return_value=100):
            with mock.patch('builtins.open', mock.mock_open(read_data=self._make_json())):
                compare_rule_object.parse_fusion_rule(compare_data)

    def test_parse_fusion_rule2(self):
        fusion_json_file_path = "/home/demo/1.json"
        quant_fusion_rule_file_path = ""
        compare_rule_object = compare_rule.CompareRule(fusion_json_file_path, quant_fusion_rule_file_path)
        compare_data = mock.Mock(left_dump_info=mock.Mock(ffts=False))
        with mock.patch("os.path.getsize", return_value=100):
            with mock.patch('builtins.open', mock.mock_open(read_data=self._make_json())):
                compare_rule_object.parse_fusion_rule(compare_data)

    def test_parse_fusion_rule3(self):
        fusion_json_file_path = ""
        quant_fusion_rule_file_path = "/home/demo/2.json"
        compare_rule_object = compare_rule.CompareRule(fusion_json_file_path, quant_fusion_rule_file_path)
        compare_data = mock.Mock(left_dump_info=mock.Mock(ffts=False))
        with mock.patch("os.path.getsize", return_value=100):
            with mock.patch('builtins.open', mock.mock_open(read_data=self._make_json())):
                compare_rule_object.parse_fusion_rule(compare_data)

    def test_parse_fusion_rule4(self):
        fusion_json_file_path = "/home/demo/2.json"
        quant_fusion_rule_file_path = ""
        close_fusion_rule_file_path = "/home/demo/3.json"
        compare_rule_object = compare_rule.CompareRule(fusion_json_file_path, quant_fusion_rule_file_path,
                                                       close_fusion_rule_file_path)
        compare_data = mock.Mock(left_dump_info=mock.Mock(ffts=False))
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
