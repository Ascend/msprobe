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
from unittest import mock
import uuid
import pytest
import json

from vector_cmp.fusion_manager import fusion_op
from cmp_utils import utils, utils_type, path_check
from vector_cmp.fusion_manager import fusion_rule_parser
from cmp_utils.constant.compare_error import CompareError
from dump_parse import dump, dump_utils, mapping


class TestUtilsMethods(unittest.TestCase):

    def test_analysis_fusion_rule1(self):
        with pytest.raises(CompareError) as error:
            parser = fusion_rule_parser.FusionRuleParser(
                '/home/resnet50.json')
            parser.analysis_fusion_rule()
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_OPEN_FILE_ERROR)

    def test_analysis_fusion_rule2(self):
        data = json.dumps({'name': 'resnet50'})
        with pytest.raises(CompareError) as error:
            with mock.patch("os.path.getsize", return_value=100):
                with mock.patch('builtins.open', mock.mock_open(read_data=data)):
                    parser = fusion_rule_parser.FusionRuleParser(
                        '/home/resnet50.json')
                    parser.analysis_fusion_rule()
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_PARSER_JSON_FILE_ERROR)

    def test_analysis_fusion_rule3(self):
        with pytest.raises(CompareError) as error:
            with mock.patch("os.path.getsize", return_value=100):
                with mock.patch('builtins.open', mock.mock_open(read_data=b'[{')):
                    parser = fusion_rule_parser.FusionRuleParser(
                        '/home/resnet50.json')
                    parser.analysis_fusion_rule()
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_PARSER_JSON_FILE_ERROR)

    def test_analysis_fusion_rule4(self):
        data = json.dumps({'name': 'resnet50', 'graph': {'name': 'merge1'}})
        with pytest.raises(CompareError) as error:
            with mock.patch("os.path.getsize", return_value=100):
                with mock.patch('builtins.open', mock.mock_open(read_data=data)):
                    parser = fusion_rule_parser.FusionRuleParser(
                        '/home/resnet50.json')
                    parser.analysis_fusion_rule()
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_PARSER_JSON_FILE_ERROR)

    def test_analysis_fusion_rule5(self):
        data = json.dumps({'name': 'resnet50', 'graph': [
            {'name': 'merge1', 'op': [{'name': 76, 'type': 'Data'}]}]})
        with pytest.raises(CompareError) as error:
            with mock.patch("os.path.getsize", return_value=100):
                with mock.patch('builtins.open', mock.mock_open(read_data=data)):
                    parser = fusion_rule_parser.FusionRuleParser(
                        '/home/resnet50.json')
                    parser.analysis_fusion_rule()
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_PARSER_JSON_FILE_ERROR)

    def test_analysis_fusion_rule6(self):
        data = json.dumps({'name': 'resnet50', 'graph': [
            {'name': 'merge1', 'op': [{'name': 'data', 'type': 'Data'}]}]})
        with mock.patch("os.path.getsize", return_value=100):
            with mock.patch('builtins.open', mock.mock_open(read_data=data)):
                parser = fusion_rule_parser.FusionRuleParser(
                    '/home/resnet50.json')
                parser.analysis_fusion_rule()
        self.assertEqual(len(parser.fusion_op_name_to_op_map), 1)
        op = parser.fusion_op_name_to_op_map['data'][0]
        self.assertEqual(op.op_name, 'data')
        self.assertEqual(op.op_id, 0)
        self.assertEqual(op.op_type, 'Data')
        self.assertEqual(len(op.input_list), 0)
        self.assertEqual(len(op.attr.original_op_names), 1)
        self.assertEqual(op.attr.original_op_names[0], 'data')
        self.assertEqual(len(op.output_desc), 1)

    def test_analysis_fusion_rule7(self):
        data = json.dumps({'name': 'resnet50', 'graph': [
            {'name': 'merge1', 'op':
                [{'name': 'dynamic_const_432', 'type': 'Const', "attr": [
                    {"key": "_datadump_original_op_names",
                     "value": {"list": {"val_type": 1}}}]}, ]}]})
        with mock.patch("os.path.getsize", return_value=100):
            with mock.patch('builtins.open', mock.mock_open(read_data=data)):
                parser = fusion_rule_parser.FusionRuleParser(
                    '/home/resnet50.json')
                parser.analysis_fusion_rule()
        self.assertEqual(len(parser.fusion_op_name_to_op_map), 1)
        op = parser.fusion_op_name_to_op_map['dynamic_const_432'][0]
        self.assertEqual(op.op_name, 'dynamic_const_432')
        self.assertEqual(op.op_id, 0)
        self.assertEqual(op.op_type, 'Const')
        self.assertEqual(len(op.input_list), 0)
        self.assertEqual(len(op.attr.original_op_names), 1)
        self.assertEqual(op.attr.original_op_names[0], '')
        self.assertEqual(len(op.output_desc), 0)

    def test_analysis_fusion_rule8(self):
        data = json.dumps({'name': 'resnet50', 'graph': [
            {'name': 'merge1',
             'op': [{'name': 'data', 'type': 'Data', 'input': 'data:0'}]}]})
        with pytest.raises(CompareError) as error:
            with mock.patch("os.path.getsize", return_value=100):
                with mock.patch('builtins.open', mock.mock_open(read_data=data)):
                    parser = fusion_rule_parser.FusionRuleParser(
                        '/home/resnet50.json')
                    parser.analysis_fusion_rule()
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_PARSER_JSON_FILE_ERROR)

    def test_analysis_fusion_rule9(self):
        data = json.dumps({'name': 'resnet50', 'graph': [
            {'name': 'merge1',
             'op': [{'name': 'datddda', 'type': 'Conv', 'input': ['data:0'],
                     'output_desc': [
                         {'attr': [{'key': '_datadump_origin_output_index',
                                    'value': {'i': 'xxx'}}]}]}]}]})
        with pytest.raises(CompareError) as error:
            with mock.patch("os.path.getsize", return_value=100):
                with mock.patch('builtins.open', mock.mock_open(read_data=data)):
                    parser = fusion_rule_parser.FusionRuleParser(
                        '/home/resnet50.json')
                    parser.analysis_fusion_rule()
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_PARSER_JSON_FILE_ERROR)

    def test_analysis_fusion_rule10(self):
        data = json.dumps({'name': 'resnet50', 'graph': [
            {'name': 'merge1',
             'op': [{'name': 'datddda', 'type': 'Conv', 'input': [''],
                     'output_desc': [
                         {'attr': [{'key': 'origin_format',
                                    'value': {'s': 'NCHW'}}]}]}]}]})
        with mock.patch("os.path.getsize", return_value=100):
            with mock.patch('builtins.open', mock.mock_open(read_data=data)):
                parser = fusion_rule_parser.FusionRuleParser(
                    '/home/resnet50.json')
                parser.analysis_fusion_rule()
        op = parser.fusion_op_name_to_op_map['datddda'][0]
        self.assertEqual(op.op_name, 'datddda')
        self.assertEqual(op.op_id, 0)
        self.assertEqual(op.op_type, 'Conv')
        self.assertEqual(len(op.input_list), 0)
        self.assertEqual(len(op.attr.original_op_names), 1)
        self.assertEqual(op.attr.original_op_names[0], 'datddda')
        self.assertEqual(len(op.output_desc), 1)
        self.assertEqual(op.output_desc[0].origin_name, 'datddda')
        self.assertEqual(op.output_desc[0].origin_format, 'NCHW')

    def test_analysis_fusion_rule11(self):
        data = json.dumps({'name': 'resnet50', 'graph': [
            {'name': 'merge1', 'op':
                [{'name': 'A1', 'type': 'XXX', "attr": [
                    {"key": "_L1_fusion_sub_graph_no",
                     "value": {"s": 'A'}}]},
                 {'name': 'A2', 'type': 'XXX', "attr": [
                     {"key": "_L1_fusion_sub_graph_no",
                      "value": {"s": 'A'}}]},
                 ]}]})
        with mock.patch("os.path.getsize", return_value=100):
            with mock.patch('builtins.open', mock.mock_open(read_data=data)):
                parser = fusion_rule_parser.FusionRuleParser(
                    '/home/resnet50.json')
                parser.analysis_fusion_rule()
        self.assertEqual(len(parser.fusion_op_name_to_op_map), 1)
        self.assertEqual(len(parser.fusion_op_name_to_op_map['A']), 2)
        op = parser.fusion_op_name_to_op_map['A'][1]
        self.assertEqual(op.op_name, 'A2')
        self.assertEqual(op.op_id, 0)
        self.assertEqual(op.op_type, 'XXX')
        self.assertEqual(len(op.input_list), 0)
        self.assertEqual(len(op.attr.original_op_names), 1)
        self.assertEqual(op.attr.original_op_names[0], 'A2')

    def test_analysis_fusion_rule12(self):
        data = json.dumps({'name': 'resnet50', 'graph': [
            {'name': 'merge1',
             'op': [
                 {'name': 'data', 'type': 'Data', 'input': ['data:0'], "attr": [
                     {"key": "_datadump_is_multiop",
                      "value": {"b": 'xx'}}]}, ]}]})

        with pytest.raises(CompareError) as error:
            with mock.patch("os.path.getsize", return_value=100):
                with mock.patch('builtins.open', mock.mock_open(read_data=data)):
                    parser = fusion_rule_parser.FusionRuleParser(
                        '/home/resnet50.json')
                    parser.analysis_fusion_rule()
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_PARSER_JSON_FILE_ERROR)

    def test_analysis_fusion_rule13(self):
        data = json.dumps({'name': 'resnet50', 'graph': [
            {'name': 'merge1', 'op':
                [{'name': 'conv1conv1_relu',
                  'type': 'Relu',
                  "attr": [
                      {"key": "_datadump_original_op_names",
                       "value": {"list": {"val_type": 1,
                                          "s": ["scale_conv1", "conv1",
                                                "bn_conv1", "conv1_relu"]}}}],
                  "input": [
                      "data:0",
                      "dynamic_const_471:0",
                      "dynamic_const_387:0"
                  ],
                  'output_desc': [
                      {'attr': [
                          {'key': '_datadump_origin_name',
                           'value': {'s': 'conv1_relu'}},
                          {'key': '_datadump_origin_output_index',
                           'value': {'i': 1}},
                          {'key': '_datadump_origin_format',
                           'value': {'s': 'NCHW'}},
                      ]},
                  ]
                  },
                 ]
             }]})
        with mock.patch("os.path.getsize", return_value=100):
            with mock.patch('builtins.open', mock.mock_open(read_data=data)):
                parser = fusion_rule_parser.FusionRuleParser(
                    '/home/resnet50.json')
                parser.analysis_fusion_rule()
        self.assertEqual(len(parser.fusion_op_name_to_op_map), 1)
        key = uuid.uuid3(uuid.NAMESPACE_DNS, ''.join(["scale_conv1", "conv1", "bn_conv1", "conv1_relu"]))
        op = parser.fusion_op_name_to_op_map[key][0]
        self.assertEqual(op.op_name, 'conv1conv1_relu')
        self.assertEqual(op.op_id, 0)
        self.assertEqual(op.op_type, 'Relu')
        self.assertEqual(len(op.input_list), 3)
        self.assertEqual(len(op.attr.original_op_names), 4)
        self.assertEqual(op.attr.original_op_names[3], 'conv1_relu')
        self.assertEqual(len(op.output_desc), 1)
        self.assertEqual(op.output_desc[0].origin_name, 'conv1_relu')
        self.assertEqual(op.output_desc[0].origin_format, 'NCHW')

    def test_analysis_fusion_rule14(self):
        data = json.dumps({'name': 'resnet50', 'graph': [
            {'name': 'merge1', 'op':
                [{'name': 'conv1conv1_relu',
                  'type': 'Relu',
                  "attr": [
                      {"key": "_datadump_original_op_names",
                       "value": {"list": {"val_type": 1,
                                          "s": ["scale_conv1", "conv1",
                                                "bn_conv1", "conv1_relu"]}}},
                      {"key": "use_cudnn_on_gpu",
                       "value": {"b": True}}
                  ],
                  "input": [
                      "data:0",
                      "dynamic_const_471:0",
                      "dynamic_const_387:0"
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
        with mock.patch("os.path.getsize", return_value=100):
            with mock.patch('builtins.open', mock.mock_open(read_data=data)):
                parser = fusion_rule_parser.FusionRuleParser(
                    '/home/resnet50.json')
                parser.analysis_fusion_rule()
        self.assertEqual(len(parser.fusion_op_name_to_op_map), 1)
        key = uuid.uuid3(uuid.NAMESPACE_DNS, ''.join(["scale_conv1", "conv1", "bn_conv1", "conv1_relu"]))
        op = parser.fusion_op_name_to_op_map[key][0]
        self.assertEqual(op.op_name, 'conv1conv1_relu')
        self.assertEqual(op.op_id, 0)
        self.assertEqual(op.op_type, 'Relu')
        self.assertEqual(len(op.input_list), 3)
        self.assertEqual(len(op.attr.original_op_names), 4)
        self.assertEqual(op.attr.original_op_names[3], 'conv1_relu')
        self.assertEqual(len(op.output_desc), 1)
        self.assertEqual(op.output_desc[0].origin_name, 'conv1_relu')
        self.assertEqual(op.output_desc[0].origin_format, 'NHWC')

    def test_analysis_fusion_rule15(self):
        data = json.dumps({'name': 'resnet50', 'graph': [
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
        with mock.patch("os.path.getsize", return_value=100):
            with mock.patch('builtins.open', mock.mock_open(read_data=data)):
                parser = fusion_rule_parser.FusionRuleParser(
                    '/home/resnet50.json')
                parser.analysis_fusion_rule()
        self.assertEqual(len(parser.fusion_op_name_to_op_map), 1)
        key = uuid.uuid3(uuid.NAMESPACE_DNS, ''.join(["scale_conv1", "conv1", "bn_conv1", "conv1_relu"]))
        op = parser.fusion_op_name_to_op_map[key][0]
        self.assertEqual(op.op_name, 'conv1conv1_relu')
        self.assertEqual(op.op_id, 0)
        self.assertEqual(op.op_type, 'Relu')
        self.assertEqual(len(op.input_list), 3)
        self.assertEqual(len(op.attr.original_op_names), 4)
        self.assertEqual(op.attr.original_op_names[3], 'conv1_relu')
        self.assertEqual(len(op.output_desc), 1)
        self.assertEqual(op.output_desc[0].origin_name, 'conv1_relu')
        self.assertEqual(op.output_desc[0].origin_format, 'NHWC')

    def test_get_fusion_op_list1(self):
        with pytest.raises(CompareError) as error:
            with mock.patch("os.path.getsize", return_value=100):
                with mock.patch('builtins.open',
                                mock.mock_open(read_data=self._make_json())):
                    parser = fusion_rule_parser.FusionRuleParser(
                        '/home/resnet50.json')
                    parser.analysis_fusion_rule()
                    parser.get_fusion_op_list('conv1conv1_relu2dd')
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_get_fusion_op_list2(self):
        with mock.patch("os.path.getsize", return_value=100):
            with mock.patch('builtins.open',
                            mock.mock_open(read_data=self._make_json())):
                parser = fusion_rule_parser.FusionRuleParser(
                    '/home/resnet50.json')
                parser.analysis_fusion_rule()
                fusion_op_list, fusion_op = parser.get_fusion_op_list(
                    'conv1conv1_relu2')
        self.assertEqual(len(fusion_op_list), 2)
        self.assertEqual(fusion_op.op_name, 'conv1conv1_relu2')

    def test_get_origin_tensor1(self):
        fusion_op_info = fusion_op.FusionOp(
            12, 'conv1colu', ['a:0'], 'Relu', [], fusion_op.OpAttr(['conv1', 'conelu'], '', False, 12))
        with pytest.raises(CompareError) as error:
            fusion_op_info.get_origin_tensor(0)
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR)

    def test_get_origin_tensor2(self):
        output_desc_list = []
        output_desc = fusion_op.OutputDesc('', 0, 'NCHW', [1, 3, 4, 4])
        output_desc_list.append(output_desc)
        fusion_op_info = fusion_op.FusionOp(
            12, 'conv1colu', ['a:0'], 'Relu', output_desc_list, fusion_op.OpAttr(['conv1', 'conelu'], '', False, 12))
        with pytest.raises(CompareError) as error:
            fusion_op_info.get_origin_tensor(1)
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR)

    def test_get_origin_tensor3(self):
        output_desc_list = []
        output_desc = fusion_op.OutputDesc('conv1_relu', None, 'NCHW', [1, 3, 33, 33])
        output_desc_list.append(output_desc)
        fusion_op_info = fusion_op.FusionOp(
            12, 'conv1colu', ['a:0'], 'Relu', output_desc_list, fusion_op.OpAttr(['conv1', 'conelu'], '', False, 12))
        origin_tensor = fusion_op_info.get_origin_tensor(2)
        self.assertEqual(origin_tensor.name, 'conv1_relu')
        self.assertEqual(origin_tensor.index, 2)
        self.assertEqual(origin_tensor.tensor_format, 'NCHW')
        self.assertEqual(origin_tensor.shape, [1, 3, 33, 33])

    def test_get_input_tensor1(self):
        output_desc_list = []
        output_desc = fusion_op.OutputDesc('', 0, 'NCHW', [1, 2, 3, 4])
        output_desc_list.append(output_desc)
        fusion_op_info = fusion_op.FusionOp(
            12, 'conv1colu', ['a:0'], 'Relu', output_desc_list, fusion_op.OpAttr(['conv1', 'conelu'], '', False, 12))
        with pytest.raises(CompareError) as error:
            fusion_op_info.get_input_tensor(1)
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INDEX_OUT_OF_BOUNDS_ERROR)

    def test_get_input_tensor2(self):
        output_desc_list = []
        output_desc = fusion_op.OutputDesc('', 0, 'NCHW', [1, 2, 3, 4])
        output_desc_list.append(output_desc)
        input_fusion_op = fusion_op.FusionOp(
            12, 'a', [], 'Relu', output_desc_list, fusion_op.OpAttr(['conv1', 'conelu'], '', True, 12))
        fusion_op_info = fusion_op.FusionOp(
            12, 'conv1colu', ['a:0'], 'Relu', output_desc_list, fusion_op.OpAttr(['conv1', 'conelu'], '', True, 12))
        with mock.patch(
                'vector_cmp.fusion_manager.fusion_rule_parser.FusionRuleParser.get_fusion_op_list',
                return_value=[[], input_fusion_op]):
            input_tensor = fusion_op_info.get_input_tensor(0)
        self.assertEqual(input_tensor.name, 'a')
        self.assertEqual(input_tensor.index, 0)

    def test_get_input_tensor3(self):
        output_desc_list = []
        output_desc = fusion_op.OutputDesc('', 0, 'NCHW', [1, 2, 3, 4])
        output_desc_list.append(output_desc)
        fusion_op_info = fusion_op.FusionOp(
            12, 'conv1colu', ['0'], 'Relu', output_desc_list, fusion_op.OpAttr(['conv1', 'conelu'], '', False, 12))
        with pytest.raises(CompareError) as error:
            fusion_op_info.get_input_tensor(0)
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_JSON_FILE_ERROR)

    def test_get_input_tensor4(self):
        output_desc_list = []
        output_desc = fusion_op.OutputDesc('', 0, 'NCHW', [1, 2, 3, 4])
        output_desc_list.append(output_desc)
        fusion_op_info = fusion_op.FusionOp(
            12, 'conv1colu', ['rrr:d0'], 'Relu', output_desc_list,
            fusion_op.OpAttr(['conv1', 'conelu'], '', False, 12))
        with pytest.raises(CompareError) as error:
            fusion_op_info.get_input_tensor(0)
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_JSON_FILE_ERROR)

    def test_get_input_tensor5(self):
        output_desc_list = []
        output_desc = fusion_op.OutputDesc('', 0, 'NCHW', [1, 2, 3, 4])
        output_desc_list.append(output_desc)
        fusion_op_info = fusion_op.FusionOp(
            12, 'conv1colu', [':d0'], 'Relu', output_desc_list, fusion_op.OpAttr(['conv1', 'conelu'], '', False, 12))
        with pytest.raises(CompareError) as error:
            fusion_op_info.get_input_tensor(0)
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_JSON_FILE_ERROR)

    def test_make_right_to_left_multi_map(self):
        fusion_op_list = [
            fusion_op.FusionOp(0, 'C', ['a:0'], 'Relu', [], fusion_op.OpAttr(['A', 'B'], '', True, 0)),
            fusion_op.FusionOp(0, 'D', ['a:0'], 'Relu', [], fusion_op.OpAttr(['A', 'B'], '', True, 0)),
            fusion_op.FusionOp(1, 'F', ['a:0'], 'Relu', [], fusion_op.OpAttr(['G', 'H', 'I'], '', False, 1)),
            fusion_op.FusionOp(2, 'E', ['a:0'], 'Relu', [], fusion_op.OpAttr(['E'], '', False, 1))]
        right_to_left_map = fusion_rule_parser.make_right_to_left_multi_map(fusion_op_list)
        self.assertEqual(len(right_to_left_map), 3)
        self.assertEqual(len(right_to_left_map['A,B']), 2)
        self.assertEqual(len(right_to_left_map['G,H,I']), 1)
        self.assertEqual(len(right_to_left_map['E']), 1)

    def test_make_left_and_right_string(self):
        fusion_op_list = [
            fusion_op.FusionOp(0, 'C', ['a:0'], 'Relu', [], fusion_op.OpAttr(['A', 'B'], '', True, 0)),
            fusion_op.FusionOp(0, 'D', ['a:0'], 'Relu', [], fusion_op.OpAttr(['A', 'B'], '', True, 0))]
        right_to_left_map = fusion_rule_parser.make_right_to_left_multi_map(fusion_op_list)
        left_ops_str, right_ops_str = \
            fusion_rule_parser.make_left_and_right_string(right_to_left_map)
        self.assertEqual(left_ops_str, 'C,D')
        self.assertEqual(right_ops_str, 'A,B')

    def test_get_relation_for_fusion1(self):
        fusion_op_list = [
            fusion_op.FusionOp(0, 'C', ['a:0'], 'Relu', [], fusion_op.OpAttr(['A', 'B'], '', True, 0)),
            fusion_op.FusionOp(0, 'D', ['a:0'], 'Relu', [], fusion_op.OpAttr(['A', 'B'], '', True, 0))]
        relation = fusion_rule_parser.get_relation_for_fusion(fusion_op_list)
        self.assertEqual(relation, utils_type.FusionRelation.MultiToMulti)

    def test_get_relation_for_fusion2(self):
        fusion_op_list = [
            fusion_op.FusionOp(0, 'C', ['a:0'], 'Relu', [], fusion_op.OpAttr(['A'], '', False, 0)),
            fusion_op.FusionOp(0, 'D', ['a:0'], 'Relu', [], fusion_op.OpAttr(['A'], '', False, 0))]
        relation = fusion_rule_parser.get_relation_for_fusion(fusion_op_list)
        self.assertEqual(relation, utils_type.FusionRelation.OneToMulti)

    def test_get_relation_for_fusion3(self):
        fusion_op_list = [
            fusion_op.FusionOp(0, 'A', ['a:0'], 'Relu', [], fusion_op.OpAttr(['A'], '', False, 0))]
        relation = fusion_rule_parser.get_relation_for_fusion(fusion_op_list)
        self.assertEqual(relation, utils_type.FusionRelation.OneToOne)

    def test_get_relation_for_fusion4(self):
        fusion_op_list = [
            fusion_op.FusionOp(0, 'A', ['a:0'], 'Relu', [], fusion_op.OpAttr(['A', 'B'], '', False, 0))]
        relation = fusion_rule_parser.get_relation_for_fusion(fusion_op_list)
        self.assertEqual(relation, utils_type.FusionRelation.MultiToOne)

    def test_get_relation_for_fusion5(self):
        fusion_op_list = [
            fusion_op.FusionOp(0, 'A1', ['a:0'], 'Relu', [], fusion_op.OpAttr(['A'], '1', True, 0)),
            fusion_op.FusionOp(0, 'A2', ['a:0'], 'Relu', [], fusion_op.OpAttr(['A'], '1', True, 0))]
        relation = fusion_rule_parser.get_relation_for_fusion(fusion_op_list)
        self.assertEqual(relation, utils_type.FusionRelation.L1Fusion)

    def test_get_origin_name_to_op_name_map(self):
        with mock.patch("os.path.getsize", return_value=100):
            with mock.patch('builtins.open',
                            mock.mock_open(read_data=self._make_json())):
                parser = fusion_rule_parser.FusionRuleParser(
                    '/home/resnet50.json')
                parser.analysis_fusion_rule()
                map1 = parser.get_origin_name_to_op_name_map()
        self.assertEqual(map1.get('bn_conv1'), 'conv1conv1_relu2')

    def test_make_open_fusion_original_op_names1(self):
        op1 = fusion_op.FusionOp(0, 'A1', ['a:0'], 'Relu', [], fusion_op.OpAttr([''], '1', True, 0))
        value = fusion_rule_parser._make_open_fusion_original_op_names(op1, {})
        self.assertEqual(value, [''])

    def test_make_open_fusion_original_op_names2(self):
        op1 = fusion_op.FusionOp(0, 'A1', ['a:0'], 'Relu', [], fusion_op.OpAttr(['A', 'B', 'C'], '1', True, 0))
        value = fusion_rule_parser._make_open_fusion_original_op_names(op1, {'A': 'A1', 'B': 'BIN'})
        self.assertEqual(len(value), 2)
        self.assertTrue('A1' in value)
        self.assertTrue('BIN' in value)

    def test_get_close_fusion_origin_output_index1(self):
        output_desc = fusion_op.OutputDesc('B', 1, 'NCHW', [1, 3, 4, 4])
        op_list = [fusion_op.FusionOp(0, 'B2', ['a:0'], 'Relu', [output_desc],
                                      fusion_op.OpAttr(['A', 'B', 'C'], '1', True, 0))]
        value = fusion_rule_parser._get_close_fusion_origin_output_index(op_list, output_desc)
        self.assertTrue(value)
        self.assertEqual(0, output_desc.origin_output_index)

    def test_get_close_fusion_origin_output_index2(self):
        op_list = [fusion_op.FusionOp(0, 'B2', ['a:0'], 'Relu', [fusion_op.OutputDesc('A', 1, 'NCHW', [1, 3, 4, 4])],
                                      fusion_op.OpAttr(['A', 'B', 'C'], '1', True, 0))]
        value = fusion_rule_parser._get_close_fusion_origin_output_index(op_list, fusion_op.OutputDesc('B', 1, 'NCHW',
                                                                                                       [1, 3, 4, 4]))
        self.assertFalse(value)

    def test_make_open_fusion_output_desc1(self):
        output_desc_list = [fusion_op.OutputDesc('B', 1, 'NCHW', [1, 3, 4, 4]),
                            fusion_op.OutputDesc('', 0, 'NCHW', [1, 3, 4, 4]),
                            fusion_op.OutputDesc('C', 0, 'NCHW', [1, 3, 4, 4])]
        op1 = fusion_op.FusionOp(0, 'B2', ['a:0'], 'Relu', output_desc_list,
                                 fusion_op.OpAttr(['A', 'B', 'C'], '1', True, 0))
        close_fusion_rule = mock.Mock()
        close_fusion_rule.fusion_op_name_to_op_map = {
            'CIn': [fusion_op.FusionOp(0, 'CIn', ['a:0'], 'Relu', output_desc_list,
                                       fusion_op.OpAttr(['C'], '1', True, 0))]}
        close_fusion_rule.op_name_to_fusion_op_name_map = {'CIn': 'CIn'}
        fusion_rule_parser._make_open_fusion_output_desc(op1, {'C': 'CIn'}, close_fusion_rule)
        self.assertEqual('CIn', op1.output_desc[2].origin_name)

    def test_merge_close_and_open_fusion_rule(self):
        close_fusion_rule = mock.Mock()
        close_op = fusion_op.FusionOp(0, 'CIn', ['a:0'], 'Relu', [fusion_op.OutputDesc('C', 1, 'NCHW', [1, 3, 4, 4])],
                                      fusion_op.OpAttr(['C'], '1', True, 0))
        close_fusion_rule.fusion_op_name_to_op_map = {'CIn': [close_op]}
        close_fusion_rule.op_name_to_fusion_op_name_map = {'CIn': 'CIn'}
        close_fusion_rule.get_origin_name_to_op_name_map = mock.Mock(return_value={'C': 'CIn'})
        open_fusion_rule = mock.Mock()
        open_op = fusion_op.FusionOp(0, 'BCIn', ['a:0'], 'Relu', [fusion_op.OutputDesc('C', 1, 'NCHW', [1, 3, 4, 4])],
                                     fusion_op.OpAttr(['B', 'C'], '1', True, 0))
        open_fusion_rule.fusion_op_name_to_op_map = {'CIn': [open_op]}
        open_fusion_rule.op_name_to_fusion_op_name_map = {'CIn': 'CIn'}
        fusion_rule_parser.merge_close_and_open_fusion_rule(open_fusion_rule, close_fusion_rule)
        self.assertEqual(['CIn'], open_op.attr.original_op_names)
        self.assertEqual('CIn', open_op.output_desc[0].origin_name)
        self.assertEqual(0, open_op.output_desc[0].origin_output_index)

    def test_merge_fusion_rule(self):
        offline_json = json.dumps({'graph': [{'op': [
            {'name': 'data', 'type': 'Data',
             'output_desc': [{'attr': [{'key': 'layout', 'value': 'NCHW'}]}]},
            {'name': 'res2a_branch1_quant_layer', 'type': 'AscendQuant',
             'output_desc': [
                 {'attr': [{'key': 'layout', 'value': 'NCHW'}]}, ],
             },
            {'name': 'res2a_branch1res2a_branch1_dequant_layer',
             'type': 'AscendDequant', 'attr': [{
                'key': '_datadump_original_op_names',
                'value': {'list': {'val_type': 1,
                                   's': ['res2a_branch1',
                                         'res2a_branch1_dequant_layer']}}}, ],
             'output_desc': [
                 {'attr': [
                     {'key': '_datadump_origin_name',
                      'value': {'s': 'res2a_branch1_dequant_layer'}},
                     {'key': '_datadump_origin_output_index',
                      'value': {'i': 0}},
                     {'key': '_datadump_origin_format',
                      'value': {'s': 'NHWC'}},
                 ]}, ]
             },
        ]}]})
        with mock.patch("os.path.getsize", return_value=100):
            with mock.patch('builtins.open',
                            mock.mock_open(read_data=offline_json)):
                offline_fusion_rule = fusion_rule_parser.FusionRuleParser(
                    '/home/resnet50.json')
                offline_fusion_rule.analysis_fusion_rule()
        quant_json = json.dumps({'graph': [{'op': [
            {'name': 'data', 'type': 'Input'},
            {'name': 'res2a_branch1_quant_layer', 'type': 'Quant', 'attr': [{
                'key': '_datadump_original_op_names',
                'value': {'list': {'val_type': 1,
                                   's': ['res2a_branch1', 'bn2a_branch1',
                                         'scale2a_branch1']}}
            }]},
            {'name': 'res2a_branch1', 'type': 'Convolution', 'attr': [{
                'key': '_datadump_original_op_names',
                'value': {'list': {'val_type': 1,
                                   's': ['res2a_branch1', 'bn2a_branch1',
                                         'scale2a_branch1']}}
            }]},
            {'name': 'res2a_branch1_dequant_layer', 'type': 'DeQuant',
             'attr': [{
                 'key': '_datadump_original_op_names',
                 'value': {'list': {'val_type': 1,
                                    's': ['res2a_branch1', 'bn2a_branch1',
                                          'scale2a_branch1']}}}],
             'output_desc': [
                 {'attr': [
                     {'key': '_datadump_origin_name',
                      'value': {'s': 'scale2a_branch1'}},
                     {'key': '_datadump_origin_output_index',
                      'value': {'i': 0}},
                     {'key': '_datadump_origin_format',
                      'value': {'s': 'NHWC'}},
                 ]}, ]
             }
        ]}]})
        with mock.patch("os.path.getsize", return_value=100):
            with mock.patch('builtins.open',
                            mock.mock_open(read_data=quant_json)):
                quant_fusion_rule = fusion_rule_parser.FusionRuleParser(
                    '/home/resnet50.json')
                quant_fusion_rule.analysis_fusion_rule()
        fusion_rule = fusion_rule_parser.merge_fusion_rule(offline_fusion_rule,
                                                           quant_fusion_rule)
        self.assertEqual(len(fusion_rule.fusion_op_name_to_op_map), 2)
        key = fusion_rule.op_name_to_fusion_op_name_map[
            'res2a_branch1_quant_layer']
        self.assertEqual(len(fusion_rule.fusion_op_name_to_op_map[key]), 2)
        op = fusion_rule.fusion_op_name_to_op_map[key][0]
        self.assertEqual(op.op_name, 'res2a_branch1_quant_layer')
        self.assertEqual(op.op_id, 1)
        self.assertEqual(op.op_type, 'AscendQuant')
        self.assertEqual(len(op.attr.original_op_names), 3)
        self.assertEqual(len(op.output_desc), 1)
        self.assertEqual(op.output_desc[0].origin_name, '')
        self.assertEqual(op.output_desc[0].origin_format, '')

    def test_is_inner_node1(self):
        fusion_op_info = fusion_op.FusionOp(
            12, 'conv1colu', ['a:0'], 'Relu', [], fusion_op.OpAttr(['conv1', 'conelu'], '', False, 0))
        self.assertEqual(False, fusion_op_info.is_inner_node())

    def test_is_inner_node2(self):
        output_desc_list = []
        output_desc = fusion_op.OutputDesc('', 0, 'NCHW', [1, 3, 4, 4])
        output_desc_list.append(output_desc)
        fusion_op_info = fusion_op.FusionOp(
            12, 'conv1colu', ['a:0'], 'Relu', output_desc_list, fusion_op.OpAttr(['conv1', 'conelu'], '', True, 12))
        self.assertEqual(True, fusion_op_info.is_inner_node())

    def test_is_inner_node3(self):
        output_desc_list = []
        output_desc = fusion_op.OutputDesc('xxx', 0, 'NCHW', [1, 3, 4, 4])
        output_desc_list.append(output_desc)
        fusion_op_info = fusion_op.FusionOp(
            12, 'conv1colu', ['a:0'], 'Relu', output_desc_list, fusion_op.OpAttr(['conv1', 'conelu'], '', True, 12))
        self.assertEqual(False, fusion_op_info.is_inner_node())

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
                                                "bn_conv1", "conv1_relu"]}}},
                      {"key": "use_cudnn_on_gpu",
                       "value": {"b": True}}
                  ],
                  "input": [
                      "data:0",
                      "dynamic_const_471:0",
                      "dynamic_const_387:0"
                  ],
                  'output_desc': [
                      {'attr': [
                          {'key': '_datadump_origin_name',
                           'value': {'s': 'conv1_relu'}},
                          {'key': '_datadump_origin_output_index',
                           'value': {'i': 0}},
                          {'key': '_datadump_origin_format',
                           'value': {'s': 'NHWC'}},
                      ]},
                  ]
                  },
                 {'name': 'conv1conv1_relu2',
                  'type': 'Relu',
                  "attr": [
                      {"key": "_datadump_original_op_names",
                       "value": {"list": {"val_type": 1,
                                          "s": ["scale_conv1", "conv1",
                                                "bn_conv1", "conv1_relu"]}}},
                  ],
                  "input": [
                      "data:0",
                      "dynamic_const_471:0",
                      "dynamic_const_387:0"
                  ],
                  'output_desc': [
                      {'attr': [
                          {'key': '_datadump_origin_name',
                           'value': {'s': 'conv1_relu'}},
                          {'key': '_datadump_origin_output_index',
                           'value': {'i': 1}},
                          {'key': '_datadump_origin_format',
                           'value': {'s': 'NHWC'}},
                          {"key": "origin_shape",
                           "value": {"list": {"val_type": 1,
                                              "i": [1, 3, 4, 5]}}},
                      ]},
                  ]
                  },
                 {'name': 'dynamic_const_432', 'type': 'Const', "attr": [
                     {"key": "_datadump_original_op_names",
                      "value": {"list": {"val_type": 1}}}]},
                 {'name': 'prob', 'type': 'Const'},
                 ]
             }]})


if __name__ == '__main__':
    unittest.main()
