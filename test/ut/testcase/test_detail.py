import json

import unittest
import pytest
import dump_data_pb2 as DD
from unittest import mock

from vector_cmp.compare_detail import detail
from vector_cmp.fusion_manager import fusion_rule_parser
from cmp_utils.constant.compare_error import CompareError
from cmp_utils.constant.const_manager import ConstManager


class TestUtilsMethods(unittest.TestCase):
    def test_tensor_id(self):
        with pytest.raises(CompareError) as error:
            detail.TensorId('xx', 'xxx', 'xx')
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_tensor_id_check_arguments_valid1(self):
        with pytest.raises(CompareError) as error:
            tensor_id = detail.TensorId('!!@##$', 'input', '0')
            tensor_id.check_arguments_valid()
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_tensor_id_check_arguments_valid2(self):
        with pytest.raises(CompareError) as error:
            tensor_id = detail.TensorId('xx', 'xx', '0')
            tensor_id.check_arguments_valid()
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_get_tensor_id(self):
        tensor_id = detail.TensorId('xx', 'input', '0')
        tensor_id.check_arguments_valid()
        self.assertEqual('xx:input:0', tensor_id.get_tensor_id())

    def test_is_input1(self):
        tensor_id = detail.TensorId('xx', 'input', '0')
        ret = tensor_id.is_input()
        self.assertEqual(ret, True)

    def test_is_input2(self):
        tensor_id = detail.TensorId('xx', 'output', '0')
        ret = tensor_id.is_input()
        self.assertEqual(ret, False)

    def test_get_file_prefix(self):
        tensor_id = detail.TensorId('A/B/C', 'output', '3')
        self.assertEqual('A_B_C_output_3', tensor_id.get_file_prefix())

    def test_check_arguments_valid1(self):
        tensor_id = detail.TensorId('xx', 'input', '0')
        detail_info = detail.DetailInfo(tensor_id, -1, True, ConstManager.MAX_DETAIL_INFO_LINE_COUNT)
        with pytest.raises(CompareError) as error:
            detail_info.check_arguments_valid()
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_INDEX_OUT_OF_BOUNDS_ERROR)

    def test_check_arguments_valid2(self):
        tensor_id = detail.TensorId('xx', 'input', '1')
        detail_info = detail.DetailInfo(tensor_id, 10200, True, ConstManager.MAX_DETAIL_INFO_LINE_COUNT)
        with pytest.raises(CompareError) as error:
            detail_info.check_arguments_valid()
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_INDEX_OUT_OF_BOUNDS_ERROR)

    def test_set_detail_format1(self):
        tensor_id = detail.TensorId('prob', 'output', '3')
        detail_info = detail.DetailInfo(tensor_id, 10, True, ConstManager.MAX_DETAIL_INFO_LINE_COUNT)
        detail_info.set_detail_format('(1,3,224,224)', DD.FORMAT_HWCN, 'NCHW')
        self.assertEqual(detail_info.detail_format, 'N C H W')
        self.assertEqual(detail_info.make_detail_header(),
                         'Index,N C H W,NPUDump,GroundTruth,AbsoluteError,RelativeError\n')

    def test_set_detail_format2(self):
        tensor_id = detail.TensorId('prob', 'output', '3')
        detail_info = detail.DetailInfo(tensor_id, 10, True, ConstManager.MAX_DETAIL_INFO_LINE_COUNT)
        detail_info.set_detail_format('()', DD.FORMAT_NCHW, 'ND')
        self.assertEqual(detail_info.detail_format, 'ID')
        self.assertEqual(detail_info.make_detail_header(),
                         'Index,ID,NPUDump,GroundTruth,AbsoluteError,RelativeError\n')

    def test_set_detail_format3(self):
        tensor_id = detail.TensorId('prob', 'output', '3')
        detail_info = detail.DetailInfo(tensor_id, 10, True, ConstManager.MAX_DETAIL_INFO_LINE_COUNT)
        detail_info.set_detail_format('(1,3,4,5,4,4,4)', DD.FORMAT_NDHWC, 'NDHWC')
        self.assertEqual(detail_info.detail_format, 'N C D H W')
        self.assertEqual(detail_info.make_detail_header(),
                         'Index,N C D H W,NPUDump,GroundTruth,AbsoluteError,RelativeError\n')

    def test_set_detail_format4(self):
        tensor_id = detail.TensorId('prob', 'output', '3')
        detail_info = detail.DetailInfo(tensor_id, 10, True, ConstManager.MAX_DETAIL_INFO_LINE_COUNT)
        detail_info.set_detail_format('(1,3,4,5,4)', DD.FORMAT_FRACTAL_Z, 'HWCN')
        self.assertEqual(detail_info.detail_format, 'H W C N')
        self.assertEqual(detail_info.make_detail_header(),
                         'Index,H W C N,NPUDump,GroundTruth,AbsoluteError,RelativeError\n')

    def test_get_detail_op1(self):
        tensor_id = detail.TensorId('dynamic_const_471', 'output', '0')
        detail_info = detail.DetailInfo(tensor_id, 10, False, ConstManager.MAX_DETAIL_INFO_LINE_COUNT)
        with mock.patch("os.path.getsize", return_value=100):
            fusion_op, _ = detail_info.get_detail_op(self._make_fusion_rule())
            detail_info.set_detail_format('()', DD.FORMAT_ND, 'ND')
        self.assertEqual(detail_info.my_output_ops, 'dynamic_const_471')
        self.assertEqual(detail_info.ground_truth_ops, '*')
        self.assertEqual('NPUDump:dynamic_const_471\nGroundTruth:*\nFormat:ID\n',
                         detail_info.get_detail_info())

    def test_get_detail_op2(self):
        tensor_id = detail.TensorId('conv1conv1_relu', 'output', '0')
        detail_info = detail.DetailInfo(tensor_id, 10, False, ConstManager.MAX_DETAIL_INFO_LINE_COUNT)
        with mock.patch("os.path.getsize", return_value=100):
            fusion_op, _ = detail_info.get_detail_op(self._make_fusion_rule())
            detail_info.set_detail_format('(1,3,4,5,4,4,4)', DD.FORMAT_NDHWC, 'NDHWC')
        self.assertEqual(detail_info.my_output_ops, 'conv1conv1_relu')
        self.assertEqual(detail_info.ground_truth_ops,
                         'scale_conv1 conv1 bn_conv1 conv1_relu')
        self.assertEqual('NPUDump:conv1conv1_relu\nGroundTruth:scale_conv1 conv1 bn_conv1 conv1_relu\nFormat:N C D H W\n',
                         detail_info.get_detail_info())

    @staticmethod
    def _make_fusion_rule():
        offline_json = json.dumps({'name': 'resnet50', 'graph': [
            {'name': 'merge1', 'op':
                [{'name': 'data',
                  'type': 'Input',
                  "attr": [
                      {"key": "xxx",
                       "value": 'xxx'},
                  ],
                  'output_desc': [
                      {'attr': [
                          {'key': 'xxx',
                           'value': 'xxx'},
                      ]},
                  ]
                  },
                 {'name': 'dynamic_const_471', 'type': 'Const', "attr": [
                     {"key": "_datadump_original_op_names",
                      "value": {"list": {"val_type": 1}}}]},
                 {'name': 'dynamic_const_387', 'type': 'Const', "attr": [
                     {"key": "_datadump_original_op_names",
                      "value": {"list": {"val_type": 1}}}]},
                 {'name': 'conv1conv1_relu',
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
                      "dynamic_const_387:0"
                  ],
                  'output_desc': [
                      {'attr': [
                          {'key': '_datadump_origin_name',
                           'value': {'s': 'conv1_relu'}},
                          {'key': '_datadump_origin_output_index',
                           'value': {'i': 0}},
                          {'key': '_datadump_origin_format',
                           'value': {'s': 'NCHW'}},
                      ]},
                  ]
                  },

                 ],
             },
        ]})
        with mock.patch('builtins.open',
                        mock.mock_open(read_data=offline_json)):
            offline_fusion_rule = fusion_rule_parser.FusionRuleParser(
                '/home/resnet50.json')
            offline_fusion_rule.analysis_fusion_rule()
        return offline_fusion_rule
