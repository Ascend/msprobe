import json
import unittest
from unittest import mock
import pytest
from src.compare.cmp_utils import utils
from src.compare.vector_cmp.fusion_manager import quant_filter
from src.compare.vector_cmp.fusion_manager import fusion_op


class TestUtilsMethods(unittest.TestCase):

    def test_check_name_type(self):
        op_list = mock.Mock
        filtering = quant_filter.QuantFilter(op_list)
        name_type = filtering._check_name_type('resnet_v1_50/pool1/MaxPoolresnet_v1_50/block1/unit_1/'
            'bottleneck_v1/shortcut/act_quant/AscendQuant')
        self.assertEqual(name_type, quant_filter.QuantFilter.QUANT_OP)
        name_type = filtering._check_name_type('pool1res2a_branch1_0_quant_layer')
        self.assertEqual(name_type, quant_filter.QuantFilter.QUANT_OP)
        name_type = filtering._check_name_type('resnet_v1_50/logits/Conv2Dresnet_v1_50/logits/dequant/AscendDequant')
        self.assertEqual(name_type, quant_filter.QuantFilter.DEQUANT_OP)
        name_type = filtering._check_name_type('res2a_branch2cres2a_branch2c_dequant_layer')
        self.assertEqual(name_type, quant_filter.QuantFilter.DEQUANT_OP)
        name_type = filtering._check_name_type('resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/Conv2Dresnet_v1_50/'
                                               'block4/unit_3/bottleneck_v1/conv1/dequant/AscendDequantresnet_v1_50'
                                               '/block4/unit_3/bottleneck_v1/conv2/act_quant/AscendQuant')
        self.assertEqual(name_type, quant_filter.QuantFilter.QUANT_DEQUANT_OP)
        name_type = filtering._check_name_type('res2a_branch2ares2a_branch2a_dequant_layerres2a_branch2b_0_quant_layer')
        self.assertEqual(name_type, quant_filter.QuantFilter.QUANT_DEQUANT_OP)
        name_type = filtering._check_name_type('trans_TransData_1')
        self.assertEqual(name_type, quant_filter.QuantFilter.NORMAL_OP)

    def test_check_in_pairs(self):
        op_list = mock.Mock
        filtering = quant_filter.QuantFilter(op_list)
        filtering._op_output_type_map = {
            'node1': [quant_filter.QuantFilter.QUANT_OP],
            'node2': [quant_filter.QuantFilter.DEQUANT_OP],
            'node3': [quant_filter.QuantFilter.NORMAL_OP],
            'node4': [quant_filter.QuantFilter.MIDDLE_OP],
            'node5': [quant_filter.QuantFilter.QUANT_OP, quant_filter.QuantFilter.DEQUANT_OP]
        }

        op = mock.Mock
        op.input_list = ['']
        in_pair = filtering._check_in_pairs(op)
        self.assertEqual(in_pair, False)

        op.input_list = ['node1:0']
        in_pair = filtering._check_in_pairs(op)
        self.assertEqual(in_pair, True)

        op.input_list = ['node2:0']
        in_pair = filtering._check_in_pairs(op)
        self.assertEqual(in_pair, False)

        op.input_list = ['node3:0']
        in_pair = filtering._check_in_pairs(op)
        self.assertEqual(in_pair, False)

        op.input_list = ['node4:0']
        in_pair = filtering._check_in_pairs(op)
        self.assertEqual(in_pair, True)

        op.input_list = ['node5:0']
        in_pair = filtering._check_in_pairs(op)
        self.assertEqual(in_pair, True)

        op.input_list = ['node5:1']
        in_pair = filtering._check_in_pairs(op)
        self.assertEqual(in_pair, False)

        op.input_list = ['node3:0', 'node5:0']
        in_pair = filtering._check_in_pairs(op)
        self.assertEqual(in_pair, True)

        op.input_list = ['node3:0', 'node5:1']
        in_pair = filtering._check_in_pairs(op)
        self.assertEqual(in_pair, False)

    def test_check_out_type(self):
        op_list = mock.Mock
        filtering = quant_filter.QuantFilter(op_list)
        output_node = mock.Mock

        param_result_list = [
            ("", quant_filter.QuantFilter.QUANT_OP, False, 'DT_INT8', quant_filter.QuantFilter.QUANT_OP),
            ("", quant_filter.QuantFilter.QUANT_OP, True, 'DT_INT8', quant_filter.QuantFilter.QUANT_OP),
            ("", quant_filter.QuantFilter.DEQUANT_OP, False, 'DT_INT8', quant_filter.QuantFilter.NORMAL_OP),
            ("", quant_filter.QuantFilter.DEQUANT_OP, True, 'DT_INT8', quant_filter.QuantFilter.MIDDLE_OP),
            ("", quant_filter.QuantFilter.QUANT_DEQUANT_OP, False, 'DT_INT8', quant_filter.QuantFilter.QUANT_OP),
            ("", quant_filter.QuantFilter.QUANT_DEQUANT_OP, True, 'DT_INT8', quant_filter.QuantFilter.QUANT_OP),
            ("", quant_filter.QuantFilter.NORMAL_OP, False, 'DT_INT8', quant_filter.QuantFilter.NORMAL_OP),
            ("", quant_filter.QuantFilter.NORMAL_OP, True, 'DT_INT8', quant_filter.QuantFilter.MIDDLE_OP),
            ("", quant_filter.QuantFilter.QUANT_OP, False, 'DT_FLOAT16', quant_filter.QuantFilter.NORMAL_OP),
            ("", quant_filter.QuantFilter.QUANT_OP, True, 'DT_FLOAT16', quant_filter.QuantFilter.MIDDLE_OP),
            ("", quant_filter.QuantFilter.DEQUANT_OP, False, 'DT_FLOAT16', quant_filter.QuantFilter.DEQUANT_OP),
            ("", quant_filter.QuantFilter.DEQUANT_OP, True, 'DT_FLOAT16', quant_filter.QuantFilter.DEQUANT_OP),
            ("", quant_filter.QuantFilter.QUANT_DEQUANT_OP, False, 'DT_FLOAT16', quant_filter.QuantFilter.DEQUANT_OP),
            ("", quant_filter.QuantFilter.QUANT_DEQUANT_OP, True, 'DT_FLOAT16', quant_filter.QuantFilter.DEQUANT_OP),
            ("", quant_filter.QuantFilter.NORMAL_OP, False, 'DT_FLOAT16', quant_filter.QuantFilter.NORMAL_OP),
            ("", quant_filter.QuantFilter.NORMAL_OP, True, 'DT_FLOAT16', quant_filter.QuantFilter.MIDDLE_OP),
            ("", quant_filter.QuantFilter.NORMAL_OP, False, 'DT_INT32', quant_filter.QuantFilter.NORMAL_OP),
            ("", quant_filter.QuantFilter.NORMAL_OP, True, 'DT_INT32', quant_filter.QuantFilter.MIDDLE_OP),
            ("", quant_filter.QuantFilter.QUANT_OP, False, 'DT_INT32', quant_filter.QuantFilter.NORMAL_OP),
            ("", quant_filter.QuantFilter.QUANT_OP, True, 'DT_INT32', quant_filter.QuantFilter.MIDDLE_OP),
            ("", quant_filter.QuantFilter.DEQUANT_OP, False, 'DT_INT32', quant_filter.QuantFilter.NORMAL_OP),
            ("", quant_filter.QuantFilter.DEQUANT_OP, True, 'DT_INT32', quant_filter.QuantFilter.MIDDLE_OP),
            ("", quant_filter.QuantFilter.QUANT_DEQUANT_OP, False, 'DT_INT32', quant_filter.QuantFilter.NORMAL_OP),
            ("", quant_filter.QuantFilter.QUANT_DEQUANT_OP, True, 'DT_INT32', quant_filter.QuantFilter.MIDDLE_OP)
        ]

        for i, param_result in enumerate(param_result_list):
            output_node.data_type = param_result[3]
            out_type = filtering._check_out_type(param_result[0], param_result[1], param_result[2], output_node)
            self.assertEqual(out_type, param_result[4], '{} th case not equal.'.format(i+1))

    def test_add_filter_list(self):
        op = mock.Mock
        op.attr = fusion_op.OpAttr('', 0, '', [])

        filtering = quant_filter.QuantFilter(mock.Mock)
        filtering._op_output_type_map = {
            'node1': [quant_filter.QuantFilter.QUANT_OP],
            'node2': [quant_filter.QuantFilter.DEQUANT_OP],
            'node3': [quant_filter.QuantFilter.NORMAL_OP],
            'node4': [quant_filter.QuantFilter.MIDDLE_OP],
            'node5': [quant_filter.QuantFilter.QUANT_OP, quant_filter.QuantFilter.DEQUANT_OP],
        }

        param_result_list = [
            ('node1', False, False),
            ('node1', True, True),
            ('node2', False, False),
            ('node2', True, False),
            ('node3', False, False),
            ('node3', True, True),
            ('node4', False, False),
            ('node4', True, True),
            ('node5', False, False),
            ('node5', True, True),
        ]

        for i, param_result in enumerate(param_result_list):
            op.op_name = param_result[0]
            filtering._add_filter_list(op, param_result[1])
            self.assertEqual(op.attr.quant_filter, param_result[2])

    def test_process_filtering(self):
        op_list, filter_result_list = self._make_op_list()
        self.assertEqual(len(op_list), len(filter_result_list))
        filtering = quant_filter.QuantFilter(op_list)
        filtering.process_filtering()
        for i, op, expect_result in zip(range(len(op_list)), op_list, filter_result_list):
            self.assertEqual(op.attr.quant_filter, expect_result, '{} th op attr not match'.format(i+1))

    def _make_op_list(self):
        # 'op': {
        #     'op_name': 'name_str',
        #     'attr': {
        #         'quant_filter': False
        #     },
        #     'input_list': ['node:0', ...],
        #     'output_desc': [{'data_type': 'DT_INT8'}, {'data_type': 'DT_FLOAT16'}, ...]
        # }
        op_list = [
            self._init_mock_op('input', [], ['DT_FLOAT']),
            self._init_mock_op('trans_Cast_0', ['input:0'], ['DT_FLOAT16']),
            self._init_mock_op('trans_TransData_1', ['trans_Cast_0:0'], ['DT_FLOAT16']),
            self._init_mock_op('resnet_v1_50/conv1/Conv2Dresnet_v1_50/conv1/Relu', ['trans_TransData_1:0'],
                               ['DT_FLOAT16']),
            self._init_mock_op('resnet_v1_50/pool1/MaxPoolresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/act_quant'
                               '/AscendQuant', ['resnet_v1_50/conv1/Conv2Dresnet_v1_50/conv1/Relu:0'],
                               ['DT_INT8']),
            self._init_mock_op('resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/Conv2Dresnet_v1_50/block1/unit_1'
                               '/bottleneck_v1/conv1/dequant/AscendDequantresnet_v1_50/block1/unit_1/bottleneck_v1'
                               '/conv2/act_quant/AscendQuant',
                               ['resnet_v1_50/pool1/MaxPoolresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/act_quant'
                                '/AscendQuant:0'],
                               ['DT_INT8']),
            self._init_mock_op('resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/Conv2Dresnet_v1_50/block1/unit_1'
                               '/bottleneck_v1/conv2/dequant/AscendDequantresnet_v1_50/block1/unit_1/bottleneck_v1'
                               '/conv3/act_quant/AscendQuant',
                               ['resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/Conv2Dresnet_v1_50/block1/unit_1'
                                '/bottleneck_v1/conv1/dequant/AscendDequantresnet_v1_50/block1/unit_1/bottleneck_v1'
                                '/conv2/act_quant/AscendQuant:0'],
                               ['DT_INT8']),
            self._init_mock_op('resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/Conv2Dresnet_v1_50/block1/unit_1'
                               '/bottleneck_v1/conv3/dequant/AscendDequant',
                               ['resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/Conv2Dresnet_v1_50/block1/unit_1'
                                '/bottleneck_v1/conv2/dequant/AscendDequantresnet_v1_50/block1/unit_1/bottleneck_v1'
                                '/conv3/act_quant/AscendQuant:0'],
                               ['DT_FLOAT16']),
            self._init_mock_op('resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/Conv2Dresnet_v1_50/block1/unit_1'
                               '/bottleneck_v1/shortcut/dequant/AscendDequantresnet_v1_50/block1/unit_1/bottleneck_v1'
                               '/addresnet_v1_50/block1/unit_1/bottleneck_v1/Reluresnet_v1_50/block1/unit_2/bottleneck'
                               '_v1/conv1/act_quant/AscendQuant',
                               ['resnet_v1_50/pool1/MaxPoolresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/act_quant'
                                '/AscendQuant:0', 'resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/Conv2Dresnet_v1_50'
                                '/block1/unit_1/bottleneck_v1/conv3/dequant/AscendDequant:0'],
                               ['DT_FLOAT16', 'DT_INT8']),
            self._init_mock_op('resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/Conv2Dresnet_v1_50/block1/unit_2'
                               '/bottleneck_v1/conv1/dequant/AscendDequantresnet_v1_50/block1/unit_2/bottleneck_v1'
                               '/conv2/act_quant/AscendQuant',
                               ['resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/Conv2Dresnet_v1_50/block1/unit_1'
                                '/bottleneck_v1/shortcut/dequant/AscendDequantresnet_v1_50/block1/unit_1/bottleneck_v1'
                                '/addresnet_v1_50/block1/unit_1/bottleneck_v1/Reluresnet_v1_50/block1/unit_2'
                                '/bottleneck_v1/conv1/act_quant/AscendQuant:1'],
                               ['DT_INT8']),
            self._init_mock_op('resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/Conv2Dresnet_v1_50/block1/unit_2'
                               '/bottleneck_v1/conv2/dequant/AscendDequantresnet_v1_50/block1/unit_2/bottleneck_v1'
                               '/conv3/act_quant/AscendQuant',
                               ['resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/Conv2Dresnet_v1_50/block1/unit_2'
                                '/bottleneck_v1/conv1/dequant/AscendDequantresnet_v1_50/block1/unit_2/bottleneck_v1'
                                '/conv2/act_quant/AscendQuant:0'],
                               ['DT_INT8']),
            self._init_mock_op('resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/Conv2Dresnet_v1_50/block1/unit_2'
                               '/bottleneck_v1/conv3/dequant/AscendDequantresnet_v1_50/block1/unit_2/bottleneck_v1'
                               '/addresnet_v1_50/block1/unit_2/bottleneck_v1/Reluresnet_v1_50/block1/unit_3'
                               '/bottleneck_v1/conv1/act_quant_1/AscendQuant',
                               ['resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/Conv2Dresnet_v1_50/block1/unit_2'
                                '/bottleneck_v1/conv2/dequant/AscendDequantresnet_v1_50/block1/unit_2/bottleneck_v1'
                                '/conv3/act_quant/AscendQuant:0', 'resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut'
                                '/Conv2Dresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/dequant/AscendDequant'
                                'resnet_v1_50/block1/unit_1/bottleneck_v1/addresnet_v1_50/block1/unit_1/bottleneck_v1'
                                '/Reluresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/act_quant/AscendQuant:0'],
                               ['DT_INT8']),
            self._init_mock_op('resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/anti_quant/AscendAntiQuantresnet_v1_50'
                               '/block1/unit_3/bottleneck_v1/shortcut/MaxPool',
                               ['resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/Conv2Dresnet_v1_50/block1/unit_2'
                                '/bottleneck_v1/conv3/dequant/AscendDequantresnet_v1_50/block1/unit_2/bottleneck_v1'
                                '/addresnet_v1_50/block1/unit_2/bottleneck_v1/Reluresnet_v1_50/block1/unit_3'
                                '/bottleneck_v1/conv1/act_quant_1/AscendQuant:0'],
                               ['DT_FLOAT16']),
            self._init_mock_op('resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/Conv2Dresnet_v1_50/block1/unit_3'
                               '/bottleneck_v1/conv1/dequant/AscendDequantresnet_v1_50/block1/unit_3/bottleneck_v1'
                               '/conv2/act_quant/AscendQuant',
                               ['resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/Conv2Dresnet_v1_50/block1/unit_2'
                                '/bottleneck_v1/conv3/dequant/AscendDequantresnet_v1_50/block1/unit_2/bottleneck_v1'
                                '/addresnet_v1_50/block1/unit_2/bottleneck_v1/Reluresnet_v1_50/block1/unit_3'
                                '/bottleneck_v1/conv1/act_quant_1/AscendQuant:0'],
                               ['DT_INT8']),
            self._init_mock_op('resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/Conv2Dresnet_v1_50/block1/unit_3'
                               '/bottleneck_v1/conv2/dequant/AscendDequantresnet_v1_50/block1/unit_3/bottleneck_v1'
                               '/conv3/act_quant/AscendQuant',
                               ['resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/Conv2Dresnet_v1_50/block1/unit_3'
                                '/bottleneck_v1/conv1/dequant/AscendDequantresnet_v1_50/block1/unit_3/bottleneck_v1'
                                '/conv2/act_quant/AscendQuant:0'],
                               ['DT_INT8']),
            self._init_mock_op('resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/Conv2Dresnet_v1_50/block1/unit_3'
                               '/bottleneck_v1/conv3/dequant/AscendDequantresnet_v1_50/block1/unit_3/bottleneck_v1'
                               '/addresnet_v1_50/block1/unit_3/bottleneck_v1/Reluresnet_v1_50/block2/unit_1'
                               '/bottleneck_v1/shortcut/act_quant/AscendQuant',
                               ['resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/Conv2Dresnet_v1_50/block1/unit_3'
                                '/bottleneck_v1/conv2/dequant/AscendDequantresnet_v1_50/block1/unit_3/bottleneck_v1'
                                '/conv3/act_quant/AscendQuant:0', 'resnet_v1_50/block1/unit_3/bottleneck_v1/conv1'
                                '/anti_quant/AscendAntiQuantresnet_v1_50/block1/unit_3/bottleneck_v1/shortcut'
                                '/MaxPool:0'],
                               ['DT_INT8']),
            self._init_mock_op('resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/Conv2Dresnet_v1_50/block2/unit_1'
                               '/bottleneck_v1/conv1/dequant/AscendDequantresnet_v1_50/block2/unit_1/bottleneck_v1'
                               '/conv2/act_quant/AscendQuant',
                               ['resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/Conv2Dresnet_v1_50/block1/unit_3'
                                '/bottleneck_v1/conv3/dequant/AscendDequantresnet_v1_50/block1/unit_3/bottleneck_v1'
                                '/addresnet_v1_50/block1/unit_3/bottleneck_v1/Reluresnet_v1_50/block2/unit_1'
                                '/bottleneck_v1/shortcut/act_quant/AscendQuant:0'],
                               ['DT_INT8']),
            self._init_mock_op('resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/Conv2Dresnet_v1_50/block2/unit_1'
                               '/bottleneck_v1/conv2/dequant/AscendDequantresnet_v1_50/block2/unit_1/bottleneck_v1'
                               '/conv3/act_quant/AscendQuant',
                               ['resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/Conv2Dresnet_v1_50/block2/unit_1'
                                '/bottleneck_v1/conv1/dequant/AscendDequantresnet_v1_50/block2/unit_1/bottleneck_v1'
                                '/conv2/act_quant/AscendQuant:0'],
                               ['DT_INT8']),
            self._init_mock_op('resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/Conv2Dresnet_v1_50/block2/unit_1'
                               '/bottleneck_v1/conv3/dequant/AscendDequant',
                               ['resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/Conv2Dresnet_v1_50/block2/unit_1'
                                '/bottleneck_v1/conv2/dequant/AscendDequantresnet_v1_50/block2/unit_1/bottleneck_v1'
                                '/conv3/act_quant/AscendQuant:0'],
                               ['DT_FLOAT16']),
            self._init_mock_op('resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/Conv2Dresnet_v1_50/block2/unit_1'
                               '/bottleneck_v1/shortcut/dequant/AscendDequantresnet_v1_50/block2/unit_1/bottleneck_v1'
                               '/addresnet_v1_50/block2/unit_1/bottleneck_v1/Reluresnet_v1_50/block2/unit_2'
                               '/bottleneck_v1/conv1/act_quant/AscendQuant',
                               ['resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/Conv2Dresnet_v1_50/block1/unit_3'
                                '/bottleneck_v1/conv3/dequant/AscendDequantresnet_v1_50/block1/unit_3/bottleneck_v1'
                                '/addresnet_v1_50/block1/unit_3/bottleneck_v1/Reluresnet_v1_50/block2/unit_1'
                                '/bottleneck_v1/shortcut/act_quant/AscendQuant:0', 'resnet_v1_50/block2/unit_1'
                                '/bottleneck_v1/conv3/Conv2Dresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/dequant'
                                '/AscendDequant:0'],
                               ['DT_FLOAT16', 'DT_INT8']),
            self._init_mock_op('resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/Conv2Dresnet_v1_50/block2/unit_2'
                               '/bottleneck_v1/conv1/dequant/AscendDequantresnet_v1_50/block2/unit_2/bottleneck_v1'
                               '/conv2/act_quant/AscendQuant',
                               ['resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/Conv2Dresnet_v1_50/block2/unit_1'
                                '/bottleneck_v1/shortcut/dequant/AscendDequantresnet_v1_50/block2/unit_1/bottleneck_v1'
                                '/addresnet_v1_50/block2/unit_1/bottleneck_v1/Reluresnet_v1_50/block2/unit_2'
                                '/bottleneck_v1/conv1/act_quant/AscendQuant:1'],
                               ['DT_INT8']),
            self._init_mock_op('resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/Conv2Dresnet_v1_50/block2/unit_2'
                               '/bottleneck_v1/conv2/dequant/AscendDequantresnet_v1_50/block2/unit_2/bottleneck_v1'
                               '/conv3/act_quant/AscendQuant',
                               ['resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/Conv2Dresnet_v1_50/block2/unit_2'
                                '/bottleneck_v1/conv1/dequant/AscendDequantresnet_v1_50/block2/unit_2/bottleneck_v1'
                                '/conv2/act_quant/AscendQuant:0'],
                               ['DT_INT8']),
            self._init_mock_op('resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/Conv2Dresnet_v1_50/block2/unit_2'
                               '/bottleneck_v1/conv3/dequant/AscendDequantresnet_v1_50/block2/unit_2/bottleneck_v1'
                               '/addresnet_v1_50/block2/unit_2/bottleneck_v1/Reluresnet_v1_50/block2/unit_3'
                               '/bottleneck_v1/conv1/act_quant/AscendQuant',
                               ['resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/Conv2Dresnet_v1_50/block2/unit_2'
                                '/bottleneck_v1/conv2/dequant/AscendDequantresnet_v1_50/block2/unit_2/bottleneck_v1'
                                '/conv3/act_quant/AscendQuant:0', 'resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut'
                                '/Conv2Dresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/dequant/AscendDequant'
                                'resnet_v1_50/block2/unit_1/bottleneck_v1/addresnet_v1_50/block2/unit_1/bottleneck_v1'
                                '/Reluresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/act_quant/AscendQuant:0'],
                               ['DT_FLOAT16', 'DT_INT8']),
            self._init_mock_op('resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/Conv2Dresnet_v1_50/block2/unit_3'
                               '/bottleneck_v1/conv1/dequant/AscendDequantresnet_v1_50/block2/unit_3/bottleneck_v1'
                               '/conv2/act_quant/AscendQuant',
                               ['resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/Conv2Dresnet_v1_50/block2/unit_2'
                                '/bottleneck_v1/conv3/dequant/AscendDequantresnet_v1_50/block2/unit_2/bottleneck_v1'
                                '/addresnet_v1_50/block2/unit_2/bottleneck_v1/Reluresnet_v1_50/block2/unit_3'
                                '/bottleneck_v1/conv1/act_quant/AscendQuant:1'],
                               ['DT_INT8']),
            self._init_mock_op('resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/Conv2Dresnet_v1_50/block2/unit_3'
                               '/bottleneck_v1/conv2/dequant/AscendDequantresnet_v1_50/block2/unit_3/bottleneck_v1'
                               '/conv3/act_quant/AscendQuant',
                               ['resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/Conv2Dresnet_v1_50/block2/unit_3'
                                '/bottleneck_v1/conv1/dequant/AscendDequantresnet_v1_50/block2/unit_3/bottleneck_v1'
                                '/conv2/act_quant/AscendQuant:0'],
                               ['DT_INT8']),
            self._init_mock_op('resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/Conv2Dresnet_v1_50/block2/unit_3'
                               '/bottleneck_v1/conv3/dequant/AscendDequantresnet_v1_50/block2/unit_3/bottleneck_v1'
                               '/addresnet_v1_50/block2/unit_3/bottleneck_v1/Reluresnet_v1_50/block2/unit_4'
                               '/bottleneck_v1/conv1/act_quant_1/AscendQuant',
                               ['resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/Conv2Dresnet_v1_50/block2/unit_3'
                                '/bottleneck_v1/conv2/dequant/AscendDequantresnet_v1_50/block2/unit_3/bottleneck_v1'
                                '/conv3/act_quant/AscendQuant:0', 'resnet_v1_50/block2/unit_2/bottleneck_v1/conv3'
                                '/Conv2Dresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/dequant/AscendDequant'
                                'resnet_v1_50/block2/unit_2/bottleneck_v1/addresnet_v1_50/block2/unit_2/bottleneck_v1'
                                '/Reluresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/act_quant/AscendQuant:0'],
                               ['DT_INT8']),
            self._init_mock_op('resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/anti_quant/AscendAntiQuantresnet_v1_50'
                               '/block2/unit_4/bottleneck_v1/shortcut/MaxPool',
                               ['resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/Conv2Dresnet_v1_50/block2/unit_3'
                                '/bottleneck_v1/conv3/dequant/AscendDequantresnet_v1_50/block2/unit_3/bottleneck_v1'
                                '/addresnet_v1_50/block2/unit_3/bottleneck_v1/Reluresnet_v1_50/block2/unit_4'
                                '/bottleneck_v1/conv1/act_quant_1/AscendQuant:0'],
                               ['DT_FLOAT16']),
            self._init_mock_op('resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/Conv2Dresnet_v1_50/block2/unit_4'
                               '/bottleneck_v1/conv1/dequant/AscendDequantresnet_v1_50/block2/unit_4/bottleneck_v1'
                               '/conv2/act_quant/AscendQuant',
                               ['resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/Conv2Dresnet_v1_50/block2/unit_3'
                                '/bottleneck_v1/conv3/dequant/AscendDequantresnet_v1_50/block2/unit_3/bottleneck_v1'
                                '/addresnet_v1_50/block2/unit_3/bottleneck_v1/Reluresnet_v1_50/block2/unit_4'
                                '/bottleneck_v1/conv1/act_quant_1/AscendQuant:0'],
                               ['DT_INT8']),
            self._init_mock_op('resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/Conv2Dresnet_v1_50/block2/unit_4'
                               '/bottleneck_v1/conv2/dequant/AscendDequantresnet_v1_50/block2/unit_4/bottleneck_v1'
                               '/conv3/act_quant/AscendQuant',
                               ['resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/Conv2Dresnet_v1_50/block2/unit_4'
                                '/bottleneck_v1/conv1/dequant/AscendDequantresnet_v1_50/block2/unit_4/bottleneck_v1'
                                '/conv2/act_quant/AscendQuant:0'],
                               ['DT_INT8']),
            self._init_mock_op('resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/Conv2Dresnet_v1_50/block2/unit_4'
                               '/bottleneck_v1/conv3/dequant/AscendDequantresnet_v1_50/block2/unit_4/bottleneck_v1'
                               '/addresnet_v1_50/block2/unit_4/bottleneck_v1/Reluresnet_v1_50/block3/unit_1'
                               '/bottleneck_v1/shortcut/act_quant/AscendQuant',
                               ['resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/Conv2Dresnet_v1_50/block2/unit_4'
                                '/bottleneck_v1/conv2/dequant/AscendDequantresnet_v1_50/block2/unit_4/bottleneck_v1'
                                '/conv3/act_quant/AscendQuant:0', 'resnet_v1_50/block2/unit_4/bottleneck_v1/conv1'
                                '/anti_quant/AscendAntiQuantresnet_v1_50/block2/unit_4/bottleneck_v1/shortcut'
                                '/MaxPool:0'], ['DT_INT8']),
            self._init_mock_op('resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/Conv2Dresnet_v1_50/block3/unit_1'
                               '/bottleneck_v1/conv1/dequant/AscendDequantresnet_v1_50/block3/unit_1/bottleneck_v1'
                               '/conv2/act_quant/AscendQuant',
                               ['resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/Conv2Dresnet_v1_50/block2/unit_4'
                                '/bottleneck_v1/conv3/dequant/AscendDequantresnet_v1_50/block2/unit_4/bottleneck_v1'
                                '/addresnet_v1_50/block2/unit_4/bottleneck_v1/Reluresnet_v1_50/block3/unit_1'
                                '/bottleneck_v1/shortcut/act_quant/AscendQuant:0'],
                               ['DT_INT8']),
            self._init_mock_op('resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/Conv2Dresnet_v1_50/block3/unit_1'
                               '/bottleneck_v1/conv2/dequant/AscendDequantresnet_v1_50/block3/unit_1/bottleneck_v1'
                               '/conv3/act_quant/AscendQuant',
                               ['resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/Conv2Dresnet_v1_50/block3/unit_1'
                                '/bottleneck_v1/conv1/dequant/AscendDequantresnet_v1_50/block3/unit_1/bottleneck_v1'
                                '/conv2/act_quant/AscendQuant:0'],
                               ['DT_INT8']),
            self._init_mock_op('resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/Conv2Dresnet_v1_50/block3/unit_1'
                               '/bottleneck_v1/conv3/dequant/AscendDequant',
                               ['resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/Conv2Dresnet_v1_50/block3/unit_1'
                                '/bottleneck_v1/conv2/dequant/AscendDequantresnet_v1_50/block3/unit_1/bottleneck_v1'
                                '/conv3/act_quant/AscendQuant:0'],
                               ['DT_FLOAT16']),
            self._init_mock_op('resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/Conv2Dresnet_v1_50/block3/unit_1'
                               '/bottleneck_v1/shortcut/dequant/AscendDequantresnet_v1_50/block3/unit_1/bottleneck_v1'
                               '/addresnet_v1_50/block3/unit_1/bottleneck_v1/Reluresnet_v1_50/block3/unit_2'
                               '/bottleneck_v1/conv1/act_quant/AscendQuant',
                               ['resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/Conv2Dresnet_v1_50/block2/unit_4'
                                '/bottleneck_v1/conv3/dequant/AscendDequantresnet_v1_50/block2/unit_4/bottleneck_v1'
                                '/addresnet_v1_50/block2/unit_4/bottleneck_v1/Reluresnet_v1_50/block3/unit_1'
                                '/bottleneck_v1/shortcut/act_quant/AscendQuant:0', 'resnet_v1_50/block3/unit_1'
                                '/bottleneck_v1/conv3/Conv2Dresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/dequant'
                                '/AscendDequant:0'],
                               ['DT_FLOAT16', 'DT_INT8']),
            self._init_mock_op('resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/Conv2Dresnet_v1_50/block3/unit_2'
                               '/bottleneck_v1/conv1/dequant/AscendDequantresnet_v1_50/block3/unit_2/bottleneck_v1'
                               '/conv2/act_quant/AscendQuant',
                               ['resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/Conv2Dresnet_v1_50/block3/unit_1'
                                '/bottleneck_v1/shortcut/dequant/AscendDequantresnet_v1_50/block3/unit_1/bottleneck_v1'
                                '/addresnet_v1_50/block3/unit_1/bottleneck_v1/Reluresnet_v1_50/block3/unit_2'
                                '/bottleneck_v1/conv1/act_quant/AscendQuant:1'],
                               ['DT_INT8']),
            self._init_mock_op('resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/Conv2Dresnet_v1_50/block3/unit_2'
                               '/bottleneck_v1/conv2/dequant/AscendDequantresnet_v1_50/block3/unit_2/bottleneck_v1'
                               '/conv3/act_quant/AscendQuant',
                               ['resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/Conv2Dresnet_v1_50/block3/unit_2'
                                '/bottleneck_v1/conv1/dequant/AscendDequantresnet_v1_50/block3/unit_2/bottleneck_v1'
                                '/conv2/act_quant/AscendQuant:0'],
                               ['DT_INT8']),
            self._init_mock_op('resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/Conv2Dresnet_v1_50/block3/unit_2'
                               '/bottleneck_v1/conv3/dequant/AscendDequantresnet_v1_50/block3/unit_2/bottleneck_v1'
                               '/addresnet_v1_50/block3/unit_2/bottleneck_v1/Reluresnet_v1_50/block3/unit_3'
                               '/bottleneck_v1/conv1/act_quant/AscendQuant',
                               ['resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/Conv2Dresnet_v1_50/block3/unit_2'
                                '/bottleneck_v1/conv2/dequant/AscendDequantresnet_v1_50/block3/unit_2/bottleneck_v1'
                                '/conv3/act_quant/AscendQuant:0',
                                'resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/Conv2Dresnet_v1_50/block3/unit_1'
                                '/bottleneck_v1/shortcut/dequant/AscendDequantresnet_v1_50/block3/unit_1/bottleneck_v1'
                                '/addresnet_v1_50/block3/unit_1/bottleneck_v1/Reluresnet_v1_50/block3/unit_2'
                                '/bottleneck_v1/conv1/act_quant/AscendQuant:0'],
                               ['DT_FLOAT16', 'DT_INT8']),
            self._init_mock_op('resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/Conv2Dresnet_v1_50/block3/unit_3'
                               '/bottleneck_v1/conv1/dequant/AscendDequantresnet_v1_50/block3/unit_3/bottleneck_v1'
                               '/conv2/act_quant/AscendQuant',
                               ['resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/Conv2Dresnet_v1_50/block3/unit_2'
                                '/bottleneck_v1/conv3/dequant/AscendDequantresnet_v1_50/block3/unit_2/bottleneck_v1'
                                '/addresnet_v1_50/block3/unit_2/bottleneck_v1/Reluresnet_v1_50/block3/unit_3'
                                '/bottleneck_v1/conv1/act_quant/AscendQuant:1'],
                               ['DT_INT8']),
            self._init_mock_op('resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/Conv2Dresnet_v1_50/block3/unit_3'
                               '/bottleneck_v1/conv2/dequant/AscendDequantresnet_v1_50/block3/unit_3/bottleneck_v1'
                               '/conv3/act_quant/AscendQuant',
                               ['resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/Conv2Dresnet_v1_50/block3/unit_3'
                                '/bottleneck_v1/conv1/dequant/AscendDequantresnet_v1_50/block3/unit_3/bottleneck_v1'
                                '/conv2/act_quant/AscendQuant:0'],
                               ['DT_INT8']),
            self._init_mock_op('resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/Conv2Dresnet_v1_50/block3/unit_3'
                               '/bottleneck_v1/conv3/dequant/AscendDequantresnet_v1_50/block3/unit_3/bottleneck_v1'
                               '/addresnet_v1_50/block3/unit_3/bottleneck_v1/Reluresnet_v1_50/block3/unit_4'
                               '/bottleneck_v1/conv1/act_quant/AscendQuant',
                               ['resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/Conv2Dresnet_v1_50/block3/unit_3'
                                '/bottleneck_v1/conv2/dequant/AscendDequantresnet_v1_50/block3/unit_3/bottleneck_v1'
                                '/conv3/act_quant/AscendQuant:0',
                                'resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/Conv2Dresnet_v1_50/block3/unit_2'
                                '/bottleneck_v1/conv3/dequant/AscendDequantresnet_v1_50/block3/unit_2/bottleneck_v1'
                                '/addresnet_v1_50/block3/unit_2/bottleneck_v1/Reluresnet_v1_50/block3/unit_3'
                                '/bottleneck_v1/conv1/act_quant/AscendQuant:0'],
                               ['DT_FLOAT16', 'DT_INT8']),
            self._init_mock_op('resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/Conv2Dresnet_v1_50/block3/unit_4'
                               '/bottleneck_v1/conv1/dequant/AscendDequantresnet_v1_50/block3/unit_4/bottleneck_v1'
                               '/conv2/act_quant/AscendQuant',
                               ['resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/Conv2Dresnet_v1_50/block3/unit_3'
                                '/bottleneck_v1/conv3/dequant/AscendDequantresnet_v1_50/block3/unit_3/bottleneck_v1'
                                '/addresnet_v1_50/block3/unit_3/bottleneck_v1/Reluresnet_v1_50/block3/unit_4'
                                '/bottleneck_v1/conv1/act_quant/AscendQuant:1'],
                               ['DT_INT8']),
            self._init_mock_op('resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/Conv2Dresnet_v1_50/block3/unit_4'
                               '/bottleneck_v1/conv2/dequant/AscendDequantresnet_v1_50/block3/unit_4/bottleneck_v1'
                               '/conv3/act_quant/AscendQuant',
                               ['resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/Conv2Dresnet_v1_50/block3/unit_4'
                                '/bottleneck_v1/conv1/dequant/AscendDequantresnet_v1_50/block3/unit_4/bottleneck_v1'
                                '/conv2/act_quant/AscendQuant:0'],
                               ['DT_INT8']),
            self._init_mock_op('resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/Conv2Dresnet_v1_50/block3/unit_4'
                               '/bottleneck_v1/conv3/dequant/AscendDequantresnet_v1_50/block3/unit_4/bottleneck_v1'
                               '/addresnet_v1_50/block3/unit_4/bottleneck_v1/Reluresnet_v1_50/block3/unit_5'
                               '/bottleneck_v1/conv1/act_quant/AscendQuant',
                               ['resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/Conv2Dresnet_v1_50/block3/unit_4'
                                '/bottleneck_v1/conv2/dequant/AscendDequantresnet_v1_50/block3/unit_4/bottleneck_v1'
                                '/conv3/act_quant/AscendQuant:0',
                                'resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/Conv2Dresnet_v1_50/block3/unit_3'
                                '/bottleneck_v1/conv3/dequant/AscendDequantresnet_v1_50/block3/unit_3/bottleneck_v1'
                                '/addresnet_v1_50/block3/unit_3/bottleneck_v1/Reluresnet_v1_50/block3/unit_4'
                                '/bottleneck_v1/conv1/act_quant/AscendQuant:0'],
                               ['DT_FLOAT16', 'DT_INT8']),
            self._init_mock_op('resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/Conv2Dresnet_v1_50/block3/unit_5'
                               '/bottleneck_v1/conv1/dequant/AscendDequantresnet_v1_50/block3/unit_5/bottleneck_v1'
                               '/conv2/act_quant/AscendQuant',
                               ['resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/Conv2Dresnet_v1_50/block3/unit_4'
                                '/bottleneck_v1/conv3/dequant/AscendDequantresnet_v1_50/block3/unit_4/bottleneck_v1'
                                '/addresnet_v1_50/block3/unit_4/bottleneck_v1/Reluresnet_v1_50/block3/unit_5'
                                '/bottleneck_v1/conv1/act_quant/AscendQuant:1'],
                               ['DT_INT8']),
            self._init_mock_op('resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/Conv2Dresnet_v1_50/block3/unit_5'
                               '/bottleneck_v1/conv2/dequant/AscendDequantresnet_v1_50/block3/unit_5/bottleneck_v1'
                               '/conv3/act_quant/AscendQuant',
                               ['resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/Conv2Dresnet_v1_50/block3/unit_5'
                                '/bottleneck_v1/conv1/dequant/AscendDequantresnet_v1_50/block3/unit_5/bottleneck_v1'
                                '/conv2/act_quant/AscendQuant:0'],
                               ['DT_INT8']),
            self._init_mock_op('resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/Conv2Dresnet_v1_50/block3/unit_5'
                               '/bottleneck_v1/conv3/dequant/AscendDequantresnet_v1_50/block3/unit_5/bottleneck_v1'
                               '/addresnet_v1_50/block3/unit_5/bottleneck_v1/Reluresnet_v1_50/block3/unit_6'
                               '/bottleneck_v1/conv1/act_quant_1/AscendQuant',
                               ['resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/Conv2Dresnet_v1_50/block3/unit_5'
                                '/bottleneck_v1/conv2/dequant/AscendDequantresnet_v1_50/block3/unit_5/bottleneck_v1'
                                '/conv3/act_quant/AscendQuant:0',
                                'resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/Conv2Dresnet_v1_50/block3/unit_4'
                                '/bottleneck_v1/conv3/dequant/AscendDequantresnet_v1_50/block3/unit_4/bottleneck_v1'
                                '/addresnet_v1_50/block3/unit_4/bottleneck_v1/Reluresnet_v1_50/block3/unit_5'
                                '/bottleneck_v1/conv1/act_quant/AscendQuant:0'],
                               ['DT_INT8']),
            self._init_mock_op('resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/anti_quant/AscendAntiQuantresnet_v1_50'
                               '/block3/unit_6/bottleneck_v1/shortcut/MaxPool',
                               ['resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/Conv2Dresnet_v1_50/block3/unit_5'
                                '/bottleneck_v1/conv3/dequant/AscendDequantresnet_v1_50/block3/unit_5/bottleneck_v1'
                                '/addresnet_v1_50/block3/unit_5/bottleneck_v1/Reluresnet_v1_50/block3/unit_6'
                                '/bottleneck_v1/conv1/act_quant_1/AscendQuant:0'],
                               ['DT_FLOAT16']),
            self._init_mock_op('resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/Conv2Dresnet_v1_50/block3/unit_6'
                               '/bottleneck_v1/conv1/dequant/AscendDequantresnet_v1_50/block3/unit_6/bottleneck_v1'
                               '/conv2/act_quant/AscendQuant',
                               ['resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/Conv2Dresnet_v1_50/block3/unit_5'
                                '/bottleneck_v1/conv3/dequant/AscendDequantresnet_v1_50/block3/unit_5/bottleneck_v1'
                                '/addresnet_v1_50/block3/unit_5/bottleneck_v1/Reluresnet_v1_50/block3/unit_6'
                                '/bottleneck_v1/conv1/act_quant_1/AscendQuant:0'],
                               ['DT_INT8']),
            self._init_mock_op('resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/Conv2Dresnet_v1_50/block3/unit_6'
                               '/bottleneck_v1/conv2/dequant/AscendDequantresnet_v1_50/block3/unit_6/bottleneck_v1'
                               '/conv3/act_quant/AscendQuant',
                               ['resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/Conv2Dresnet_v1_50/block3/unit_6'
                                '/bottleneck_v1/conv1/dequant/AscendDequantresnet_v1_50/block3/unit_6/bottleneck_v1'
                                '/conv2/act_quant/AscendQuant:0'],
                               ['DT_INT8']),
            self._init_mock_op('resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/Conv2Dresnet_v1_50/block3/unit_6'
                               '/bottleneck_v1/conv3/dequant/AscendDequantresnet_v1_50/block3/unit_6/bottleneck_v1'
                               '/addresnet_v1_50/block3/unit_6/bottleneck_v1/Reluresnet_v1_50/block4/unit_1'
                               '/bottleneck_v1/shortcut/act_quant/AscendQuant',
                               ['resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/Conv2Dresnet_v1_50/block3/unit_6'
                                '/bottleneck_v1/conv2/dequant/AscendDequantresnet_v1_50/block3/unit_6/bottleneck_v1'
                                '/conv3/act_quant/AscendQuant:0',
                                'resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/anti_quant/AscendAntiQuantresnet_v1_50'
                                '/block3/unit_6/bottleneck_v1/shortcut/MaxPool:0'],
                               ['DT_INT8']),
            self._init_mock_op('resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/Conv2Dresnet_v1_50/block4/unit_1'
                               '/bottleneck_v1/conv1/dequant/AscendDequantresnet_v1_50/block4/unit_1/bottleneck_v1'
                               '/conv2/act_quant/AscendQuant',
                               ['resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/Conv2Dresnet_v1_50/block3/unit_6'
                                '/bottleneck_v1/conv3/dequant/AscendDequantresnet_v1_50/block3/unit_6/bottleneck_v1'
                                '/addresnet_v1_50/block3/unit_6/bottleneck_v1/Reluresnet_v1_50/block4/unit_1'
                                '/bottleneck_v1/shortcut/act_quant/AscendQuant:0'],
                               ['DT_INT8']),
            self._init_mock_op('resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/Conv2Dresnet_v1_50/block4/unit_1'
                               '/bottleneck_v1/conv2/dequant/AscendDequantresnet_v1_50/block4/unit_1/bottleneck_v1'
                               '/conv3/act_quant/AscendQuant',
                               ['resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/Conv2Dresnet_v1_50/block4/unit_1'
                                '/bottleneck_v1/conv1/dequant/AscendDequantresnet_v1_50/block4/unit_1/bottleneck_v1'
                                '/conv2/act_quant/AscendQuant:0'],
                               ['DT_INT8']),
            self._init_mock_op('resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/Conv2Dresnet_v1_50/block4/unit_1'
                               '/bottleneck_v1/conv3/dequant/AscendDequant',
                               ['resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/Conv2Dresnet_v1_50/block4/unit_1'
                                '/bottleneck_v1/conv2/dequant/AscendDequantresnet_v1_50/block4/unit_1/bottleneck_v1'
                                '/conv3/act_quant/AscendQuant:0'],
                               ['DT_FLOAT16']),
            self._init_mock_op('resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/Conv2Dresnet_v1_50/block4/unit_1'
                               '/bottleneck_v1/shortcut/dequant/AscendDequantresnet_v1_50/block4/unit_1/bottleneck_v1'
                               '/addresnet_v1_50/block4/unit_1/bottleneck_v1/Reluresnet_v1_50/block4/unit_2'
                               '/bottleneck_v1/conv1/act_quant/AscendQuant',
                               ['resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/Conv2Dresnet_v1_50/block3/unit_6'
                                '/bottleneck_v1/conv3/dequant/AscendDequantresnet_v1_50/block3/unit_6/bottleneck_v1'
                                '/addresnet_v1_50/block3/unit_6/bottleneck_v1/Reluresnet_v1_50/block4/unit_1'
                                '/bottleneck_v1/shortcut/act_quant/AscendQuant:0',
                                'resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/Conv2Dresnet_v1_50/block4/unit_1'
                                '/bottleneck_v1/conv3/dequant/AscendDequant:0'],
                               ['DT_FLOAT16', 'DT_INT8']),
            self._init_mock_op('resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/Conv2Dresnet_v1_50/block4/unit_2'
                               '/bottleneck_v1/conv1/dequant/AscendDequantresnet_v1_50/block4/unit_2/bottleneck_v1'
                               '/conv2/act_quant/AscendQuant',
                               ['resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/Conv2Dresnet_v1_50/block4/unit_1'
                                '/bottleneck_v1/shortcut/dequant/AscendDequantresnet_v1_50/block4/unit_1/bottleneck_v1'
                                '/addresnet_v1_50/block4/unit_1/bottleneck_v1/Reluresnet_v1_50/block4/unit_2'
                                '/bottleneck_v1/conv1/act_quant/AscendQuant:1'],
                               ['DT_INT8']),
            self._init_mock_op('resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/Conv2Dresnet_v1_50/block4/unit_2'
                               '/bottleneck_v1/conv2/dequant/AscendDequantresnet_v1_50/block4/unit_2/bottleneck_v1'
                               '/conv3/act_quant/AscendQuant',
                               ['resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/Conv2Dresnet_v1_50/block4/unit_2'
                                '/bottleneck_v1/conv1/dequant/AscendDequantresnet_v1_50/block4/unit_2/bottleneck_v1'
                                '/conv2/act_quant/AscendQuant:0'],
                               ['DT_INT8']),
            self._init_mock_op('resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/Conv2Dresnet_v1_50/block4/unit_2'
                               '/bottleneck_v1/conv3/dequant/AscendDequantresnet_v1_50/block4/unit_2/bottleneck_v1'
                               '/addresnet_v1_50/block4/unit_2/bottleneck_v1/Reluresnet_v1_50/block4/unit_3'
                               '/bottleneck_v1/conv1/act_quant/AscendQuant',
                               ['resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/Conv2Dresnet_v1_50/block4/unit_2'
                                '/bottleneck_v1/conv2/dequant/AscendDequantresnet_v1_50/block4/unit_2/bottleneck_v1'
                                '/conv3/act_quant/AscendQuant:0',
                                'resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/Conv2Dresnet_v1_50/block4/unit_1'
                                '/bottleneck_v1/shortcut/dequant/AscendDequantresnet_v1_50/block4/unit_1/bottleneck_v1'
                                '/addresnet_v1_50/block4/unit_1/bottleneck_v1/Reluresnet_v1_50/block4/unit_2'
                                '/bottleneck_v1/conv1/act_quant/AscendQuant:0'],
                               ['DT_FLOAT16', 'DT_INT8']),
            self._init_mock_op('resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/Conv2Dresnet_v1_50/block4/unit_3'
                               '/bottleneck_v1/conv1/dequant/AscendDequantresnet_v1_50/block4/unit_3/bottleneck_v1'
                               '/conv2/act_quant/AscendQuant',
                               ['resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/Conv2Dresnet_v1_50/block4/unit_2'
                                '/bottleneck_v1/conv3/dequant/AscendDequantresnet_v1_50/block4/unit_2/bottleneck_v1'
                                '/addresnet_v1_50/block4/unit_2/bottleneck_v1/Reluresnet_v1_50/block4/unit_3'
                                '/bottleneck_v1/conv1/act_quant/AscendQuant:1'],
                               ['DT_INT8']),
            self._init_mock_op('resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/Conv2Dresnet_v1_50/block4/unit_3'
                               '/bottleneck_v1/conv2/dequant/AscendDequantresnet_v1_50/block4/unit_3/bottleneck_v1'
                               '/conv3/act_quant/AscendQuant',
                               ['resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/Conv2Dresnet_v1_50/block4/unit_3'
                                '/bottleneck_v1/conv1/dequant/AscendDequantresnet_v1_50/block4/unit_3/bottleneck_v1'
                                '/conv2/act_quant/AscendQuant:0'],
                               ['DT_INT8']),
            self._init_mock_op('resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/Conv2Dresnet_v1_50/block4/unit_3'
                               '/bottleneck_v1/conv3/dequant/AscendDequantresnet_v1_50/block4/unit_3/bottleneck_v1'
                               '/addresnet_v1_50/block4/unit_3/bottleneck_v1/Relu',
                               ['resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/Conv2Dresnet_v1_50/block4/unit_3'
                                '/bottleneck_v1/conv2/dequant/AscendDequantresnet_v1_50/block4/unit_3/bottleneck_v1'
                                '/conv3/act_quant/AscendQuant:0',
                                'resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/Conv2Dresnet_v1_50/block4/unit_2'
                                '/bottleneck_v1/conv3/dequant/AscendDequantresnet_v1_50/block4/unit_2/bottleneck_v1'
                                '/addresnet_v1_50/block4/unit_2/bottleneck_v1/Reluresnet_v1_50/block4/unit_3'
                                '/bottleneck_v1/conv1/act_quant/AscendQuant:0'],
                               ['DT_FLOAT16']),
            self._init_mock_op('resnet_v1_50/pool5',
                               ['resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/Conv2Dresnet_v1_50/block4/unit_3'
                                '/bottleneck_v1/conv3/dequant/AscendDequantresnet_v1_50/block4/unit_3/bottleneck_v1'
                                '/addresnet_v1_50/block4/unit_3/bottleneck_v1/Relu:0'],
                               ['DT_FLOAT16']),
            self._init_mock_op('resnet_v1_50/logits/act_quant/AscendQuant', ['resnet_v1_50/pool5:0'], ['DT_INT8']),
            self._init_mock_op('resnet_v1_50/logits/Conv2Dresnet_v1_50/logits/dequant/AscendDequant',
                               ['resnet_v1_50/logits/act_quant/AscendQuant:0'], ['DT_FLOAT16']),
            self._init_mock_op('trans_Cast_629trans_TransData_627',
                               ['resnet_v1_50/logits/Conv2Dresnet_v1_50/logits/dequant/AscendDequant:0'],
                               ['DT_FLOAT16']),
            self._init_mock_op('resnet_v1_50/predictions/Softmax', ['trans_Cast_629trans_TransData_627:0'],
                               ['DT_FLOAT16']),
            self._init_mock_op('trans_Cast_630', ['resnet_v1_50/predictions/Softmax:0'], ['DT_FLOAT']),
            self._init_mock_op('Node_Output', ['trans_Cast_630:0'], [])
        ]

        filter_result_list = [False, False, False, False, False, True, True, False, True, True,
                              True, True, True, True, True, True, True, True, False, True,
                              True, True, True, True, True, True, True, True, True, True,
                              True, True, False, True, True, True, True, True, True, True,
                              True, True, True, True, True, True, True, True, True, True,
                              True, True, False, True, True, True, True, True, True, False,
                              False, False, False, False, False, False, False]

        return op_list, filter_result_list

    @staticmethod
    def _init_mock_op(name: str, input_list: list, out_type: list):
        op = mock.Mock()
        op.op_name = name
        op.attr = mock.Mock()
        op.attr.quant_filter = False
        op.input_list = input_list
        op.output_desc = []
        for ot in out_type:
            desc = mock.Mock()
            desc.data_type = ot
            op.output_desc.append(desc)

        return op
