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

import struct
import multiprocessing
import os
from tempfile import TemporaryDirectory

import pytest
import numpy as np
import compare_vector
import json
import unittest
from unittest import mock

import compare_vector
from cmp_utils import utils, path_check, log
from cmp_utils.constant.compare_error import CompareError
from vector_cmp.fusion_manager.fusion_op import OutputDesc, FusionOp, OpAttr
from vector_cmp.fusion_manager.compare_result import SingleOpCmpResult
from dump_parse import dump_utils
from vector_cmp import vector_comparison
from dump_parse.proto_dump_data import DumpData, OpInput, OpOutput
from cmp_utils.constant.const_manager import DD


vector_comparison.VectorComparison.MULTI_THREAD_MAX_NUM = 1


class TestUtilsMethods(unittest.TestCase):
    mock_stat_result = os.stat_result((0, 0, 0, 0, os.getuid(), os.getgid(), 0, 0, 0, 0)) 
    
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
    def _fake_listdir(path):
        # 内置格式转换脚本目录
        if 'builtin_format_convert' in path:
            return ['convert_NC1HWC0_to_NCHW.py']

        # 左侧 dump 目录（my_dump_path）
        if path.startswith('/home/demo'):
            return [
                'xxxx.conv1conv1_relu.0.1111111111111111',
                'xxxx.ccccc.0.1111111111111111'
            ]

        # 右侧 dump 目录（golden_dump_path）
        if path.startswith('/home/dt'):
            return [
                'xxxx.conv1conv1_relu.0.1111111111111111',
                'xxxx.ddddd.0.1111111111111111'
            ]

        # 其余情况：算法脚本目录
        return ['alg_CosineSimilarity.py']

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

    @staticmethod
    def _make_json():
        return {'name': 'resnet50', 'graph': [
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
                      "dynamic_const_387:0"
                  ],
                  'id': 1,
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
                  'id': 2,
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
                 {'name': 'res2s_branch1',
                  'type': 'Relu',
                  "attr": [
                      {"key": "_datadump_original_op_names",
                       "value": {"list": {"val_type": 1,
                                          "s": ["res2s_branch1",
                                                "res2s_branch1_relu"]}}},
                  ],
                  "input": [
                      "data:0",
                      "dynamic_const_471:0",
                      "dynamic_const_387:0"
                  ],
                  'id': 3,
                  'output_desc': [
                      {'attr': [
                          {'key': '_datadump_origin_name',
                           'value': {'s': 'res2s_branch1_relu'}},
                          {'key': '_datadump_origin_output_index',
                           'value': {'i': 0}},
                          {'key': '_datadump_origin_format',
                           'value': {'s': 'NCHW'}},
                      ]},
                  ]
                  },
                 {'name': 'dynamic_const_432', 'type': 'Const', 'id': 4, "attr": [
                     {"key": "_datadump_original_op_names",
                      "value": {"list": {"val_type": 1}}}]},
                 {'name': 'prob', 'id': 5, 'type': 'PORR'},
                 {'name': 'conv1_quant', 'id': 6, 'type': 'AscendQuant'},
                 {'name': 'pool5', 'id': 7, 'type': 'pool'},
                 ],
             },
            {
                'name': 'merge2', 'op': []
            }
        ]}

    @staticmethod
    def _make_input_json():
        return json.dumps({'name': 'resnet50', 'graph': [
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
                          {'key': 'origin_format',
                           'value': {'s': 'NCHW'}},
                          {'key': 'origin_shape',
                           'value': {"list": {"val_type": 1,
                                              "i": [1, 3, 4, 4]}}},
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
                          {'key': 'origin_shape',
                           'value': {"list": {"val_type": 1,
                                              "i": [1, 3, 4, 4]}}},
                      ]},
                  ]
                  },

                 ],
             },
        ]})

    @staticmethod
    def _make_csv_content():
        content = 'Index,LeftOp,RightOp,TensorIndex,MaxAbsoluteError,MaxAbsoluteError,CompareFailReason'
        return content

    @staticmethod
    def _make_json_object():
        json_object = {'name': 'resnet50', 'graph': [
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
                          {'key': 'origin_format',
                           'value': {'s': 'NCHW'}},
                          {'key': 'origin_shape',
                           'value': {"list": {"val_type": 1,
                                              "i": [1, 3, 4, 4]}}},
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
                          {'key': 'origin_shape',
                           'value': {"list": {"val_type": 1,
                                              "i": [1, 3, 4, 4]}}},
                      ]},
                  ]
                  },

                 ],
             },
        ]}
        return json_object

    @staticmethod
    def _make_L1_fusion_json():
        return json.dumps({'name': 'resnet50', 'graph': [
            {'name': 'merge1', 'op':
                [{'name': 'data', 'type': 'input'},
                 {'name': 'A1',
                  'type': 'Relu',
                  "attr": [
                      {"key": "_datadump_original_op_names",
                       "value": {"list": {"val_type": 1,
                                          "s": ["A_conv1", "A_relu"]}}},
                      {"key": "_L1_fusion_sub_graph_no",
                       "value": {'s': '1'}},
                  ],
                  "input": [
                      "data:0",
                      "dynamic_const_471:0",
                      "dynamic_const_387:0"
                  ],
                  'output_desc': [
                      {'attr': [
                          {'key': '_datadump_origin_name',
                           'value': {'s': 'A_relu'}},
                          {'key': '_datadump_origin_output_index',
                           'value': {'i': 0}},
                          {'key': '_datadump_origin_format',
                           'value': {'s': 'NCHW'}},
                      ]},
                  ]
                  },
                 {'name': 'A2',
                  'type': 'Relu',
                  "attr": [
                      {"key": "_datadump_original_op_names",
                       "value": {"list": {"val_type": 1,
                                          "s": ["A_conv1", "A_relu"]}}},
                      {"key": "_L1_fusion_sub_graph_no",
                       "value": {'s': '1'}},
                  ],
                  "input": [
                      "data:0",
                      "dynamic_const_471:0",
                      "dynamic_const_387:0"
                  ],
                  'output_desc': [
                      {'attr': [
                          {'key': '_datadump_origin_name',
                           'value': {'s': 'A_relu'}},
                          {'key': '_datadump_origin_output_index',
                           'value': {'i': 0}},
                          {'key': '_datadump_origin_format',
                           'value': {'s': 'NCHW'}},
                      ]},
                  ]
                  },

                 {'name': 'B1',
                  'type': 'Relu',
                  "attr": [
                      {"key": "_datadump_original_op_names",
                       "value": {"list": {"val_type": 1,
                                          "s": ["B"]}}},
                      {"key": "_L1_fusion_sub_graph_no",
                       "value": {'s': '1'}},
                  ],
                  "input": [
                      "A1:0",
                      "dynamic_const_471:0",
                      "dynamic_const_387:0"
                  ],
                  'output_desc': [
                      {'attr': [
                          {'key': '_datadump_origin_name',
                           'value': {'s': 'B'}},
                          {'key': '_datadump_origin_output_index',
                           'value': {'i': 0}},
                          {'key': '_datadump_origin_format',
                           'value': {'s': 'NCHW'}},
                      ]},
                  ]
                  },
                 {'name': 'B2',
                  'type': 'Relu',
                  "attr": [
                      {"key": "_datadump_original_op_names",
                       "value": {"list": {"val_type": 1,
                                          "s": ["B"]}}},
                      {"key": "_L1_fusion_sub_graph_no",
                       "value": {'s': '1'}},
                  ],
                  "input": [
                      "A2:0",
                      "dynamic_const_471:0",
                      "dynamic_const_387:0"
                  ],
                  'output_desc': [
                      {'attr': [
                          {'key': '_datadump_origin_name',
                           'value': {'s': 'B'}},
                          {'key': '_datadump_origin_output_index',
                           'value': {'i': 0}},
                          {'key': '_datadump_origin_format',
                           'value': {'s': 'NCHW'}},
                      ]},
                  ]
                  },
                 {'name': 'C1',
                  'type': 'Relu',
                  "attr": [
                      {"key": "_datadump_original_op_names",
                       "value": {"list": {"val_type": 1,
                                          "s": ["C"]}}},
                      {"key": "_L1_fusion_sub_graph_no",
                       "value": {'s': '1'}},
                  ],
                  "input": [
                      "B1:0",
                      "dynamic_const_471:0",
                      "dynamic_const_387:0"
                  ],
                  'output_desc': [
                      {'attr': [
                          {'key': '_datadump_origin_name',
                           'value': {'s': 'C'}},
                          {'key': '_datadump_origin_output_index',
                           'value': {'i': 0}},
                          {'key': '_datadump_origin_format',
                           'value': {'s': 'NCHW'}},
                      ]},
                  ]
                  },
                 {'name': 'C2',
                  'type': 'Relu',
                  "attr": [
                      {"key": "_datadump_original_op_names",
                       "value": {"list": {"val_type": 1,
                                          "s": ["C"]}}},
                      {"key": "_L1_fusion_sub_graph_no",
                       "value": {'s': '1'}},
                  ],
                  "input": [
                      "B2:0",
                      "dynamic_const_471:0",
                      "dynamic_const_387:0"
                  ],
                  'output_desc': [
                      {'attr': [
                          {'key': '_datadump_origin_name',
                           'value': {'s': 'C'}},
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

    @staticmethod
    def _make_L1_fusion_json_object():
        return {'name': 'resnet50', 'graph': [
            {'name': 'merge1', 'op':
                [{'name': 'data', 'type': 'input'},
                 {'name': 'A1',
                  'type': 'Relu',
                  "attr": [
                      {"key": "_datadump_original_op_names",
                       "value": {"list": {"val_type": 1,
                                          "s": ["A_conv1", "A_relu"]}}},
                      {"key": "_L1_fusion_sub_graph_no",
                       "value": {'s': '1'}},
                  ],
                  "input": [
                      "data:0",
                      "dynamic_const_471:0",
                      "dynamic_const_387:0"
                  ],
                  'output_desc': [
                      {'attr': [
                          {'key': '_datadump_origin_name',
                           'value': {'s': 'A_relu'}},
                          {'key': '_datadump_origin_output_index',
                           'value': {'i': 0}},
                          {'key': '_datadump_origin_format',
                           'value': {'s': 'NCHW'}},
                      ]},
                  ]
                  },
                 {'name': 'A2',
                  'type': 'Relu',
                  "attr": [
                      {"key": "_datadump_original_op_names",
                       "value": {"list": {"val_type": 1,
                                          "s": ["A_conv1", "A_relu"]}}},
                      {"key": "_L1_fusion_sub_graph_no",
                       "value": {'s': '1'}},
                  ],
                  "input": [
                      "data:0",
                      "dynamic_const_471:0",
                      "dynamic_const_387:0"
                  ],
                  'output_desc': [
                      {'attr': [
                          {'key': '_datadump_origin_name',
                           'value': {'s': 'A_relu'}},
                          {'key': '_datadump_origin_output_index',
                           'value': {'i': 0}},
                          {'key': '_datadump_origin_format',
                           'value': {'s': 'NCHW'}},
                      ]},
                  ]
                  },

                 {'name': 'B1',
                  'type': 'Relu',
                  "attr": [
                      {"key": "_datadump_original_op_names",
                       "value": {"list": {"val_type": 1,
                                          "s": ["B"]}}},
                      {"key": "_L1_fusion_sub_graph_no",
                       "value": {'s': '1'}},
                  ],
                  "input": [
                      "A1:0",
                      "dynamic_const_471:0",
                      "dynamic_const_387:0"
                  ],
                  'output_desc': [
                      {'attr': [
                          {'key': '_datadump_origin_name',
                           'value': {'s': 'B'}},
                          {'key': '_datadump_origin_output_index',
                           'value': {'i': 0}},
                          {'key': '_datadump_origin_format',
                           'value': {'s': 'NCHW'}},
                      ]},
                  ]
                  },
                 {'name': 'B2',
                  'type': 'Relu',
                  "attr": [
                      {"key": "_datadump_original_op_names",
                       "value": {"list": {"val_type": 1,
                                          "s": ["B"]}}},
                      {"key": "_L1_fusion_sub_graph_no",
                       "value": {'s': '1'}},
                  ],
                  "input": [
                      "A2:0",
                      "dynamic_const_471:0",
                      "dynamic_const_387:0"
                  ],
                  'output_desc': [
                      {'attr': [
                          {'key': '_datadump_origin_name',
                           'value': {'s': 'B'}},
                          {'key': '_datadump_origin_output_index',
                           'value': {'i': 0}},
                          {'key': '_datadump_origin_format',
                           'value': {'s': 'NCHW'}},
                      ]},
                  ]
                  },
                 {'name': 'C1',
                  'type': 'Relu',
                  "attr": [
                      {"key": "_datadump_original_op_names",
                       "value": {"list": {"val_type": 1,
                                          "s": ["C"]}}},
                      {"key": "_L1_fusion_sub_graph_no",
                       "value": {'s': '1'}},
                  ],
                  "input": [
                      "B1:0",
                      "dynamic_const_471:0",
                      "dynamic_const_387:0"
                  ],
                  'output_desc': [
                      {'attr': [
                          {'key': '_datadump_origin_name',
                           'value': {'s': 'C'}},
                          {'key': '_datadump_origin_output_index',
                           'value': {'i': 0}},
                          {'key': '_datadump_origin_format',
                           'value': {'s': 'NCHW'}},
                      ]},
                  ]
                  },
                 {'name': 'C2',
                  'type': 'Relu',
                  "attr": [
                      {"key": "_datadump_original_op_names",
                       "value": {"list": {"val_type": 1,
                                          "s": ["C"]}}},
                      {"key": "_L1_fusion_sub_graph_no",
                       "value": {'s': '1'}},
                  ],
                  "input": [
                      "B2:0",
                      "dynamic_const_471:0",
                      "dynamic_const_387:0"
                  ],
                  'output_desc': [
                      {'attr': [
                          {'key': '_datadump_origin_name',
                           'value': {'s': 'C'}},
                          {'key': '_datadump_origin_output_index',
                           'value': {'i': 0}},
                          {'key': '_datadump_origin_format',
                           'value': {'s': 'NCHW'}},
                      ]},
                  ]
                  },
                 ],
             },

        ]}

    @staticmethod
    def get_result_list(result):
        result_list = []
        for single_res in result:
            for item in single_res.result_list:
                result_list.append(item)
        return result_list

    def test_compare1(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-o',
                '/home/result.txt', '-d', 'prob', '-t', 'xxx']
        with pytest.raises(CompareError) as error:
            with mock.patch('sys.argv', args):
                main = compare_vector.VectorComparison()
                main.compare()
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_compare3(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result.txt']
        with pytest.raises(CompareError) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('os.listdir', side_effect=[['alg_CosineSimilarity.py'], []]):
                    with mock.patch('os.path.exists', return_value=True):
                        with mock.patch('os.access', return_value=False):
                            main = compare_vector.VectorComparison()
                            main.compare()
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_INVALID_PATH_ERROR)

    def test_compare4(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-q',
                '/home/a.json', '-o', '/home/result.txt']
        with pytest.raises(CompareError) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('os.listdir', side_effect=[['alg_CosineSimilarity.py'], []]):
                    with mock.patch('os.path.exists', return_value=True):
                        with mock.patch('os.access', return_value=False):
                            main = compare_vector.VectorComparison()
                            main.compare()
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_INVALID_PATH_ERROR)

    def test_compare6(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-q',
                '/home/a.json', '-o', '/home/result', '-d', 'prob', '-t', 'xxx']
        with pytest.raises(CompareError) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=True), \
                     mock.patch('os.access', return_value=True), \
                     mock.patch('os.path.isfile', return_value=True), \
                     mock.patch('os.stat', return_value=self.mock_stat_result):
                    main = compare_vector.VectorComparison()
                    main.compare()
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_compare7(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-q',
                '/home/a.json', '-o', '/home/result', '-d', 'prob', '-i', 'xxx']
        with pytest.raises(CompareError) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=True), \
                     mock.patch('os.access', return_value=True), \
                     mock.patch('os.path.isfile', return_value=True):
                    main = compare_vector.VectorComparison()
                    main.compare()
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_compare8(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-q',
                '/home/a.json', '-o', '/home/result', '-d', 'prob%$']
        with pytest.raises(CompareError) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=True), \
                     mock.patch('os.access', return_value=True), \
                     mock.patch('os.path.isfile', return_value=True), \
                     mock.patch('os.stat', return_value=self.mock_stat_result):
                    main = compare_vector.VectorComparison()
                    main.compare()
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_compare9(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result']
        with pytest.raises(CompareError) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=True), \
                     mock.patch('os.access', return_value=True), \
                     mock.patch('os.remove'), \
                     mock.patch('os.path.isdir', return_value=False), \
                     mock.patch('os.path.isfile', return_value=True), \
                     mock.patch('os.stat', return_value=self.mock_stat_result):
                    main = compare_vector.VectorComparison()
                    main.compare()
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_INVALID_PATH_ERROR)

    def test_compare10(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result']
        with pytest.raises(CompareError) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=True), \
                     mock.patch('os.access', return_value=True), \
                     mock.patch('os.remove'), \
                     mock.patch('os.listdir', side_effect=[['alg_CosineSimilarity.py'], []]), \
                     mock.patch('os.path.isdir', return_value=True), \
                     mock.patch('os.path.isfile', return_value=True), \
                     mock.patch('os.path.getsize', return_value=1024), \
                     mock.patch('os.stat', return_value=self.mock_stat_result):
                        main = compare_vector.VectorComparison()
                        main.compare()
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_DUMP_FILE_ERROR)

    def test_compare11(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result']
        with pytest.raises(CompareError) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=True), \
                     mock.patch('os.access', return_value=True), \
                     mock.patch('os.remove'), \
                     mock.patch('os.listdir', side_effect=[['alg_CosineSimilarity.py'], ['xx.g.gg.xx']]), \
                     mock.patch('os.path.isdir', return_value=True), \
                     mock.patch('os.path.isfile', return_value=True), \
                     mock.patch('os.path.getsize', return_value=1024), \
                     mock.patch('os.stat', return_value=self.mock_stat_result):
                    main = compare_vector.VectorComparison()
                    main.compare()
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_DUMP_FILE_ERROR)

    def test_compare12(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result']
        with pytest.raises(CompareError) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=True), \
                     mock.patch('os.access', return_value=True), \
                     mock.patch('os.remove'), \
                     mock.patch('os.listdir',
                                side_effect=[['alg_CosineSimilarity.py'], ['aaa.aaa.0.1111111111111111',
                                                                           'aaa.0.1111111111111111.quant']]), \
                     mock.patch('os.path.isdir', return_value=True), \
                     mock.patch('os.path.isfile', return_value=True), \
                     mock.patch('os.path.getsize', return_value=1024), \
                     mock.patch('os.stat', return_value=self.mock_stat_result):
                    main = compare_vector.VectorComparison()
                    main.compare()
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_DUMP_FILE_ERROR)

    def test_compare13(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result']
        with pytest.raises(CompareError) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=True), \
                     mock.patch('os.access', return_value=True), \
                     mock.patch('os.remove'), \
                     mock.patch('os.listdir',
                                side_effect=[['alg_CosineSimilarity.py'], ['aaa.0.1111111111111111.pb']]), \
                     mock.patch('os.path.isdir', side_effect=[True, False]), \
                     mock.patch('os.path.isfile', return_value=True), \
                     mock.patch('os.path.getsize', return_value=1024), \
                     mock.patch('os.stat', return_value=self.mock_stat_result):
                    main = compare_vector.VectorComparison()
                    main.compare()
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_INVALID_PATH_ERROR)

    def test_compare14(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result']
        with pytest.raises(CompareError) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=True), \
                     mock.patch('os.access', return_value=True), \
                     mock.patch('os.remove'), \
                     mock.patch('os.listdir',
                                side_effect=[['alg_CosineSimilarity.py'], ['aaa.0.1111111111111111.pb'],
                                             ['aaa.0.1111111111111111.pb']]), \
                     mock.patch('os.path.isdir', return_value=True), \
                     mock.patch('os.path.isfile', return_value=True), \
                     mock.patch('os.path.getsize', return_value=1024), \
                     mock.patch('os.stat', return_value=self.mock_stat_result):
                    main = compare_vector.VectorComparison()
                    main.compare()
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_INVALID_DUMP_TYPE_ERROR)

    def test_compare15(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result']
        with pytest.raises(CompareError) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=True), \
                     mock.patch('os.access', return_value=True), \
                     mock.patch('os.remove'), \
                     mock.patch('os.listdir',
                                side_effect=[['alg_CosineSimilarity.py'], ['ccc.aaa.0.1111111111111111',
                                                                           'ccc.aaa.1.1111111111111111'],
                                             ['ccc.aaa.0.1111111111111111',
                                              'convert_failed_file_list.txt']]), \
                     mock.patch('os.path.isdir', return_value=True), \
                     mock.patch('os.path.isfile', return_value=True), \
                     mock.patch('os.path.getsize', return_value=1024), \
                     mock.patch('os.stat', return_value=self.mock_stat_result):
                    main = compare_vector.VectorComparison()
                    main.compare()
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_compare16(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result']
        with pytest.raises(CompareError) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=True), \
                     mock.patch('os.access', return_value=True), \
                     mock.patch('os.remove'), \
                     mock.patch('os.listdir',
                                side_effect=[['alg_CosineSimilarity.py'], ['aaa.0.1111111111111111.dump'],
                                             ['aaa.0.1111111111111111.dump']]), \
                     mock.patch('os.path.isdir', return_value=True), \
                     mock.patch('os.path.isfile', return_value=True), \
                     mock.patch('os.path.getsize', return_value=1024), \
                     mock.patch('os.stat', return_value=self.mock_stat_result):
                    main = compare_vector.VectorComparison()
                    main.compare()
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_DUMP_FILE_ERROR)

    def test_compare17(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-q', '/honm/b.json', '-o', '/home/result']
        with pytest.raises(CompareError) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=True), \
                     mock.patch('os.access', return_value=True), \
                     mock.patch('os.remove'), \
                     mock.patch('os.listdir',
                                side_effect=[['alg_CosineSimilarity.py'], ['aaa.0.1111111111111111.quant'],
                                             ['aaa.0.1111111111111111.dump']]), \
                     mock.patch('os.path.isdir', return_value=True), \
                     mock.patch('os.path.isfile', return_value=True), \
                     mock.patch('os.path.getsize', return_value=1024), \
                     mock.patch('os.stat', return_value=self.mock_stat_result):
                    main = compare_vector.VectorComparison()
                    main.compare()
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_DUMP_FILE_ERROR)

    def test_compare18(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-q', '/honm/b.json', '-o', '/home/result']
        with pytest.raises(CompareError) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=True), \
                     mock.patch('os.access', return_value=True), \
                     mock.patch('os.remove'), \
                     mock.patch('os.listdir',
                                side_effect=[['alg_CosineSimilarity.py'], ['aaa.0.1111111111111111.dump'],
                                             ['aaa.0.1111111111111111.quant']]), \
                     mock.patch('os.path.isdir', return_value=True), \
                     mock.patch('os.path.isfile', return_value=True), \
                     mock.patch('os.path.getsize', return_value=1024), \
                     mock.patch('os.stat', return_value=self.mock_stat_result):
                    main = compare_vector.VectorComparison()
                    main.compare()
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_DUMP_FILE_ERROR)

    def test_compare19(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result']
        with pytest.raises(CompareError) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=True), \
                     mock.patch('os.access', return_value=True), \
                     mock.patch('os.remove'), \
                     mock.patch('os.listdir',
                                side_effect=[['alg_CosineSimilarity.py'], ['aaa.0.1111111111111111.quant'],
                                             ['aaa.0.1111111111111111.dump']]), \
                     mock.patch('os.path.isdir', return_value=True), \
                     mock.patch('os.path.isfile', return_value=True), \
                     mock.patch('os.path.getsize', return_value=1024), \
                     mock.patch('os.stat', return_value=self.mock_stat_result):
                    main = compare_vector.VectorComparison()
                    main.compare()
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_DUMP_FILE_ERROR)

    def test_compare20(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-q',
                '/home/a.json', '-o', '/home/result']
        with pytest.raises(CompareError) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=True), \
                     mock.patch('os.access', return_value=True), \
                     mock.patch('os.remove'), \
                     mock.patch('os.listdir',
                                side_effect=[['alg_CosineSimilarity.py'], ['aaa.0.1111111111111111.dump'],
                                             ['aaa.0.1111111111111111.pb']]), \
                     mock.patch('os.path.isdir', return_value=True), \
                     mock.patch('os.path.isfile', return_value=True), \
                     mock.patch('os.path.getsize', return_value=1024), \
                     mock.patch('os.stat', return_value=self.mock_stat_result):
                    main = compare_vector.VectorComparison()
                    main.compare()
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_DUMP_FILE_ERROR)

    def test_compare21(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-q',
                '/home/a.json', '-o', '/home/result']
        with pytest.raises(CompareError) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=True), \
                     mock.patch('os.access', return_value=True), \
                     mock.patch('os.remove'), \
                     mock.patch('os.listdir',
                                side_effect=[['alg_CosineSimilarity.py'], ['aaa.0.1111111111111111.quant'],
                                             ['aaa.0.1111111111111111.dump']]), \
                     mock.patch('os.path.isdir', return_value=True), \
                     mock.patch('os.path.isfile', return_value=True), \
                     mock.patch('os.path.getsize', return_value=1024), \
                     mock.patch('os.stat', return_value=self.mock_stat_result):
                    main = compare_vector.VectorComparison()
                    main.compare()
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_DUMP_FILE_ERROR)

    def test_compare22(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-q',
                '/home/a.json', '-o', '/home/result']
        with pytest.raises(CompareError) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=True), \
                     mock.patch('os.access', return_value=True), \
                     mock.patch('os.remove'), \
                     mock.patch('os.listdir',
                                side_effect=[['alg_CosineSimilarity.py'], ['aaa.0.1111111111111111.quant'],
                                             ['aaa.0.1111111111111111.pb'],
                                             ['convert_NC1HWC0_to_NCHW.py']]), \
                     mock.patch('os.path.isdir', return_value=True), \
                     mock.patch('os.path.isfile', return_value=True), \
                     mock.patch('os.stat', return_value=self.mock_stat_result), \
                     mock.patch("os.path.getsize", side_effect=FileNotFoundError):
                    main = compare_vector.VectorComparison()
                    ret = main.compare()
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_OPEN_FILE_ERROR)

    def test_compare23(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result']
        with pytest.raises(CompareError) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=True), \
                     mock.patch('os.access', return_value=True), \
                     mock.patch('os.remove'), \
                     mock.patch('os.listdir',
                                side_effect=[['alg_CosineSimilarity.py'], ['ccc.aaa.0.1111111111111111'],
                                             ['aaa.0.1111111111111111.pb'],
                                             ['convert_NC1HWC0_to_NCHW.py']]), \
                     mock.patch('os.path.isdir', return_value=True), \
                     mock.patch('os.path.isfile', return_value=True), \
                     mock.patch('os.stat', return_value=self.mock_stat_result):
                    with mock.patch("os.path.getsize", return_value=100):
                        with mock.patch('builtins.open',
                                        mock.mock_open(read_data=b'01x03')):
                            main = compare_vector.VectorComparison()
                            ret = main.compare()
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_PARSER_JSON_FILE_ERROR)

    def test_compare24(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-o',
                '/home/result']
        with pytest.raises(CompareError) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=True), \
                     mock.patch('os.access', return_value=True), \
                     mock.patch('os.remove'), \
                     mock.patch('os.listdir',
                                side_effect=[['alg_CosineSimilarity.py'], ['aaa.0.1111111111111111.dump'],
                                             ['aaa.0.1111111111111111.dump']]), \
                     mock.patch('os.path.isdir', return_value=True), \
                     mock.patch('os.path.isfile', return_value=True), \
                     mock.patch('os.path.getsize', return_value=1024), \
                     mock.patch('os.stat', return_value=self.mock_stat_result):
                    main = compare_vector.VectorComparison()
                    main.compare()
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_DUMP_FILE_ERROR)

    def test_compare25(self):
        args = ['aaa.py', '-l', '/home/left', '-s', '/home/right', '-o',
                '/home/result.txt', '-d', 'prob', '-t', 'xxx']
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                main = compare_vector.VectorComparison()
                main.compare()
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR)

    def test_compare_vector1(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result.txt']
        arguments = mock.Mock()
        arguments.fusion_rule_file = "/home/b.json"
        arguments.quant_fusion_rule_file = ""
        arguments.close_fusion_rule_file = ""
        arguments.my_dump_path = "/home/demo"
        arguments.golden_dump_path = "/home/dt"
        arguments.dump_version = 1
        arguments.max_cmp_size = 1024
        arguments.op_name = ""
        arguments.output_path = "/home/de"
        arguments.custom_script_path = ""
        arguments.algorithm = 'all'
        arguments.algorithm_options = ''
        arguments.range = None
        arguments.select = None
        dump_data = DumpData()
        dump_data.input.append(
            self._make_op_input(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data.output.append(
            self._make_op_output(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data = dump_utils.convert_dump_data(dump_data)
        result = [[0, True, "data&message"]]
        multiprocessing.Manager = mock.Mock
        multiprocessing.Manager.RLock = mock.Mock
        with mock.patch('sys.argv', args):
            with mock.patch('os.path.exists', return_value=True), \
                 mock.patch('os.access', return_value=True), \
                 mock.patch('os.remove'), \
                 mock.patch('os.listdir',
                            side_effect=[['alg_CosineSimilarity.py'],
                                         ['ReduceMeanD.conv1conv1_relu.6.4.1613727240764749'],
                                         ['input.4.1613727240736566.pb',
                                          'trans_Cast_1167.4.1613727241293941.pb'],
                                         ['convert_NC1HWC0_to_NCHW.py']]), \
                 mock.patch('os.path.isdir', return_value=True), \
                 mock.patch('os.path.isfile', return_value=True), \
                 mock.patch('dump_parse.dump_utils.parse_dump_file', return_value=dump_data), \
                 mock.patch('os.stat', return_value=self.mock_stat_result):
                with mock.patch("os.path.getsize", return_value=100):
                    with mock.patch("json.load", return_value=self._make_json_object()):
                        with mock.patch('builtins.open', mock.mock_open(
                                read_data=self._make_csv_content())):
                            with mock.patch('os.open') as open_file, \
                                    mock.patch('os.fdopen'):
                                with mock.patch("multiprocessing.pool.ApplyResult.get", return_value=result):
                                    with mock.patch("os.path.realpath", return_value="/home/demo/a.json"):
                                        open_file.write = None
                                        main = compare_vector.VectorComparison(arguments)
                                        main.compare()
                                        key = \
                                            main.compare_rule.fusion_info.op_name_to_fusion_op_name_map[
                                                'conv1conv1_relu']
                                        ret, dump_match, result = main._compare_by_fusion_op(
                                            key)
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)
        self.assertEqual(dump_match, True)

    def test_compare_vector2(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result.txt']
        arguments = mock.Mock()
        arguments.fusion_rule_file = "/home/b.json"
        arguments.quant_fusion_rule_file = ""
        arguments.close_fusion_rule_file = ""
        arguments.my_dump_path = "/home/demo"
        arguments.golden_dump_path = "/home/dt"
        arguments.dump_version = 1
        arguments.max_cmp_size = 0
        arguments.op_name = ""
        arguments.output_path = "/home/de"
        arguments.custom_script_path = ""
        arguments.algorithm = 'all'
        arguments.algorithm_options = ''
        arguments.range = None
        arguments.select = None
        multiprocessing.Manager = mock.Mock
        multiprocessing.Manager.RLock = mock.Mock
        dump_data = DumpData()
        dump_data.input.append(
            self._make_op_input(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data.output.append(
            self._make_op_output(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data = dump_utils.convert_dump_data(dump_data)
        result = [[0, True, "data&message"]]
        with mock.patch('sys.argv', args):
            with mock.patch('os.path.exists', return_value=True), \
                 mock.patch('os.access', return_value=True), \
                 mock.patch('os.remove'), \
                 mock.patch('os.listdir',
                            side_effect=[
                                ['alg_CosineSimilarity.py'],
                                ['xxxx.conv1conv1_relu.0.1111111111111111'],
                                ['data.0.1111111111111111.pb',
                                 'conv1_relu.0.1111111111111111.pb'],
                                ['convert_NC1HWC0_to_NCHW.py']]), \
                 mock.patch('os.path.isdir', return_value=True), \
                 mock.patch('os.path.isfile', return_value=True), \
                 mock.patch('dump_parse.dump_utils.parse_dump_file', return_value=dump_data), \
                 mock.patch('os.stat', return_value=self.mock_stat_result):
                with mock.patch("os.path.getsize", return_value=100):
                    with mock.patch("json.load", return_value=self._make_json_object()):
                        with mock.patch('builtins.open',
                                        mock.mock_open(
                                            read_data=self._make_csv_content())):
                            with mock.patch("multiprocessing.pool.ApplyResult.get", return_value=result):
                                with mock.patch('os.open') as open_file, \
                                        mock.patch('os.fdopen'):
                                    open_file.write = None
                                    main = compare_vector.VectorComparison(arguments)
                                    main.compare()
                                    key = \
                                        main.compare_rule.fusion_info.op_name_to_fusion_op_name_map[
                                            'conv1conv1_relu']
                                    ret, dump_match, result = main._compare_by_fusion_op(
                                        key)
                                    result_list = self.get_result_list(result)
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)
        self.assertEqual(dump_match, True)
        self.assertEqual(len(result_list), 2)

    def test_compare_vector3(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-o',
                '/home/result.txt']
        arguments = mock.Mock()
        arguments.fusion_rule_file = ""
        arguments.quant_fusion_rule_file = ""
        arguments.close_fusion_rule_file = ""
        arguments.my_dump_path = "/home/demo"
        arguments.golden_dump_path = "/home/dt"
        arguments.dump_version = 1
        arguments.max_cmp_size = 0
        arguments.op_name = ""
        arguments.output_path = "/home/de"
        arguments.custom_script_path = ""
        arguments.algorithm = 'all'
        arguments.algorithm_options = ''
        arguments.range = None
        arguments.select = None
        multiprocessing.Manager = mock.Mock
        multiprocessing.Manager.RLock = mock.Mock
        dump_data = DumpData()
        dump_data.input.append(
            self._make_op_input(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data.output.append(
            self._make_op_output(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data = dump_utils.convert_dump_data(dump_data)
        result = [[0, True, "data&message"]]
        with mock.patch('sys.argv', args):
            with mock.patch('os.path.exists', return_value=True), \
                mock.patch('os.access', return_value=True), \
                mock.patch('os.remove'), \
                mock.patch('os.listdir', side_effect=self._fake_listdir), \
                 mock.patch('os.path.isdir', return_value=True), \
                 mock.patch('os.path.isfile', return_value=True), \
                 mock.patch('dump_parse.dump_utils.parse_dump_file', return_value=dump_data), \
                 mock.patch('os.stat', return_value=self.mock_stat_result):
                with mock.patch("json.load", return_value=self._make_json_object()):
                    with mock.patch('builtins.open',
                                    mock.mock_open(
                                        read_data=self._make_csv_content())):
                        with mock.patch('os.open') as open_file, \
                                mock.patch('os.fdopen'), \
                                mock.patch('os.path.getsize', return_value=1024):
                            with mock.patch("multiprocessing.pool.ApplyResult.get", return_value=result):
                                open_file.write = None
                                main = compare_vector.VectorComparison(arguments)
                                main.compare()
                                ret, dump_match, result = main._compare_by_fusion_op(
                                    'conv1conv1_relu')
                                result_list = self.get_result_list(result)
        print("******************")
        print(ret)
        print("******************")
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)
        self.assertEqual(dump_match, True)
        self.assertEqual(len(result_list), 2)

    def test_compare_vector4(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-o',
                '/home/result.txt']
        arguments = mock.Mock()
        arguments.fusion_rule_file = ""
        arguments.quant_fusion_rule_file = ""
        arguments.close_fusion_rule_file = ""
        arguments.my_dump_path = "/home/demo"
        arguments.golden_dump_path = "/home/dt"
        arguments.dump_version = 1
        arguments.max_cmp_size = 0
        arguments.op_name = ""
        arguments.output_path = "/home/de"
        arguments.custom_script_path = ""
        arguments.algorithm = 'all'
        arguments.algorithm_options = ''
        arguments.range = None
        arguments.select = None
        multiprocessing.Manager = mock.Mock
        multiprocessing.Manager.RLock = mock.Mock
        dump_data = DumpData()
        dump_data.input.append(
            self._make_op_input(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data.output.append(
            self._make_op_output(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data = dump_utils.convert_dump_data(dump_data)
        result = [[0, True, "data&message"]]
        with mock.patch('sys.argv', args):
            with mock.patch('os.path.exists', return_value=True), \
                mock.patch('os.access', return_value=True), \
                mock.patch('os.remove'), \
                mock.patch('os.listdir', side_effect=self._fake_listdir), \
                 mock.patch('os.path.isdir', return_value=True), \
                 mock.patch('os.path.isfile', return_value=True), \
                 mock.patch('dump_parse.dump_utils.parse_dump_file', return_value=dump_data), \
                 mock.patch('os.stat', return_value=self.mock_stat_result):
                with mock.patch("json.load", return_value=self._make_json_object()):
                    with mock.patch('builtins.open',
                                    mock.mock_open(
                                        read_data=self._make_csv_content())):
                        with mock.patch('os.open') as open_file, \
                                mock.patch('os.fdopen'), \
                                mock.patch('os.path.getsize', return_value=1024):
                            with mock.patch("multiprocessing.pool.ApplyResult.get", return_value=result):
                                open_file.write = None
                                main = compare_vector.VectorComparison(arguments)
                                main.compare()
                                ret, dump_match, result = main._compare_by_fusion_op(
                                    'ccccc')
                                result_list = self.get_result_list(result)
        self.assertEqual(ret, CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR)
        self.assertEqual(dump_match, False)
        self.assertEqual(len(result_list), 1)

    def test_compare_vector5(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-o',
                '/home/result.txt']
        arguments = mock.Mock()
        arguments.fusion_rule_file = ""
        arguments.quant_fusion_rule_file = ""
        arguments.close_fusion_rule_file = ""
        arguments.my_dump_path = "/home/demo"
        arguments.golden_dump_path = "/home/dt"
        arguments.dump_version = 1
        arguments.max_cmp_size = 0
        arguments.op_name = ""
        arguments.output_path = "/home/de"
        arguments.custom_script_path = ""
        arguments.algorithm = 'all'
        arguments.algorithm_options = ''
        arguments.range = None
        arguments.select = None
        multiprocessing.Manager = mock.Mock
        multiprocessing.Manager.RLock = mock.Mock
        dump_data = DumpData()
        dump_data.input.append(
            self._make_op_input(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data.output.append(
            self._make_op_output(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data = dump_utils.convert_dump_data(dump_data)
        result = [[0, True, "data&message"]]
        with mock.patch('sys.argv', args):
            with mock.patch('os.path.exists', return_value=True), \
                mock.patch('os.access', return_value=True), \
                mock.patch('os.remove'), \
                mock.patch('os.listdir', side_effect=self._fake_listdir), \
                 mock.patch('os.path.isdir', return_value=True), \
                 mock.patch('os.path.isfile', return_value=True), \
                 mock.patch('dump_parse.dump_utils.parse_dump_file', return_value=dump_data), \
                 mock.patch('os.stat', return_value=self.mock_stat_result):
                with mock.patch("json.load", return_value=self._make_json_object()):
                    with mock.patch('builtins.open',
                                    mock.mock_open(
                                        read_data=self._make_csv_content())):
                        with mock.patch('os.open') as open_file, \
                                mock.patch('os.fdopen'), \
                                mock.patch('os.path.getsize', return_value=1024):
                            with mock.patch("multiprocessing.pool.ApplyResult.get", return_value=result):
                                open_file.write = None
                                main = compare_vector.VectorComparison(arguments)
                                main.compare()
                                ret, dump_match, result = main._compare_by_fusion_op(
                                    'ddddd')
                                result_list = self.get_result_list(result)
        self.assertEqual(ret, CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR)
        self.assertEqual(dump_match, False)
        self.assertEqual(len(result_list), 1)

    def test_compare_vector6(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result.txt']
        arguments = mock.Mock()
        arguments.fusion_rule_file = "/home/a.json"
        arguments.quant_fusion_rule_file = ""
        arguments.close_fusion_rule_file = ""
        arguments.my_dump_path = "/home/demo"
        arguments.golden_dump_path = "/home/dt"
        arguments.dump_version = 1
        arguments.max_cmp_size = 0
        arguments.op_name = ""
        arguments.output_path = "/home/de"
        arguments.custom_script_path = ""
        arguments.algorithm = 'all'
        arguments.algorithm_options = ''
        arguments.range = None
        arguments.select = None
        multiprocessing.Manager = mock.Mock
        multiprocessing.Manager.RLock = mock.Mock
        dump_data = DumpData()
        dump_data.input.append(
            self._make_op_input(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data.output.append(
            self._make_op_output(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data = dump_utils.convert_dump_data(dump_data)
        result = [[0, True, "data&message"]]
        with mock.patch('sys.argv', args):
            with mock.patch('os.path.exists', return_value=True), \
                 mock.patch('os.access', return_value=True), \
                 mock.patch('os.remove'), \
                 mock.patch('os.listdir',
                            side_effect=[['alg_CosineSimilarity.py'],
                                         ['xxxx.conv1conv1_relu.0.1111111111111111'],
                                         ['data.0.1111111111111111.pb',
                                          'conv1_relu.0.1111111111111111.pb'],
                                         ['convert_NC1HWC0_to_NCHW.py']]), \
                 mock.patch('os.path.isdir', return_value=True), \
                 mock.patch('os.path.isfile', return_value=True), \
                 mock.patch('dump_parse.dump_utils.parse_dump_file', return_value=dump_data), \
                 mock.patch('os.stat', return_value=self.mock_stat_result):
                with mock.patch("os.path.getsize", return_value=100):
                    with mock.patch("json.load", return_value=self._make_json()):
                        with mock.patch('builtins.open',
                                        mock.mock_open(
                                            read_data=self._make_csv_content())):
                            with mock.patch('os.open') as open_file, \
                                    mock.patch('os.fdopen'):
                                with mock.patch("multiprocessing.pool.ApplyResult.get", return_value=result):
                                    open_file.write = None
                                    main = compare_vector.VectorComparison(arguments)
                                    main.compare()
                                    key = \
                                        main.compare_rule.fusion_info.op_name_to_fusion_op_name_map[
                                            'conv1conv1_relu']
                                    ret, dump_match, result = main._compare_by_fusion_op(
                                        key)
                                    result_list = self.get_result_list(result)
        self.assertEqual(ret, CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR)
        self.assertEqual(dump_match, True)
        self.assertEqual(len(result_list), 2)

    def test_compare_multi_result_empty(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result.txt']
        arguments = mock.Mock()
        arguments.fusion_rule_file = "/home/a.json"
        arguments.quant_fusion_rule_file = ""
        arguments.close_fusion_rule_file = ""
        arguments.my_dump_path = "/home/demo"
        arguments.golden_dump_path = "/home/dt"
        arguments.dump_version = 1
        arguments.max_cmp_size = 0
        arguments.op_name = ""
        arguments.output_path = "/home/de"
        arguments.custom_script_path = ""
        arguments.algorithm = 'all'
        arguments.algorithm_options = ''
        arguments.range = None
        arguments.select = None
        multiprocessing.Manager = mock.Mock
        multiprocessing.Manager.RLock = mock.Mock
        dump_data = DumpData()
        dump_data = dump_utils.convert_dump_data(dump_data)
        result = [[0, True, "data&message"]]
        with mock.patch('sys.argv', args):
            with mock.patch('os.path.exists', return_value=True), \
                 mock.patch('os.access', return_value=True), \
                 mock.patch('os.remove'), \
                 mock.patch('os.listdir',
                            side_effect=[['alg_CosineSimilarity.py'],
                                         ['xxxx.conv1conv1_relu.0.1111111111111111'],
                                         ['data.0.1111111111111111.pb',
                                          'conv1_relu.0.1111111111111111.pb'],
                                         ['convert_NC1HWC0_to_NCHW.py']]), \
                 mock.patch('os.path.isdir', return_value=True), \
                 mock.patch('os.path.isfile', return_value=True), \
                 mock.patch('dump_parse.dump_utils.parse_dump_file', return_value=dump_data), \
                 mock.patch('os.stat', return_value=self.mock_stat_result):
                with mock.patch("os.path.getsize", return_value=100):
                    with mock.patch("json.load", return_value=self._make_json()):
                        with mock.patch('builtins.open',
                                        mock.mock_open(
                                            read_data=self._make_csv_content())):
                            with mock.patch('os.open') as open_file, \
                                    mock.patch('os.fdopen'):
                                with mock.patch("multiprocessing.pool.ApplyResult.get", return_value=result):
                                    open_file.write = None
                                    main = compare_vector.VectorComparison(arguments)
                                    main.compare()
                                    key = \
                                        main.compare_rule.fusion_info.op_name_to_fusion_op_name_map[
                                            'conv1conv1_relu']
                                    ret, dump_match, result = main._compare_by_fusion_op(
                                        key)
                                    result_list = self.get_result_list(result)
        self.assertEqual(ret, CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR)
        self.assertEqual(dump_match, True)
        self.assertEqual(len(result_list), 1)

    def test_compare_vector7(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result.txt']
        arguments = mock.Mock()
        arguments.fusion_rule_file = "/home/a.json"
        arguments.quant_fusion_rule_file = ""
        arguments.close_fusion_rule_file = ""
        arguments.my_dump_path = "/home/demo"
        arguments.golden_dump_path = "/home/dt"
        arguments.dump_version = 1
        arguments.max_cmp_size = 0
        arguments.op_name = ""
        arguments.output_path = "/home/de"
        arguments.custom_script_path = ""
        arguments.algorithm = 'all'
        arguments.algorithm_options = ''
        arguments.range = None
        arguments.select = None
        multiprocessing.Manager = mock.Mock
        multiprocessing.Manager.RLock = mock.Mock
        dump_data = DumpData()
        dump_data.input.append(
            self._make_op_input(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data.output.append(
            self._make_op_output(DD.FORMAT_NCHW, [1, 1, 1, 1]))
        result = [[0, True, "data&message"]]
        dump_data = dump_utils.convert_dump_data(dump_data)
        with mock.patch('sys.argv', args):
            with mock.patch('os.path.exists', return_value=True), \
                 mock.patch('os.access', return_value=True), \
                 mock.patch('os.remove'), \
                 mock.patch('os.listdir',
                            side_effect=[['alg_CosineSimilarity.py'],
                                         ['xxxx.conv1conv1_relu.0.1111111111111111'],
                                         ['data.0.1111111111111111.pb',
                                          'conv1_relu.0.1111111111111111.pb'],
                                         ['convert_NC1HWC0_to_NCHW.py']]), \
                 mock.patch('os.path.isdir', return_value=True), \
                 mock.patch('os.path.isfile', return_value=True), \
                 mock.patch('dump_parse.dump_utils.parse_dump_file', return_value=dump_data), \
                 mock.patch('os.stat', return_value=self.mock_stat_result):
                with mock.patch("os.path.getsize", return_value=100):
                    with mock.patch("json.load", return_value=self._make_json()):
                        with mock.patch('builtins.open',
                                        mock.mock_open(
                                            read_data=self._make_csv_content())):
                            with mock.patch('os.open') as open_file, \
                                    mock.patch('os.fdopen'):
                                with mock.patch("multiprocessing.pool.ApplyResult.get", return_value=result):
                                    open_file.write = None
                                    main = compare_vector.VectorComparison(arguments)
                                    main.compare()
                                    key = \
                                        main.compare_rule.fusion_info.op_name_to_fusion_op_name_map[
                                            'conv1conv1_relu']
                                    ret, dump_match, result = main._compare_by_fusion_op(
                                        key)
                                    result_list = self.get_result_list(result)
        self.assertEqual(ret, CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR)
        self.assertEqual(dump_match, True)
        self.assertEqual(len(result_list), 2)

    def test_compare_vector8(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result.txt']
        arguments = mock.Mock()
        arguments.fusion_rule_file = "/home/a.json"
        arguments.quant_fusion_rule_file = ""
        arguments.close_fusion_rule_file = ""
        arguments.my_dump_path = "/home/demo"
        arguments.golden_dump_path = "/home/dt"
        arguments.dump_version = 1
        arguments.max_cmp_size = 0
        arguments.op_name = ""
        arguments.output_path = "/home/de"
        arguments.custom_script_path = ""
        arguments.algorithm = 'all'
        arguments.algorithm_options = ''
        arguments.range = None
        arguments.select = None
        multiprocessing.Manager = mock.Mock
        multiprocessing.Manager.RLock = mock.Mock
        dump_data = DumpData()
        dump_data.input.append(
            self._make_op_input(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data.output.append(
            self._make_op_output(DD.FORMAT_NCHW, [0, 1, 1]))
        dump_data = dump_utils.convert_dump_data(dump_data)
        result = [[0, True, "data&message"]]
        with mock.patch('sys.argv', args):
            with mock.patch('os.path.exists', return_value=True), \
                 mock.patch('os.access', return_value=True), \
                 mock.patch('os.remove'), \
                 mock.patch('os.listdir',
                            side_effect=[['alg_CosineSimilarity.py'],
                                         ['xxxx.conv1conv1_relu.0.1111111111111111'],
                                         ['data.0.1111111111111111.pb',
                                          'conv1_relu.0.1111111111111111.pb'],
                                         ['convert_NC1HWC0_to_NCHW.py']]), \
                 mock.patch('os.path.isdir', return_value=True), \
                 mock.patch('os.path.isfile', return_value=True), \
                 mock.patch('dump_parse.dump_utils.parse_dump_file', return_value=dump_data), \
                 mock.patch('os.stat', return_value=self.mock_stat_result):
                with mock.patch("os.path.getsize", return_value=100):
                    with mock.patch("json.load", return_value=self._make_json()):
                        with mock.patch('builtins.open',
                                        mock.mock_open(
                                            read_data=self._make_csv_content())):
                            with mock.patch('os.open') as open_file, \
                                    mock.patch('os.fdopen'):
                                with mock.patch("multiprocessing.pool.ApplyResult.get", return_value=result):
                                    open_file.write = None
                                    main = compare_vector.VectorComparison(arguments)
                                    main.compare()
                                    key = \
                                        main.compare_rule.fusion_info.op_name_to_fusion_op_name_map[
                                            'conv1conv1_relu']
                                    ret, dump_match, result = main._compare_by_fusion_op(
                                        key)
                                    result_list = self.get_result_list(result)
        self.assertEqual(ret, CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR)
        self.assertEqual(dump_match, True)
        self.assertEqual(len(result_list), 2)

    def test_compare_vector9(self):
        arguments = mock.Mock()
        arguments.fusion_rule_file = "/home/a.json"
        arguments.quant_fusion_rule_file = ""
        arguments.close_fusion_rule_file = ""
        arguments.my_dump_path = "/home/demo"
        arguments.golden_dump_path = "/home/dt"
        arguments.dump_version = 1
        arguments.max_cmp_size = 0
        arguments.op_name = ""
        arguments.output_path = "/home/de"
        arguments.custom_script_path = ""
        arguments.algorithm = 'all'
        arguments.algorithm_options = ''
        arguments.range = None
        arguments.select = None
        with mock.patch("compare_vector.VectorComparison._compare_by_multi_process", return_value=(0, False)):
                with mock.patch("cmp_utils.utils.sort_result_file_by_index"):
                    compare_instance = compare_vector.VectorComparison(arguments)
                    compare_instance._compare_vector()
                    compare_instance.args["range"] = '[1,-1,2]'
                    compare_instance._compare_vector()

    def test_compare_vector_range1(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result.txt', '-r', ',,']
        arguments = mock.Mock()
        arguments.fusion_rule_file = "/home/a.json"
        arguments.quant_fusion_rule_file = ""
        arguments.close_fusion_rule_file = ""
        arguments.my_dump_path = "/home/demo"
        arguments.golden_dump_path = "/home/dt"
        arguments.dump_version = 1
        arguments.max_cmp_size = 0
        arguments.op_name = ""
        arguments.output_path = "/home/de"
        arguments.custom_script_path = ""
        arguments.algorithm = 'all'
        arguments.algorithm_options = ''
        arguments.range = ',,'
        arguments.select = None
        multiprocessing.Manager = mock.Mock
        multiprocessing.Manager.RLock = mock.Mock
        dump_data = DumpData()
        dump_data.input.append(
            self._make_op_input(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data.output.append(
            self._make_op_output(DD.FORMAT_NCHW, [0, 1, 1]))
        dump_data = dump_utils.convert_dump_data(dump_data)
        result = [[0, True, "data&message"]]
        with mock.patch('sys.argv', args):
            with mock.patch('os.path.exists', return_value=True), \
                 mock.patch('os.access', return_value=True), \
                 mock.patch('os.remove'), \
                 mock.patch('os.listdir',
                            side_effect=[['alg_CosineSimilarity.py'],
                                         ['xxxx.conv1conv1_relu.0.1111111111111111'],
                                         ['data.0.1111111111111111.pb',
                                          'conv1_relu.0.1111111111111111.pb'],
                                         ['convert_NC1HWC0_to_NCHW.py']]), \
                 mock.patch('os.path.isdir', return_value=True), \
                 mock.patch('os.path.isfile', return_value=True), \
                 mock.patch('dump_parse.dump_utils.parse_dump_file', return_value=dump_data), \
                 mock.patch('os.stat', return_value=self.mock_stat_result):
                with mock.patch("os.path.getsize", return_value=100):
                    with mock.patch("json.load", return_value=self._make_json()):
                        with mock.patch('builtins.open',
                                        mock.mock_open(
                                            read_data=self._make_csv_content())):
                            with mock.patch('os.open') as open_file, \
                                    mock.patch('os.fdopen'):
                                with mock.patch("multiprocessing.pool.ApplyResult.get", return_value=result):
                                    open_file.write = None
                                    main = compare_vector.VectorComparison(arguments)
                                    main.compare()
                                    key = \
                                        main.compare_rule.fusion_info.op_name_to_fusion_op_name_map[
                                            'conv1conv1_relu']
                                    ret, dump_match, result = main._compare_by_fusion_op(
                                        key)
                                    result_list = self.get_result_list(result)
        self.assertEqual(ret, CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR)
        self.assertEqual(dump_match, True)
        self.assertEqual(len(result_list), 2)

    def test_compare_vector_l1_fusion1(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result.txt']
        arguments = mock.Mock()
        arguments.fusion_rule_file = "/home/a.json"
        arguments.quant_fusion_rule_file = ""
        arguments.close_fusion_rule_file = ""
        arguments.my_dump_path = "/home/demo"
        arguments.golden_dump_path = "/home/dt"
        arguments.dump_version = 1
        arguments.max_cmp_size = 0
        arguments.op_name = ""
        arguments.output_path = "/home/de"
        arguments.custom_script_path = ""
        arguments.algorithm = 'all'
        arguments.algorithm_options = ''
        arguments.range = None
        arguments.select = None
        multiprocessing.Manager = mock.Mock
        multiprocessing.Manager.RLock = mock.Mock
        dump_data = DumpData()
        dump_data.input.append(
            self._make_op_input(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data.output.append(
            self._make_op_output(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data = dump_utils.convert_dump_data(dump_data)
        result = [[0, True, "data&message"]]
        with mock.patch('sys.argv', args):
            with mock.patch('os.path.exists', return_value=True), \
                 mock.patch('os.access', return_value=True), \
                 mock.patch('os.remove'), \
                 mock.patch('os.listdir',
                            side_effect=[['alg_CosineSimilarity.py'],
                                         ['xxxx.A1.0.1111111111111111',
                                          'xxxx.A2.0.1111111111111121',
                                          'xxxx.C1.0.1111111111111131',
                                          'xxxx.C2.0.1111111111111141'],
                                         ['A_relu.0.1111111111111111.pb',
                                          'data.0.1111111111111111.pb',
                                          'C.0.1111111111111111.pb'],
                                         ['convert_NC1HWC0_to_NCHW.py']]), \
                 mock.patch('os.path.isdir', return_value=True), \
                 mock.patch('os.path.isfile', return_value=True), \
                 mock.patch('dump_parse.dump_utils.parse_dump_file', return_value=dump_data), \
                 mock.patch('os.stat', return_value=self.mock_stat_result):
                with mock.patch("os.path.getsize", return_value=100):
                    with mock.patch("json.load", return_value=self._make_L1_fusion_json_object()):
                        with mock.patch('builtins.open',
                                        mock.mock_open(
                                            read_data=self._make_csv_content())):
                            with mock.patch('os.open') as open_file, \
                                    mock.patch('os.fdopen'):
                                with mock.patch("multiprocessing.pool.ApplyResult.get", return_value=result):
                                    open_file.write = None
                                    main = compare_vector.VectorComparison(arguments)
                                    main.compare()
                                    key = \
                                        main.compare_rule.fusion_info.op_name_to_fusion_op_name_map[
                                            'A1']
                                    ret, dump_match, result = main._compare_by_fusion_op(
                                        key)
                                    result_list = self.get_result_list(result)
        self.assertEqual(ret, 0)
        self.assertEqual(dump_match, True)
        self.assertEqual(len(result_list), 4)

    def test_compare_detail1(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result', '-d', 'aaa']
        with mock.patch('sys.argv', args):
            with mock.patch('os.path.exists', return_value=True), \
                 mock.patch('os.access', return_value=True), \
                 mock.patch('os.remove'), \
                 mock.patch('os.listdir',
                            side_effect=[['alg_CosineSimilarity.py'],
                                         ['ccc.aaa.0.1111111111111111'],
                                         ['aaa.0.1111111111111111.pb'],
                                         ['convert_NC1HWC0_to_NCHW.py']]), \
                 mock.patch('os.path.isdir', return_value=True), \
                 mock.patch('os.path.isfile', return_value=True), \
                 mock.patch('os.stat', return_value=self.mock_stat_result):
                with mock.patch("os.path.getsize", return_value=100):
                    with mock.patch('builtins.open',
                                    mock.mock_open(
                                        read_data=self._make_input_json().encode(
                                            'utf-8'))):
                        main = compare_vector.VectorComparison()
                        ret = main.compare()
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_compare_detail2(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result', '-d', 'data']
        with pytest.raises(CompareError) as err:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=True), \
                     mock.patch('os.access', return_value=True), \
                     mock.patch('os.path.getsize', return_value=100), \
                     mock.patch('os.remove'), \
                     mock.patch('os.listdir',
                                side_effect=[['alg_CosineSimilarity.py'],
                                             ['ccc.data.0.1111111111111111'],
                                             ['data.0.1111111111111111.pb'],
                                             ['convert_NC1HWC0_to_NCHW.py']]), \
                     mock.patch('os.path.isdir', return_value=True), \
                     mock.patch('os.path.isfile', return_value=True), \
                     mock.patch('os.stat', return_value=self.mock_stat_result):
                    with mock.patch('builtins.open',
                                    mock.mock_open(
                                        read_data=self._make_input_json().encode(
                                            'utf-8'))):
                        main = compare_vector.VectorComparison()
                        ret = main.compare()
        self.assertEqual(err.value.args[0], CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR)

    def test_compare_detail3(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result', '-d', 'dynamic_const_471']
        with pytest.raises(CompareError) as err:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=True), \
                     mock.patch('os.access', return_value=True), \
                     mock.patch('os.remove'), \
                     mock.patch('os.listdir',
                                side_effect=[['alg_CosineSimilarity.py'],
                                             ['ccc.data.0.1111111111111111'],
                                             ['data.0.1111111111111111.pb'],
                                             ['convert_NC1HWC0_to_NCHW.py']]), \
                     mock.patch('os.path.isdir', return_value=True), \
                     mock.patch('os.path.isfile', return_value=True), \
                     mock.patch('os.stat', return_value=self.mock_stat_result):
                    with mock.patch("os.path.getsize", return_value=100):
                        with mock.patch('builtins.open',
                                        mock.mock_open(
                                            read_data=self._make_input_json().encode(
                                                'utf-8'))):
                            main = compare_vector.VectorComparison()
                            ret = main.compare()
        self.assertEqual(err.value.args[0], CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR)

    def test_compare_detail4(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result', '-d', 'conv1conv1_relu',
                '-i', '10']
        with pytest.raises(CompareError) as err:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=True), \
                     mock.patch('os.access', return_value=True), \
                     mock.patch('os.remove'), \
                     mock.patch('os.listdir',
                                side_effect=[['alg_CosineSimilarity.py'],
                                             ['ccc.data.0.1111111111111111'],
                                             ['data.0.1111111111111111.pb'],
                                             ['convert_NC1HWC0_to_NCHW.py']]), \
                     mock.patch('os.path.isdir', return_value=True), \
                     mock.patch('os.path.isfile', return_value=True), \
                     mock.patch('os.stat', return_value=self.mock_stat_result):
                    with mock.patch("os.path.getsize", return_value=100):
                        with mock.patch('builtins.open',
                                        mock.mock_open(
                                            read_data=self._make_input_json().encode(
                                                'utf-8'))):
                            main = compare_vector.VectorComparison()
                            ret = main.compare()
        self.assertEqual(err.value.args[0], CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR)

    def test_compare_detail6(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result', '-d', 'data',
                '-i', '10']
        dump_data = DumpData()
        dump_data.output.append(
            self._make_op_output(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data = dump_utils.convert_dump_data(dump_data)
        with pytest.raises(CompareError) as err:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=True), \
                     mock.patch('os.access', return_value=True), \
                     mock.patch('os.remove'), \
                     mock.patch('os.listdir',
                                side_effect=[['alg_CosineSimilarity.py'],
                                             ['ccc.data.0.1111111111111111'],
                                             ['data.0.1111111111111111.pb'],
                                             ['convert_NC1HWC0_to_NCHW.py']]), \
                     mock.patch('os.path.isdir', return_value=True), \
                     mock.patch('os.path.isfile', return_value=True), \
                     mock.patch('dump_parse.dump_utils.parse_dump_file', return_value=dump_data), \
                     mock.patch('os.stat', return_value=self.mock_stat_result):
                    with mock.patch("os.path.getsize", return_value=100):
                        with mock.patch('builtins.open',
                                        mock.mock_open(
                                            read_data=self._make_input_json().encode(
                                                'utf-8'))):
                            main = compare_vector.VectorComparison()
                            ret = main.compare()
        self.assertEqual(err.value.args[0], CompareError.MSACCUCMP_INDEX_OUT_OF_BOUNDS_ERROR)

    def test_compare_detail7(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result', '-d', 'data']
        dump_data = DumpData()
        dump_data.output.append(
            self._make_op_output(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data = dump_utils.convert_dump_data(dump_data)
        with mock.patch('sys.argv', args):
            with mock.patch('os.path.exists', return_value=True), \
                 mock.patch('os.access', return_value=True), \
                 mock.patch('os.remove'), \
                 mock.patch('os.listdir',
                            side_effect=[['alg_CosineSimilarity.py'],
                                         ['ccc.data.0.1111111111111111'],
                                         ['data.0.1111111111111111.pb'],
                                         ['convert_NC1HWC0_to_NCHW.py'], []]), \
                 mock.patch('os.path.isdir', return_value=True), \
                 mock.patch('os.path.isfile', return_value=True), \
                 mock.patch('dump_parse.dump_utils.parse_dump_file', return_value=dump_data), \
                     mock.patch('os.stat', return_value=self.mock_stat_result):
                with mock.patch("os.path.getsize", return_value=100):
                    with mock.patch('builtins.open',
                                    mock.mock_open(
                                        read_data=self._make_input_json().encode(
                                            'utf-8'))):
                        with mock.patch('os.open') as open_file, mock.patch('os.fdopen'):
                            open_file.write = None
                            main = compare_vector.VectorComparison()
                            ret = main.compare()
        self.assertEqual(ret, 0)

    def test_compare_detail8(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result', '-d', 'data', '-t',
                'input']
        dump_data = DumpData()
        dump_data.input.append(
            self._make_op_input(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data = dump_utils.convert_dump_data(dump_data)
        with pytest.raises(CompareError) as err:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=True), \
                     mock.patch('os.access', return_value=True), \
                     mock.patch('os.remove'), \
                     mock.patch('os.listdir',
                                side_effect=[['alg_CosineSimilarity.py'],
                                             ['ccc.data.0.1111111111111111'],
                                             ['data.0.1111111111111111.pb'],
                                             ['convert_NC1HWC0_to_NCHW.py'], []]), \
                     mock.patch('os.path.isdir', return_value=True), \
                     mock.patch('os.path.isfile', return_value=True), \
                     mock.patch('dump_parse.dump_utils.parse_dump_file', return_value=dump_data), \
                     mock.patch('os.stat', return_value=self.mock_stat_result):
                    with mock.patch("os.path.getsize", return_value=100):
                        with mock.patch('builtins.open',
                                        mock.mock_open(
                                            read_data=self._make_input_json().encode(
                                                'utf-8'))):
                            with mock.patch('os.open') as open_file, \
                                    mock.patch('os.fdopen'):
                                open_file.write = None
                                main = compare_vector.VectorComparison()
                                ret = main.compare()
        self.assertEqual(err.value.args[0], CompareError.MSACCUCMP_INDEX_OUT_OF_BOUNDS_ERROR)

    def test_compare_detail9(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result', '-d', 'conv1conv1_relu',
                '-t', 'input']
        dump_data = DumpData()
        dump_data.input.append(
            self._make_op_input(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data.output.append(
            self._make_op_output(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data = dump_utils.convert_dump_data(dump_data)
        with mock.patch('sys.argv', args):
            with mock.patch('os.path.exists', return_value=True), \
                 mock.patch('os.access', return_value=True), \
                 mock.patch('os.remove'), \
                 mock.patch('os.listdir',
                            side_effect=[['alg_CosineSimilarity.py'],
                                         ['ccc.conv1conv1_relu.0.1111111111111111'],
                                         ['data.0.1111111111111111.pb'],
                                         ['convert_NC1HWC0_to_NCHW.py'], []]), \
                 mock.patch('os.path.isdir', return_value=True), \
                 mock.patch('os.path.isfile', return_value=True), \
                 mock.patch('dump_parse.dump_utils.parse_dump_file', return_value=dump_data), \
                     mock.patch('os.stat', return_value=self.mock_stat_result):
                with mock.patch("os.path.getsize", return_value=100):
                    with mock.patch('builtins.open',
                                    mock.mock_open(
                                        read_data=self._make_input_json().encode(
                                            'utf-8'))):
                        with mock.patch('os.open') as open_file, \
                                mock.patch('os.fdopen'):
                            open_file.write = None
                            main = compare_vector.VectorComparison()
                            ret = main.compare()
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_compare_detail10(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result', '-d', 'conv1conv1_relu']
        dump_data = DumpData()
        dump_data.input.append(
            self._make_op_input(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data.output.append(
            self._make_op_output(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data = dump_utils.convert_dump_data(dump_data)
        with pytest.raises(CompareError) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=True), \
                     mock.patch('os.access', return_value=True), \
                     mock.patch('os.remove'), \
                     mock.patch('os.listdir',
                                side_effect=[['alg_CosineSimilarity.py'],
                                             ['conv1_relu.0.1111111111111111.dump'],
                                             ['data.0.1111111111111111.pb',
                                              'conv1_relu.0.1111111111111111.pb'],
                                             ['convert_NC1HWC0_to_NCHW.py']]), \
                     mock.patch('os.path.isdir', return_value=True), \
                     mock.patch('os.path.isfile', return_value=True), \
                     mock.patch('dump_parse.dump_utils.parse_dump_file', return_value=dump_data), \
                     mock.patch('os.stat', return_value=self.mock_stat_result):
                    with mock.patch('builtins.open',
                                    mock.mock_open(
                                        read_data=self._make_input_json().encode(
                                            'utf-8'))):
                        with mock.patch('os.open') as open_file, \
                                mock.patch('os.fdopen'), \
                                mock.patch('os.path.getsize', return_value=1024):
                            open_file.write = None
                            main = compare_vector.VectorComparison()
                            ret = main.compare()
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_DUMP_FILE_ERROR)

    def test_compare_detail_l1_fusion1(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result', '-d', 'A1']
        dump_data = DumpData()
        dump_data.input.append(
            self._make_op_input(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data = dump_utils.convert_dump_data(dump_data)
        with pytest.raises(CompareError) as err:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=True), \
                     mock.patch('os.access', return_value=True), \
                     mock.patch('os.remove'), \
                     mock.patch('os.listdir',
                                side_effect=[['alg_CosineSimilarity.py'],
                                             ['xxxx.A1.0.1111111111111111',
                                              'xxxx.A2.0.1111111111111121',
                                              'xxxx.C1.0.1111111111111131',
                                              'xxxx.C2.0.1111111111111141'],
                                             ['A_relu.0.1111111111111111.pb',
                                              'data.0.1111111111111111.pb',
                                              'C.0.1111111111111111.pb'],
                                             ['convert_NC1HWC0_to_NCHW.py']]), \
                     mock.patch('os.path.isdir', return_value=True), \
                     mock.patch('os.path.isfile', return_value=True), \
                     mock.patch('dump_parse.dump_utils.parse_dump_file', return_value=dump_data), \
                     mock.patch('os.stat', return_value=self.mock_stat_result):
                    with mock.patch("os.path.getsize", return_value=100):
                        with mock.patch('builtins.open',
                                        mock.mock_open(
                                            read_data=self._make_L1_fusion_json().encode(
                                                'utf-8'))):
                            with mock.patch('os.open') as open_file, \
                                    mock.patch('os.fdopen'):
                                open_file.write = None
                                main = compare_vector.VectorComparison()
                                ret = main.compare()
        self.assertEqual(err.value.args[0], CompareError.MSACCUCMP_INDEX_OUT_OF_BOUNDS_ERROR)

    def test_compare_detail_l1_fusion2(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result', '-d', 'A1', '-t', 'input']
        dump_data = DumpData()
        dump_data.input.append(
            self._make_op_input(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data.output.append(
            self._make_op_output(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data = dump_utils.convert_dump_data(dump_data)
        with mock.patch('sys.argv', args):
            with mock.patch('os.path.exists', return_value=True), \
                 mock.patch('os.access', return_value=True), \
                 mock.patch('os.remove'), \
                 mock.patch('os.listdir',
                            side_effect=[['alg_CosineSimilarity.py'],
                                         ['xxxx.A1.0.1111111111111111',
                                          'xxxx.A2.0.1111111111111121',
                                          'xxxx.C1.0.1111111111111131',
                                          'xxxx.C2.0.1111111111111141'],
                                         ['A_relu.0.1111111111111111.pb',
                                          'data.0.1111111111111111.pb',
                                          'C.0.1111111111111111.pb'],
                                         ['convert_NC1HWC0_to_NCHW.py'], []]), \
                 mock.patch('os.path.isdir', return_value=True), \
                 mock.patch('os.path.isfile', return_value=True), \
                 mock.patch('dump_parse.dump_utils.parse_dump_file', return_value=dump_data), \
                     mock.patch('os.stat', return_value=self.mock_stat_result):
                with mock.patch("os.path.getsize", return_value=100):
                    with mock.patch('builtins.open',
                                    mock.mock_open(
                                        read_data=self._make_L1_fusion_json().encode(
                                            'utf-8'))):
                        with mock.patch('os.open') as open_file, \
                                mock.patch('os.fdopen'):
                            open_file.write = None
                            main = compare_vector.VectorComparison()
                            ret = main.compare()
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_compare_detail_l1_fusion3(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result', '-d', 'B1']
        dump_data = DumpData()
        dump_data.input.append(
            self._make_op_input(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data.output.append(
            self._make_op_output(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data = dump_utils.convert_dump_data(dump_data)
        with pytest.raises(CompareError) as err:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=True), \
                     mock.patch('os.access', return_value=True), \
                     mock.patch('os.remove'), \
                     mock.patch('os.listdir',
                                side_effect=[['alg_CosineSimilarity.py'],
                                             ['xxxx.A1.0.1111111111111111',
                                              'xxxx.A2.0.1111111111111121',
                                              'xxxx.C1.0.1111111111111131',
                                              'xxxx.C2.0.1111111111111141'],
                                             ['A_relu.0.1111111111111111.pb',
                                              'data.0.1111111111111111.pb',
                                              'C.0.1111111111111111.pb'],
                                             ['convert_NC1HWC0_to_NCHW.py']]), \
                     mock.patch('os.path.isdir', return_value=True), \
                     mock.patch('os.path.isfile', return_value=True), \
                     mock.patch('dump_parse.dump_utils.parse_dump_file', return_value=dump_data), \
                     mock.patch('os.stat', return_value=self.mock_stat_result):
                    with mock.patch("os.path.getsize", return_value=100):
                        with mock.patch('builtins.open',
                                        mock.mock_open(
                                            read_data=self._make_L1_fusion_json().encode(
                                                'utf-8'))):
                            with mock.patch('os.open') as open_file, \
                                    mock.patch('os.fdopen'):
                                open_file.write = None
                                main = compare_vector.VectorComparison()
                                ret = main.compare()
        self.assertEqual(err.value.args[0], CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR)

    def test_compare_detail_l1_fusion4(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result', '-d', 'C1']
        dump_data = DumpData()
        dump_data.output.append(
            self._make_op_output(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data = dump_utils.convert_dump_data(dump_data)
        with mock.patch('sys.argv', args):
            with mock.patch('os.path.exists', return_value=True), \
                 mock.patch('os.access', return_value=True), \
                 mock.patch('os.remove'), \
                 mock.patch('os.listdir',
                            side_effect=[['alg_CosineSimilarity.py'],
                                         ['xxxx.A1.0.1111111111111111',
                                          'xxxx.A2.0.1111111111111121',
                                          'xxxx.C1.0.1111111111111131',
                                          'xxxx.C2.0.1111111111111141'],
                                         ['A_relu.0.1111111111111111.pb',
                                          'data.0.1111111111111111.pb',
                                          'C.0.1111111111111111.pb'],
                                         ['convert_NC1HWC0_to_NCHW.py'], []]), \
                 mock.patch('os.path.isdir', return_value=True), \
                 mock.patch('os.path.isfile', return_value=True), \
                 mock.patch('dump_parse.dump_utils.parse_dump_file', return_value=dump_data), \
                     mock.patch('os.stat', return_value=self.mock_stat_result):
                with mock.patch("os.path.getsize", return_value=100):
                    with mock.patch('builtins.open',
                                    mock.mock_open(
                                        read_data=self._make_L1_fusion_json().encode(
                                            'utf-8'))):
                        with mock.patch('os.open') as open_file, \
                                mock.patch('os.fdopen'):
                            open_file.write = None
                            main = compare_vector.VectorComparison()
                            ret = main.compare()
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_compare_detail_l1_fusion5(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result', '-d', 'C2']
        dump_data = DumpData()
        dump_data.output.append(
            self._make_op_output(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data = dump_utils.convert_dump_data(dump_data)
        with mock.patch('sys.argv', args):
            with mock.patch('os.path.exists', return_value=True), \
                 mock.patch('os.access', return_value=True), \
                 mock.patch('os.remove'), \
                 mock.patch('os.listdir',
                            side_effect=[['alg_CosineSimilarity.py'],
                                         ['xxxx.A1.0.1111111111111111',
                                          'xxxx.A2.0.1111111111111121',
                                          'xxxx.C1.0.1111111111111131',
                                          'xxxx.C2.0.1111111111111141'],
                                         ['A_relu.0.1111111111111111.pb',
                                          'data.0.1111111111111111.pb',
                                          'C.0.1111111111111111.pb'],
                                         ['convert_NC1HWC0_to_NCHW.py'], []]), \
                 mock.patch('os.path.isdir', return_value=True), \
                 mock.patch('os.path.isfile', return_value=True), \
                 mock.patch('dump_parse.dump_utils.parse_dump_file', return_value=dump_data), \
                     mock.patch('os.stat', return_value=self.mock_stat_result):
                with mock.patch("os.path.getsize", return_value=100):
                    with mock.patch('builtins.open',
                                    mock.mock_open(
                                        read_data=self._make_L1_fusion_json().encode(
                                            'utf-8'))):
                        with mock.patch('os.open') as open_file, \
                                mock.patch('os.fdopen'):
                            open_file.write = None
                            main = compare_vector.VectorComparison()
                            ret = main.compare()
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_compare_result_empty(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result.txt']
        arguments = mock.Mock()
        arguments.fusion_rule_file = "/home/b.json"
        arguments.quant_fusion_rule_file = ""
        arguments.close_fusion_rule_file = ""
        arguments.my_dump_path = "/home/demo"
        arguments.golden_dump_path = "/home/dt"
        arguments.dump_version = 1
        arguments.max_cmp_size = 0
        arguments.op_name = ""
        arguments.output_path = "/home/de"
        arguments.custom_script_path = ""
        arguments.algorithm = 'all'
        arguments.algorithm_options = ''
        arguments.range = None
        arguments.select = None
        multiprocessing.Manager = mock.Mock
        multiprocessing.Manager.RLock = mock.Mock
        dump_data = DumpData()
        dump_data = dump_utils.convert_dump_data(dump_data)
        result = [[0, True, "data&message"]]
        with mock.patch('sys.argv', args):
            with mock.patch('os.path.exists', return_value=True), \
                 mock.patch('os.access', return_value=True), \
                 mock.patch('os.remove'), \
                 mock.patch('os.listdir',
                            side_effect=[['alg_CosineSimilarity.py'],
                                         ['xxxx.conv1conv1_relu.0.1111111111111111'],
                                         ['data.0.1111111111111111.pb',
                                          'conv1_relu.0.1111111111111111.pb'],
                                         ['convert_NC1HWC0_to_NCHW.py']]), \
                 mock.patch('os.path.isdir', return_value=True), \
                 mock.patch('os.path.isfile', return_value=True), \
                 mock.patch('dump_parse.dump_utils.parse_dump_file', return_value=dump_data), \
                     mock.patch('os.stat', return_value=self.mock_stat_result):
                with mock.patch("os.path.getsize", return_value=100):
                    with mock.patch("json.load", return_value=self._make_json_object()):
                        with mock.patch('builtins.open',
                                        mock.mock_open(
                                            read_data=self._make_csv_content())):
                            with mock.patch("multiprocessing.pool.ApplyResult.get", return_value=result):
                                with mock.patch('os.open') as open_file, \
                                        mock.patch('os.fdopen'):
                                    open_file.write = None
                                    main = compare_vector.VectorComparison(arguments)
                                    main.compare()
                                    key = \
                                        main.compare_rule.fusion_info.op_name_to_fusion_op_name_map[
                                            'conv1conv1_relu']
                                    ret, dump_match, result = main._compare_by_fusion_op(
                                        key)
                                    result_list = self.get_result_list(result)
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)
        self.assertEqual(dump_match, False)
        self.assertEqual(len(result_list), 1)

    def test_make_mapping_table_by_op_name(self):
        arguments = mock.Mock()
        arguments.fusion_rule_file = "/home/b.json"
        arguments.quant_fusion_rule_file = ""
        arguments.close_fusion_rule_file = ""
        arguments.my_dump_path = "/home/demo"
        arguments.golden_dump_path = "/home/dt"
        arguments.dump_version = 1
        arguments.max_cmp_size = 0
        arguments.op_name = ""
        arguments.output_path = "/home/de"
        arguments.custom_script_path = ""
        arguments.algorithm = "all"
        arguments.algorithm_options = ""
        arguments.range = None
        arguments.select = None
        compare = compare_vector.VectorComparison(arguments)
        compare.compare_rule = mock.Mock
        compare.compare_rule.fusion_info = mock.Mock
        op_name = "input_ids"
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
        output_desc = OutputDesc(origin_name, origin_output_index, origin_format, origin_shape)
        attr = OpAttr(original_op_names, l1_fusion_no, is_multi_op, 1)
        fusion_list = [FusionOp(op_id, op_name, input_list, op_type, output_desc, attr)]
        compare.compare_rule.fusion_info.fusion_op_name_to_op_map = {"demo": fusion_list}
        compare._make_mapping_table_by_op_name(["demo"])

    def test_make_mapping_table_by_op_name1(self):
        arguments = mock.Mock()
        arguments.fusion_rule_file = "/home/b.json"
        arguments.quant_fusion_rule_file = ""
        arguments.close_fusion_rule_file = ""
        arguments.my_dump_path = "/home/demo"
        arguments.golden_dump_path = "/home/dt"
        arguments.dump_version = 1
        arguments.max_cmp_size = 0
        arguments.op_name = ""
        arguments.output_path = "/home/de"
        arguments.custom_script_path = ""
        arguments.algorithm = "all"
        arguments.algorithm_options = ""
        arguments.range = None
        arguments.select = None
        compare = compare_vector.VectorComparison(arguments)
        compare.compare_rule = mock.Mock
        compare.compare_rule.fusion_info = mock.Mock
        compare.compare_data = mock.Mock()
        compare.left_dump_info = mock.Mock
        compare.left_dump_info.get_op_dump_file = mock.Mock(side_effect=ValueError)
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
        output_desc = OutputDesc(origin_name, origin_output_index, origin_format, origin_shape)
        attr = OpAttr(original_op_names, l1_fusion_no, is_multi_op, 1)
        fusion_list = [FusionOp(op_id, op_name, input_list, op_type, output_desc, attr)]
        compare.compare_rule.fusion_info.fusion_op_name_to_op_map = {"demo": fusion_list}
        compare._make_mapping_table_by_op_name(["demo"])

    def test_save_cmp_result1(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result', '-d', 'C2']
        with mock.patch('sys.argv', args):
            main = compare_vector.VectorComparison()
            result = ""
            lock = mock.Mock()
            lock.acquire = mock.Mock()
            lock.release = mock.Mock()
            with mock.patch('os.open', side_effect=IOError):
                main._save_cmp_result(result, lock)

    def test_save_cmp_result2(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result', '-d', 'C2']
        with mock.patch('sys.argv', args):
            main = compare_vector.VectorComparison()
            result = [""]
            lock = mock.Mock()
            lock.acquire = mock.Mock()
            lock.release = mock.Mock()
            with mock.patch('os.open'), mock.patch("os.fdopen"):
                main._save_cmp_result(result, lock)

    def test_save_cmp_result3(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result', '-d', 'C2']
        result = '0 input_ids input_ids input_ids:output:0 1.000000 0.000000 0.000000 0.000000 0.000000' \
                 ' \'(1171.945;2293.594) (1171.945;2293.594)\''
        single_op_cmp_result = SingleOpCmpResult()
        result_info = utils.ResultInfo("opname", False, [[result]], 1, [], [], [], False, {}, False)
        single_op_cmp_result.update_attr(result_info)
        with mock.patch('sys.argv', args):
            main = compare_vector.VectorComparison()
            result = [[1, "", [single_op_cmp_result]]]
            lock = mock.Mock()
            lock.acquire = mock.Mock()
            lock.release = mock.Mock()
            with mock.patch('os.open'), mock.patch("os.fdopen"):
                main._save_cmp_result(result, lock)

    def test_write_header_to_file1(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result', '-d', 'C2']
        with mock.patch('sys.argv', args):
            with mock.patch('os.open'), mock.patch("os.fdopen"):
                main = compare_vector.VectorComparison()
                result = main._write_header_to_file()
        self.assertEqual(result, True)

    def test_write_header_to_file2(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result', '-d', 'C2', "-csv"]
        with mock.patch('sys.argv', args):
            with mock.patch('os.open'), mock.patch("os.fdopen"):
                main = compare_vector.VectorComparison()
                result = main._write_header_to_file()
        self.assertEqual(result, True)

    def test_write_header_to_file3(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result', '-d', 'C2']
        with mock.patch('sys.argv', args):
            with mock.patch('os.open', side_effect=IOError):
                main = compare_vector.VectorComparison()
                result = main._write_header_to_file()
        self.assertEqual(result, False)

    def test_set_output_path(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result', '-d', 'C2']
        with mock.patch('sys.argv', args):
            main = compare_vector.VectorComparison()
            main.set_output_path("/home/demo")

    def test_ffts_find_pre_op1(self):
        single_op_cmp_result1 = SingleOpCmpResult()
        single_op_cmp_result2 = SingleOpCmpResult()
        result_list1 = [
            ['0', 'Test', 'test_op1', 'NaN', 'NaN', 'test_op1', 'NaN', 'test_op1:input:0',
             'NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', ''],
            ['0', 'Test', 'test_op1', 'NaN', 'NaN', 'test_op1', 'NaN', 'test_op1:output:0',
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, '']
        ]
        input_result_list = [
            ['0', 'Test', 'test_op1', 'NaN', 'NaN', 'test_op1', 'NaN', 'test_op1:input:0', 'NaN', 'NaN',
             'NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', '']
        ]
        output_result_list1 = [
            ['0', 'Test', 'test_op1', 'NaN', 'NaN', 'test_op1', 'NaN', 'test_op1:output:0',
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, '']
        ]
        op_name_origin_output_index_map = {"test_op1:input:0": ("test_op2", 0)}
        result_info1 = utils.ResultInfo("test_op1", False, result_list1, 1, [], input_result_list, output_result_list1,
                                        True, op_name_origin_output_index_map, False)
        result_list2 = [
            ['1', 'Test', 'test_op2', 'NaN', 'NaN', 'test_op2', 'NaN', 'test_op2:output:0',
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, '']
        ]
        output_result_list2 = [
            ['1', 'Test', 'test_op2', 'NaN', 'NaN', 'test_op2', 'NaN', 'test_op2:output:0',
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, '']
        ]
        result_info2 = utils.ResultInfo("test_op2", False, result_list2, 1, [], [], output_result_list2,
                                        True, op_name_origin_output_index_map, False)
        single_op_cmp_result1.update_attr(result_info1)
        single_op_cmp_result2.update_attr(result_info2)
        result_mapping = {"test_op1": single_op_cmp_result1, "test_op2": single_op_cmp_result2}
        single_op_cmp_result1.find_pre_op(result_mapping)
        result = [
            ['0', 'Test', 'test_op1', 'NaN', 'NaN', 'test_op1', 'NaN', 'test_op1:input:0',
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ''],
            ['0', 'Test', 'test_op1', 'NaN', 'NaN', 'test_op1', 'NaN', 'test_op1:output:0',
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, '']
        ]
        self.assertEqual(result, single_op_cmp_result1.result_list)

    def test_ffts_find_pre_op2(self):
        single_op_cmp_result = SingleOpCmpResult()
        result_list = [
            ['0', 'Test', 'test_op1', 'NaN', 'NaN', 'test_op1', 'NaN', 'test_op1:input:0',
             'NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', '']
        ]
        input_result_list = [
            ['0', 'Test', 'test_op1', 'NaN', 'NaN', 'test_op1', 'NaN', 'test_op1:input:0',
             'NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', '']
        ]
        output_result_list1 = [
            ['0', 'Test', 'test_op1', 'NaN', 'NaN', 'test_op1', 'NaN', 'test_op1:output:0',
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, '']
        ]
        result_info = utils.ResultInfo(
            "test_op", False, result_list, 1, [], input_result_list, output_result_list1, True, {}, False)
        single_op_cmp_result.update_attr(result_info)
        result_mapping = {}
        with pytest.raises(CompareError) as error:
            single_op_cmp_result.find_pre_op(result_mapping)
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_INVALID_INPUT_MAPPING)

    def test_ffts_find_pre_op3(self):
        single_op_cmp_result = SingleOpCmpResult()
        result_list = [
            ['0', 'Test', 'test_op1', 'NaN', 'NaN', 'test_op1', 'NaN', 'test_op1:input:0',
             'NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', ''],
            ['0', 'Test', 'test_op1', 'NaN', 'NaN', 'test_op1', 'NaN', 'test_op1:output:0',
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, '']
        ]
        input_result_list = [
            ['0', 'Test', 'test_op1', 'NaN', 'NaN', 'test_op1', 'NaN', 'test_op1:input:0',
             'NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', '']
        ]
        output_result_list1 = [
            ['0', 'Test', 'test_op1', 'NaN', 'NaN', 'test_op1', 'NaN', 'test_op1:output:0',
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, '']
        ]
        result_info = utils.ResultInfo("test_op", False, result_list, 1, [], input_result_list, output_result_list1,
                                       True, {}, False)
        single_op_cmp_result.update_attr(result_info)
        result_mapping = {}
        with pytest.raises(CompareError) as error:
            single_op_cmp_result.find_pre_op(result_mapping)
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_INVALID_INPUT_MAPPING)

    def test_set_output_path_softlink_path_raises_error(self):
        with TemporaryDirectory() as tmpdir:
            # 创建目标文件和符号链接
            target_path = os.path.join(tmpdir, 'target')
            open(target_path, 'w').close()
            link_path = os.path.join(tmpdir, 'link')
            os.symlink(target_path, link_path)

            with mock.patch.object(log, 'print_error_log') as mock_log:
                with pytest.raises(CompareError) as exc_info:
                    args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                            '/home/a.json', '-o', '/home/result', '-d', 'C2']
                    with mock.patch('sys.argv', args):
                        main = compare_vector.VectorComparison()
                        main.set_output_path(link_path)

                # 验证异常类型和错误码
                assert exc_info.value.code == CompareError.MSACCUCMP_INVALID_PATH_ERROR
                # 验证错误日志内容
                expected_msg = f'The path "{link_path!r}" is a softlink, not permitted.'
                mock_log.assert_called_once_with(expected_msg)

    def test_init_by_input_parse_softlink_path_raises_error(self):
        with TemporaryDirectory() as tmpdir:
            # 创建目标文件和符号链接
            target_path = os.path.join(tmpdir, 'target')
            open(target_path, 'w').close()
            link_path = os.path.join(tmpdir, 'link')
            os.symlink(target_path, link_path)

            with mock.patch.object(log, 'print_error_log') as mock_log:
                with pytest.raises(CompareError) as exc_info:
                    args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                            '/home/a.json', '-o', link_path, '-d', 'C2']
                    with mock.patch('sys.argv', args):
                        main = compare_vector.VectorComparison()
                        main._init_by_input_parse()

                # 验证异常类型和错误码
                assert exc_info.value.code == CompareError.MSACCUCMP_INVALID_PATH_ERROR
                # 验证错误日志内容
                expected_msg = f'The path "{link_path!r}" is a softlink, not permitted.'
                mock_log.assert_called_once_with(expected_msg)

    def test_process_output_path_parameter_softlink_path_raises_error(self):
        with TemporaryDirectory() as tmpdir:
            # 创建目标文件和符号链接
            target_path = os.path.join(tmpdir, 'target')
            open(target_path, 'w').close()
            link_path = os.path.join(tmpdir, 'link')
            os.symlink(target_path, link_path)

            with mock.patch.object(log, 'print_error_log') as mock_log:
                with pytest.raises(CompareError) as exc_info:
                    arguments = mock.Mock()
                    arguments.fusion_rule_file = "/home/b.json"
                    arguments.quant_fusion_rule_file = ""
                    arguments.close_fusion_rule_file = ""
                    arguments.my_dump_path = "/home/demo"
                    arguments.golden_dump_path = "/home/dt"
                    arguments.dump_version = 1
                    arguments.max_cmp_size = 0
                    arguments.op_name = ""
                    arguments.output_path = link_path
                    main = compare_vector.VectorComparison(arguments)
                    main._process_output_path_parameter(arguments)

                # 验证异常类型和错误码
                assert exc_info.value.code == CompareError.MSACCUCMP_INVALID_PATH_ERROR
                # 验证错误日志内容
                expected_msg = f'The path "{link_path!r}" is a softlink, not permitted.'
                mock_log.assert_called_once_with(expected_msg)


    def test_process_single_op_parameters_softlink_path_raises_error(self):
        with TemporaryDirectory() as tmpdir:
            # 创建目标文件和符号链接
            target_path = os.path.join(tmpdir, 'target')
            open(target_path, 'w').close()
            link_path = os.path.join(tmpdir, 'link')
            os.symlink(target_path, link_path)
            with mock.patch.object(vector_comparison.VectorComparison, '__init__', return_value=None):
                with pytest.raises(CompareError) as exc_info:
                    arguments = mock.Mock()
                    arguments.output_path = link_path
                    arguments.max_line = None
                    main = compare_vector.VectorComparison(arguments)
                    main._process_single_op_parameters(arguments)

                # 验证异常类型和错误码
                assert exc_info.value.code == CompareError.MSACCUCMP_INVALID_PATH_ERROR


if __name__ == '__main__':
    unittest.main()
