# coding=utf-8
import json
import os
import shutil
import threading
import unittest
from unittest.mock import patch, MagicMock
import tempfile
from unittest import mock

import numpy as np
import pandas as pd
import torch

from msprobe.core.common.file_utils import load_json
from msprobe.core.common.const import CompareConst, Const
from msprobe.core.common.utils import CompareException
from msprobe.core.compare.acc_compare import ModeConfig, MappingConfig, MappingDict, Comparator, ParseData, \
    ProcessDf, Match, CreateTable, CalcStatsDiff
from msprobe.core.compare.stats_diff_calc import ValType, ALL_TYPES


npu_op_item_data_fuzzy = {
    'op_name': 'Functional.conv2d.0.forward.input.0',
    'dtype': 'torch.float32',
    'shape': [1, 1, 28, 28],
    'summary': [3.029174327850342, -2.926689624786377, -0.06619918346405029],
    'stack_info': [],
    'data_name': 'Functional.conv2d.0.forward.input.0.pt',
    'compare_key': 'Functional.conv2d.0.forward.input.0',
    'compare_shape': [1, 1, 28, 28],
}
npu_op_item_fuzzy = pd.Series(npu_op_item_data_fuzzy)
npu_op_item_data_fuzzy_2 = {
    'op_name': 'Functional.conv2d.0.forward.input.1',
    'dtype': 'torch.float32',
    'shape': [1, 1, 28, 28],
    'summary': [3.029174327850342, -2.926689624786377, -0.06619918346405029],
    'stack_info': [],
    'data_name': 'Functional.conv2d.0.forward.input.1.pt',
    'compare_key': 'Functional.conv2d.0.forward.input.1',
    'compare_shape': [1, 1, 28, 28],
}
npu_op_item_fuzzy_2 = pd.Series(npu_op_item_data_fuzzy_2)
bench_op_item_data_fuzzy = {
    'op_name': 'Functional.conv2d.1.forward.input.0',
    'dtype': 'torch.float32',
    'shape': [1, 1, 28, 28],
    'summary': [3.029174327850342, -2.926689624786377, -0.06619918346405029],
    'stack_info': [],
    'data_name': 'Functional.conv2d.1.forward.input.0.pt',
    'compare_key': 'Functional.conv2d.1.forward.input.0',
    'compare_shape': [1, 1, 28, 28],
}
bench_op_item_fuzzy = pd.Series(bench_op_item_data_fuzzy)

npu_dict = {'op_name': ['Functional.conv2d.0.forward.input.0', 'Functional.conv2d.0.forward.input.1',
                        'Functional.conv2d.0.forward.input.2', 'Functional.conv2d.0.forward.output'],
            'input_struct': [('torch.float32', [1, 1, 28, 28]), ('torch.float32', [16, 1, 5, 5]),
                             ('torch.float32', [16])],
            'output_struct': [('torch.float32', [1, 16, 28, 28])],
            'summary': [[3.029174327850342, -2.926689624786377, -0.06619918346405029],
                        [0.19919930398464203, -0.19974489510059357, 0.006269412115216255],
                        [0.19734230637550354, -0.18177609145641327, 0.007903944700956345],
                        [2.1166646480560303, -2.190781354904175, -0.003579073818400502]], 'stack_info': []}

npu_dict2 = {'op_name': ['Functional.conv2d.1.forward.input.0', 'Functional.conv2d.0.forward.input.1',
                         'Functional.conv2d.0.forward.input.2', 'Functional.conv2d.0.forward.output'],
             'input_struct': [('torch.float32', [1, 1, 28, 28]), ('torch.float32', [16, 1, 5, 5]),
                              ('torch.float32', [16])],
             'output_struct': [('torch.float32', [1, 16, 28, 28])],
             'summary': [[3.029174327850342, -2.926689624786377, -0.06619918346405029],
                         [0.19919930398464203, -0.19974489510059357, 0.006269412115216255],
                         [0.19734230637550354, -0.18177609145641327, 0.007903944700956345],
                         [2.1166646480560303, -2.190781354904175, -0.003579073818400502]], 'stack_info': []}

bench_dict = {'op_name': ['Functional.conv2d.0.forward.input.0', 'Functional.conv2d.0.forward.input.1',
                          'Functional.conv2d.0.forward.input.2', 'Functional.conv2d.0.forward.output'],
              'input_struct': [('torch.float32', [1, 1, 28, 28]), ('torch.float32', [16, 1, 5, 5]),
                               ('torch.float32', [16])],
              'output_struct': [('torch.float32', [1, 16, 28, 28])],
              'summary': [[3.029174327850342, -2.926689624786377, -0.06619918346405029],
                          [0.19919930398464203, -0.19974489510059357, 0.006269412115216255],
                          [0.19734230637550354, -0.18177609145641327, 0.007903944700956345],
                          [2.1166646480560303, -2.190781354904175, -0.003579073818400502]], 'stack_info': []}

tensor_list = [
    {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3], 'Max': 0.33033010363578796,
     'Min': -0.331031858921051, 'Mean': -0.030964046716690063, 'Norm': 2.2533628940582275, 'requires_grad': True,
     'full_op_name': 'Tensor.add_.0.forward.input.0'},
    {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3],
     'Max': 0.003992878366261721, 'Min': -0.008102823048830032, 'Mean': -0.0002002553956117481,
     'Norm': 0.02844562754034996, 'requires_grad': False, 'full_op_name': 'Tensor.add_.0.forward.input.1'},
    {'full_op_name': 'Tensor.add_.0.forward.input.alpha.0', 'dtype': "<class 'float'>", "shape": '[]', 'md5': None,
     'Max': -0.1, 'Min': -0.1, 'Mean': -0.1, 'Norm': -0.1, 'data_name': '-1'},
    {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3],
     'Max': 0.33033010363578796, 'Min': -0.331031858921051, 'Mean': -0.030964046716690063,
     'Norm': 2.2533628940582275, 'requires_grad': True, 'full_op_name': 'Tensor.add_.0.forward.output.0'}
]

result_op_dict = {'op_name': ['Tensor.add_.0.forward.input.0', 'Tensor.add_.0.forward.input.1',
                              'Tensor.add_.0.forward.input.alpha.0', 'Tensor.add_.0.forward.output.0'],
                  'input_struct': [('torch.float32', [16, 1, 3, 3]), ('torch.float32', [16, 1, 3, 3]),
                                   ("<class 'float'>", '[]')],
                  'output_struct': [('torch.float32', [16, 1, 3, 3])],
                  'summary': [[0.33033010363578796, -0.331031858921051, -0.030964046716690063, 2.2533628940582275],
                              [0.003992878366261721, -0.008102823048830032, -0.0002002553956117481,
                               0.02844562754034996],
                              [-0.1, -0.1, -0.1, -0.1],
                              [0.33033010363578796, -0.331031858921051, -0.030964046716690063, 2.2533628940582275]],
                  'stack_info': []}

o_result = [
    ['Functional.conv2d.0.forward.input.0', 'Functional.conv2d.0.forward.input.0', 'torch.float32', 'torch.float32',
     [1, 1, 28, 28], [1, 1, 28, 28], 0.0, 0.0, 0.0, ' ', '0.0%', '0.0%', '0.0%', ' ', 3.029174327850342,
     -2.926689624786377,
     -0.06619918346405029, 3.029174327850342, -2.926689624786377, -0.06619918346405029, '', '', 'None'],
    ['Functional.conv2d.0.forward.input.1', 'Functional.conv2d.0.forward.input.1', 'torch.float32', 'torch.float32',
     [16, 1, 5, 5], [16, 1, 5, 5], 0.0, 0.0, 0.0, ' ', '0.0%', '0.0%', '0.0%', ' ', 0.19919930398464203,
     -0.19974489510059357,
     0.006269412115216255, 0.19919930398464203, -0.19974489510059357, 0.006269412115216255, '', '', 'None'],
    ['Functional.conv2d.0.forward.input.2', 'Functional.conv2d.0.forward.input.2', 'torch.float32', 'torch.float32',
     [16], [16], 0.0, 0.0, 0.0, ' ', '0.0%', '0.0%', '0.0%', ' ', 0.19734230637550354, -0.18177609145641327,
     0.007903944700956345,
     0.19734230637550354, -0.18177609145641327, 0.007903944700956345, '', '', 'None'],
    ['Functional.conv2d.0.forward.output', 'Functional.conv2d.0.forward.output', 'torch.float32', 'torch.float32',
     [1, 16, 28, 28], [1, 16, 28, 28], 0.0, 0.0, 0.0, ' ', '0.0%', '0.0%', '0.0%', ' ', 2.1166646480560303,
     -2.190781354904175,
     -0.003579073818400502, 2.1166646480560303, -2.190781354904175, -0.003579073818400502, '', '', 'None']]

npu_dict_aten = {'op_name': ['Aten__native_batch_norm_legit_functional.default_0_forward.input.0',
                             'Aten__native_batch_norm_legit_functional.default_0_forward.input.1',
                             'Aten__native_batch_norm_legit_functional.default_0_forward.input.2',
                             'Aten__native_batch_norm_legit_functional.default_0_forward.input.3',
                             'Aten__native_batch_norm_legit_functional.default_0_forward.input.4',
                             'Aten__native_batch_norm_legit_functional.default_0_forward.output.0',
                             'Aten__native_batch_norm_legit_functional.default_0_forward.output.1',
                             'Aten__native_batch_norm_legit_functional.default_0_forward.output.2',
                             'Aten__native_batch_norm_legit_functional.default_0_forward.output.3',
                             'Aten__native_batch_norm_legit_functional.default_0_forward.output.4'],
                 'input_struct': [('torch.float16', [256, 256, 14, 14]), ('torch.float32', [256]),
                                  ('torch.float32', [256]), ('torch.float32', [256]), ('torch.float32', [256])],
                 'output_struct': [('torch.float16', [256, 256, 14, 14]), ('torch.float32', [256]),
                                   ('torch.float32', [256]), ('torch.float32', [256]), ('torch.float32', [256])],
                 'summary': [[139.625, -127.5625, -0.0103607177734375],
                             [2.5276029109954834, -2.1788690090179443, -0.0008259844034910202],
                             [2.472219944000244, -2.845968723297119, -0.008756577968597412],
                             [2.763145923614502, -3.398397922515869, -0.052132632583379745],
                             [2.673110008239746, -3.149275064468384, 0.01613386906683445],
                             [13.5546875, -10.640625, -0.008758544921875],
                             [0.30550330877304077, -0.24485322833061218, -0.010361209511756897],
                             [623.9192504882812, 432.96826171875, 520.2276611328125],
                             [2.4797861576080322, -3.055997371673584, -0.04795549064874649],
                             [61.7945556640625, 42.59713363647461, 52.03831481933594]]}

bench_dict_functional = {
    'op_name': ['Functional_batch_norm_0_forward.input.0', 'Functional_batch_norm_0_forward.input.1',
                'Functional_batch_norm_0_forward.input.2', 'Functional_batch_norm_0_forward.input.3',
                'Functional_batch_norm_0_forward.input.4', 'Functional_batch_norm_0_forward.output'],
    'input_struct': [('torch.float32', [256, 256, 14, 14]), ('torch.float32', [256]), ('torch.float32', [256]),
                     ('torch.float32', [256]), ('torch.float32', [256])],
    'output_struct': [('torch.float32', [256, 256, 14, 14])],
    'summary': [[3.061628818511963, -3.22507381439209, 3.634914173744619e-05],
                [0.0005779837374575436, -0.0006301702815108001, 3.634906533989124e-06],
                [0.9338104128837585, 0.9277191162109375, 0.930335283279419],
                [1.0, 1.0, 1.0], [0.0, 0.0, 0.0],
                [5.397906303405762, -5.796811580657959, 2.5283952709287405e-10]]
}

aten_result = [
    ['Aten__native_batch_norm_legit_functional.default_0_forward.input.0', 'Functional_batch_norm_0_forward.input.0',
     'torch.float16', 'torch.float32', [256, 256, 14, 14], [256, 256, 14, 14], 136.56337118148804, -124.33742618560791,
     -0.010397066915174946, ' ', '4460.480981749501%', '3855.335826136584%', '28603.33536971545%', ' ', 139.625,
     -127.5625, -0.0103607177734375, 3.061628818511963, -3.22507381439209, 3.634914173744619e-05, 'Warning',
     'Need double check api accuracy.', 'None'],
    ['Aten__native_batch_norm_legit_functional.default_0_forward.input.1', 'Functional_batch_norm_0_forward.input.1',
     'torch.float32', 'torch.float32', [256], [256], 2.527024927258026, -2.1782388387364335, -0.0008296193100250093,
     ' ', '437213.84590749856%', '345658.76916858414%', '22823.676544842117%', ' ', 2.5276029109954834,
     -2.1788690090179443, -0.0008259844034910202, 0.0005779837374575436, -0.0006301702815108001, 3.634906533989124e-06,
     'Warning', 'Need double check api accuracy.', 'None'],
    ['Aten__native_batch_norm_legit_functional.default_0_forward.input.2', 'Functional_batch_norm_0_forward.input.2',
     'torch.float32', 'torch.float32', [256], [256], 1.5384095311164856, -3.7736878395080566, -0.9390918612480164, ' ',
     '164.74538192025793%', '406.7705163736246%', '100.94122819224167%', ' ', 2.472219944000244, -2.845968723297119,
     -0.008756577968597412, 0.9338104128837585, 0.9277191162109375, 0.930335283279419, 'Warning',
     'Need double check api accuracy.', 'None'],
    ['Aten__native_batch_norm_legit_functional.default_0_forward.input.3', 'Functional_batch_norm_0_forward.input.3',
     'torch.float32', 'torch.float32', [256], [256], 1.763145923614502, -4.398397922515869, -1.0521326325833797, ' ',
     '176.3145923614502%', '439.8397922515869%', '105.21326325833797%', ' ', 2.763145923614502, -3.398397922515869,
     -0.052132632583379745, 1.0, 1.0, 1.0, 'Warning', 'Need double check api accuracy.', 'None'],
    ['Aten__native_batch_norm_legit_functional.default_0_forward.input.4', 'Functional_batch_norm_0_forward.input.4',
     'torch.float32', 'torch.float32', [256], [256], 2.673110008239746, -3.149275064468384, 0.01613386906683445, ' ',
     'N/A', 'N/A', 'N/A', ' ', 2.673110008239746, -3.149275064468384, 0.01613386906683445, 0.0, 0.0, 0.0, 'Warning',
     'Need double check api accuracy.', 'None'],
    ['Aten__native_batch_norm_legit_functional.default_0_forward.output.0', 'Functional_batch_norm_0_forward.output',
     'torch.float16', 'torch.float32', [256, 256, 14, 14], [256, 256, 14, 14], 8.156781196594238, -4.843813419342041,
     -0.008758545174714527, ' ', '151.11009228611078%', '83.55995967687207%', '3464072756.115108%', ' ', 13.5546875,
     -10.640625, -0.008758544921875, 5.397906303405762, -5.796811580657959, 2.5283952709287405e-10, 'Warning',
     'Need double check api accuracy.', 'None'],
    ['Aten__native_batch_norm_legit_functional.default_0_forward.output.1', 'Nan', 'torch.float32', 'Nan', [256], 'Nan',
     ' ', ' ', ' ', ' ', ' ', ' ', 0.30550330877304077, -0.24485322833061218, -0.010361209511756897, 'Nan', 'Nan',
     'Nan',
     'Yes', '', 'None'],
    ['Aten__native_batch_norm_legit_functional.default_0_forward.output.2', 'Nan', 'torch.float32', 'Nan', [256], 'Nan',
     ' ', ' ', ' ', ' ', ' ', ' ', 623.9192504882812, 432.96826171875, 520.2276611328125, 'Nan', 'Nan', 'Nan',
     'Yes', '', 'None'],
    ['Aten__native_batch_norm_legit_functional.default_0_forward.output.3', 'Nan', 'torch.float32', 'Nan', [256], 'Nan',
     ' ', ' ', ' ', ' ', ' ', ' ', 2.4797861576080322, -3.055997371673584, -0.04795549064874649, 'Nan', 'Nan', 'Nan',
     'Yes', '', 'None'],
    ['Aten__native_batch_norm_legit_functional.default_0_forward.output.4', 'Nan', 'torch.float32', 'Nan', [256], 'Nan',
     ' ', ' ', ' ', ' ', ' ', ' ', 61.7945556640625, 42.59713363647461, 52.03831481933594, 'Nan', 'Nan', 'Nan',
     'Yes', '', 'None']]

op_data = {
    'input_args': [{'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3],
                    'Max': 0.33033010363578796, 'Min': -0.331031858921051, 'Mean': -0.030964046716690063,
                    'Norm': 2.2533628940582275, 'requires_grad': True},
                   {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3],
                    'Max': 0.003992878366261721, 'Min': -0.008102823048830032, 'Mean': -0.0002002553956117481,
                    'Norm': 0.02844562754034996, 'requires_grad': False}],
    'input_kwargs': {'alpha': {'type': 'float', 'value': -0.1}},
    'output': [{'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3],
                'Max': 0.33033010363578796, 'Min': -0.331031858921051, 'Mean': -0.030964046716690063,
                'Norm': 2.2533628940582275, 'requires_grad': True}]}

op_name = "Tensor.add_0.0.forward"

op_result = [
    {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3],
     'Max': 0.33033010363578796, 'Min': -0.331031858921051, 'Mean': -0.030964046716690063,
     'Norm': 2.2533628940582275, 'requires_grad': True, 'full_op_name': 'Tensor.add_0.0.forward.input.0'},
    {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3],
     'Max': 0.003992878366261721, 'Min': -0.008102823048830032, 'Mean': -0.0002002553956117481,
     'Norm': 0.02844562754034996, 'requires_grad': False, 'full_op_name': 'Tensor.add_0.0.forward.input.1'},
    {'full_op_name': 'Tensor.add_0.0.forward.input.alpha.0', 'dtype': "<class 'float'>", 'shape': '[]', 'md5': None,
     'Max': -0.1, 'Min': -0.1, 'Mean': -0.1, 'Norm': -0.1, 'data_name': '-1'},
    {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3],
     'Max': 0.33033010363578796, 'Min': -0.331031858921051, 'Mean': -0.030964046716690063,
     'Norm': 2.2533628940582275, 'requires_grad': True, 'full_op_name': 'Tensor.add_0.0.forward.output.0'}]

base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'test_acc_compare_data')
base_dir2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'test_acc_compare_data2')
base_dir3 = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'test_acc_compare_data3')
pt_dir = os.path.join(base_dir3, f'dump_data_dir')


def generate_dump_json(base_dir):
    data_path = os.path.join(base_dir, 'dump.json')
    data = {
        'task': 'statistics',
        'framework': 'pytorch',
        'level': 'L1',
        'dump_data_dir': '',
        'data': {
            'Functional.linear.0.forward': {
                'input_args': [
                    {'type': 'torch.Tensor',
                     'dtype': 'torch.float32',
                     'shape': [2, 2],
                     'Max': 2,
                     'Min': 0,
                     'Mean': 1,
                     'Norm': 1,
                     'requires_grad': False,
                     'data_name': 'Functional.linear.0.forward.input.0.pt'
                     }
                ]
            }
        }
    }
    with open(data_path, 'w') as json_file:
        json.dump(data, json_file)


def generate_dump_json_md5(base_dir):
    data_path = os.path.join(base_dir, 'dump_md5.json')
    data = {
        'task': 'statistics',
        'level': 'L1',
        'dump_data_dir': '',
        'data': {
            'Functional.linear.0.forward': {
                'input_args': [
                    {'type': 'torch.Tensor',
                     'dtype': 'torch.float32',
                     'shape': [2, 2],
                     'Max': 2,
                     'Min': 0,
                     'Mean': 1,
                     'Norm': 1,
                     'requires_grad': False,
                     'md5': 123456
                     }
                ]
            }
        }
    }
    with open(data_path, 'w') as json_file:
        json.dump(data, json_file)


def generate_stack_json(base_dir):
    data_path = os.path.join(base_dir, 'stack.json')
    data = {'Functional.linear.0.forward': ['File']}
    with open(data_path, 'w') as json_file:
        json.dump(data, json_file)


def generate_pt(base_dir):
    data_path = os.path.join(base_dir, 'Functional.linear.0.forward.input.0.pt')
    data = torch.Tensor([1, 2, 3, 4])
    torch.save(data, data_path)


class TestUtilsMethods(unittest.TestCase):

    def setUp(self):
        os.makedirs(base_dir, mode=0o750, exist_ok=True)
        os.makedirs(base_dir2, mode=0o750, exist_ok=True)
        os.makedirs(base_dir3, mode=0o750, exist_ok=True)
        os.makedirs(pt_dir, mode=0o750, exist_ok=True)

        self.lock = threading.Lock()

    def tearDown(self):
        if os.path.exists(base_dir):
            shutil.rmtree(base_dir)
        if os.path.exists(base_dir2):
            shutil.rmtree(base_dir2)
        if os.path.exists(pt_dir):
            shutil.rmtree(pt_dir)
        if os.path.exists(base_dir3):
            shutil.rmtree(base_dir3)

    def test_gen_merge_list(self):
        op_data = {
            'input_args': [
                {
                    'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [2, 2],
                    'Max': 1, 'Min': 1, 'Mean': 1, 'Norm': 1, 'requires_grad': False,
                    'data_name': 'Functional.linear.0.forward.input.0.pt',
                    'full_op_name': 'Functional.linear.0.forward.input.0'
                }
            ]
        }
        op_name = 'Functional.linear.0.forward'
        stack_json_data = {'Functional.linear.0.forward': ['File']}
        target_merge_list = [
            {
                'full_op_name': 'Functional.linear.0.forward.input.0',
                'type': 'torch.Tensor',
                'dtype': 'torch.float32',
                'shape': [2, 2],
                'requires_grad': 'False',
                'Max': 1,
                'Min': 1,
                'Mean': 1,
                'Norm': 1,
                'md5': '00000000',
                'data_name': 'Functional.linear.0.forward.input.0.pt',
                'state': 'input'
            },
            {
                'full_op_name': 'Functional.linear.0.forward',
                'full_info': ['File']
            }
        ]

        config_dict = {
            'stack_mode':  True,
            'auto_analyze': True,
            'fuzzy_match': False,
            'dump_mode': Const.SUMMARY,
        }
        mode_config = ModeConfig(**config_dict)
        result = ParseData(mode_config, 'rank0', {}).gen_merge_list(op_data, op_name, stack_json_data)
        self.assertEqual(result, target_merge_list)

    def test_check_op_item_fuzzy(self):
        config_dict = {
            'stack_mode': False,
            'auto_analyze': True,
            'fuzzy_match': True,
            'dump_mode': Const.SUMMARY,
        }
        mode_config = ModeConfig(**config_dict)
        mapping_config = MappingConfig()

        match = Match(mode_config, mapping_config, cross_frame=False)
        result = match.check_op_item(npu_op_item_fuzzy, bench_op_item_fuzzy)
        self.assertEqual(result, True)

    def test_compare_statistics(self):
        generate_dump_json(base_dir)
        generate_stack_json(base_dir)
        file_list = [os.path.join(base_dir, 'dump.json'), os.path.join(base_dir, 'dump.json'),
                     os.path.join(base_dir, 'stack.json')]

        config_dict = {
            'stack_mode': True,
            'auto_analyze': True,
            'fuzzy_match': False,
            'dump_mode': Const.SUMMARY,
        }
        mode_config = ModeConfig(**config_dict)
        mapping_config = MappingConfig()

        from msprobe.pytorch.compare.pt_compare import read_real_data
        comparator = Comparator(read_real_data, mode_config, mapping_config)
        parse_data = ParseData(mode_config, '', {})
        npu_df, bench_df = parse_data.parse(file_list)
        result = comparator.compare_statistics(npu_df, bench_df)
        o_data = [
            ['Functional.linear.0.forward.input.0', 'Functional.linear.0.forward.input.0',
             'torch.float32', 'torch.float32', '[2, 2]', '[2, 2]', 'False', 'False',
             '0', '0', '0', '0', '0', '0', '0', '0', 2, 0, 1, 1, 2, 0, 1, 1,
             True, '', '', ['File'], 'input', 'Functional.linear.0.forward'
             ]
        ]
        columns = CompareConst.SUMMARY_COMPARE_RESULT_HEADER + ['NPU_Stack_Info'] + ['state', 'api_origin_name']
        o_result = pd.DataFrame(o_data, columns=columns, dtype=object)
        self.assertEquals(result.loc[0, CompareConst.NPU_NAME], o_result.loc[0, CompareConst.NPU_NAME])
        self.assertEquals(result.loc[0, CompareConst.NPU_DTYPE], o_result.loc[0, CompareConst.NPU_DTYPE])
        self.assertEquals(result.loc[0, CompareConst.NPU_SHAPE], o_result.loc[0, CompareConst.NPU_SHAPE])
        self.assertEquals(result.loc[0, CompareConst.NPU_MEAN], o_result.loc[0, CompareConst.NPU_MEAN])
        self.assertEquals(result.loc[0, CompareConst.REQ_GRAD_CONSIST], o_result.loc[0, CompareConst.REQ_GRAD_CONSIST])


class TestParseData(unittest.TestCase):

    def setUp(self):
        os.makedirs(base_dir, mode=0o750, exist_ok=True)
        generate_dump_json(base_dir)
        generate_dump_json_md5(base_dir)
        generate_stack_json(base_dir)

        self.lock = threading.Lock()

        mode_config = ModeConfig(stack_mode=True)
        self.parser = ParseData(mode_config, "rank0", {})

        # consistent_check相关parse实例
        self.mock_mode_config = MagicMock()
        self.mock_mode_config.consistent_check = True
        self.mock_mode_config.backend = Const.FSDP
        self.parser_consistent_check = ParseData(mode_config=self.mock_mode_config, rank=0, data_dirs={})

    def tearDown(self):
        if os.path.exists(base_dir):
            shutil.rmtree(base_dir)

    def test_parse(self):
        file_list = [os.path.join(base_dir, 'dump.json'), os.path.join(base_dir, 'dump.json'),
                     os.path.join(base_dir, 'stack.json')]

        stack_mode = True
        mode_config = ModeConfig(stack_mode=stack_mode)
        parse_data = ParseData(mode_config, 'rank0', {})
        npu_df, bench_df = parse_data.parse(file_list)

        target_df = pd.DataFrame(
            [
                ['Functional.linear.0.forward.input.0', 'torch.float32', [2, 2], [2, 0, 1, 1], ['File'], 'input',
                 'Functional.linear.0.forward', 'False',
                 'forward', '0.forward', 'Functional.linear', 0, 0, '.input.0',]
            ],
            columns=[
                'op_name', 'dtype', 'shape', 'summary', 'stack_info', 'state',
                'api_origin_name', 'requires_grad',
                'direction', 'call_direction', 'op_no_number', 'forward_call_order', 'backward_call_order', 'suffix'
            ]
        )
        self.assertTrue(npu_df.equals(target_df))
        self.assertTrue(bench_df.equals(target_df))

    def test_gen_data_df_summary(self):
        npu_json_path = os.path.join(base_dir, 'dump.json')
        stack_json_path = os.path.join(base_dir, 'stack.json')
        npu_json_data = load_json(npu_json_path)
        stack_json_data = load_json(stack_json_path)

        stack_mode = True
        mode_config = ModeConfig(stack_mode=stack_mode)
        parse_data = ParseData(mode_config, 'rank0', {})
        npu_df = parse_data.gen_data_df(npu_json_data, stack_json_data, 'NPU')

        target_df = pd.DataFrame(
            [
                ['Functional.linear.0.forward.input.0', 'torch.float32', [2, 2], [2, 0, 1, 1], ['File'], 'input',
                 'Functional.linear.0.forward', 'False',
                 'forward', '0.forward', 'Functional.linear', 0, 0, '.input.0',]
            ],
            columns=[
                'op_name', 'dtype', 'shape', 'summary', 'stack_info', 'state',
                'api_origin_name', 'requires_grad',
                'direction', 'call_direction', 'op_no_number', 'forward_call_order', 'backward_call_order', 'suffix'
            ]
        )
        self.assertTrue(npu_df.equals(target_df))

    def test_gen_data_df_all(self):
        npu_json_path = os.path.join(base_dir, 'dump.json')
        stack_json_path = os.path.join(base_dir, 'stack.json')
        npu_json_data = load_json(npu_json_path)
        stack_json_data = load_json(stack_json_path)

        stack_mode = True
        mode_config = ModeConfig(stack_mode=stack_mode, dump_mode=Const.ALL)
        parse_data = ParseData(mode_config, 'rank0', {})
        npu_df = parse_data.gen_data_df(npu_json_data, stack_json_data, 'NPU')

        target_df = pd.DataFrame(
            [
                ['Functional.linear.0.forward.input.0', 'torch.float32', [2, 2], [2, 0, 1, 1], ['File'], 'input',
                 'Functional.linear.0.forward', 'False',
                 'forward', '0.forward', 'Functional.linear', 0, 0, '.input.0',
                 'Functional.linear.0.forward.input.0.pt', -1]
            ],
            columns=[
                'op_name', 'dtype', 'shape', 'summary', 'stack_info', 'state',
                'api_origin_name', 'requires_grad',
                'direction', 'call_direction', 'op_no_number', 'forward_call_order', 'backward_call_order', 'suffix',
                'data_name', 'dirty_valid_len'
            ]
        )
        self.assertTrue(npu_df.equals(target_df))

    def test_gen_data_df_md5(self):
        npu_json_path = os.path.join(base_dir, 'dump_md5.json')
        stack_json_path = os.path.join(base_dir, 'stack.json')
        npu_json_data = load_json(npu_json_path)
        stack_json_data = load_json(stack_json_path)

        stack_mode = True
        mode_config = ModeConfig(stack_mode=stack_mode, dump_mode=Const.MD5)
        parse_data = ParseData(mode_config, 'rank0', {})
        npu_df = parse_data.gen_data_df(npu_json_data, stack_json_data, 'NPU')

        target_df = pd.DataFrame(
            [
                ['Functional.linear.0.forward.input.0', 'torch.float32', [2, 2], [2, 0, 1, 1], ['File'], 'input',
                 'Functional.linear.0.forward', 'False',
                 'forward', '0.forward', 'Functional.linear', 0, 0, '.input.0',
                 123456]
            ],
            columns=[
                'op_name', 'dtype', 'shape', 'summary', 'stack_info', 'state',
                'api_origin_name', 'requires_grad',
                'direction', 'call_direction', 'op_no_number', 'forward_call_order', 'backward_call_order', 'suffix',
                'md5'
            ]
        )
        self.assertTrue(npu_df.equals(target_df))

    def test_gen_merge_list(self):
        npu_json_path = os.path.join(base_dir, 'dump.json')
        stack_json_path = os.path.join(base_dir, 'stack.json')
        npu_json_data = load_json(npu_json_path)
        stack_json_data = load_json(stack_json_path)

        stack_mode = True
        mode_config = ModeConfig(stack_mode=stack_mode)
        parse_data = ParseData(mode_config, 'rank0', {})
        op_data = npu_json_data.get('data').get('Functional.linear.0.forward')
        merge_list = parse_data.gen_merge_list(op_data, 'Functional.linear.0.forward', stack_json_data)

        target_merge_list = [
            {
                'full_op_name': 'Functional.linear.0.forward.input.0',
                'type': 'torch.Tensor',
                'dtype': 'torch.float32',
                'shape': [2, 2],
                'requires_grad': 'False',
                'Max': 2,
                'Min': 0,
                'Mean': 1,
                'Norm': 1,
                'md5': '00000000',
                'data_name': 'Functional.linear.0.forward.input.0.pt',
                'state': 'input'
            },
            {
                'full_op_name': 'Functional.linear.0.forward',
                'full_info': ['File']
            }
        ]
        self.assertEqual(merge_list, target_merge_list)

    def test_split_data_name_components_parameters_grad(self):
        """场景：仅有 parameters_grad"""
        op, call_dir, direction = self.parser.split_data_name_components("model.parameters_grad")
        self.assertEqual(op, 'model')
        self.assertEqual(call_dir, 'parameters_grad')
        self.assertEqual(direction, 'parameters_grad')

    def test_split_data_name_components_forward_after_number(self):
        """场景：...数字.forward (API 级别数字在前)"""
        # 输入：op.123.forward
        # 期望：op_no='op', call_dir='123.forward', dir='forward'
        op, call_dir, direction = self.parser.split_data_name_components("op.123.forward")
        self.assertEqual(op, 'op')
        self.assertEqual(call_dir, '123.forward')
        self.assertEqual(direction, 'forward')

    def test_split_data_name_components_backward_after_number(self):
        """场景：...数字.backward"""
        op, call_dir, direction = self.parser.split_data_name_components("module.99.backward")
        self.assertEqual(op, 'module')
        self.assertEqual(call_dir, '99.backward')
        self.assertEqual(direction, 'backward')

    def test_split_data_name_components_forward_before_number(self):
        """场景：...forward.数字 (模块级别数字在后)"""
        # 输入：op.forward.1
        # 期望：op_no='op', call_dir='forward.1', dir='forward'
        op, call_dir, direction = self.parser.split_data_name_components("op.forward.1")
        self.assertEqual(op, 'op')
        self.assertEqual(call_dir, 'forward.1')
        self.assertEqual(direction, 'forward')

    def test_split_data_name_components_backward_before_number(self):
        """场景：...backward.数字"""
        op, call_dir, direction = self.parser.split_data_name_components("api.backward.5")
        self.assertEqual(op, 'api')
        self.assertEqual(call_dir, 'backward.5')
        self.assertEqual(direction, 'backward')

    def test_split_data_name_components_no_match(self):
        """场景：不匹配任何规则（控制用例）"""
        op, call_dir, direction = self.parser.split_data_name_components("op.123.gradient")
        self.assertEqual(op, '')
        self.assertEqual(call_dir, '')
        self.assertEqual(direction, '')

    @patch('msprobe.core.compare.acc_compare.logger')
    def test_case_1_consistent_check_false(self, mock_logger):
        """
        场景 1: consistent_check 为 False
        预期：无论其他参数如何，直接返回 True
        """
        self.mock_mode_config.consistent_check = False

        result = self.parser_consistent_check.should_parse_op(parse_flag=False, op_name="any.name", device="Npu")

        self.assertTrue(result)
        mock_logger.error.assert_not_called()

    @patch('msprobe.core.compare.acc_compare.logger')
    def test_case_2_device_is_bench(self, mock_logger):
        """
        场景 2: device 为 'Bench'
        预期：直接返回 True
        """
        self.mock_mode_config.consistent_check = True

        result = self.parser_consistent_check.should_parse_op(parse_flag=False, op_name="any.name", device="Bench")

        self.assertTrue(result)
        mock_logger.error.assert_not_called()

    @patch('msprobe.core.compare.acc_compare.logger')
    def test_case_3_data_name_length_invalid(self, mock_logger):
        """
        场景 3: data_name 分割后长度 < 3
        预期：记录错误日志，返回 False
        """
        self.mock_mode_config.consistent_check = True

        # 构造只有 2 段的数据
        invalid_data_name = f"part1.part2"

        result = self.parser_consistent_check.should_parse_op(parse_flag=True, op_name=invalid_data_name, device="Npu")

        self.assertFalse(result)
        mock_logger.error.assert_called_once()
        # 验证日志内容包含关键信息
        call_args = mock_logger.error.call_args[0][0]
        self.assertIn("dump.json", call_args)
        self.assertIn(invalid_data_name, call_args)

    @patch('msprobe.core.compare.acc_compare.logger')
    def test_case_4_stop_parse_list_match(self, mock_logger):
        """
        场景 4: 命中停止解析列表 (VERL_FSDP_STOP_PARSE_LIST) 且为 backward
        预期：返回 False
        """
        self.mock_mode_config.consistent_check = True
        self.mock_mode_config.backend = Const.FSDP

        # 构造数据：third_last 在列表中，second_last 为 'backward'
        # 格式： "p1.Qwen3Model.backward.suffix" -> split: [p1, Qwen3Model, backward, suffix]
        # [-3] = Qwen3Model, [-2] = backward
        data_name = f"prefix1.Qwen3Model.backward.suffix"

        result = self.parser_consistent_check.should_parse_op(parse_flag=True, op_name=data_name, device="Npu")

        self.assertFalse(result)
        mock_logger.error.assert_not_called()

    @patch('msprobe.core.compare.acc_compare.logger')
    def test_case_5_embedding_forward_match(self, mock_logger):
        """
        场景 5: 命中 Embedding + forward
        预期：返回 True
        """
        self.mock_mode_config.consistent_check = True
        self.mock_mode_config.backend = Const.FSDP

        # 构造：third_last='Embedding', second_last='forward'
        data_name = f"prefix1.Embedding.forward.suffix"

        result = self.parser_consistent_check.should_parse_op(parse_flag=False, op_name=data_name, device="Npu")

        self.assertTrue(result)
        mock_logger.error.assert_not_called()

    @patch('msprobe.core.compare.acc_compare.logger')
    def test_case_6_default_return_flag(self, mock_logger):
        """
        场景 6: 未命中任何特殊条件
        预期：返回原始 parse_flag
        """
        self.mock_mode_config.consistent_check = True
        self.mock_mode_config.backend = Const.FSDP

        # 构造普通数据，不命中 List 也不命中 Embedding
        data_name = f"prefix1.MatMul.forward.suffix"

        # 测试输入 True
        result_true = self.parser_consistent_check.should_parse_op(parse_flag=True, op_name=data_name, device="Npu")
        self.assertTrue(result_true)

        # 测试输入 False
        result_false = self.parser_consistent_check.should_parse_op(parse_flag=False, op_name=data_name, device="Npu")
        self.assertFalse(result_false)

        mock_logger.error.assert_not_called()


class TestProcessDf(unittest.TestCase):

    def setUp(self):
        self.mode_config = ModeConfig()
        mapping_config = MappingConfig()
        mapping_dict = MappingDict(mapping_config)
        self.process_df = ProcessDf(self.mode_config, mapping_config, mapping_dict)

    @staticmethod
    def _create_test_df(data_dict):
        """辅助方法：创建测试 DataFrame"""
        return pd.DataFrame(data_dict)

    def test_get_api_name_success(self):
        api_list = ['Functional', 'linear', '0', 'forward', 'input', '0']

        mode_config = ModeConfig()
        mapping_config = MappingConfig()
        mapping_dict = MappingDict(mapping_config)
        process_df = ProcessDf(mode_config, mapping_config, mapping_dict)
        api_name = process_df.get_api_name(api_list)

        target_api_name = 'Functional.linear'
        self.assertEqual(api_name, target_api_name)

    @patch('msprobe.core.compare.acc_compare.logger')
    def test_get_api_name_index_error(self, mock_logger):
        api_list = ['Functional']
        with self.assertRaises(CompareException) as context:
            mode_config = ModeConfig()
            mapping_config = MappingConfig()
            mapping_dict = MappingDict(mapping_config)
            process_df = ProcessDf(mode_config, mapping_config, mapping_dict)
            api_name = process_df.get_api_name(api_list)
        self.assertEqual(context.exception.code, CompareException.INDEX_OUT_OF_BOUNDS_ERROR)
        mock_logger.error.assert_called_once_with('Failed to retrieve API name, please check if the dump data is reasonable')

    def test_process_compare_key_and_shape(self):
        npu_df_o = bench_df_o = pd.DataFrame(
            [['Functional.linear.0.forward.input.0', 'torch.float32', [2, 2], [2, 0, 1, 1], ['File'], 'Functional.linear.0.forward.input.0']],
            columns=['op_name', 'dtype', 'shape', 'summary', 'stack_info', 'op_name_update']
        )

        mode_config = ModeConfig()
        mapping_config = MappingConfig()
        mapping_dict = MappingDict(mapping_config)
        process_df = ProcessDf(mode_config, mapping_config, mapping_dict)
        npu_df, bench_df = process_df.process_compare_key_and_shape(npu_df_o, bench_df_o)

        target_df = pd.DataFrame(
            [['Functional.linear.0.forward.input.0', 'torch.float32', [2, 2], [2, 0, 1, 1], ['File'], 'Functional.linear.0.forward.input.0', 'Functional.linear.0.forward.input.0', [2, 2]]],
            columns=['op_name', 'dtype', 'shape', 'summary', 'stack_info', 'op_name_update', 'compare_key', 'compare_shape']
        )
        self.assertTrue(npu_df.equals(target_df))
        self.assertTrue(bench_df.equals(target_df))

    def test_process_internal_api_mapping(self):
        mode_config = ModeConfig()
        mapping_config = MappingConfig()
        mapping_dict = MappingDict(mapping_config)
        process_df = ProcessDf(mode_config, mapping_config, mapping_dict)

        # mint to torch
        npu_op_name = 'Mint.mean.0.input.0'
        target_name = 'Torch.mean.0.input.0'
        name = process_df.process_internal_api_mapping(npu_op_name)
        self.assertEqual(name, target_name)

        # mintfunctional to functional
        npu_op_name = 'MintFunctional.mean.0.input.0'
        target_name = 'Functional.mean.0.input.0'
        name = process_df.process_internal_api_mapping(npu_op_name)
        self.assertEqual(name, target_name)

        # inner mapping exists
        npu_op_name = 'Functional.abs.0.input.0'
        mapping_dict.ms_to_pt_mapping = {'Functional.abs': 'Torch.abs'}
        target_name = 'Torch.abs.0.input.0'
        name = process_df.process_internal_api_mapping(npu_op_name)
        self.assertEqual(name, target_name)

        # inner mapping not found
        npu_op_name = 'Functional.abs.0.input.0'
        mapping_dict.ms_to_pt_mapping = {}
        target_name = 'Functional.abs.0.input.0'
        name = process_df.process_internal_api_mapping(npu_op_name)
        self.assertEqual(name, target_name)

    def test_modify_compare_data_with_user_mapping(self):
        mode_config = ModeConfig()
        mapping_config = MappingConfig()
        mapping_dict = MappingDict(mapping_config)
        process_df = ProcessDf(mode_config, mapping_config, mapping_dict)
        mapping_dict.api_mapping_dict = [{
            'ms_api': 'Functional.conv2d',
            'pt_api': 'Torch.conv2d',
            'ms_args': [0],
            'pt_args': [0]
        }]

        npu_df = pd.DataFrame([
            ['Functional.conv2d.0.forward.input.0', 'float32', [1, 2], 'summary', 'stack_info', 'Functional.conv2d.0.forward.input.0', 'input', 'Functional.conv2d.0.forward'],
            ['Functional.amax.0.forward.input.0', 'float32', [1, 2], 'summary', 'stack_info', 'Functional.amax.0.forward.input.0', 'input', 'Functional.amax.0.forward']
        ], columns=['op_name', 'dtype', 'shape', 'summary', 'stack_info', 'compare_key', 'state', 'api_origin_name'])
        bench_df = pd.DataFrame([
            ['Torch.conv2d.0.forward.input.0', 'float32', [1, 2], 'summary', 'stack_info', 'Torch.conv2d.0.forward.input.0', 'input', 'Functional.conv2d.0.forward'],
            ['Torch.amax.0.forward.input.0', 'float32', [1, 2], 'summary', 'stack_info', 'Torch.amax.0.forward.input.0', 'input', 'Functional.amax.0.forward']
        ], columns=['op_name', 'dtype', 'shape', 'summary', 'stack_info', 'compare_key', 'state', 'api_origin_name'])

        process_df.modify_compare_data_with_user_mapping(npu_df, bench_df)

    def test_get_api_indices_dict(self):
        mode_config = ModeConfig()
        mapping_config = MappingConfig()
        mapping_dict = MappingDict(mapping_config)
        process_df = ProcessDf(mode_config, mapping_config, mapping_dict)

        op_name_df = pd.DataFrame([
            ['Functional.conv2d.0.forward.input.0', 'float32', [1, 2], 'summary', 'stack_info', 'Functional.conv2d.0.forward.input.0'],
            ['Functional.amax.0.forward.input.0', 'float32', [1, 2], 'summary', 'stack_info', 'Functional.amax.0.forward.input.0']
        ], columns=['op_name', 'dtype', 'shape', 'summary', 'stack_info', 'compare_key'])

        api_indices_dict = process_df.get_api_indices_dict(op_name_df)
        expected = {
            'Functional.conv2d': [0],
            'Functional.amax': [1]
        }
        self.assertEqual(api_indices_dict, expected)

    def test_process_cell_mapping(self):
        mode_config = ModeConfig()
        mapping_config = MappingConfig()
        mapping_dict = MappingDict(mapping_config)
        process_df = ProcessDf(mode_config, mapping_config, mapping_dict)

        # not name
        npu_op_name = None
        name = process_df.process_cell_mapping(npu_op_name)
        self.assertEqual(name, CompareConst.N_A)

        # not params_grad
        npu_op_name = 'MintFunctional.embedding.0.input.0'
        name = process_df.process_cell_mapping(npu_op_name)
        self.assertEqual(name, CompareConst.N_A)

        # default replace
        npu_op_name = 'Cell.network_with_loss.module.GPTModel.forward.1.input.0'
        name = process_df.process_cell_mapping(npu_op_name)
        self.assertEqual(name, 'Module.network_with_loss.module.GPTModel.forward.1.input.0')

        # mapping_dict
        npu_op_name = 'Cell.fc1.Dense.forward.0.input.0'
        mapping_dict.cell_mapping_dict = {'fc1.Dense': 'module.name'}
        name = process_df.process_cell_mapping(npu_op_name)
        self.assertEqual(name, 'Module.module.name.forward.0.input.0')

    def test_process_data_mapping(self):
        mode_config = ModeConfig()
        mapping_config = MappingConfig()
        mapping_dict = MappingDict(mapping_config)
        process_df = ProcessDf(mode_config, mapping_config, mapping_dict)

        npu_op_name = 'Functional.flash_attention_score.4.forward.input.0'
        mapping_dict.data_mapping_dict = {'Functional.flash_attention_score.4.forward.input.0': 'NPU.npu_fusion_attention.4.forward.input.0'}
        name = process_df.process_data_mapping(npu_op_name)
        self.assertEqual(name, 'NPU.npu_fusion_attention.4.forward.input.0')

    def test_update_backward_call_digit_at_head(self):
        cmp_df = pd.DataFrame({
            CompareConst.OP_NAME: ["conv.1.backward.fp32"],
            Const.CALL_DIRECTION: ["1.backward"],
            Const.DIRECTION: [Const.BACKWARD],
            Const.BACKWARD_CALL_ORDER: [3],
            Const.OP_NO_NUMBER: ["conv"],
            Const.SUFFIX: [".fp32"],
        })

        result = self.process_df.update_backward_call(cmp_df)

        self.assertEqual(result.loc[0, Const.CALL_DIRECTION], "3.backward")
        self.assertEqual(
            result.loc[0, CompareConst.OP_NAME_UPDATE],
            "conv.3.backward.fp32"
        )

    def test_update_backward_call_digit_at_tail(self):
        cmp_df = pd.DataFrame({
            CompareConst.OP_NAME: ["matmul.backward.2.fp16"],
            Const.CALL_DIRECTION: ["backward.2"],
            Const.DIRECTION: [Const.BACKWARD],
            Const.BACKWARD_CALL_ORDER: [5],
            Const.OP_NO_NUMBER: ["matmul"],
            Const.SUFFIX: [".fp16"],
        })

        result = self.process_df.update_backward_call(cmp_df)

        self.assertEqual(result.loc[0, Const.CALL_DIRECTION], "backward.5")
        self.assertEqual(
            result.loc[0, CompareConst.OP_NAME_UPDATE],
            "matmul.backward.5.fp16"
        )

    def test_update_backward_call_non_backward_should_not_change(self):
        cmp_df = pd.DataFrame({
            CompareConst.OP_NAME: ["add.1.forward.fp32"],
            Const.CALL_DIRECTION: ["1.forward"],
            Const.DIRECTION: [Const.FORWARD],
            Const.BACKWARD_CALL_ORDER: [7],
            Const.OP_NO_NUMBER: ["add"],
            Const.SUFFIX: [".fp32"],
        })

        result = self.process_df.update_backward_call(cmp_df)

        self.assertEqual(result.loc[0, Const.CALL_DIRECTION], "1.forward")
        self.assertEqual(
            result.loc[0, CompareConst.OP_NAME],
            "add.1.forward.fp32"
        )

    def test_update_forward_call_head_is_digit(self):
        """
        场景 1: forward 调用，head 是数字，需要替换
        输入：call_direction="1.xxx", direction="forward", forward_call_order="5"
        预期：call_direction="5.xxx"
        """
        df = self._create_test_df({
            Const.CALL_DIRECTION: ["1.xxx", "2.yyy", "abc.zzz"],
            Const.DIRECTION: ["forward", "forward", "forward"],
            Const.FORWARD_CALL_ORDER: ["5", "6", "7"]
        })

        result = ProcessDf.update_forward_call(df)

        # 验证 head 是数字的被替换
        self.assertEqual(result.loc[0, Const.CALL_DIRECTION], "5.xxx")
        self.assertEqual(result.loc[1, Const.CALL_DIRECTION], "6.yyy")
        # head 不是数字的保持不变
        self.assertEqual(result.loc[2, Const.CALL_DIRECTION], "abc.zzz")

    def test_update_forward_call_tail_is_digit(self):
        """
        场景 2: forward 调用，tail 是数字，需要替换
        输入：call_direction="xxx.1", direction="forward", forward_call_order="5"
        预期：call_direction="xxx.5"
        """
        df = self._create_test_df({
            Const.CALL_DIRECTION: ["xxx.1", "yyy.2", "zzz.abc"],
            Const.DIRECTION: ["forward", "forward", "forward"],
            Const.FORWARD_CALL_ORDER: ["5", "6", "7"]
        })

        result = ProcessDf.update_forward_call(df)

        # 验证 tail 是数字的被替换
        self.assertEqual(result.loc[0, Const.CALL_DIRECTION], "xxx.5")
        self.assertEqual(result.loc[1, Const.CALL_DIRECTION], "yyy.6")
        # tail 不是数字的保持不变
        self.assertEqual(result.loc[2, Const.CALL_DIRECTION], "zzz.abc")

    def test_update_forward_call_backward_not_modified(self):
        """
        场景 3: backward 调用，不应该被修改
        """
        df = self._create_test_df({
            Const.CALL_DIRECTION: ["1.2", "3.4"],
            Const.DIRECTION: ["backward", "backward"],
            Const.FORWARD_CALL_ORDER: ["5", "6"]
        })

        result = ProcessDf.update_forward_call(df)

        # backward 调用保持不变
        self.assertEqual(result.loc[0, Const.CALL_DIRECTION], "1.2")
        self.assertEqual(result.loc[1, Const.CALL_DIRECTION], "3.4")

    def test_update_forward_call_mixed_directions(self):
        """
        场景 4: 混合 forward 和 backward，只有 forward 被修改
        """
        df = self._create_test_df({
            Const.CALL_DIRECTION: ["1.xxx", "2.yyy", "3.zzz", "4.aaa"],
            Const.DIRECTION: ["forward", "backward", "forward", "backward"],
            Const.FORWARD_CALL_ORDER: ["5", "6", "7", "8"]
        })

        result = ProcessDf.update_forward_call(df)

        # forward 且 head 是数字的被修改
        self.assertEqual(result.loc[0, Const.CALL_DIRECTION], "5.xxx")
        # backward 保持不变
        self.assertEqual(result.loc[1, Const.CALL_DIRECTION], "2.yyy")
        # forward 且 head 是数字的被修改
        self.assertEqual(result.loc[2, Const.CALL_DIRECTION], "7.zzz")
        # backward 保持不变
        self.assertEqual(result.loc[3, Const.CALL_DIRECTION], "4.aaa")

    def test_update_forward_call_no_digit_in_call_direction(self):
        """
        场景 5: forward 调用，但 head 和 tail 都不是数字，不修改
        """
        df = self._create_test_df({
            Const.CALL_DIRECTION: ["abc.def", "xyz.uvw"],
            Const.DIRECTION: ["forward", "forward"],
            Const.FORWARD_CALL_ORDER: ["5", "6"]
        })

        result = ProcessDf.update_forward_call(df)

        # 没有数字，保持不变
        self.assertEqual(result.loc[0, Const.CALL_DIRECTION], "abc.def")
        self.assertEqual(result.loc[1, Const.CALL_DIRECTION], "xyz.uvw")

    def test_update_forward_call_returns_same_dataframe_object(self):
        """
        场景 6: 验证返回的是同一个 DataFrame 对象（原地修改）
        """
        df = self._create_test_df({
            Const.CALL_DIRECTION: ["1.xxx"],
            Const.DIRECTION: ["forward"],
            Const.FORWARD_CALL_ORDER: ["5"]
        })

        result = ProcessDf.update_forward_call(df)

        # 验证是同一个对象
        self.assertIs(result, df)

    def test_get_op_layer_normal_match(self):
        """
        场景 1: 正常匹配层数
        输入：op_no_number = "model.layers.0.attention"
        预期：layer = 0
        """
        df = self._create_test_df({
            Const.OP_NO_NUMBER: ["model.layers.0.attention", "model.layers.123.mlp"]
        })

        self.process_df.get_op_layer(df)

        self.assertEqual(df.loc[0, Const.LAYER], 0)
        self.assertEqual(df.loc[1, Const.LAYER], 123)

    def test_get_op_layer_no_match_default_value(self):
        """
        场景 2: 不匹配层数，返回默认值 -1
        输入：op_no_number = "embedding.weight"
        预期：layer = -1
        """
        df = self._create_test_df({
            Const.OP_NO_NUMBER: ["embedding.weight", "lm_head.bias"]
        })

        self.process_df.get_op_layer(df)

        self.assertEqual(df.loc[0, Const.LAYER], -1)
        self.assertEqual(df.loc[1, Const.LAYER], -1)

    def test_get_op_layer_mixed_match(self):
        """
        场景 3: 混合情况，部分匹配部分不匹配
        """
        df = self._create_test_df({
            Const.OP_NO_NUMBER: [
                "model.layers.0.attention",
                "embedding.weight",
                "model.layers.5.mlp"
            ]
        })

        self.process_df.get_op_layer(df)

        self.assertEqual(df.loc[0, Const.LAYER], 0)
        self.assertEqual(df.loc[1, Const.LAYER], -1)
        self.assertEqual(df.loc[2, Const.LAYER], 5)

    def test_process_module_mapping_infer_engine(self):
        """
        场景 1: engine == 'infer'，直接赋值 MODULE_MAPPING = MODULE
        """
        df = self._create_test_df({
            Const.MODULE: ["model.attention", "model.mlp"],
            Const.MODULE_LEN: [2, 2]
        })

        self.process_df.process_module_mapping(df, engine='infer', backend='fsdp')

        # infer 模式下，MODULE_MAPPING 直接等于 MODULE
        self.assertEqual(df.loc[0, Const.MODULE_MAPPING], "attention")
        self.assertEqual(df.loc[1, Const.MODULE_MAPPING], "mlp")

    def test_process_module_mapping_module_len_1(self):
        """
        场景 2: MODULE_LEN == 1，设置默认值（prefix 为空，last = MODULE）
        """
        self.mode_config.backend = Const.FSDP
        df = self._create_test_df({
            Const.MODULE: ["attention", "mlp"],
            Const.MODULE_LEN: [1, 1]
        })

        self.process_df.process_module_mapping(df, engine='train', backend='fsdp')

        # MODULE_LEN == 1 时，MODULE_MAPPING = MODULE（无 prefix）
        self.assertEqual(df.loc[0, Const.MODULE_PART_PREFIX], "")
        self.assertEqual(df.loc[0, Const.MODULE_LAST], "attention")
        self.assertEqual(df.loc[0, Const.MODULE_MAPPING], "attention")

    def test_process_module_mapping_qkv_proj_mapping(self):
        """
        场景 3: q_proj/k_proj/v_proj 映射到 qkv_proj
        """
        self.mode_config.backend = Const.FSDP
        df = self._create_test_df({
            Const.MODULE: ["model.q_proj", "model.k_proj", "model.v_proj"],
            Const.MODULE_LEN: [2, 2, 2]
        })

        self.process_df.process_module_mapping(df, engine='train', backend='fsdp')

        # 验证拆分正确
        self.assertEqual(df.loc[0, Const.MODULE_PART_PREFIX], "model")
        self.assertEqual(df.loc[0, Const.MODULE_LAST], "q_proj")
        # 验证映射后 q_proj -> qkv_proj
        self.assertEqual(df.loc[0, Const.MODULE_LAST_MAPPING], "qkv_proj")
        self.assertEqual(df.loc[0, Const.MODULE_MAPPING], "qkv_proj")

        # k_proj 也映射到 qkv_proj
        self.assertEqual(df.loc[1, Const.MODULE_LAST_MAPPING], "qkv_proj")
        self.assertEqual(df.loc[1, Const.MODULE_MAPPING], "qkv_proj")

        # v_proj 也映射到 qkv_proj
        self.assertEqual(df.loc[2, Const.MODULE_LAST_MAPPING], "qkv_proj")
        self.assertEqual(df.loc[2, Const.MODULE_MAPPING], "qkv_proj")

    def test_process_module_mapping_gate_up_proj_mapping(self):
        """
        场景 4: gate_proj/up_proj 映射到 gate_up_proj
        """
        self.mode_config.backend = Const.FSDP
        df = self._create_test_df({
            Const.MODULE: ["model.gate_proj", "model.up_proj"],
            Const.MODULE_LEN: [2, 2]
        })

        self.process_df.process_module_mapping(df, engine='train', backend='fsdp')

        # gate_proj -> gate_up_proj
        self.assertEqual(df.loc[0, Const.MODULE_LAST], "gate_proj")
        self.assertEqual(df.loc[0, Const.MODULE_LAST_MAPPING], "gate_up_proj")
        self.assertEqual(df.loc[0, Const.MODULE_MAPPING], "gate_up_proj")

        # up_proj -> gate_up_proj
        self.assertEqual(df.loc[1, Const.MODULE_LAST], "up_proj")
        self.assertEqual(df.loc[1, Const.MODULE_LAST_MAPPING], "gate_up_proj")
        self.assertEqual(df.loc[1, Const.MODULE_MAPPING], "gate_up_proj")

    def test_process_module_mapping_no_mapping_fallback(self):
        """
        场景 5: 无映射关系的模块，使用原值
        """
        self.mode_config.backend = Const.FSDP
        df = self._create_test_df({
            Const.MODULE: ["model.attention", "model.mlp", "model.norm"],
            Const.MODULE_LEN: [2, 2, 2]
        })

        self.process_df.process_module_mapping(df, engine='train', backend='fsdp')

        # 无映射时，MODULE_LAST_MAPPING = MODULE_LAST，MODULE_MAPPING 保持原样
        self.assertEqual(df.loc[0, Const.MODULE_LAST], "attention")
        self.assertEqual(df.loc[0, Const.MODULE_LAST_MAPPING], "attention")
        self.assertEqual(df.loc[0, Const.MODULE_MAPPING], "attention")

        self.assertEqual(df.loc[1, Const.MODULE_LAST], "mlp")
        self.assertEqual(df.loc[1, Const.MODULE_LAST_MAPPING], "mlp")
        self.assertEqual(df.loc[1, Const.MODULE_MAPPING], "mlp")

    @patch.object(ProcessDf, 'get_op_layer')
    @patch.object(ProcessDf, 'get_op_module_and_class')
    @patch.object(ProcessDf, 'process_module_mapping')
    @patch.object(ProcessDf, 'update_forward_call')
    def test_process_consistent_df_fsdp_backend(self, mock_update_forward, mock_process_mapping,
                                                mock_get_module, mock_get_layer):
        self.mode_config.backend = Const.FSDP
        npu_df = self._create_test_df({Const.OP_NO_NUMBER: ["layers.0.model.attention.Attention"]})
        bench_df = self._create_test_df({Const.OP_NO_NUMBER: ["layers.0.model.attention.Attention"]})

        result_npu, result_bench = self.process_df.process_consistent_df(npu_df, bench_df)

        # 验证 get_op_layer 被调用 2 次 (npu 和 bench)
        self.assertEqual(mock_get_layer.call_count, 2)
        # 验证 get_op_module_and_class 被调用 2 次 (train 和 infer)
        self.assertEqual(mock_get_module.call_count, 2)
        # 验证 process_module_mapping 被调用 2 次 (train 和 infer)
        self.assertEqual(mock_process_mapping.call_count, 2)
        # 验证 update_forward_call 只调用 1 次 (仅 npu)
        self.assertEqual(mock_update_forward.call_count, 1)


class TestMatch(unittest.TestCase):

    def setUp(self):
        """初始化测试固件"""
        self.mode_config = ModeConfig()
        self.mapping_config = MappingConfig()
        self.match = Match(self.mode_config, self.mapping_config, cross_frame=None)

    @staticmethod
    def _create_test_df(data_dict):
        """辅助方法：创建测试 DataFrame"""
        return pd.DataFrame(data_dict)

    def test_put_unmatched_in_table(self):
        mode_config = ModeConfig()
        mapping_config = MappingConfig()
        match = Match(mode_config, mapping_config, cross_frame=False)

        match_result = pd.DataFrame(columns=CompareConst.MATCH_RESULT_COLUMNS)
        npu_op_item = pd.Series(['op', 'float32', [1, 2], 'summary', 'stack_info',
                                 'input', 'op_origin', 'False', 'data_name', 'op', [1, 2]],
                                index=['op_name_x', 'dtype_x', 'shape_x', 'summary_x', 'stack_info_x',
                                       'state_x', 'api_origin_name_x', 'data_name_x', 'requires_grad_x',
                                       'compare_key', 'compare_shape']
                                )
        match_result = match.put_unmatched_in_table(match_result, npu_op_item)
        target_match_result = pd.DataFrame([['op', 'float32', [1, 2], 'summary', 'stack_info',
                                             'input', 'op_origin', 'False', 'data_name', 'op', [1, 2],
                                             'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A']],
                                           columns=CompareConst.MATCH_RESULT_COLUMNS)
        self.assertTrue(match_result.equals(target_match_result))

    def test_put_matched_in_table(self):
        mode_config = ModeConfig()
        mapping_config = MappingConfig()
        match = Match(mode_config, mapping_config, cross_frame=False)

        match_result = pd.DataFrame(columns=CompareConst.MATCH_RESULT_COLUMNS)
        npu_op_item = pd.Series(['op', 'float32', [1, 2], 'summary', 'stack_info',
                                 'input', 'op_origin', 'False', 'data_name', 'op', [1, 2]],
                                index=['op_name_x', 'dtype_x', 'shape_x', 'summary_x', 'stack_info_x',
                                       'state_x', 'api_origin_name_x', 'requires_grad_x', 'data_name_x',
                                       'compare_key', 'compare_shape']
                                )
        bench_op_item = pd.Series(['op', 'float32', [1, 2], 'summary', 'stack_info',
                                   'input', 'op_origin', 'False', 'data_name', 'op', [1, 2]],
                                  index=['op_name_y', 'dtype_y', 'shape_y', 'summary_y', 'stack_info_y',
                                         'state_y', 'api_origin_name_y', 'requires_grad_y', 'data_name_y',
                                         'compare_key', 'compare_shape']
                                  )
        match_result = match.put_matched_in_table(match_result, npu_op_item, bench_op_item)
        target_match_result = pd.DataFrame([['op', 'float32', [1, 2], 'summary', 'stack_info',
                                             'input', 'op_origin', 'False', 'data_name', 'op', [1, 2],
                                             'op', 'float32', [1, 2], 'summary', 'stack_info',
                                             'input', 'op_origin', 'False', 'data_name']],
                                           columns=CompareConst.MATCH_RESULT_COLUMNS)
        self.assertTrue(match_result.equals(target_match_result))

    def test_rename_api(self):
        mode_config = ModeConfig()
        mapping_config = MappingConfig()
        match = Match(mode_config, mapping_config, cross_frame=False)

        op_name_1 = 'Functional.linear.0.forward.input.0'
        result_1 = match.rename_api(op_name_1)
        self.assertTrue(result_1, 'Functional.linear.input.0')

        op_name_2 = 'Functional.linear.0.backward.input.0'
        result_2 = match.rename_api(op_name_2)
        self.assertTrue(result_2, 'Functional.linear.input.0')

        op_name_3 = 'Functional.linear.0.x.input.0'
        result_3 = match.rename_api(op_name_3)
        self.assertTrue(result_3, 'Functional.linear.0.x.input.0')

    def test_check_op_item(self):
        mode_config = ModeConfig()
        mapping_config = MappingConfig()
        match = Match(mode_config, mapping_config, cross_frame=False)

        npu_op_item = pd.Series(['op', 'float32', [1, 2], 'summary', 'stack_info', 'data_name', 'Functional.linear.0.forward.input.0', [1, 2]],
                                index=['op_name_x', 'dtype_x', 'shape_x', 'summary_x', 'stack_info_x', 'data_name_x',
                                       'compare_key', 'compare_shape']
                                )
        bench_op_item = pd.Series(['op', 'float32', [1, 2], 'summary', 'stack_info', 'data_name', 'Functional.linear.1.forward.input.0', [1, 2]],
                                  index=['op_name_y', 'dtype_y', 'shape_y', 'summary_y', 'stack_info_y', 'data_name_y',
                                         'compare_key', 'compare_shape']
                                  )
        result = match.check_op_item(npu_op_item, bench_op_item)
        self.assertTrue(result)

    def test_process_fuzzy_match(self):
        mode_config = ModeConfig()
        mapping_config = MappingConfig()
        match = Match(mode_config, mapping_config, cross_frame=False)

        npu_df = pd.DataFrame([
            ['Functional.conv2d.3.forward.input.0', 'float32', [1, 2], 'summary', 'stack_info',
             'input', 'Functional.conv2d.3.forward', 'True', 'Functional.conv2d.3.forward.input.0.pt',
             'Functional.conv2d.3.forward.input.0', [1, 2]],
            ['Functional.amax.1.forward.input.0', 'float32', [1, 2], 'summary', 'stack_info',
             'input', 'Functional.amax.1.forward', 'True', 'Functional.amax.0.forward.input.0.pt',
             'Functional.amax.1.forward.input.0', [1, 2]]
        ], columns=['op_name', 'dtype', 'shape', 'summary', 'stack_info',
                    'state', 'api_origin_name', 'requires_grad', 'data_name',
                    'compare_key', 'compare_shape'])
        bench_df = pd.DataFrame([
            ['Functional.conv2d.0.forward.input.0', 'float32', [1, 2], 'summary', 'stack_info',
             'input', 'Functional.conv2d.0.forward', 'True', 'Functional.conv2d.0.forward.input.0.pt',
             'Functional.conv2d.0.forward.input.0', [1, 2]],
            ['Functional.amax.0.forward.input.0', 'float32', [1, 2], 'summary', 'stack_info',
             'input', 'Functional.amax.0.forward', 'True', 'Functional.amax.0.forward.input.0.pt',
             'Functional.amax.0.forward.input.0', [1, 2]]
        ], columns=['op_name', 'dtype', 'shape', 'summary', 'stack_info',
                    'state', 'api_origin_name', 'requires_grad', 'data_name', 'compare_key', 'compare_shape'])

        match_result = match.process_fuzzy_match(npu_df, bench_df)
        expected = pd.DataFrame(
            [
                ['Functional.conv2d.3.forward.input.0', 'float32', [1, 2], 'summary', 'stack_info',
                 'input', 'Functional.conv2d.3.forward', 'True', 'Functional.conv2d.3.forward.input.0.pt',
                 'Functional.conv2d.3.forward.input.0', [1, 2],
                 'Functional.conv2d.0.forward.input.0', 'float32', [1, 2], 'summary', 'stack_info',
                 'input', 'Functional.conv2d.0.forward', 'True', 'Functional.conv2d.0.forward.input.0.pt'],
                ['Functional.amax.1.forward.input.0', 'float32', [1, 2], 'summary', 'stack_info',
                 'input', 'Functional.amax.1.forward', 'True', 'Functional.amax.0.forward.input.0.pt',
                 'Functional.amax.1.forward.input.0', [1, 2],
                 'Functional.amax.0.forward.input.0', 'float32', [1, 2], 'summary', 'stack_info',
                 'input', 'Functional.amax.0.forward', 'True', 'Functional.amax.0.forward.input.0.pt']
            ]
        , columns=CompareConst.MATCH_RESULT_COLUMNS)

        self.assertTrue(match_result.equals(expected))

    def test_match_op_both_last_element(self):
        config_dict = {
            'stack_mode': False,
            'auto_analyze': True,
            'fuzzy_match': False,
            'dump_mode': Const.SUMMARY,
        }
        mode_config = ModeConfig(**config_dict)
        mapping_config = MappingConfig()

        match = Match(mode_config, mapping_config, cross_frame=False)
        a, b = match.match_op([npu_op_item_fuzzy], [bench_op_item_fuzzy])
        self.assertEqual(a, 0)
        self.assertEqual(b, 0)

    def test_match_op_only_npu_last_element(self):
        config_dict = {
            'stack_mode': False,
            'auto_analyze': True,
            'fuzzy_match': False,
            'dump_mode': Const.SUMMARY,
        }
        mode_config = ModeConfig(**config_dict)
        mapping_config = MappingConfig()

        match = Match(mode_config, mapping_config, cross_frame=False)
        a, b = match.match_op([npu_op_item_fuzzy], [bench_op_item_fuzzy, 1])
        self.assertEqual(a, 0)
        self.assertEqual(b, 0)

    def test_match_op_only_bench_last_element(self):
        config_dict = {
            'stack_mode': False,
            'auto_analyze': True,
            'fuzzy_match': False,
            'dump_mode': Const.SUMMARY,
        }
        mode_config = ModeConfig(**config_dict)
        mapping_config = MappingConfig()

        match = Match(mode_config, mapping_config, cross_frame=False)
        a, b = match.match_op([npu_op_item_fuzzy, npu_op_item_data_fuzzy_2], [bench_op_item_fuzzy])
        self.assertEqual(a, 0)
        self.assertEqual(b, 0)

    def test_gen_dtype_condition(self):
        mode_config = ModeConfig()
        mapping_config = MappingConfig()
        match = Match(mode_config, mapping_config, cross_frame=True)

        # data mapping
        mapping_config.data_mapping = True
        match_result = pd.DataFrame([1, 2, 3])
        result = match.gen_dtype_condition(match_result)
        expected = pd.Series([True, True, True])
        self.assertTrue(result.equals(expected))

        # normal
        mapping_config.data_mapping = None
        match_result = pd.DataFrame([['Float16', 'Float32'], ['torch.float32', 'torch.bfloat16']], columns=['dtype_x', 'dtype_y'])
        result = match.gen_dtype_condition(match_result)
        expected = pd.Series([True, True])
        self.assertTrue(result.equals(expected))

    def test_process_cross_frame_dtype(self):
        mode_config = ModeConfig()
        mapping_config = MappingConfig()
        match = Match(mode_config, mapping_config, cross_frame=True)

        dtype_o = pd.Series(['Int8', 'Float16', 'torch.bool', 'Complex64', 'unknown'])
        dtype = match.process_cross_frame_dtype(dtype_o)
        self.assertTrue(dtype.equals(pd.Series(['int', 'float', 'bool', 'complex', 'unknown'])))

    def test_dual_monotonic_sort_normal_case(self):
        """
        场景 1: 正常情况，两列都能单调递增
        输入：npu_pos 和 bench_pos 都能同步递增
        预期：按正确顺序返回
        """
        df = self._create_test_df({
            'npu_pos': [1, 2, 3],
            'bench_pos': [1, 2, 3],
            'data': ['a', 'b', 'c']
        })

        result = Match.dual_monotonic_sort(df, 'npu_pos', 'bench_pos')

        # 验证顺序不变（已经有序）
        self.assertEqual(result.loc[0, 'data'], 'a')
        self.assertEqual(result.loc[1, 'data'], 'b')
        self.assertEqual(result.loc[2, 'data'], 'c')

    def test_dual_monotonic_sort_with_nan(self):
        """
        场景 2: 包含 NaN 值，NaN 应该排在最后
        输入：部分行包含 NaN
        预期：NaN 行排在最后，非 NaN 行保持单调
        """
        df = self._create_test_df({
            'npu_pos': [1, np.nan, 2],
            'bench_pos': [1, 2, np.nan],
            'data': ['a', 'b', 'c']
        })

        result = Match.dual_monotonic_sort(df, 'npu_pos', 'bench_pos')

        # 验证非 NaN 行在前，NaN 行在后
        # 第一行应该是 (1, 1)
        self.assertEqual(result.loc[0, 'npu_pos'], 1)
        self.assertEqual(result.loc[0, 'bench_pos'], 1)
        # 最后一行应该包含 NaN
        self.assertTrue(pd.isna(result.loc[2, 'npu_pos']) or pd.isna(result.loc[2, 'bench_pos']))

    def test_dual_monotonic_sort_conflicting_data(self):
        """
        场景 3: 冲突数据，无法同时满足两列单调递增
        输入：npu_pos 和 bench_pos 顺序冲突
        预期：抛出 ValueError
        """
        df = self._create_test_df({
            'npu_pos': [1, 2],
            'bench_pos': [2, 1],  # 与 npu_pos 顺序冲突
            'data': ['a', 'b']
        })

        # 验证抛出 ValueError
        with self.assertRaises(ValueError) as context:
            Match.dual_monotonic_sort(df, 'npu_pos', 'bench_pos')

        # 验证错误信息包含关键内容
        self.assertIn("Conflicting", str(context.exception))

    def test_dual_monotonic_sort_empty_dataframe(self):
        """
        场景 4: 空 DataFrame，不应报错
        """
        df = self._create_test_df({
            'npu_pos': [],
            'bench_pos': [],
            'data': []
        })

        result = Match.dual_monotonic_sort(df, 'npu_pos', 'bench_pos')

        # 验证返回空 DataFrame
        self.assertEqual(len(result), 0)

    def test_dual_monotonic_sort_single_row(self):
        """
        场景 5: 单行数据，应正常返回
        """
        df = self._create_test_df({
            'npu_pos': [1],
            'bench_pos': [1],
            'data': ['a']
        })

        result = Match.dual_monotonic_sort(df, 'npu_pos', 'bench_pos')

        # 验证单行正常返回
        self.assertEqual(len(result), 1)
        self.assertEqual(result.loc[0, 'data'], 'a')
        self.assertEqual(result.loc[0, 'npu_pos'], 1)
        self.assertEqual(result.loc[0, 'bench_pos'], 1)


class TestCreateTable(unittest.TestCase):

    def test_process_data_name(self):
        mode_config = ModeConfig()
        create_table = CreateTable(mode_config)

        data = {
            'data_name_x': ['A', 'B', 'C'],
            'data_name_y': ['X', 'Y', 'Z'],
            'dirty_valid_len_x': [21, -1, 21],
            'dirty_valid_len_y': [-1, -1, 21]
        }
        result_o = pd.DataFrame(data)
        result = create_table.process_all_mode_addition_header(result_o)
        target_data = {
            'data_name_x': [['A', 'X'], ['B', 'Y'], ['C', 'Z']],
            'data_name_y': ['X', 'Y', 'Z'],
            'dirty_valid_len_x': [[21, -1], [-1, -1], [21, 21]],
            'dirty_valid_len_y': [-1, -1, 21]
        }
        target_result = pd.DataFrame(target_data)
        self.assertTrue(result.equals(target_result))

    def test_set_summary(self):
        mode_config = ModeConfig()
        create_table = CreateTable(mode_config)

        # all nan
        result = create_table.set_summary(['nan', 'NaN', 'nAn'])
        expected = [CompareConst.NAN, CompareConst.NAN, CompareConst.NAN]
        self.assertEqual(result, expected)

        # mixed values
        result = create_table.set_summary([1, 'nan', 2.0, 'NaN'])
        expected = [1, CompareConst.NAN, 2.0, CompareConst.NAN]
        self.assertEqual(result, expected)

        # NA case
        result = create_table.set_summary(CompareConst.N_A)
        expected = [CompareConst.N_A, CompareConst.N_A, CompareConst.N_A, CompareConst.N_A]
        self.assertEqual(result, expected)

        # empty input
        result = create_table.set_summary([])
        expected = []
        self.assertEqual(result, expected)


class TestCalcStatsDiff(unittest.TestCase):

    def setUp(self):
        """
        每个 test 都会创建一个新的实例
        """
        self.mode_config = ModeConfig()  # 如果构造需要参数，这里补
        self.calc = CalcStatsDiff(self.mode_config)

    # ===============================
    # is_same_value
    # ===============================
    def test_is_same_value(self):
        a = pd.Series(["1", "2", "nan", "INF"])
        b = pd.Series(["1", "3", "NaN", "INF"])

        result = self.calc.is_same_value(a, b)
        expected = pd.Series([True, False, False, True])

        pd.testing.assert_series_equal(result, expected)

    # ===============================
    # is_number
    # ===============================
    def test_is_number(self):
        s = pd.Series(["1", "2.5", "abc", np.nan, "inf", True, False, 'false', 'True'])

        result = self.calc.is_number(s)
        expected = pd.Series([True, True, False, False, True, False, False, False, False])

        pd.testing.assert_series_equal(result, expected)

    # ===============================
    # is_nan
    # ===============================
    def test_is_nan(self):
        s = pd.Series(["nan", "NaN", np.nan, "1", "none"])

        result = self.calc.is_nan(s)
        expected = pd.Series([True, True, True, False, False])

        pd.testing.assert_series_equal(result, expected)

    # ===============================
    # is_inf
    # ===============================
    def test_is_inf(self):
        s = pd.Series(["inf", "INF", np.inf, "-inf", 1])

        result = self.calc.is_inf(s)
        expected = pd.Series([True, True, True, False, False])

        pd.testing.assert_series_equal(result, expected)

    # ===============================
    # is_neg_inf
    # ===============================
    def test_is_neg_inf(self):
        s = pd.Series(["-inf", "-INF", -np.inf, "inf", 1])

        result = self.calc.is_neg_inf(s)
        expected = pd.Series([True, True, True, False, False])

        pd.testing.assert_series_equal(result, expected)

    # ===============================
    # is_device
    # ===============================
    def test_is_device(self):
        s = pd.Series(["npu:0", "CPU", "cuda:1", "gpu", None])

        result = self.calc.is_device(s)
        expected = pd.Series([True, True, True, False, False])

        pd.testing.assert_series_equal(result, expected)

    # ===============================
    # is_na
    # ===============================
    def test_is_na(self):
        s = pd.Series([CompareConst.N_A, "N/A", "na", 1])

        result = self.calc.is_na(s)
        expected = pd.Series([True, True, False, False])

        pd.testing.assert_series_equal(result, expected)

    # ===============================
    # rule_num_num
    # ===============================
    def test_rule_num_num(self):
        npu = pd.Series([10.0, 5.0, 3.0])
        bench = pd.Series([5.0, 0.0, 3.0])

        diff, rel = self.calc.rule_num_num(npu, bench)

        expected_diff = pd.Series([5.0, 5.0, 0.0])
        expected_rel = pd.Series(["100.0%", CompareConst.INF, "0.0%"])

        pd.testing.assert_series_equal(diff, expected_diff)
        pd.testing.assert_series_equal(rel, expected_rel)

    # ===============================
    # static_diff / DEFAULT_RULE
    # ===============================
    def test_static_diff(self):
        diff, rel = self.calc.static_diff("A", "B")
        self.assertEqual(diff, "A")
        self.assertEqual(rel, "B")

    def test_default_rule(self):
        diff, rel = self.calc.DEFAULT_RULE
        self.assertEqual(diff, CompareConst.N_A)
        self.assertEqual(rel, CompareConst.N_A)

    # ===============================
    # get_number
    # ===============================
    def test_get_number(self):
        mode_config = ModeConfig()
        calc_stats_diff = CalcStatsDiff(mode_config)

        series = pd.Series([1, '2', 3.5, 'text', None])
        result = calc_stats_diff.get_number(series)
        expected = pd.Series([1, 2, 3.5, float('nan'), float('nan')])
        self.assertTrue(result.equals(expected))


class TestCalcStatsDiffBuildRules(unittest.TestCase):

    def setUp(self):
        self.calc = CalcStatsDiff(ModeConfig())

    def test_rules_initialized(self):
        self.assertIsInstance(self.calc.rules, dict)
        self.assertTrue(len(self.calc.rules) > 0)

    def test_num_num_rule(self):
        rule = self.calc.rules.get((ValType.NUM, ValType.NUM))
        self.assertTrue(callable(rule))
        self.assertEqual(rule, self.calc.rule_num_num)

    def test_nan_rules(self):
        nan_rule = self.calc.static_diff(CompareConst.NAN)

        for t in (ValType.NUM, ValType.INF, ValType.NEG_INF, ValType.NAN):
            self.assertEqual(self.calc.rules[(ValType.NAN, t)], nan_rule)
            self.assertEqual(self.calc.rules[(t, ValType.NAN)], nan_rule)

    def test_inf_rules(self):
        pos_inf = self.calc.static_diff(CompareConst.INF)
        neg_inf = self.calc.static_diff(CompareConst.NEG_INF)

        self.assertEqual(self.calc.rules[(ValType.INF, ValType.NUM)], pos_inf)
        self.assertEqual(self.calc.rules[(ValType.NEG_INF, ValType.NUM)], neg_inf)
        self.assertEqual(self.calc.rules[(ValType.NUM, ValType.INF)], neg_inf)
        self.assertEqual(self.calc.rules[(ValType.NUM, ValType.NEG_INF)], pos_inf)

    def test_device_device_rule(self):
        rule = self.calc.rules[(ValType.DEVICE, ValType.DEVICE)]
        self.assertEqual(rule, self.calc.static_diff(CompareConst.N_A))


class TestCalcStatsDiffClassify(unittest.TestCase):

    def setUp(self):
        self.calc = CalcStatsDiff(ModeConfig())

    def test_classify_all_types(self):
        s = pd.Series([
            "1.23",        # NUM
            "nan",         # NAN
            np.nan,        # NAN
            "inf",         # INF
            "-inf",        # NEG_INF
            "npu:0",       # DEVICE
            CompareConst.N_A,  # NA
            "abc",         # OTHER
            True,          # OTHER
            False,         # OTHER
            "false",       # OTHER
            "True"         # OTHER
        ])

        result = self.calc.classify(s)

        expected = pd.Series([
            ValType.NUM,
            ValType.NAN,
            ValType.NAN,
            ValType.INF,
            ValType.NEG_INF,
            ValType.DEVICE,
            ValType.NA,
            ValType.OTHER,
            ValType.OTHER,
            ValType.OTHER,
            ValType.OTHER,
            ValType.OTHER,
        ])

        pd.testing.assert_series_equal(result, expected)


class TestCalcStatsDiffCalcSummaryDiff(unittest.TestCase):

    def setUp(self):
        self.calc = CalcStatsDiff(ModeConfig())

    def test_calc_summary_diff_num_num(self):
        df = pd.DataFrame({
            "NPU mean":   [10.0, 5.0, 3.0],
            "Bench mean": [5.0, 5.0, 3.0],
        })

        self.calc.calc_summary_diff(df, "mean")

        self.assertIn("Mean diff", df.columns)
        self.assertIn("MeanRelativeErr", df.columns)

        # equal
        self.assertEqual(df.loc[1, "Mean diff"], '0')
        self.assertEqual(df.loc[2, "Mean diff"], '0')

        # unequal num × num
        self.assertEqual(df.loc[0, "Mean diff"], '5.0')
        self.assertEqual(df.loc[0, "MeanRelativeErr"], "100.0%")

    def test_calc_summary_diff_special_types(self):
        df = pd.DataFrame({
            "NPU mean":   ["nan", "inf", CompareConst.N_A],
            "Bench mean": [1.0,   1.0,   2.0],
        })

        self.calc.calc_summary_diff(df, "mean")

        self.assertEqual(df.loc[0, "Mean diff"], CompareConst.NAN)
        self.assertEqual(df.loc[1, "Mean diff"], CompareConst.INF)
        self.assertEqual(df.loc[2, "Mean diff"], CompareConst.N_A)

    def test_calc_summary_diff_device(self):
        df = pd.DataFrame({
            "NPU mean":   ["npu:0", "abc"],
            "Bench mean": ["cpu",   1.0],
        })

        self.calc.calc_summary_diff(df, "mean")

        self.assertEqual(df.loc[0, "Mean diff"], CompareConst.N_A)
        self.assertEqual(df.loc[1, "Mean diff"], CompareConst.DIFF_FLAG)


if __name__ == '__main__':
    unittest.main()
