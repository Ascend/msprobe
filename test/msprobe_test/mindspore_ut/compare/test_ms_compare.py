# coding=utf-8
import os
import shutil
import random
import unittest
from unittest.mock import patch

import torch
import numpy as np

from msprobe.mindspore.compare.ms_compare import check_cross_framework, read_real_data, ms_compare
from msprobe.core.common.const import Const
from msprobe.core.common.utils import CompareException
from ...core_ut.compare.test_acc_compare import generate_dump_json, generate_stack_json


npu_dict = {'op_name': ['Functional.conv2d.0.forward.input.0', 'Functional.conv2d.0.forward.input.1',
                        'Functional.conv2d.0.forward.input.2', 'Functional.conv2d.0.forward.output'],
           'input_struct': [('Float32', [1, 1, 28, 28]), ('Float32', [16, 1, 5, 5]),
                             ('Float32', [16])],
            'output_struct': [('Float32', [1, 16, 28, 28])],
            'summary': [[3.029174327850342, -2.926689624786377, -0.06619918346405029],
                        [0.19919930398464203, -0.19974489510059357, 0.006269412115216255],
                        [0.19734230637550354, -0.18177609145641327, 0.007903944700956345],
                        [2.1166646480560303, -2.190781354904175, -0.003579073818400502]], 'stack_info': []}

npu_dict_MintFunctional = {'op_name': ['MintFunctional.conv2d.0.forward.input.0', 'MintFunctional.conv2d.0.forward.input.1',
                        'MintFunctional.conv2d.0.forward.input.2', 'MintFunctional.conv2d.0.forward.output'],
           'input_struct': [('Float32', [1, 1, 28, 28]), ('Float32', [16, 1, 5, 5]),
                             ('Float32', [16])],
            'output_struct': [('Float32', [1, 16, 28, 28])],
            'summary': [[3.029174327850342, -2.926689624786377, -0.06619918346405029],
                        [0.19919930398464203, -0.19974489510059357, 0.006269412115216255],
                        [0.19734230637550354, -0.18177609145641327, 0.007903944700956345],
                        [2.1166646480560303, -2.190781354904175, -0.003579073818400502]], 'stack_info': []}

npu_dict_Mint = {'op_name': ['Mint.conv2d.0.forward.input.0', 'Mint.conv2d.0.forward.input.1',
                        'Mint.conv2d.0.forward.input.2', 'Mint.conv2d.0.forward.output'],
           'input_struct': [('Float32', [1, 1, 28, 28]), ('Float32', [16, 1, 5, 5]),
                             ('Float32', [16])],
            'output_struct': [('Float32', [1, 16, 28, 28])],
            'summary': [[3.029174327850342, -2.926689624786377, -0.06619918346405029],
                        [0.19919930398464203, -0.19974489510059357, 0.006269412115216255],
                        [0.19734230637550354, -0.18177609145641327, 0.007903944700956345],
                        [2.1166646480560303, -2.190781354904175, -0.003579073818400502]], 'stack_info': []}

bench_dict = {'op_name': ['Functional.conv2d.0.forward.input.0', 'Functional.conv2d.0.forward.input.1',
                          'Functional.conv2d.0.forward.input.2', 'Functional.conv2d.0.forward.output'],
             'input_struct': [('Float32', [1, 1, 28, 28]), ('Float32', [16, 1, 5, 5]),
                               ('Float32', [16])],
              'output_struct': [('Float32', [1, 16, 28, 28])],
              'summary': [[3.029174327850342, -2.926689624786377, -0.06619918346405029],
                          [0.19919930398464203, -0.19974489510059357, 0.006269412115216255],
                          [0.19734230637550354, -0.18177609145641327, 0.007903944700956345],
                          [2.1166646480560303, -2.190781354904175, -0.003579073818400502]], 'stack_info': []}

npu_op_name_list = ['Functional.conv2d.0.forward.input.0', 'Functional.conv2d.0.forward.input.1',
                          'Functional.conv2d.0.forward.input.2', 'Functional.conv2d.0.forward.output']

npu_op_name_Mint = ['Mint.conv2d.0.forward.input.0', 'Mint.conv2d.0.forward.input.1',
                          'Mint.conv2d.0.forward.input.2', 'Mint.conv2d.0.forward.output']

bench_op_name = ['Functional.conv2d.0.forward.input.0', 'Functional.conv2d.0.forward.input.1',
                          'Functional.conv2d.0.forward.input.2', 'Functional.conv2d.0.forward.output']

data_mapping = {'Functional.flash_attention_score.4.forward.input.0': 'NPU.npu_fusion_attention.4.forward.input.0',
                'Functional.flash_attention_score.4.forward.output.0': 'NPU.npu_fusion_attention.4.forward.output.0'}

npu_cell_dict = {'op_name': ['Cell.fc1.Dense.forward.0.input.0', 'Cell.fc1.Dense.forward.0.input.1',
                             'Cell.fc1.Dense.forward.0.input.2', 'Cell.fc1.Dense.forward.0.output.0'],
                 'input_struct': [('Float32', [1, 1, 28, 28]), ('Float32', [16, 1, 5, 5]),
                                  ('Float32', [16])],
                 'output_struct': [('Float32', [1, 16, 28, 28])],
                 'summary': [[3.029174327850342, -2.926689624786377, -0.06619918346405029],
                          [0.19919930398464203, -0.19974489510059357, 0.006269412115216255],
                          [0.19734230637550354, -0.18177609145641327, 0.007903944700956345],
                          [2.1166646480560303, -2.190781354904175, -0.003579073818400502]], "stack_info": []}

npu_json_data = {
    'task': 'statistics',
    'level': 'L1',
    'dump_data_dir': '',
    'data': {
        'Functional.flash_attention_score.4.forward': {
            'input_args': [
                {
                    'type': 'mindspore.Tensor',
                    'dtype': 'BFloat16',
                    'shape': [
                        4096,
                        1,
                        2048
                    ],
                    'Max': 4.1875,
                    'Min': -4.4375,
                    'Mean': -4.550282028503716e-05,
                    'Norm': 2316.379150390625,
                    'data_name': '',
                    'md5': ''
                }
            ],
            'output': [
                {
                    'type': 'mindspore.Tensor',
                    'dtype': 'BFloat16',
                    'shape': [
                        4096,
                        1,
                        2048
                    ],
                    'Max': 4.1875,
                    'Min': -4.4375,
                    'Mean': -4.550282028503716e-05,
                    'Norm': 2316.379150390625,
                    'data_name': '',
                    'md5': ''
                }
            ]
        }
    }
}

bench_json_data = {
    'task': 'statistics',
    'level': 'L1',
    'dump_data_dir': '',
    'data': {
        'NPU.npu_fusion_attention.4.forward': {
            'input_args': [
                {
                    'type': 'torch.Tensor',
                    'dtype': 'torch.bfloat16',
                    'shape': [
                        4096,
                        1,
                        2048
                    ],
                    'Max': 4.1875,
                    'Min': -4.4375,
                    'Mean': -4.553794860839844e-05,
                    'Norm': 2320.0,
                    'data_name': '',
                    'md5': ''
                }
            ],
            'output': [
                {
                    'type': 'torch.Tensor',
                    'dtype': 'torch.bfloat16',
                    'shape': [
                        4096,
                        1,
                        2048
                    ],
                    'Max': 4.1875,
                    'Min': -4.4375,
                    'Mean': -4.553794860839844e-05,
                    'Norm': 2320.0,
                    'data_name': '',
                    'md5': ''
                }
            ]
        }
    }
}


json_data_template = {
    'task': 'statistics',
    'level': 'L1',
    'dump_data_dir': '',
    'data': {}
}

base_dir1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'test_ms_compare1')


def gen_data(is_ms=True):
    type_value = 'mindspore.Tensor' if is_ms else 'torch.Tensor'
    dtype_value = 'BFloat16' if is_ms else 'torch.bfloat16'
    return {
        'type': type_value,
        'dtype': dtype_value,
        'shape': [4096, 1, 2048],
        'Max': random.uniform(0, 4),
        'Min': random.uniform(-4, 0),
        'Mean': random.random() / 10000,
        'Norm': random.random() * 1000
    }


class TestUtilsMethods(unittest.TestCase):

    def setUp(self):
        os.makedirs(base_dir1, mode=0o750, exist_ok=True)
        np.save(os.path.join(base_dir1, 'numpy_data.npy'), np.array([1, 2, 3]))
        torch.save(torch.tensor([2, 3, 4]), os.path.join(base_dir1, 'torch_data.pt'))

    def tearDown(self):
        if os.path.exists(base_dir1):
            shutil.rmtree(base_dir1)

    @patch('msprobe.mindspore.compare.utils.detect_framework_by_dump_json')
    def test_check_cross_framework_valid_pytorch(self, mock_detect_framework):
        mock_detect_framework.return_value = Const.PT_FRAMEWORK

        result = check_cross_framework("dummy_path")

        self.assertTrue(result)

    @patch('msprobe.mindspore.compare.utils.detect_framework_by_dump_json')
    def test_check_cross_framework_invalid_framework(self, mock_detect_framework):
        mock_detect_framework.return_value = Const.MS_FRAMEWORK

        result = check_cross_framework("dummy_path")

        self.assertFalse(result)

    def test_read_real_data_ms(self):
        data_path_dict = {
            "npu_dir": base_dir1,
            "bench_dir": base_dir1,
            "npu_data_name": "numpy_data.npy",
            "bench_data_name": "numpy_data.npy"
        }
        n_value, b_value = read_real_data(data_path_dict, False, '')
        self.assertTrue(np.array_equal(n_value, np.array([1, 2, 3])))
        self.assertTrue(np.array_equal(b_value, np.array([1, 2, 3])))

    def test_read_real_data_cross_frame(self):
        data_path_dict = {
            "npu_dir": base_dir1,
            "bench_dir": base_dir1,
            "npu_data_name": "numpy_data.npy",
            "bench_data_name": "torch_data.pt"
        }
        n_value, b_value = read_real_data(data_path_dict, True, '')
        self.assertTrue(np.array_equal(n_value, np.array([1, 2, 3])))
        self.assertTrue(np.array_equal(b_value, np.array([2, 3, 4])))

    def test_ms_compare(self):
        generate_dump_json(base_dir1)
        generate_stack_json(base_dir1)

        dump_path = os.path.join(base_dir1, 'dump.json')

        input_param = {
            'npu_path': dump_path,
            'bench_path': dump_path,
            'is_print_compare_log': True
        }
        output_path = base_dir1

        ms_compare(input_param, output_path)
        output_files = os.listdir(output_path)
        self.assertTrue(any(f.endswith(".csv") for f in output_files))

        input_param2 = {
            'npu_path': '',
            'bench_path': dump_path,
            'is_print_compare_log': True
        }
        with self.assertRaises(CompareException) as context:
            ms_compare(input_param2, output_path)
        self.assertEqual(context.exception.code, 1)
