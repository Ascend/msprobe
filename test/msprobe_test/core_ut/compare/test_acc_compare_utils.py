# coding=utf-8
import argparse
import json
import os
import shutil
import unittest
from unittest.mock import patch, MagicMock
import zlib
import tempfile

import numpy as np
import pandas as pd

from msprobe.core.common.cli_help import MindStudioArgumentParser
from msprobe.core.common.const import CompareConst, Const
from msprobe.core.common.utils import CompareException
from msprobe.core.compare.utils import ApiItemInfo, _compare_parser, check_and_return_dir_contents, extract_json, \
    count_struct, get_accuracy, get_rela_diff_summary_mode, merge_tensor, op_item_parse, read_op, result_item_init, \
    stack_column_process, table_value_is_valid, reorder_op_name_list, gen_op_item, ApiBatch, get_paired_dirs, \
    reorder_index, gen_api_batches, check_input_param_path, check_input_param_path_and_framework, compare_entry, \
    multi_ranks_compare, mp_logger_init, make_result_table, get_sorted_ranks

# test_read_op_1
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
    {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3], 'md5': '00000000',
     'Max': 0.33033010363578796, 'Min': -0.331031858921051, 'Mean': -0.030964046716690063, 'data_name': '-1',
     'Norm': 2.2533628940582275, 'requires_grad': 'True', 'full_op_name': 'Tensor.add_0.0.forward.input.0',
     'state': 'input'},
    {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3], 'md5': '00000000',
     'Max': 0.003992878366261721, 'Min': -0.008102823048830032, 'Mean': -0.0002002553956117481, 'data_name': '-1',
     'Norm': 0.02844562754034996, 'requires_grad': 'False', 'full_op_name': 'Tensor.add_0.0.forward.input.1',
     'state': 'input'},
    {'full_op_name': 'Tensor.add_0.0.forward.input.alpha', 'dtype': "<class 'float'>", 'shape': '[]', 'md5': '0dae4479',
     'Max': -0.1, 'Min': -0.1, 'Mean': -0.1, 'Norm': -0.1, 'requires_grad': None, 'data_name': '-1', 'type': 'float',
     'value': -0.1, 'state': 'input'},
    {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3], 'md5': '00000000',
     'Max': 0.33033010363578796, 'Min': -0.331031858921051, 'Mean': -0.030964046716690063, 'data_name': '-1',
     'Norm': 2.2533628940582275, 'requires_grad': 'True', 'full_op_name': 'Tensor.add_0.0.forward.output.0',
     'state': 'output'}]

# test_read_op_1
op_data_b = {
    'input': [{'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3],
               'Max': 0.33033010363578796, 'Min': -0.331031858921051, 'Mean': -0.030964046716690063,
               'Norm': 2.2533628940582275, 'requires_grad': True},
              {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3],
               'Max': 0.003992878366261721, 'Min': -0.008102823048830032, 'Mean': -0.0002002553956117481,
               'Norm': 0.02844562754034996, 'requires_grad': False}],
    'output': [{'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3],
                'Max': 0.33033010363578796, 'Min': -0.331031858921051, 'Mean': -0.030964046716690063,
                'Norm': 2.2533628940582275, 'requires_grad': True}]}
op_name_b = "Tensor.add_0.0.backward"
op_result_b = [
    {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3], 'data_name': '-1', 'md5': '00000000',
     'Max': 0.33033010363578796, 'Min': -0.331031858921051, 'Mean': -0.030964046716690063,
     'Norm': 2.2533628940582275, 'requires_grad': 'True', 'full_op_name': 'Tensor.add_0.0.backward.input.0',
     'state': 'input'},
    {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3], 'data_name': '-1', 'md5': '00000000',
     'Max': 0.003992878366261721, 'Min': -0.008102823048830032, 'Mean': -0.0002002553956117481,
     'Norm': 0.02844562754034996, 'requires_grad': 'False', 'full_op_name': 'Tensor.add_0.0.backward.input.1',
     'state': 'input'},
    {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3], 'data_name': '-1', 'md5': '00000000',
     'Max': 0.33033010363578796, 'Min': -0.331031858921051, 'Mean': -0.030964046716690063,
     'Norm': 2.2533628940582275, 'requires_grad': 'True', 'full_op_name': 'Tensor.add_0.0.backward.output.0',
     'state': 'output'}]

# test_op_item_parse
parse_item = [
    {'Max': 4097.0, 'Mean': 820.2, 'Min': 0.0, 'Norm': 4097.0, 'dtype': 'torch.int64', 'requires_grad': False,
     'shape': [5], 'type': 'torch.Tensor'},
    {'type': 'int', 'value': 0},
    {'type': 'slice', 'value': [None, None, None]}
]
parse_op_name = 'Distributed.broadcast.0.forward.input'
parse_index = None
parse_item_list = None
parse_top_bool = True
o_result_parse = [
    {'Max': 4097.0, 'Mean': 820.2, 'Min': 0.0, 'Norm': 4097.0, 'dtype': 'torch.int64', 'requires_grad': 'False',
     'shape': [5], 'type': 'torch.Tensor', 'full_op_name': 'Distributed.broadcast.0.forward.input.0',
     'data_name': '-1', 'md5': '00000000', 'state': 'input'},
    {'full_op_name': 'Distributed.broadcast.0.forward.input.1', 'dtype': "<class 'int'>", 'shape': '[]',
     'md5': 'f4dbdf21', 'Max': 0, 'Min': 0, 'Mean': 0, 'Norm': 0, 'data_name': '-1', 'type': 'int', 'value': 0,
     'state': 'input', 'requires_grad': None},
    {'Max': None, 'Mean': None, 'Min': None, 'Norm': None, 'data_name': '-1', 'dtype': 'slice', 'type': 'slice',
     'full_op_name': 'Distributed.broadcast.0.forward.input.2', 'md5': '5fbbe87f', 'shape': '(3,)',
     'value': [None, None, None], 'state': 'input', 'requires_grad': None}
]

# test_resolve_api_special_parameters
data_dict = {
    "last_hidden_state":
        {"type": "torch.Tensor", "dtype": "torch.bfloat16"},
    "loss":
        {"type": "torch.Tensor", "dtype": "torch.float32"}
}
full_op_name = "Tensor.add_0.0.forward.input.0"
o_result_api_special = [
    {"type": "torch.Tensor", "dtype": "torch.bfloat16",
     "full_op_name": "Tensor.add_0.0.forward.input.last_hidden_state.0"},
    {"type": "torch.Tensor", "dtype": "torch.float32", "full_op_name": "Tensor.add_0.0.forward.input.loss.0"}
]

# test_get_accuracy
npu_dict = {'op_name': ['Functional.conv2d.0.forward.input.0', 'Functional.conv2d.0.forward.input.1',
                        'Functional.conv2d.0.forward.input.2', 'Functional.conv2d.0.forward.output.0',
                        'Functional.conv2d.0.forward.parameters.weight', 'Functional.conv2d.0.forward.parameters.bias',
                        'Functional.conv2d.0.parameters_grad.weight', 'Functional.conv2d.0.parameters_grad.bias'],
            'input_struct': [('torch.float32', [1, 1, 28, 28]), ('torch.float32', [16, 1, 5, 5]),
                             ('torch.float32', [16])],
            'output_struct': [('torch.float32', [1, 16, 28, 28])],
            'params_struct': [('torch.float32', [1, 16, 28, 28]), ('torch.float32', [1, 16, 28, 28])],
            'params_grad_struct': [('torch.float32', [1, 16, 28, 28]), ('torch.float32', [1, 16, 28, 28])],
            'summary': [[3.029174327850342, -2.926689624786377, -0.06619918346405029, 1.0],
                        [0.19919930398464203, -0.19974489510059357, 0.006269412115216255, 1.0],
                        [0.19734230637550354, -0.18177609145641327, 0.007903944700956345, 1.0],
                        [2.1166646480560303, -2.190781354904175, -0.003579073818400502, 1.0],
                        [1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0]],
            'stack_info': [],
            'requires_grad': [True, False, True, True, True, True, True, True]}
bench_dict = {'op_name': ['Functional.conv2d.0.forward.input.0', 'Functional.conv2d.0.forward.input.1',
                          'Functional.conv2d.0.forward.input.2', 'Functional.conv2d.0.forward.output.0',
                          'Functional.conv2d.0.forward.parameters.weight', 'Functional.conv2d.0.forward.parameters.bias',
                          'Functional.conv2d.0.parameters_grad.weight', 'Functional.conv2d.0.parameters_grad.bias'],
              'input_struct': [('torch.float32', [1, 1, 28, 28]), ('torch.float32', [16, 1, 5, 5]),
                               ('torch.float32', [16])],
              'output_struct': [('torch.float32', [1, 16, 28, 28])],
              'params_struct': [('torch.float32', [1, 16, 28, 28]), ('torch.float32', [1, 16, 28, 28])],
              'params_grad_struct': [('torch.float32', [1, 16, 28, 28]), ('torch.float32', [1, 16, 28, 28])],
              'summary': [[3.029174327850342, -2.926689624786377, -0.06619918346405029, 1.0],
                          [0.19919930398464203, -0.19974489510059357, 0.006269412115216255, 1.0],
                          [0.19734230637550354, -0.18177609145641327, 0.007903944700956345, 1.0],
                          [2.1166646480560303, -2.190781354904175, -0.003579073818400502, 1.0],
                          [1.0, 1.0, 1.0, 1.0],
                          [1.0, 1.0, 1.0, 1.0],
                          [1.0, 1.0, 1.0, 1.0],
                          [1.0, 1.0, 1.0, 1.0]],
              'stack_info': [],
              'requires_grad': [True, False, True, True, True, True, True, True]}

o_result = [
    ['Functional.conv2d.0.forward.input.0', 'Functional.conv2d.0.forward.input.0', 'torch.float32', 'torch.float32',
     [1, 1, 28, 28], [1, 1, 28, 28], True, True, 0.0, 0.0, 0.0, 0.0, '0.0%', '0.0%', '0.0%', '0.0%',
     3.029174327850342, -2.926689624786377, -0.06619918346405029, 1.0,
     3.029174327850342, -2.926689624786377, -0.06619918346405029, 1.0, True, '', '', 'None'],
    ['Functional.conv2d.0.forward.input.1', 'Functional.conv2d.0.forward.input.1', 'torch.float32', 'torch.float32',
     [16, 1, 5, 5], [16, 1, 5, 5], False, False, 0.0, 0.0, 0.0, 0.0, '0.0%', '0.0%', '0.0%', '0.0%',
     0.19919930398464203, -0.19974489510059357, 0.006269412115216255, 1.0,
     0.19919930398464203, -0.19974489510059357, 0.006269412115216255, 1.0, True, '', '', 'None'],
    ['Functional.conv2d.0.forward.input.2', 'Functional.conv2d.0.forward.input.2', 'torch.float32', 'torch.float32',
     [16], [16], True, True, 0.0, 0.0, 0.0, 0.0, '0.0%', '0.0%', '0.0%', '0.0%',
     0.19734230637550354, -0.18177609145641327, 0.007903944700956345, 1.0,
     0.19734230637550354, -0.18177609145641327, 0.007903944700956345, 1.0, True, '', '', 'None'],
    ['Functional.conv2d.0.forward.output.0', 'Functional.conv2d.0.forward.output.0', 'torch.float32', 'torch.float32',
     [1, 16, 28, 28], [1, 16, 28, 28], True, True, 0.0, 0.0, 0.0, 0.0, '0.0%', '0.0%', '0.0%', '0.0%',
     2.1166646480560303, -2.190781354904175, -0.003579073818400502, 1.0,
     2.1166646480560303, -2.190781354904175, -0.003579073818400502, 1.0, True, '', '', 'None'],
    ['Functional.conv2d.0.forward.parameters.weight', 'Functional.conv2d.0.forward.parameters.weight', 'torch.float32',
     'torch.float32',
     [1, 16, 28, 28], [1, 16, 28, 28], True, True, 0.0, 0.0, 0.0, 0.0, '0.0%', '0.0%', '0.0%', '0.0%',
     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, True, '', '', 'None'],
    ['Functional.conv2d.0.forward.parameters.bias', 'Functional.conv2d.0.forward.parameters.bias', 'torch.float32',
     'torch.float32',
     [1, 16, 28, 28], [1, 16, 28, 28], True, True, 0.0, 0.0, 0.0, 0.0, '0.0%', '0.0%', '0.0%', '0.0%',
     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, True, '', '', 'None'],
    ['Functional.conv2d.0.parameters_grad.weight', 'Functional.conv2d.0.parameters_grad.weight', 'torch.float32',
     'torch.float32',
     [1, 16, 28, 28], [1, 16, 28, 28], True, True, 0.0, 0.0, 0.0, 0.0, '0.0%', '0.0%', '0.0%', '0.0%',
     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, True, '', '', 'None'],
    ['Functional.conv2d.0.parameters_grad.bias', 'Functional.conv2d.0.parameters_grad.bias', 'torch.float32',
     'torch.float32',
     [1, 16, 28, 28], [1, 16, 28, 28], True, True, 0.0, 0.0, 0.0, 0.0, '0.0%', '0.0%', '0.0%', '0.0%',
     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, True, '', '', 'None'],
]

# test_get_un_match_accuracy
o_result_unmatch_1 = [
    ['Functional.conv2d.0.forward.input.0', 'N/A', 'torch.float32', 'N/A', [1, 1, 28, 28], 'N/A', 'N/A', 'N/A', 'N/A',
     'None'],
    ['Functional.conv2d.0.forward.input.1', 'N/A', 'torch.float32', 'N/A', [16, 1, 5, 5], 'N/A', 'N/A', 'N/A', 'N/A',
     'None'],
    ['Functional.conv2d.0.forward.input.2', 'N/A', 'torch.float32', 'N/A', [16], 'N/A', 'N/A', 'N/A', 'N/A', 'None'],
    ['Functional.conv2d.0.forward.parameters.weight', 'N/A', 'torch.float32', 'N/A', [1, 16, 28, 28], 'N/A', 'N/A',
     'N/A', 'N/A',
     'None'],
    ['Functional.conv2d.0.forward.parameters.bias', 'N/A', 'torch.float32', 'N/A', [1, 16, 28, 28], 'N/A', 'N/A', 'N/A',
     'N/A',
     'None'],
    ['Functional.conv2d.0.forward.output.0', 'N/A', 'torch.float32', 'N/A', [1, 16, 28, 28], 'N/A', 'N/A', 'N/A', 'N/A',
     'None'],
    ['Functional.conv2d.0.parameters_grad.weight', 'N/A', 'torch.float32', 'N/A', [1, 16, 28, 28], 'N/A', 'N/A', 'N/A',
     'N/A',
     'None'],
    ['Functional.conv2d.0.parameters_grad.bias', 'N/A', 'torch.float32', 'N/A', [1, 16, 28, 28], 'N/A', 'N/A', 'N/A',
     'N/A',
     'None']
]
o_result_unmatch_2 = [
    ['Functional.conv2d.0.forward.input.0', 'N/A', 'torch.float32', 'N/A', [1, 1, 28, 28], 'N/A', 'N/A', 'N/A', 'N/A',
     'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 3.029174327850342, -2.926689624786377, -0.06619918346405029, 1.0, 'N/A', 'N/A',
     'N/A', 'N/A', 'N/A', 'No bench data matched.', 'None'],
    ['Functional.conv2d.0.forward.input.1', 'N/A', 'torch.float32', 'N/A', [16, 1, 5, 5], 'N/A', 'N/A', 'N/A', 'N/A',
     'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 0.19919930398464203, -0.19974489510059357, 0.006269412115216255, 1.0, 'N/A',
     'N/A',
     'N/A', 'N/A', 'N/A', 'No bench data matched.', 'None'],
    ['Functional.conv2d.0.forward.input.2', 'N/A', 'torch.float32', 'N/A', [16], 'N/A', 'N/A', 'N/A', 'N/A', 'N/A',
     'N/A', 'N/A', 'N/A', 'N/A', 0.19734230637550354, -0.18177609145641327, 0.007903944700956345, 1.0, 'N/A', 'N/A',
     'N/A',
     'N/A', 'N/A', 'No bench data matched.', 'None'],
    ['Functional.conv2d.0.forward.parameters.weight', 'N/A', 'torch.float32', 'N/A', [1, 16, 28, 28], 'N/A', 'N/A',
     'N/A', 'N/A',
     'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 1.0, 1.0, 1.0, 1.0, 'N/A',
     'N/A', 'N/A', 'N/A', 'N/A', 'No bench data matched.', 'None'],
    ['Functional.conv2d.0.forward.parameters.bias', 'N/A', 'torch.float32', 'N/A', [1, 16, 28, 28], 'N/A', 'N/A', 'N/A',
     'N/A',
     'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 1.0, 1.0, 1.0, 1.0, 'N/A',
     'N/A', 'N/A', 'N/A', 'N/A', 'No bench data matched.', 'None'],
    ['Functional.conv2d.0.forward.output.0', 'N/A', 'torch.float32', 'N/A', [1, 16, 28, 28], 'N/A', 'N/A', 'N/A', 'N/A',
     'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 2.1166646480560303, -2.190781354904175, -0.003579073818400502, 1.0, 'N/A',
     'N/A',
     'N/A', 'N/A', 'N/A', 'No bench data matched.', 'None'],
    ['Functional.conv2d.0.parameters_grad.weight', 'N/A', 'torch.float32', 'N/A', [1, 16, 28, 28], 'N/A', 'N/A', 'N/A',
     'N/A',
     'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 1.0, 1.0, 1.0, 1.0, 'N/A',
     'N/A', 'N/A', 'N/A', 'N/A', 'No bench data matched.', 'None'],
    ['Functional.conv2d.0.parameters_grad.bias', 'N/A', 'torch.float32', 'N/A', [1, 16, 28, 28], 'N/A', 'N/A', 'N/A',
     'N/A',
     'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 1.0, 1.0, 1.0, 1.0, 'N/A',
     'N/A', 'N/A', 'N/A', 'N/A', 'No bench data matched.', 'None']
]
o_result_unmatch_3 = [
    ['Functional.conv2d.0.forward.input.0', 'N/A', 'torch.float32', 'N/A', [1, 1, 28, 28], 'N/A',
     'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A',
     3.029174327850342, -2.926689624786377, -0.06619918346405029, 1.0, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A',
     'No bench data matched.', 'None', ['-1', '-1']],
    ['Functional.conv2d.0.forward.input.1', 'N/A', 'torch.float32', 'N/A', [16, 1, 5, 5], 'N/A',
     'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A',
     0.19919930398464203, -0.19974489510059357, 0.006269412115216255, 1.0, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A',
     'No bench data matched.', 'None', ['-1', '-1']],
    ['Functional.conv2d.0.forward.input.2', 'N/A', 'torch.float32', 'N/A', [16], 'N/A',
     'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A',
     0.19734230637550354, -0.18177609145641327, 0.007903944700956345, 1.0, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A',
     'No bench data matched.', 'None', ['-1', '-1']],
    ['Functional.conv2d.0.forward.parameters.weight', 'N/A', 'torch.float32', 'N/A', [1, 16, 28, 28], 'N/A',
     'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A',
     1.0, 1.0, 1.0, 1.0, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'No bench data matched.', 'None', ['-1', '-1']],
    ['Functional.conv2d.0.forward.parameters.bias', 'N/A', 'torch.float32', 'N/A', [1, 16, 28, 28], 'N/A',
     'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A',
     1.0, 1.0, 1.0, 1.0, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'No bench data matched.', 'None', ['-1', '-1']],
    ['Functional.conv2d.0.forward.output.0', 'N/A', 'torch.float32', 'N/A', [1, 16, 28, 28], 'N/A',
     'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A',
     2.1166646480560303, -2.190781354904175, -0.003579073818400502, 1.0, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A',
     'No bench data matched.', 'None', ['-1', '-1']],
    ['Functional.conv2d.0.parameters_grad.weight', 'N/A', 'torch.float32', 'N/A', [1, 16, 28, 28], 'N/A',
     'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A',
     1.0, 1.0, 1.0, 1.0, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'No bench data matched.', 'None', ['-1', '-1']],
    ['Functional.conv2d.0.parameters_grad.bias', 'N/A', 'torch.float32', 'N/A', [1, 16, 28, 28], 'N/A',
     'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A',
     1.0, 1.0, 1.0, 1.0, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'No bench data matched.', 'None', ['-1', '-1']]
]

# test_merge_tensor
tensor_list = [
    {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3], 'Max': 0.33033010363578796,
     'Min': -0.331031858921051, 'Mean': -0.030964046716690063, 'Norm': 2.2533628940582275, 'requires_grad': True,
     'full_op_name': 'Tensor.add_.0.forward.input.0', 'state': 'input'},
    {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3],
     'Max': 0.003992878366261721, 'Min': -0.008102823048830032, 'Mean': -0.0002002553956117481,
     'Norm': 0.02844562754034996, 'requires_grad': False, 'full_op_name': 'Tensor.add_.0.forward.input.1',
     'state': 'input'},
    {'full_op_name': 'Tensor.add_.0.forward.input.alpha.0', 'dtype': "<class 'float'>", "shape": '[]', 'md5': None,
     'Max': -0.1, 'Min': -0.1, 'Mean': -0.1, 'Norm': -0.1, 'data_name': '-1', 'state': 'input'},
    {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3],
     'Max': 0.33033010363578796, 'Min': -0.331031858921051, 'Mean': -0.030964046716690063,
     'Norm': 2.2533628940582275, 'requires_grad': True, 'full_op_name': 'Tensor.add_.0.forward.output.0',
     'state': 'output'}
]
result_op_dict = {'op_name': ['Tensor.add_.0.forward.input.0', 'Tensor.add_.0.forward.input.1',
                              'Tensor.add_.0.forward.input.alpha.0', 'Tensor.add_.0.forward.output.0'],
                  'input_struct': [('torch.float32', [16, 1, 3, 3]), ('torch.float32', [16, 1, 3, 3]),
                                   ("<class 'float'>", '[]')],
                  'output_struct': [('torch.float32', [16, 1, 3, 3])],
                  'params_struct': [],
                  'params_grad_struct': [],
                  'debug_struct': [],
                  'summary': [[0.33033010363578796, -0.331031858921051, -0.030964046716690063, 2.2533628940582275],
                              [0.003992878366261721, -0.008102823048830032, -0.0002002553956117481,
                               0.02844562754034996],
                              [-0.1, -0.1, -0.1, -0.1],
                              [0.33033010363578796, -0.331031858921051, -0.030964046716690063, 2.2533628940582275]],
                  'stack_info': [],
                  'state': ['input', 'input', 'input', 'output'],
                  'requires_grad': [True, False, None, True]}

tensor_list_md5 = [
    {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3],
     'Max': 0.003992878366261721, 'Min': -0.008102823048830032, 'Mean': -0.0002002553956117481,
     'Norm': 0.02844562754034996, 'requires_grad': False, 'full_op_name': 'Tensor.add_.0.forward.input.0', 'md5': 1,
     'state': 'input'},
    {'full_op_name': 'Tensor.add_.0.forward.kwargs.alpha.0', 'dtype': "<class 'float'>", "shape": '[]', 'md5': None,
     'Max': -0.1, 'Min': -0.1, 'Mean': -0.1, 'Norm': -0.1, 'data_name': '-1', 'state': 'input'},
    {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3],
     'Max': 0.33033010363578796, 'Min': -0.331031858921051, 'Mean': -0.030964046716690063,
     'Norm': 2.2533628940582275, 'requires_grad': True, 'full_op_name': 'Tensor.add_.0.forward.output.0', 'md5': 2,
     'state': 'output'}
]
result_op_dict_md5 = {'op_name': ['Tensor.add_.0.forward.input.0', 'Tensor.add_.0.forward.kwargs.alpha.0',
                                  'Tensor.add_.0.forward.output.0'],
                      'input_struct': [('torch.float32', [16, 1, 3, 3], 1), ("<class 'float'>", '[]', None)],
                      'output_struct': [('torch.float32', [16, 1, 3, 3], 2)],
                      'params_struct': [],
                      'params_grad_struct': [],
                      'debug_struct': [],
                      'summary': [
                          [0.003992878366261721, -0.008102823048830032, -0.0002002553956117481, 0.02844562754034996],
                          [-0.1, -0.1, -0.1, -0.1],
                          [0.33033010363578796, -0.331031858921051, -0.030964046716690063, 2.2533628940582275]],
                      'stack_info': [],
                      'state': ['input', 'input', 'output'],
                      'requires_grad': [False, None, True]
                      }

base_dir1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'test_acc_compare_utils1')
base_dir2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'test_acc_compare_utils2')


def create_json_files(base_dir):
    file_names = ['dump.json', 'stack.json', 'construct.json', 'debug.json']

    for file_name in file_names:
        file_path = os.path.join(base_dir, file_name)
        with open(file_path, 'w') as f:
            json.dump({}, f)


def create_rank_dirs(base_dir):
    folder_names = ['rank0', 'rank1']

    for folder_name in folder_names:
        folder_path = os.path.join(base_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)


class TestUtilsMethods(unittest.TestCase):

    def setUp(self):
        self.parser = argparse.ArgumentParser()
        _compare_parser(self.parser)

        os.makedirs(base_dir1, mode=0o750, exist_ok=True)
        os.makedirs(base_dir2, mode=0o750, exist_ok=True)

    def tearDown(self):
        if os.path.exists(base_dir1):
            shutil.rmtree(base_dir1)
        if os.path.exists(base_dir2):
            shutil.rmtree(base_dir2)

    def test_compare_help_lists_modes_aliases_rank_scenarios_and_output(self):
        parser = MindStudioArgumentParser(prog="msprobe compare")
        _compare_parser(parser)

        help_text = parser.format_help()
        normalized_help = " ".join(help_text.split())

        self.assertIn("--mode {auto,offline_data,torchair,offline_model,atb}", help_text)
        for alias in (
            "-tensor_log", "-ofs", "-cfr", "-qfr", "-fr", "-da", "-lm",
            "-dm", "-am", "-cm", "-fm", "-tp", "-gp",
        ):
            self.assertIn(alias, help_text)
        self.assertIn('Rank ID(s) to compare when comparing MindSpore kernels in "--mode auto".', normalized_help)
        self.assertIn('Input rank ID [0, 255] for "--mode offline_model".', normalized_help)
        self.assertNotIn("torchair dumps", help_text)
        self.assertIn("<DIR>/compare_result_{timestamp}.csv (or .xlsx)", help_text)
        self.assertNotIn("<output_path>/", help_text)

    def test_extract_json_1(self):
        create_json_files(base_dir1)
        result = extract_json(base_dir1, Const.DUMP_JSON_FILE)
        self.assertEqual(result, os.path.join(base_dir1, 'dump.json'))

        result = extract_json(base_dir1, Const.STACK_JSON_FILE)
        self.assertEqual(result, os.path.join(base_dir1, 'stack.json'))

        result = extract_json(base_dir1, Const.DEBUG_JSON_FILE)
        self.assertEqual(result, os.path.join(base_dir1, 'debug.json'))

    def test_check_and_return_dir_contents(self):
        create_rank_dirs(base_dir2)
        result = check_and_return_dir_contents(base_dir2, 'rank')
        self.assertEqual(set(result), set(['rank0', 'rank1']))

    def test_read_op(self):
        result = read_op(op_data, op_name)
        self.assertEqual(result, op_result)

    def test_read_op_back(self):
        result = read_op(op_data_b, op_name_b)
        self.assertEqual(result, op_result_b)

    def test_op_item_parse(self):
        result = op_item_parse(parse_item, parse_op_name, 'input')
        self.assertEqual(result, o_result_parse)

    def test_op_item_parse_max_depth(self):
        with self.assertRaises(CompareException) as context:
            op_item_parse(parse_item, parse_op_name, 'input', depth=401)
        self.assertEqual(context.exception.code, CompareException.RECURSION_LIMIT_ERROR)

    def test_get_rela_diff_summary_mode_float_or_int(self):
        result_item = [0] * 16
        err_msg = ''
        npu_summary_data = [1, 1, 1, 1]
        bench_summary_data = [2, 2, 2, 2]
        result_item, accuracy_check, err_msg = get_rela_diff_summary_mode(result_item, npu_summary_data,
                                                                          bench_summary_data, err_msg)
        self.assertEqual(result_item, [0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, '50.0%', '50.0%', '50.0%', '50.0%'])
        self.assertEqual(accuracy_check, '')
        self.assertEqual(err_msg, '')

    def test_get_rela_diff_summary_mode_bool(self):
        result_item = [0] * 16
        err_msg = ''
        npu_summary_data = [True, True, True, True]
        bench_summary_data = [True, True, True, True]
        result_item, accuracy_check, err_msg = get_rela_diff_summary_mode(result_item, npu_summary_data,
                                                                          bench_summary_data, err_msg)
        self.assertEqual(result_item, [0, 0, 0, 0, 0, 0, 0, 0, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'])
        self.assertEqual(accuracy_check, '')
        self.assertEqual(err_msg, '')

    def test_get_rela_diff_summary_mode_nan(self):
        result_item = [0] * 16
        err_msg = ''
        npu_summary_data = [float('nan')]
        bench_summary_data = [float('nan')]
        result_item, accuracy_check, err_msg = get_rela_diff_summary_mode(result_item, npu_summary_data,
                                                                          bench_summary_data, err_msg)
        self.assertEqual(result_item, [0, 0, 0, 0, 0, 0, 0, 0, 'Nan', 0, 0, 0, 'Nan', 0, 0, 0])
        self.assertEqual(accuracy_check, '')
        self.assertEqual(err_msg, '')

    def test_count_struct_normal(self):
        op_dict = {
            CompareConst.OP_NAME: ['op1', 'op2', 'op3', 'op4', 'op5', 'op6', 'op7', 'op8'],
            CompareConst.INPUT_STRUCT: [("torch.float32", [1]), ("torch.float32", [1])],
            CompareConst.OUTPUT_STRUCT: [("torch.float32", [1]), ("torch.float32", [1])],
            CompareConst.PARAMS_STRUCT: [("torch.float32", [1]), ("torch.float32", [1])],
            CompareConst.PARAMS_GRAD_STRUCT: [("torch.float32", [1]), ("torch.float32", [1])],
        }

        result = count_struct(op_dict)

        self.assertEqual(result, (8, 2, 2, 2, 2))

    @patch('msprobe.core.compare.utils.logger')
    def test_mismatch_case(self, mock_logger):
        op_dict = {
            CompareConst.OP_NAME: ['op1', 'op2', 'op3', 'op4', 'op5', 'op6', 'op7', 'op8'],
            CompareConst.INPUT_STRUCT: [("torch.float32", [1])],
            CompareConst.OUTPUT_STRUCT: [("torch.float32", [1]), ("torch.float32", [1])],
            CompareConst.PARAMS_STRUCT: [("torch.float32", [1]), ("torch.float32", [1])],
            CompareConst.PARAMS_GRAD_STRUCT: [("torch.float32", [1]), ("torch.float32", [1])],
        }

        with self.assertRaises(CompareException) as context:
            count_struct(op_dict)
        self.assertEqual(context.exception.code, CompareException.NAMES_STRUCTS_MATCH_ERROR)

    def test_get_accuracy(self):
        result = []
        get_accuracy(result, npu_dict, bench_dict, dump_mode=Const.SUMMARY)
        self.assertEqual(result, o_result)

    def test_merge_tensor_summary(self):
        op_dict = merge_tensor(tensor_list, dump_mode=Const.SUMMARY)
        self.assertEqual(op_dict, result_op_dict)

    def test_merge_tensor_md5(self):
        op_dict = merge_tensor(tensor_list_md5, dump_mode=Const.MD5)
        self.assertEqual(op_dict, result_op_dict_md5)

    def test_stack_column_process_stack_info(self):
        result_item = []
        has_stack = True
        index = 0
        key = CompareConst.INPUT_STRUCT
        npu_stack_info = ['abc']
        result_item = stack_column_process(result_item, has_stack, index, key, npu_stack_info)
        self.assertEqual(result_item, ['abc'])

    def test_stack_column_process_None(self):
        result_item = []
        has_stack = True
        index = 1
        key = CompareConst.INPUT_STRUCT
        npu_stack_info = ['abc']
        result_item = stack_column_process(result_item, has_stack, index, key, npu_stack_info)
        self.assertEqual(result_item, ['None'])

    def test_result_item_init_all_and_summary(self):
        n_name = 'Tensor.add.0.forward.input.0'
        n_struct = ('torch.float32', [96])
        npu_stack_info = ['abc']
        b_name = 'Tensor.add.0.forward.input.0'
        b_struct = ('torch.float32', [96])
        bench_stack_info = ['abc']
        requires_grad_pair = [True, True]
        n_info = ApiItemInfo(n_name, n_struct, npu_stack_info)
        b_info = ApiItemInfo(b_name, b_struct, bench_stack_info)

        dump_mode = Const.ALL
        result_item = result_item_init(n_info, b_info, requires_grad_pair, dump_mode)
        self.assertEqual(result_item, ['Tensor.add.0.forward.input.0', 'Tensor.add.0.forward.input.0',
                                       'torch.float32', 'torch.float32', [96], [96], True, True,
                                       ' ', ' ', ' ', ' ', ' ', ' '])

        dump_mode = Const.SUMMARY
        result_item = result_item_init(n_info, b_info, requires_grad_pair, dump_mode)
        self.assertEqual(result_item, ['Tensor.add.0.forward.input.0', 'Tensor.add.0.forward.input.0',
                                       'torch.float32', 'torch.float32', [96], [96], True, True,
                                       ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '])

    def test_result_item_init_md5(self):
        n_name = 'Tensor.add.0.forward.input.0'
        n_struct = ('torch.float32', [96], 'e87000dc')
        npu_stack_info = ['abc']
        b_name = 'Tensor.add.0.forward.input.0'
        b_struct = ('torch.float32', [96], 'e87000dc')
        bench_stack_info = ['abc']
        requires_grad_pair = [True, True]
        n_info = ApiItemInfo(n_name, n_struct, npu_stack_info)
        b_info = ApiItemInfo(b_name, b_struct, bench_stack_info)

        dump_mode = Const.MD5
        result_item = result_item_init(n_info, b_info, requires_grad_pair, dump_mode)
        self.assertEqual(result_item, ['Tensor.add.0.forward.input.0', 'Tensor.add.0.forward.input.0',
                                       'torch.float32', 'torch.float32', [96], [96], True, True,
                                       'e87000dc', 'e87000dc', True, 'pass'])

    def test_result_item_init_md5_index_error(self):
        n_name = 'Tensor.add.0.forward.input.0'
        n_struct = ('torch.float32', [96])
        npu_stack_info = ['abc']
        b_name = 'Tensor.add.0.forward.input.0'
        b_struct = ('torch.float32', [96])
        bench_stack_info = ['abc']
        requires_grad_pair = [True, True]
        n_info = ApiItemInfo(n_name, n_struct, npu_stack_info)
        b_info = ApiItemInfo(b_name, b_struct, bench_stack_info)

        dump_mode = Const.MD5
        with self.assertRaises(CompareException) as context:
            result_item = result_item_init(n_info, b_info, requires_grad_pair, dump_mode)
        self.assertEqual(context.exception.code, CompareException.INDEX_OUT_OF_BOUNDS_ERROR)

    def test_table_value_is_valid_int(self):
        result = table_value_is_valid(1)
        self.assertTrue(result)

    def test_table_value_is_valid_float(self):
        result = table_value_is_valid("-1.00")
        self.assertTrue(result)

        result = table_value_is_valid("+1.00")
        self.assertTrue(result)

    def test_table_value_is_valid_invalid_str(self):
        result = table_value_is_valid("=1.00")
        self.assertFalse(result)

    @patch("msprobe.core.compare.utils.check_file_or_directory_path")
    def test_check_input_param_path_both_exist(self, mock_check):
        """两个路径都传入时，应调用两次 check_file_or_directory_path"""
        input_param = {
            "npu_path": "/path/to/npu",
            "bench_path": "/path/to/bench",
        }

        check_input_param_path(input_param)

        # 断言总共调用 2 次
        self.assertEqual(mock_check.call_count, 2)

        # 分别断言调用过指定参数
        mock_check.assert_any_call("/path/to/npu")
        mock_check.assert_any_call("/path/to/bench")

    @patch("msprobe.core.compare.utils.is_module_available")
    @patch("msprobe.core.compare.utils.get_compare_framework")
    def test_check_input_param_path_and_framework_ok(self, mock_get_framework, mock_is_available):
        """框架一致且依赖库存在，应该正常执行，不抛异常"""

        class Args:
            target_path = "/tmp/npu"
            golden_path = "/tmp/bench"

        mock_get_framework.return_value = "pytorch"
        mock_is_available.return_value = True

        # 不应该抛异常
        check_input_param_path_and_framework(Args(), "pytorch")

        mock_get_framework.assert_called_once()
        mock_is_available.assert_called_once_with("torch")

    @patch("msprobe.core.compare.utils.is_module_available")
    @patch("msprobe.core.compare.utils.get_compare_framework")
    def test_check_input_param_path_and_framework_framework_mismatch(self, mock_get_framework, mock_is_available):
        """框架不一致，应抛 CompareException"""

        class Args:
            target_path = "/tmp/a"
            golden_path = "/tmp/b"

        mock_get_framework.return_value = "mindspore"  # 故意不一致
        mock_is_available.return_value = True

        with self.assertRaises(CompareException):
            check_input_param_path_and_framework(Args(), "pytorch")

        mock_get_framework.assert_called_once()

    @patch("msprobe.core.compare.utils.is_module_available")
    @patch("msprobe.core.compare.utils.get_compare_framework")
    def test_check_input_param_path_and_framework_dependency_missing(self, mock_get_framework, mock_is_available):
        """框架一致但库不存在，应抛 Exception"""

        class Args:
            target_path = "/tmp/a"
            golden_path = "/tmp/b"

        mock_get_framework.return_value = "pytorch"
        mock_is_available.return_value = False  # 模拟 torch 不存在

        with self.assertRaises(Exception):
            check_input_param_path_and_framework(Args(), "pytorch")

        mock_is_available.assert_called_once_with("torch")

    @patch("msprobe.core.compare.utils.logger")
    def test_compare_entry_normal(self, mock_logger):
        """compare_func 正常执行时不应抛异常，也不会记录 error"""

        mock_compare = MagicMock()

        compare_entry(
            compare_func=mock_compare,
            input_param={"a": 1},
            output_path="/tmp",
            nr=1,
            kwargs={"x": 10}
        )

        mock_compare.assert_called_once_with(
            input_param={"a": 1},
            output_path="/tmp",
            suffix="_1",
            x=10
        )

        mock_logger.error.assert_not_called()

    @patch("msprobe.core.compare.utils.logger")
    def test_compare_entry_invalid_data(self, mock_logger):
        """compare_func 抛 INVALID_DATA_ERROR 时，应记录对应错误日志"""

        def raise_invalid_data(*args, **kwargs):
            raise CompareException(CompareException.INVALID_DATA_ERROR)

        mock_compare = MagicMock(side_effect=raise_invalid_data)

        compare_entry(
            compare_func=mock_compare,
            input_param={"a": 1},
            output_path="/tmp",
            nr=2,
            kwargs={}
        )

        mock_logger.error.assert_called_once()
        self.assertIn("Invalid or missing 'data' in dump.json", mock_logger.error.call_args[0][0])

    @patch("msprobe.core.compare.utils.logger")
    def test_compare_entry_invalid_task(self, mock_logger):
        """compare_func 抛 INVALID_TASK_ERROR 时，应记录对应错误日志"""

        def raise_invalid_task(*args, **kwargs):
            raise CompareException(CompareException.INVALID_TASK_ERROR)

        mock_compare = MagicMock(side_effect=raise_invalid_task)

        compare_entry(
            compare_func=mock_compare,
            input_param={"a": 1},
            output_path="/tmp",
            nr=3,
            kwargs={}
        )

        mock_logger.error.assert_called_once()
        self.assertIn("Invalid or missing 'task' in dump.json", mock_logger.error.call_args[0][0])

    @patch("msprobe.core.compare.utils.compare_entry")
    @patch("msprobe.core.compare.utils.mp_logger_init")
    def test_multi_ranks_compare(self, mock_logger_init, mock_compare_entry):
        """验证：logger 初始化正确 + compare_entry 调用次数和参数正确"""

        # 1. 构造输入
        input_param_nr_list = [
            ({"p": 1}, "0"),
            ({"p": 2}, "3"),
            ({"p": 9}, "7"),
        ]
        output_path = "/tmp"
        kwargs = {"x": 10}

        # 2. 执行
        multi_ranks_compare(
            compare_func=MagicMock(),
            input_param_nr_list=input_param_nr_list,
            output_path=output_path,
            kwargs=kwargs,
        )

        # 3. 验证 mp_logger_init 是否按 rank_list 被调用
        mock_logger_init.assert_called_once_with("[0 3 7]")

        # 4. compare_entry 调用次数应与 input_param_nr_list 相同
        self.assertEqual(mock_compare_entry.call_count, 3)

        # 5. 逐次验证 compare_entry 调用参数
        expected_calls = [
            ({"p": 1}, "0"),
            ({"p": 2}, "3"),
            ({"p": 9}, "7"),
        ]

        for call_args, expected in zip(mock_compare_entry.call_args_list, expected_calls):
            ((_, input_param, out_path, nr, kw), _) = call_args  # 解构 MagicMock 调用参数

            self.assertEqual(input_param, expected[0])
            self.assertEqual(nr, expected[1])
            self.assertEqual(out_path, output_path)
            self.assertEqual(kw, kwargs)

    @patch("msprobe.core.compare.utils.logger")   # patch logger 本身
    def test_mp_logger_init(self, mock_logger):
        """验证 logger 的 info/warning/error 都被正确 wrap 并添加前缀"""

        # 1. 创建可监测的 fake logger 方法
        mock_logger.info = MagicMock()
        mock_logger.warning = MagicMock()
        mock_logger.error = MagicMock()

        # 2. 调用 mp_logger_init
        mp_logger_init("[0] ")

        # 3. 调用 wrap 后的 logger 方法
        mock_logger.info("hello")
        mock_logger.warning("abc")
        mock_logger.error("xyz")


    @patch("msprobe.core.compare.utils.logger")
    @patch("msprobe.core.compare.utils.check_and_return_dir_contents")
    def test_get_sorted_ranks_mismatch(self, mock_check, mock_logger):
        """异常情况：两个 rank 列表长度不一致 → 触发 if 分支并抛异常"""

        mock_check.side_effect = [
            ["rank0", "rank1"],   # npu → len = 2
            ["rank0"],            # bench → len = 1
        ]

        with self.assertRaises(CompareException) as cm:
            get_sorted_ranks("npu_dir", "bench_dir")

        # 验证抛出 INVALID_PATH_ERROR
        self.assertEqual(cm.exception.code, CompareException.INVALID_PATH_ERROR)

        # 验证 logger.error 被调用
        mock_logger.error.assert_called_once()
        self.assertIn("The number of ranks", mock_logger.error.call_args[0][0])


class TestReorderIndex(unittest.TestCase):
    def test_reorder_index_mixed_states(self):
        op_parsed_list = [
            {Const.STATE: "OTHER"},
            {Const.STATE: Const.OUTPUT},
            {Const.STATE: Const.PARAMS},
            {Const.STATE: Const.PARAMS_GRAD},
            {Const.STATE: Const.INPUT},
            {"not_state": 123},  # 没有 STATE，算作 other
        ]

        reordered = reorder_index(op_parsed_list)
        self.assertTrue(reordered == [0, 4, 2, 1, 3])

    def test_reorder_index_all_params(self):
        op_parsed_list = [
            {Const.STATE: Const.PARAMS},
            {Const.STATE: Const.PARAMS},
            {Const.STATE: Const.PARAMS},
        ]
        reordered = reorder_index(op_parsed_list)
        self.assertTrue(reordered == [0, 1])

    def test_reorder_index_empty(self):
        op_parsed_list = []
        reordered = reorder_index(op_parsed_list)
        self.assertTrue(reordered == [])

    def test_reorder_index_single_element(self):
        op_parsed_list = [{Const.STATE: Const.PARAMS}]
        reordered = reorder_index(op_parsed_list)
        self.assertTrue(reordered == [])


class TestReorderOpNameList(unittest.TestCase):
    def test_reorder_op_name_list(self):
        # 标准顺序
        op_name_list = ["op.forward.input.0.0", "op.forward.output.0", "op.forward.output.1", "op.forward.parameters.1",
                        "op.forward.parameters.2", "op.parameters_grad.0"]
        state_list = ["input", "output", "output", "parameters", "parameters", "parameters_grad"]
        op_name_reorder, state_reorder = reorder_op_name_list(op_name_list, state_list)
        expected_result = ["op.forward.input.0.0", "op.forward.parameters.1", "op.forward.parameters.2",
                           "op.forward.output.0", "op.forward.output.1", "op.parameters_grad.0"]
        expected_state = ["input", "parameters", "parameters", "output", "output", "parameters_grad"]
        self.assertEqual(op_name_reorder, expected_result)
        self.assertEqual(state_reorder, expected_state)

        # 只有输入元素
        op_name_list = ["op.forward.input.0", "op.forward.input.1"]
        state_list = ["input", "input"]
        op_name_reorder, state_reorder = reorder_op_name_list(op_name_list, state_list)
        expected_result = ["op.forward.input.0", "op.forward.input.1"]
        expected_state = ["input", "input"]
        self.assertEqual(op_name_reorder, expected_result)
        self.assertEqual(state_reorder, expected_state)

        # 输入为空
        op_name_list = []
        state_list = []
        op_name_reorder, state_reorder = reorder_op_name_list(op_name_list, state_list)
        expected_result = []
        expected_state = []
        self.assertEqual(op_name_reorder, expected_result)
        self.assertEqual(state_reorder, expected_state)


class TestGenOpItem(unittest.TestCase):
    def test_gen_op_item_with_data_name(self):
        op_data = {
            'data_name': 'test_data',
            'type': 'torch.Tensor',
            'dtype': 'torch.int64',
            'shape': [3],
            'value': [1, 2, 3],
            'Max': 3,
            'Min': 1,
            'Mean': 2,
            'Norm': 2
        }
        op_name = 'op_test'

        result = gen_op_item(op_data, op_name, 'input')

        self.assertEqual(result['data_name'], 'test_data')
        self.assertEqual(result['full_op_name'], 'test_data')
        self.assertEqual(result['dtype'], 'torch.int64')
        self.assertEqual(result['shape'], [3])
        self.assertEqual(result['Max'], 3)
        self.assertEqual(result['Min'], 1)
        self.assertEqual(result['Mean'], 2)
        self.assertEqual(result['Norm'], 2)
        self.assertEqual(result['md5'], f"{zlib.crc32(str(op_data['value']).encode()):08x}")
        self.assertEqual(result['state'], 'input')

    def test_gen_op_item_with_empty_data_name(self):
        op_data = {
            'data_name': '',
            'type': 'torch.Tensor',
            'value': [1, 2, 3]
        }
        op_name = 'op_test'

        result = gen_op_item(op_data, op_name, 'input')

        # data_name为空时，应该被设置为'-1'
        self.assertEqual(result['data_name'], '-1')
        self.assertEqual(result['full_op_name'], op_name)
        self.assertEqual(result['state'], 'input')

    def test_gen_op_item_with_none_data_name(self):
        op_data = {
            'data_name': None,
            'type': 'torch.Tensor',
            'value': [1, 2, 3]
        }
        op_name = 'op_test'

        result = gen_op_item(op_data, op_name, 'input')

        # data_name为None时，应该被设置为'-1'
        self.assertEqual(result['data_name'], '-1')
        self.assertEqual(result['full_op_name'], op_name)
        self.assertEqual(result['state'], 'input')

    def test_gen_op_item_with_type_torch_size(self):
        op_data = {
            'data_name': 'test_data',
            'type': 'torch.Size',
            'value': [2, 3, 4]
        }
        op_name = 'op_test'

        result = gen_op_item(op_data, op_name, 'input')

        self.assertEqual(result['dtype'], 'torch.Size')
        self.assertEqual(result['shape'], '[2, 3, 4]')
        self.assertEqual(result['Max'], None)
        self.assertEqual(result['Min'], None)
        self.assertEqual(result['Mean'], None)
        self.assertEqual(result['Norm'], None)
        self.assertEqual(result['state'], 'input')

    def test_gen_op_item_with_type_slice(self):
        op_data = {
            'data_name': 'test_data',
            'type': 'slice',
            'value': [1, 2, 3]
        }
        op_name = 'op_test'

        result = gen_op_item(op_data, op_name, 'input')

        self.assertEqual(result['dtype'], 'slice')
        self.assertEqual(result['shape'], str(np.shape(np.array(op_data['value']))))
        self.assertEqual(result['state'], 'input')

    def test_gen_op_item_with_type_ellipsis(self):
        op_data = {
            'data_name': 'test_data',
            'type': 'ellipsis',
            'value': '...'
        }
        op_name = 'op_test'

        result = gen_op_item(op_data, op_name, 'input')

        self.assertEqual(result['dtype'], 'ellipsis')
        self.assertEqual(result['shape'], '[]')
        self.assertEqual(result['Max'], '...')
        self.assertEqual(result['Min'], '...')
        self.assertEqual(result['Mean'], '...')
        self.assertEqual(result['Norm'], '...')
        self.assertEqual(result['state'], 'input')

    def test_gen_op_item_with_type_torch_process_group(self):
        op_data = {
            'data_name': 'test_data',
            'type': 'torch.ProcessGroup',
            'group_ranks': [0, 1]
        }
        op_name = 'op_test'

        result = gen_op_item(op_data, op_name, 'input')

        self.assertEqual(result['dtype'], 'torch.ProcessGroup')
        self.assertEqual(result['shape'], '[]')
        self.assertEqual(result['Max'], '[0, 1]')
        self.assertEqual(result['Min'], '[0, 1]')
        self.assertEqual(result['Mean'], '[0, 1]')
        self.assertEqual(result['Norm'], '[0, 1]')
        self.assertEqual(result['state'], 'input')

    def test_gen_op_item_with_default_dtype(self):
        op_data = {
            'data_name': 'test_data',
            'type': 'other_type',
            'value': [1, 2, 3]
        }
        op_name = 'op_test'

        result = gen_op_item(op_data, op_name, 'input')

        self.assertEqual(result['dtype'], str(type(op_data['value'])))
        self.assertEqual(result['shape'], '[]')
        self.assertEqual(result['state'], 'input')

    def test_gen_op_item_with_md5(self):
        op_data = {
            'data_name': 'test_data',
            'type': 'torch.Tensor',
            'value': [1, 2, 3]
        }
        op_name = 'op_test'

        result = gen_op_item(op_data, op_name, 'input')

        expected_md5 = f"{zlib.crc32(str(op_data['value']).encode()):08x}"
        self.assertEqual(result['md5'], expected_md5)
        self.assertEqual(result['state'], 'input')


class TestApiBatch(unittest.TestCase):
    def test_ApiBatch_increment_input(self):
        api_name = "functional.conv2d"
        start = 2
        api_batch = ApiBatch(api_name, start)

        api_batch.increment(Const.INPUT)

        self.assertEqual(api_batch._state, Const.INPUT)
        self.assertEqual(api_batch.input_len, 2)
        self.assertEqual(api_batch.params_end_index, 4)
        self.assertEqual(api_batch.output_end_index, 4)
        self.assertEqual(api_batch.params_grad_end_index, 4)

    def test_ApiBatch_increment_output(self):
        api_name = "functional.conv2d"
        start = 2
        api_batch = ApiBatch(api_name, start)

        api_batch.increment(Const.OUTPUT)

        self.assertEqual(api_batch._state, Const.OUTPUT)
        self.assertEqual(api_batch.input_len, 1)
        self.assertEqual(api_batch.params_end_index, 3)
        self.assertEqual(api_batch.output_end_index, 4)
        self.assertEqual(api_batch.params_grad_end_index, 4)

    def test_ApiBatch_increment_kwargs(self):
        api_name = "functional.conv2d"
        start = 2
        api_batch = ApiBatch(api_name, start)

        api_batch.increment(Const.KWARGS)

        self.assertEqual(api_batch._state, Const.KWARGS)
        self.assertEqual(api_batch.input_len, 2)
        self.assertEqual(api_batch.params_end_index, 4)
        self.assertEqual(api_batch.output_end_index, 4)
        self.assertEqual(api_batch.params_grad_end_index, 4)

    def test_ApiBatch_increment_params(self):
        api_name = "functional.conv2d"
        start = 2
        api_batch = ApiBatch(api_name, start)

        api_batch.increment(Const.PARAMS)

        self.assertEqual(api_batch._state, Const.PARAMS)
        self.assertEqual(api_batch.input_len, 1)
        self.assertEqual(api_batch.params_end_index, 4)
        self.assertEqual(api_batch.output_end_index, 4)
        self.assertEqual(api_batch.params_grad_end_index, 4)

    def test_ApiBatch_increment_multiple_input(self):
        api_name = "functional.conv2d"
        start = 2
        api_batch = ApiBatch(api_name, start)

        api_batch.increment(Const.INPUT)
        api_batch.increment(Const.INPUT)

        self.assertEqual(api_batch._state, Const.INPUT)
        self.assertEqual(api_batch.input_len, 3)
        self.assertEqual(api_batch.params_end_index, 5)
        self.assertEqual(api_batch.output_end_index, 5)
        self.assertEqual(api_batch.params_grad_end_index, 5)

    def test_ApiBatch_increment_multiple_output(self):
        api_name = "functional.conv2d"
        start = 2
        api_batch = ApiBatch(api_name, start)

        api_batch.increment(Const.OUTPUT)
        api_batch.increment(Const.OUTPUT)

        self.assertEqual(api_batch._state, Const.OUTPUT)
        self.assertEqual(api_batch.input_len, 1)
        self.assertEqual(api_batch.params_end_index, 3)
        self.assertEqual(api_batch.output_end_index, 5)
        self.assertEqual(api_batch.params_grad_end_index, 5)


class TestGenApiBatches(unittest.TestCase):
    def test_gen_api_batches_normal(self):
        result_df_part1 = pd.DataFrame(o_result)
        result_df_part1.columns = CompareConst.SUMMARY_COMPARE_RESULT_HEADER_STACK
        new_columns = [
            ['input', 'Functional.conv2d.0.forward'],
            ['input', 'Functional.conv2d.0.forward'],
            ['input', 'Functional.conv2d.0.forward'],
            ['parameters', 'Functional.conv2d.0.forward'],
            ['parameters', 'Functional.conv2d.0.forward'],
            ['output', 'Functional.conv2d.0.forward'],
            ['parameters_grad', 'Functional.conv2d.0.forward'],
            ['parameters_grad', 'Functional.conv2d.0.forward']
        ]
        result_df_part2 = pd.DataFrame(new_columns)
        result_df_part2.columns = [Const.STATE, Const.API_ORIGIN_NAME]
        result_df = pd.concat([result_df_part1, result_df_part2], axis=1)
        result = result_df.values
        header = result_df.columns.tolist()
        result_api_batches = gen_api_batches(result, header)

        api_batch = ApiBatch('Functional.conv2d.0.forward', 0)
        api_batch.input_len = 3
        api_batch.output_end_index = 6
        api_batch.params_end_index = 5
        api_batch.params_grad_end_index = 8
        api_batch._state = 'parameters_grad'

        result_api_batch = result_api_batches[0]
        self.assertEqual(result_api_batch.api_name, api_batch.api_name)
        self.assertEqual(result_api_batch.start, api_batch.start)
        self.assertEqual(result_api_batch.input_len, api_batch.input_len)
        self.assertEqual(result_api_batch.params_end_index, api_batch.params_end_index)
        self.assertEqual(result_api_batch.params_grad_end_index, api_batch.params_grad_end_index)
        self.assertEqual(result_api_batch._state, api_batch._state)


class TestGetPairedSteps(unittest.TestCase):
    def setUp(self):
        self.npu_dir = tempfile.TemporaryDirectory()
        self.bench_dir = tempfile.TemporaryDirectory()

        self.npu_files = ['step1', 'step2']
        for name in self.npu_files:
            open(os.path.join(self.npu_dir.name, name), 'w').close()

        self.bench_files = ['step2', 'step3']
        for name in self.bench_files:
            open(os.path.join(self.bench_dir.name, name), 'w').close()

    def tearDown(self):
        self.npu_dir.cleanup()
        self.bench_dir.cleanup()

    def test_get_paired_steps(self):
        paired = get_paired_dirs(self.npu_dir.name, self.bench_dir.name)
        self.assertEqual(set(paired), {'step2'})


class FakeConst:
    ALL = "ALL"


class FakeCompareConst:
    HEAD_OF_COMPARE_MODE = {
        "ALL": ["a", "b"],
        "OTHER": ["x", "y"]
    }
    STACK = "stack"
    DATA_NAME = "data_name"


class TestMakeResultTable(unittest.TestCase):

    @patch("msprobe.core.compare.utils.CompareConst", FakeCompareConst)
    @patch("msprobe.core.compare.utils.Const", FakeConst)
    def test_stack_mode_all(self):
        """stack_mode=True 且 dump_mode=ALL → header += [stack, data_name]"""

        result = [[1, 2, "stack_val", "data_val"]]

        df = make_result_table(result, dump_mode="ALL", stack_mode=True)

        self.assertListEqual(df.columns.tolist(), ["a", "b", "stack", "data_name"])
        self.assertEqual(df.iloc[0].tolist(), [1, 2, "stack_val", "data_val"])

    @patch("msprobe.core.compare.utils.CompareConst", FakeCompareConst)
    @patch("msprobe.core.compare.utils.Const", FakeConst)
    def test_stack_mode_other(self):
        """stack_mode=True 且 dump_mode!=ALL → header += [stack]"""

        result = [[10, 20, "stack_info"]]

        df = make_result_table(result, dump_mode="OTHER", stack_mode=True)

        self.assertListEqual(df.columns.tolist(), ["x", "y", "stack"])
        self.assertEqual(df.iloc[0].tolist(), [10, 20, "stack_info"])

    @patch("msprobe.core.compare.utils.CompareConst", FakeCompareConst)
    @patch("msprobe.core.compare.utils.Const", FakeConst)
    def test_no_stack_all(self):
        """stack_mode=False 且 dump_mode=ALL → 删除每行倒数第二列"""

        result = [[1, 2, "stack_to_delete", "data_name_val"]]

        df = make_result_table(result, dump_mode="ALL", stack_mode=False)

        # 删除倒数第二列 → 剩 [1, 2, data_name_val]
        self.assertListEqual(result[0], [1, 2, "data_name_val"])

        self.assertListEqual(df.columns.tolist(), ["a", "b", "data_name"])
        self.assertEqual(df.iloc[0].tolist(), [1, 2, "data_name_val"])

    @patch("msprobe.core.compare.utils.CompareConst", FakeCompareConst)
    @patch("msprobe.core.compare.utils.Const", FakeConst)
    def test_no_stack_other(self):
        """stack_mode=False 且 dump_mode!=ALL → 删除每行最后一列"""

        result = [[11, 22, "stack_to_delete"]]

        df = make_result_table(result, dump_mode="OTHER", stack_mode=False)

        # 删除倒数第一列 → 剩 [11, 22]
        self.assertListEqual(result[0], [11, 22])

        self.assertListEqual(df.columns.tolist(), ["x", "y"])
        self.assertEqual(df.iloc[0].tolist(), [11, 22])


class TestGetAccuracyNLenGtBLen(unittest.TestCase):

    @patch("msprobe.core.compare.utils.process_summary_data")
    @patch("msprobe.core.compare.utils.ApiItemInfo")
    @patch("msprobe.core.compare.utils.result_item_init")
    @patch("msprobe.core.compare.utils.stack_column_process")
    @patch("msprobe.core.compare.utils.safe_get_value")
    def test_n_len_gt_b_len_md5_mode(
        self,
        mock_safe_get_value,
        mock_stack_column_process,
        mock_result_item_init,
        mock_ApiItemInfo,
        mock_process_summary_data,
    ):
        """
        覆盖 n_len > b_len 分支，dump_mode=MD5。
        mock_xxx 注入方式完全隔离内部逻辑。
        """


        # -------------------------------
        # mock 行为定义
        # -------------------------------
        def fake_safe_get_value(d, idx, name, key=None):
            if isinstance(d.get(key), list):
                return d[key][idx]
            return d[key]

        mock_safe_get_value.side_effect = fake_safe_get_value

        mock_stack_column_process.side_effect = lambda item, *a, **k: item
        mock_result_item_init.side_effect = lambda *a, **k: []
        mock_process_summary_data.side_effect = lambda s: s
        mock_ApiItemInfo.side_effect = lambda name, struct, stack: MagicMock()

        # -------------------------------
        # 输入构造：n_len = 2, b_len = 1
        # -------------------------------
        n_dict = {
            "op_name": ["n_op0", "n_op1"],
            "requires_grad": [True, False],
            CompareConst.SUMMARY: [
                [1, 2, 3],
                [4, 5, 6]
            ],
            CompareConst.INPUT_STRUCT: [
                ["n0_a", "n0_b", "n0_c"],
                ["n1_a", "n1_b", "n1_c"],
            ]
        }

        b_dict = {
            "op_name": ["b_op0"],
            "requires_grad": [True],
            CompareConst.SUMMARY: [
                [7, 8, 9]
            ],
            CompareConst.INPUT_STRUCT: [
                ["b0_a", "b0_b", "b0_c"],
            ]
        }

        result = []
        get_accuracy(result, n_dict, b_dict, Const.MD5)

        # -------------------------------
        # 断言结果：应有 2 行，第 2 行来自 n_len > b_len 分支
        # -------------------------------
        self.assertEqual(len(result), 2)

        tail = result[1]

        # MD5 模式的 tail 应为：
        # [n_name, NAN, struct[0], NAN, struct[1], NAN,
        #  requires_grad, NAN, struct[2], NAN, False, NAN, None]

        self.assertEqual(tail[0], "n_op1")
        self.assertEqual(tail[1], CompareConst.NAN)
        self.assertEqual(tail[2], "n1_a")
        self.assertEqual(tail[3], CompareConst.NAN)
        self.assertEqual(tail[4], "n1_b")
        self.assertEqual(tail[5], CompareConst.NAN)
        self.assertEqual(tail[6], False)
        self.assertEqual(tail[7], CompareConst.NAN)
        self.assertEqual(tail[8], "n1_c")
        self.assertEqual(tail[9], CompareConst.NAN)
        self.assertEqual(tail[10], False)
        self.assertEqual(tail[11], CompareConst.NAN)
        self.assertIsNone(tail[12])

    @patch("msprobe.core.compare.utils.process_summary_data")
    @patch("msprobe.core.compare.utils.ApiItemInfo")
    @patch("msprobe.core.compare.utils.result_item_init")
    @patch("msprobe.core.compare.utils.stack_column_process")
    @patch("msprobe.core.compare.utils.safe_get_value")
    def test_n_len_gt_b_len_tensor_mode(
        self,
        mock_safe_get_value,
        mock_stack_column_process,
        mock_result_item_init,
        mock_ApiItemInfo,
        mock_process_summary_data,
    ):
        """
        覆盖 n_len > b_len 分支，dump_mode=TENSOR。
        """

        # -------------------------------
        # mock 行为
        # -------------------------------
        def fake_safe_get_value(d, idx, name, key=None):
            if not isinstance(d, dict):
                return d
            if isinstance(d.get(key), list):
                return d[key][idx]
            return d[key]

        mock_safe_get_value.side_effect = fake_safe_get_value
        mock_stack_column_process.side_effect = lambda item, *a, **k: item
        mock_result_item_init.side_effect = lambda *a, **k: []
        mock_process_summary_data.side_effect = lambda s: s
        mock_ApiItemInfo.side_effect = lambda *a, **k: MagicMock()

        # -------------------------------
        # 输入构造：n_len = 2, b_len = 1
        # -------------------------------
        n_dict = {
            "op_name": ["n_op0", "n_op1"],
            "requires_grad": [True, False],
            CompareConst.SUMMARY: [
                [1, 2, 3],
                [4, 5, 6]
            ],
            CompareConst.INPUT_STRUCT: [
                ["n0_a", "n0_b", "n0_c"],
                ["n1_a", "n1_b", "n1_c"],
            ]
        }

        b_dict = {
            "op_name": ["b_op0"],
            "requires_grad": [True],
            CompareConst.SUMMARY: [
                [7, 8, 9]
            ],
            CompareConst.INPUT_STRUCT: [
                ["b0_a", "b0_b", "b0_c"],
            ]
        }

        result = []
        get_accuracy(result, n_dict, b_dict, Const.ALL)

        # -------------------------------
        # 断言：应有两行（第 2 行来自 n_len > b_len）
        # -------------------------------
        self.assertEqual(len(result), 2)
        tail = result[1]

        # TENSOR 模式应该走这个分支:
        # [n_name, NAN, struct[0], NAN, struct[1], NAN,
        #  requires_grad, NAN, " ", " ", " ", " ", " ", " "]
        self.assertEqual(tail[0], "n_op1")
        self.assertEqual(tail[1], CompareConst.NAN)
        self.assertEqual(tail[2], "n1_a")
        self.assertEqual(tail[3], CompareConst.NAN)
        self.assertEqual(tail[4], "n1_b")
        self.assertEqual(tail[5], CompareConst.NAN)
        self.assertEqual(tail[6], False)
        self.assertEqual(tail[7], CompareConst.NAN)

        # 6 个空格字段
        self.assertEqual(tail[8:14], [" ", " ", " ", " ", " ", " "])

        # summary + nan summary + False + PASS + ""（err_msg）
        self.assertEqual(tail[14:17], [4, 5, 6])
        self.assertEqual(tail[17:20], [CompareConst.NAN]*3)
        self.assertEqual(tail[20], False)
        self.assertEqual(tail[21], CompareConst.PASS)
        self.assertEqual(tail[22], "")


class TestExtractJson(unittest.TestCase):

    @patch("msprobe.core.compare.utils.logger")
    @patch("os.listdir")
    def test_invalid_json_file_type(self, mock_listdir, mock_logger):
        """覆盖 invalid json_file_type 分支"""
        mock_listdir.return_value = ["stack.json"]

        with self.assertRaises(CompareException):
            extract_json("/some/dir", "NOT_EXIST_TYPE")

        mock_logger.error.assert_called_once()
        self.assertIn("invalid json_file_type", mock_logger.error.call_args[0][0])

    @patch("msprobe.core.compare.utils.logger")
    @patch("os.listdir")
    def test_stack_json_not_found(self, mock_listdir, mock_logger):
        """覆盖 stack.json 未找到（warning）"""
        mock_listdir.return_value = []  # 空目录

        path = extract_json("/tmp", Const.STACK_JSON_FILE)

        self.assertEqual(path, "")  # 返回空
        mock_logger.warning.assert_called_once()
        self.assertIn("stack.json is not found", mock_logger.warning.call_args[0][0])

    @patch("msprobe.core.compare.utils.logger")
    @patch("os.listdir")
    def test_dump_json_not_found(self, mock_listdir, mock_logger):
        """覆盖 dump.json 未找到（error）"""
        mock_listdir.return_value = []

        path = extract_json("/tmp", Const.DUMP_JSON_FILE)

        self.assertEqual(path, "")
        mock_logger.error.assert_called_once()
        self.assertIn("dump.json is not found", mock_logger.error.call_args[0][0])

    @patch("msprobe.core.compare.utils.logger")
    @patch("os.listdir")
    def test_debug_json_not_found(self, mock_listdir, mock_logger):
        """覆盖 debug.json 未找到（warning）"""
        mock_listdir.return_value = []

        path = extract_json("/tmp", Const.DEBUG_JSON_FILE)

        self.assertEqual(path, "")
        mock_logger.warning.assert_called_once()
        self.assertIn("debug.json is not found", mock_logger.warning.call_args[0][0])

    @patch("msprobe.core.compare.utils.logger")
    @patch("os.listdir")
    def test_found_json_success(self, mock_listdir, mock_logger):
        """覆盖正常找到文件分支"""
        mock_listdir.return_value = ["abc.txt", "dump.json"]  # 目标文件存在

        path = extract_json("/tmp", Const.DUMP_JSON_FILE)

        self.assertEqual(path, "/tmp/dump.json")
        mock_logger.error.assert_not_called()
        mock_logger.warning.assert_not_called()
