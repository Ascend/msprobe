# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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
from unittest.mock import patch, MagicMock
from msprobe.core.config_check.ckpt_compare import megatron_loader
import numpy as np


class TestMegatronLoader(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_fmk_adp = MagicMock()
        self.mock_logger = MagicMock()
        self.patcher1 = patch('msprobe.core.config_check.ckpt_compare.megatron_loader.FmkAdp', self.mock_fmk_adp)
        self.patcher2 = patch('msprobe.core.common.log.logger', self.mock_logger)
        self.patcher1.start()
        self.patcher2.start()

    def tearDown(self) -> None:
        self.patcher1.stop()
        self.patcher2.stop()

    def test__get_parameter_given_nested_dict_when_recursive_then_yield_all(self):
        weights = {'a': {'b': 1, 'c': {'d': 2}}, 'e': 3}
        self.mock_fmk_adp.is_tensor.side_effect = lambda x: isinstance(x, int)
        self.mock_fmk_adp.asnumpy.side_effect = lambda x: x

        result = list(megatron_loader._get_parameter(weights))
        expected = [
            ('a.b', 1),
            ('a.c.d', 2),
            ('e', 3)
        ]
        self.assertEqual(result, expected)

    def test__get_parameter_given_flat_dict_when_no_nesting_then_yield_all(self):
        weights = {'a': 1, 'b': 2}
        self.mock_fmk_adp.is_tensor.return_value = True
        self.mock_fmk_adp.asnumpy.side_effect = lambda x: x

        result = list(megatron_loader._get_parameter(weights))
        self.assertEqual(result, [('a', 1), ('b', 2)])

    def test__parse_real_layer_idx_given_no_layer_index_when_no_match_then_return_original(self):
        param_name = 'embedding.weight/0'
        result = megatron_loader._parse_real_layer_idx(param_name, 1, 1, 0)
        self.assertEqual(result, 'embedding.weight')

    # _parse_real_expert_idx tests
    def test__parse_real_expert_idx_given_valid_name_when_exp_parallel_then_calculate_index(self):
        param_name = 'experts.0.mlp.dense_h_to_4h.weight'
        result = megatron_loader._parse_real_expert_idx(param_name, num_experts_per_rank=2, exp_rank=1)
        self.assertEqual(result, 'experts.2.mlp.dense_h_to_4h.weight')

    def test__parse_real_expert_idx_given_no_expert_index_when_no_match_then_return_original(self):
        param_name = 'non_expert.weight'
        result = megatron_loader._parse_real_expert_idx(param_name, 1, 0)
        self.assertEqual(result, 'non_expert.weight')

    # _consolidate_tp_weights tests
    def test__consolidate_tp_weights_given_column_parallel_then_concat_axis0(self):
        weights = {
            'linear_qkv': [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
        }
        result = megatron_loader._consolidate_tp_weights(weights)
        expected = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        np.testing.assert_array_equal(result['linear_qkv'], expected)

    def test__consolidate_tp_weights_given_row_parallel_then_concat_axis1(self):
        weights = {
            'linear_proj.weight': [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
        }
        result = megatron_loader._consolidate_tp_weights(weights)
        expected = np.array([[1, 2, 5, 6], [3, 4, 7, 8]])
        np.testing.assert_array_equal(result['linear_proj.weight'], expected)

    def test__consolidate_tp_weights_given_other_params_then_use_first(self):
        weights = {
            'embedding': [np.array([1, 2, 3]), np.array([1, 2, 3])]
        }
        result = megatron_loader._consolidate_tp_weights(weights)
        np.testing.assert_array_equal(result['embedding'], np.array([1, 2, 3]))

    # _parse_num_layers_per_stage tests
    def test__parse_num_layers_per_stage_given_valid_keys_then_calculate_max(self):
        tp_partition = {
            'layers.0.linear.weight': [],
            'layers.1.linear.weight': [],
            'layers.5.linear.weight': []
        }
        result = megatron_loader._parse_num_layers_per_stage(tp_partition)
        self.assertEqual(result, 6)

    @patch('os.listdir', return_value=[])
    def test_parse_parallel_size_given_empty_dir_then_raise_error(self, mock_listdir):
        with self.assertRaises(ValueError):
            megatron_loader.parse_parallel_size('empty_dir')

    @patch('os.path.exists', return_value=False)
    @patch('re.findall', return_value=[])
    def test_parse_iteration_given_invalid_path_then_raise_error(self, mock_find, mock_exists):
        with self.assertRaises(ValueError):
            megatron_loader.parse_iteration('invalid_path')


if __name__ == '__main__':
    unittest.main()
