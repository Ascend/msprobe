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
from unittest.mock import patch, mock_open
import numpy as np
from msprobe.core.config_check.ckpt_compare import metrics
from msprobe.core.config_check.ckpt_compare import megatron_loader



class TestMetrics(unittest.TestCase):

    def test_in_different_shape(self):
        a = np.zeros((2, 3))
        b = np.zeros((2, 3))
        c = np.zeros((3, 2))
        self.assertFalse(metrics.in_different_shape(a, b))
        self.assertTrue(metrics.in_different_shape(a, c))

    def test_l2_distance(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        c = np.array([4.0, 5.0, 6.0])
        self.assertAlmostEqual(metrics.l2_distance(a, b), 0.0)
        self.assertAlmostEqual(metrics.l2_distance(a, c), np.linalg.norm(a - c))
        self.assertIsNone(metrics.l2_distance(None, b))
        self.assertIsNone(metrics.l2_distance(a, None))
        self.assertIsNone(metrics.l2_distance(a, np.zeros((2, 2))))

    def test_cos_sim(self):
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        c = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        self.assertAlmostEqual(metrics.cos_sim(a, b), 1.0, places=6)
        self.assertAlmostEqual(metrics.cos_sim(a, c), 0.0, places=6)
        self.assertIsNone(metrics.cos_sim(a, np.zeros((2, 2), dtype=np.float32)))

    def test_numel(self):
        a = np.zeros((2, 3))
        b = np.zeros((2, 3))
        c = np.zeros((3, 2))
        self.assertEqual(metrics.numel(a, b), 6)
        self.assertEqual(metrics.numel(a, c), 6)
        d = np.zeros((2, 2))
        self.assertEqual(metrics.numel(a, d), (6, 4))

    def test_shape(self):
        a = np.zeros((2, 3))
        b = np.zeros((2, 3))
        c = np.zeros((3, 2))
        self.assertEqual(metrics.shape(a, b), [2, 3])
        self.assertEqual(metrics.shape(a, c), [[2, 3], [3, 2]])


class TestMegatronLoader(unittest.TestCase):

    def test__parse_real_layer_idx(self):
        name = 'layers.2.attn/1' # vpp_stage = 1
        result = megatron_loader._parse_real_layer_idx(name, num_layers_per_stage=4, pp_size=2, pp_rank=1)
        self.assertEqual(result, 'layers.14.attn')

    def test__parse_real_expert_idx(self):
        name = 'layers.0.experts.3.weight'
        result = megatron_loader._parse_real_expert_idx(name, num_experts_per_rank=4, exp_rank=2)
        self.assertIn('experts.11', result)  # 3 + 2*4 = 11

        # No expert pattern
        name2 = 'layers.0.weight'
        self.assertEqual(megatron_loader._parse_real_expert_idx(name2, 4, 2), name2)

    def test__consolidate_tp_weights(self):
        arr1 = np.ones((2,2))
        arr2 = np.zeros((2,2))
        weights = {
            'linear_fc1.weight': [arr1, arr2],
            'linear_fc2.weight': [arr1, arr2],
            'linear_fc2.bias': [arr1, arr1]
        }
        result = megatron_loader._consolidate_tp_weights(weights)
        self.assertTrue(np.allclose(result['linear_fc1.weight'], np.concatenate([arr1, arr2], axis=0)))
        self.assertTrue(np.allclose(result['linear_fc2.weight'], np.concatenate([arr1, arr2], axis=1)))
        self.assertTrue(np.allclose(result['linear_fc2.bias'], arr1))

    def test__parse_num_layers_per_stage(self):
        keys = {'layers.0.weight': None, 'layers.1.weight': None, 'layers.2.weight': None}
        self.assertEqual(megatron_loader._parse_num_layers_per_stage(keys), 3)


if __name__ == '__main__':
    unittest.main()
    