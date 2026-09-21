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

import math
import unittest
from collections import defaultdict
from unittest.mock import MagicMock, Mock, patch

import numpy as np

from msprobe.core.common.const import MonitorConst
from msprobe.mindspore.monitor.optimizer_collect import (
    DeepSpeedZeroOptimizerMon,
    DeepSpeedZeroOptimizerStage0Mon,
    DeepSpeedZeroOptimizerStage1or2Mon,
    DeepSpeedZeroOptimizerStage3Mon,
    MegatronChainedDistributedOptimizerMon,
    MegatronChainedMixPrecisionOptimizerMon,
    MegatronDistributedOptimizerMon,
    MixPrecisionOptimizerMon,
    OptimizerMon,
    OptimizerMonFactory,
)


class FakeParam:
    def __init__(self, values):
        self.data = np.array(values, dtype=float)
        self.grad = self.data.copy()
        self.main_grad = None

    def numel(self):
        return self.data.size

    def flatten(self):
        return self.data.flatten()


class FakeFlatTensor:
    def __init__(self, values):
        self.data = np.array(values, dtype=float).flatten()

    def narrow(self, dim, start, length):
        return self.data[start:start + length]


def setup_param_groups(num_groups=2, params_per_group=3):
    bit16_groups = []
    param_names = {}
    param_slice_mappings = []
    grad_position = {}
    count = 0
    for group_idx in range(num_groups):
        group = []
        param_slice_mapping = {}
        offset = 0
        for param_idx in range(params_per_group):
            name = f'param{group_idx}_{param_idx}'
            param = FakeParam([[count + param_idx, count + param_idx + 1]])
            param_slice_mapping[name] = MagicMock(start=offset, numel=param.numel())
            param_names[param] = name
            grad_position[count] = [group_idx, offset, param.numel()]
            group.append(param)
            offset += param.numel()
            count += 1
        bit16_groups.append(group)
        param_slice_mappings.append(param_slice_mapping)
    return bit16_groups, param_names, param_slice_mappings, grad_position


def create_monitor():
    monitor = Mock()
    monitor.mv_distribution = True
    monitor.mg_direction = True
    monitor.ur_distribution = True
    monitor.duplicate_param = defaultdict(bool)
    monitor.params_have_main_grad = False
    monitor.update_heatmap_visualizer = defaultdict(MagicMock)
    monitor.ratio_heatmap_visualizer = defaultdict(MagicMock)
    monitor.name2tag = {}
    monitor.register_param_call_id = MagicMock()
    return monitor


class TestOptimizerMon(unittest.TestCase):
    def test_fetch_grad_basic(self):
        monitor = create_monitor()
        param = FakeParam([[1, 2]])
        param.grad = np.array([10.0, 20.0])
        params2name = {param: 'p1'}
        monitor.name2tag = {'p1': {MonitorConst.POST_GRAD: 'tag1'}}
        optimizer_mon = OptimizerMon(MagicMock())

        grads = optimizer_mon.fetch_grad(monitor, params2name)
        self.assertTrue(np.array_equal(grads['tag1'], param.grad))
        monitor.register_param_call_id.assert_called_once()

    @patch('msprobe.mindspore.monitor.optimizer_collect.mint.sqrt', lambda x: math.sqrt(x))
    def test_fetch_mv_with_state(self):
        monitor = create_monitor()
        optimizer = MagicMock()
        optimizer.defaults = {'betas': (0.9, 0.999), 'eps': 1e-6}
        optimizer.param_groups = [{'step': 2}]
        param = FakeParam([[1, 2]])
        optimizer_mon = OptimizerMon(optimizer)
        optimizer_mon.state = {
            param: {'exp_avg': 1.0, 'exp_avg_sq': 4.0, 'step': 2}
        }

        exp_avg, exp_avg_sq, update, ratio = optimizer_mon.fetch_mv(monitor, {param: 'p1'})
        self.assertEqual(exp_avg['p1'], 1.0)
        self.assertEqual(exp_avg_sq['p1'], 4.0)
        self.assertIn('p1', update)
        self.assertIn('p1', ratio)


class TestMixPrecisionOptimizerMon(unittest.TestCase):
    def test_map_fp16_to_fp32_param(self):
        fp16_param = FakeParam([[1]])
        fp32_param = FakeParam([[2]])
        optim = MagicMock()
        optim.float16_groups = [[fp16_param]]
        optim.fp32_from_float16_groups = [[fp32_param]]

        mon = MixPrecisionOptimizerMon(optim)
        mon.map_fp16_to_fp32_param(optim)
        self.assertIs(mon.fp16_to_fp32_param[fp16_param], fp32_param)


class TestMegatronDistributedOptimizerMon(unittest.TestCase):
    def test_valid_map(self):
        optim = MagicMock()
        fp16_param = FakeParam([[1]])
        fp32_param = FakeParam([[2]])
        optim.model_float16_groups = [[fp16_param]]
        optim.shard_fp32_from_float16_groups = [[fp32_param]]

        mon = MegatronDistributedOptimizerMon(optim)
        mon.map_fp16_to_fp32_param(optim)
        self.assertIs(mon.fp16_to_fp32_param[fp16_param], fp32_param)

    def test_invalid_optimizer_raises(self):
        optim = Mock(spec=[])  # 没有相应属性，触发异常路径
        with self.assertRaises(Exception):
            MegatronDistributedOptimizerMon(optim).map_fp16_to_fp32_param(optim)


class TestMegatronChainedOptimizers(unittest.TestCase):
    def test_chained_distributed_map(self):
        chained_opt = MagicMock()
        chained_opt.chained_optimizers = []
        fp16_param = FakeParam([[1]])
        fp32_param = FakeParam([[2]])
        for _ in range(2):
            opt = MagicMock()
            opt.model_float16_groups = [[fp16_param]]
            opt.shard_fp32_from_float16_groups = [[fp32_param]]
            chained_opt.chained_optimizers.append(opt)

        mon = MegatronChainedDistributedOptimizerMon(chained_opt)
        mon.map_fp16_to_fp32_param(chained_opt)
        self.assertIs(mon.fp16_to_fp32_param[fp16_param], fp32_param)

    def test_chained_mix_precision_map(self):
        chained_opt = MagicMock()
        chained_opt.chained_optimizers = []
        fp16_param = FakeParam([[1]])
        fp32_param = FakeParam([[2]])
        for _ in range(2):
            opt = MagicMock()
            opt.float16_groups = [[fp16_param]]
            opt.fp32_from_float16_groups = [[fp32_param]]
            chained_opt.chained_optimizers.append(opt)

        mon = MegatronChainedMixPrecisionOptimizerMon(chained_opt)
        mon.map_fp16_to_fp32_param(chained_opt)
        self.assertIs(mon.fp16_to_fp32_param[fp16_param], fp32_param)


class TestDeepSpeedZeroOptimizerMon(unittest.TestCase):
    def setUp(self):
        bit16_groups, param_names, param_slice_mappings, grad_position = setup_param_groups()
        mock_opt = MagicMock()
        mock_opt.state_dict.return_value = {'param_slice_mappings': param_slice_mappings}
        mock_opt.param_names = param_names
        mock_opt.bit16_groups = bit16_groups
        mock_opt.grad_position = grad_position
        self.optimizer_mon = DeepSpeedZeroOptimizerMon(mock_opt)
        self.optimizer_mon.bit16_groups = bit16_groups
        self.optimizer_mon.param2group = self.optimizer_mon.get_group_index()
        self.param_in_partition = list(param_names.keys())[0]
        self.param_not_in_partition = FakeParam([[99]])

    def test_param_not_in_partition(self):
        self.assertFalse(self.optimizer_mon.param_not_in_partition(self.param_in_partition, 0))
        self.assertTrue(self.optimizer_mon.param_not_in_partition(self.param_not_in_partition, 0))

    def test_get_position(self):
        start, numel = self.optimizer_mon.get_position(self.param_in_partition, 0)
        self.assertEqual(start, 0)
        self.assertEqual(numel, self.param_in_partition.numel())

    def test_get_group_index(self):
        groups = self.optimizer_mon.get_group_index()
        self.assertEqual(groups[self.param_in_partition], 0)


class TestDeepSpeedZeroOptimizerStage0Mon(unittest.TestCase):
    def setUp(self):
        bit16_groups, param_names, param_slice_mappings, _ = setup_param_groups()
        flat_groups = []
        for group in bit16_groups:
            flat_data = np.concatenate([p.flatten() for p in group])
            flat_groups.append(FakeFlatTensor(flat_data))

        mock_opt = Mock()
        mock_opt.state_dict.return_value = {'param_slice_mappings': param_slice_mappings}
        mock_opt.param_names = param_names
        mock_opt.bf16_groups = bit16_groups
        mock_opt.fp32_groups_flat_partition = flat_groups
        mock_opt.fp32_groups_gradient_dict = [[FakeParam([[1]]).data for _ in group] for group in bit16_groups]
        mock_opt.state = {
            flat_group: {'exp_avg': flat_group, 'exp_avg_sq': flat_group} for flat_group in flat_groups
        }
        mock_opt.param_to_cpu_states_map = mock_opt.state  # _get_single_state 获取状态

        self.optimizer_mon = DeepSpeedZeroOptimizerStage0Mon(mock_opt)
        self.monitor = create_monitor()
        self.monitor.ur_distribution = False
        self.monitor.name2tag = {name: {MonitorConst.POST_GRAD: name} for name in param_names.values()}
        self.first_param = list(param_names.keys())[0]

    def test_get_grad_for_param(self):
        grad = self.optimizer_mon.get_grad_for_param(self.first_param, 0, 0)
        expected = self.optimizer_mon.optim.fp32_groups_gradient_dict[0][0]
        self.assertTrue(np.array_equal(grad, expected))

    def test_fetch_grad(self):
        grads = self.optimizer_mon.fetch_grad(self.monitor, self.optimizer_mon.optim.param_names)
        for param, name in self.optimizer_mon.optim.param_names.items():
            group_idx, param_idx = [int(i) for i in name.replace('param', '').split('_')]
            expected = self.optimizer_mon.optim.fp32_groups_gradient_dict[group_idx][param_idx]
            self.assertTrue(np.array_equal(grads[name], expected))

    def test_fetch_mv(self):
        # 直接使用优化器上准备好的 state，避免在 get_state 中迭代 Mock.chained_optimizers
        self.optimizer_mon.state = self.optimizer_mon.optim.param_to_cpu_states_map
        exp_avg, exp_avg_sq, _, _ = self.optimizer_mon.fetch_mv(
            self.monitor, self.optimizer_mon.optim.param_names)
        for name in self.optimizer_mon.optim.param_names.values():
            self.assertIn(name, exp_avg)
            self.assertIn(name, exp_avg_sq)


class TestDeepSpeedZeroOptimizerStage1or2Mon(unittest.TestCase):
    def setUp(self):
        bit16_groups, param_names, param_slice_mappings, _ = setup_param_groups()
        flat_groups = []
        for group in bit16_groups:
            flat_data = np.concatenate([p.flatten() for p in group])
            flat_groups.append(FakeFlatTensor(flat_data))

        averaged_gradients = {
            group_idx: [FakeParam([[group_idx]]).data for _ in group]
            for group_idx, group in enumerate(bit16_groups)
        }

        mock_opt = Mock()
        mock_opt.state_dict.return_value = {'param_slice_mappings': param_slice_mappings}
        mock_opt.param_names = param_names
        mock_opt.bit16_groups = bit16_groups
        mock_opt.single_partition_of_fp32_groups = flat_groups
        mock_opt.averaged_gradients = averaged_gradients
        mock_opt.state = {flat: {'exp_avg': flat, 'exp_avg_sq': flat} for flat in flat_groups}
        mock_opt.cpu_offload = False
        mock_opt.param_to_cpu_states_map = mock_opt.state

        self.optimizer_mon = DeepSpeedZeroOptimizerStage1or2Mon(mock_opt)
        self.monitor = create_monitor()
        self.monitor.ur_distribution = False
        self.monitor.name2tag = {name: {MonitorConst.POST_GRAD: name} for name in param_names.values()}

    def test_get_grad_for_param(self):
        param = list(self.optimizer_mon.optim.param_names.keys())[0]
        grad = self.optimizer_mon.get_grad_for_param(param, 0, 0)
        self.assertTrue(np.array_equal(grad, self.optimizer_mon.optim.averaged_gradients[0][0]))

    def test_fetch_grad(self):
        grads = self.optimizer_mon.fetch_grad(self.monitor, self.optimizer_mon.optim.param_names)
        for name, grad in grads.items():
            group_idx, param_idx = [int(i) for i in name.replace('param', '').split('_')]
            self.assertTrue(np.array_equal(
                grad, self.optimizer_mon.optim.averaged_gradients[group_idx][param_idx]))

    def test_fetch_mv(self):
        # 直接赋值 state，避免 get_state 访问 Mock.chained_optimizers
        self.optimizer_mon.state = self.optimizer_mon.optim.param_to_cpu_states_map
        exp_avg, exp_avg_sq, _, _ = self.optimizer_mon.fetch_mv(
            self.monitor, self.optimizer_mon.optim.param_names)
        for name in self.optimizer_mon.optim.param_names.values():
            self.assertIn(name, exp_avg)
            self.assertIn(name, exp_avg_sq)


class TestDeepSpeedZeroOptimizerStage3Mon(unittest.TestCase):
    def setUp(self):
        bit16_groups, param_names, _, grad_position = setup_param_groups()
        flat_groups = []
        for group in bit16_groups:
            flat_data = np.concatenate([p.flatten() for p in group])
            flat_groups.append(FakeFlatTensor(flat_data))

        averaged_gradients = {
            group_idx: [FakeParam([[group_idx]]).data for _ in group]
            for group_idx, group in enumerate(bit16_groups)
        }

        mock_opt = Mock()
        mock_opt.param_names = param_names
        mock_opt.fp16_groups = bit16_groups
        mock_opt.fp32_partitioned_groups_flat = flat_groups
        mock_opt.averaged_gradients = averaged_gradients
        mock_opt.grad_position = grad_position
        mock_opt.get_param_id = lambda p: int(param_names[p].split('_')[1])
        mock_opt.state = {flat: {'exp_avg': flat, 'exp_avg_sq': flat} for flat in flat_groups}
        mock_opt.param_to_cpu_states_map = mock_opt.state

        self.optimizer_mon = DeepSpeedZeroOptimizerStage3Mon(mock_opt)
        self.monitor = create_monitor()
        self.monitor.ur_distribution = False
        self.monitor.name2tag = {name: {MonitorConst.POST_GRAD: name} for name in param_names.values()}

    def test_fetch_grad(self):
        grads = self.optimizer_mon.fetch_grad(self.monitor, self.optimizer_mon.optim.param_names)
        for name, grad in grads.items():
            group_idx, param_idx = [int(i) for i in name.replace('param', '').split('_')]
            self.assertTrue(np.array_equal(
                grad, self.optimizer_mon.optim.averaged_gradients[group_idx][param_idx]))

    def test_fetch_mv(self):
        # 直接赋值 state，避免 get_state 访问 Mock.chained_optimizers
        self.optimizer_mon.state = self.optimizer_mon.optim.param_to_cpu_states_map
        exp_avg, exp_avg_sq, _, _ = self.optimizer_mon.fetch_mv(
            self.monitor, self.optimizer_mon.optim.param_names)
        for name in self.optimizer_mon.optim.param_names.values():
            self.assertIn(name, exp_avg)
            self.assertIn(name, exp_avg_sq)


class TestOptimizerMonFactory(unittest.TestCase):
    def test_create_optimizer_mon(self):
        mix_opt = MagicMock()
        mix_opt.__class__.__name__ = "Float16OptimizerWithFloat16Params"
        self.assertIsInstance(OptimizerMonFactory.create_optimizer_mon(mix_opt), MixPrecisionOptimizerMon)

        dist_opt = MagicMock()
        dist_opt.__class__.__name__ = "DistributedOptimizer"
        self.assertIsInstance(OptimizerMonFactory.create_optimizer_mon(dist_opt), MegatronDistributedOptimizerMon)

        chained_opt = MagicMock()
        chained_opt.__class__.__name__ = "ChainedOptimizer"
        chained_opt.chained_optimizers = [dist_opt, dist_opt]
        self.assertIsInstance(
            OptimizerMonFactory.create_optimizer_mon(chained_opt),
            MegatronChainedDistributedOptimizerMon)

        zero0_opt = MagicMock()
        zero0_opt.__class__.__name__ = "BF16_Optimizer"
        self.assertIsInstance(OptimizerMonFactory.create_optimizer_mon(zero0_opt), DeepSpeedZeroOptimizerStage0Mon)

        zero1_opt = MagicMock()
        zero1_opt.__class__.__name__ = "DeepSpeedZeroOptimizer"
        self.assertIsInstance(OptimizerMonFactory.create_optimizer_mon(zero1_opt), DeepSpeedZeroOptimizerStage1or2Mon)

        zero3_opt = MagicMock()
        zero3_opt.__class__.__name__ = "DeepSpeedZeroOptimizer_Stage3"
        self.assertIsInstance(OptimizerMonFactory.create_optimizer_mon(zero3_opt), DeepSpeedZeroOptimizerStage3Mon)

        unknown_opt = MagicMock()
        unknown_opt.__class__.__name__ = "UnknownOptimizer"
        self.assertIsInstance(OptimizerMonFactory.create_optimizer_mon(unknown_opt), OptimizerMon)


if __name__ == '__main__':
    unittest.main()
