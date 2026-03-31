import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch, MagicMock

import torch
from msprobe.core.common.const import MonitorConst
from msprobe.pytorch.monitor.optimizer_collect import OptimizerMon, \
    OptimizerMonFactory, MegatronMixPrecisionOptimizerMon, MegatronDistributedOptimizerMon, \
    MegatronChainedDistributedOptimizerMon, MegatronChainedMixPrecisionOptimizerMon, \
    DeepSpeedZeroOptimizerMon, DeepSpeedZeroOptimizerStage0Mon, \
    DeepSpeedZeroOptimizerStage1or2Mon, DeepSpeedZeroOptimizerStage3Mon
from msprobe.core.monitor.utils import MVResult


def setup_param_groups(num_groups=2, params_per_group=5):
    bit16_groups = []
    param_names = {}
    grad_position = {}
    param_slice_mappings = []
    count = 0
    for group_idx in range(num_groups):
        group = []
        param_slice_mapping = {}
        offset = 0
        for i in range(params_per_group):
            name = f'param{group_idx}_{i}'
            p = torch.nn.Parameter(torch.randn(2, 3, dtype=torch.bfloat16))
            p.ds_tensor = torch.nn.Parameter(torch.randn(1, 3, dtype=torch.bfloat16))
            p.ds_id = count
            param_slice_mapping[name] = MagicMock(start=offset, numel=p.numel())
            group.append(p)
            param_names[p] = name
            grad_position[count] = [group_idx, offset, p.numel()]
            offset += p.numel()
            count += 1
        bit16_groups.append(group)
        param_slice_mappings.append(param_slice_mapping)

    return bit16_groups, param_names, param_slice_mappings, grad_position


def setup_mock_monitor():
    mock_monitor = MagicMock()
    mock_monitor.mv_distribution = True
    mock_monitor.mg_direction = False
    mock_monitor.ur_distribution = False

    return mock_monitor


class TestOptimizerMon(unittest.TestCase):
    def setUp(self) -> None:
        self.monitor = Mock()
        self.monitor.mv_distribution = True
        self.monitor.mg_direction = True
        self.monitor.ur_distribution = True
        self.monitor.update_heatmap_visualizer = {'param1': Mock(), 'param2': Mock()}
        self.monitor.ratio_heatmap_visualizer = {'param1': Mock(), 'param2': Mock()}

    def test_get_single_state(self):
        optimizer_mon = OptimizerMon(None)
        torch_opt = Mock()
        torch_opt.param_to_cpu_states_map = {"a": 1}
        optimizer_mon._get_single_state(torch_opt)
        assert optimizer_mon.state == {"a": 1}

        optimizer_mon = OptimizerMon(None)
        torch_opt = Mock()
        del torch_opt.param_to_cpu_states_map  # 强制没有上一个属性
        torch_opt.state = {"b": 2}
        optimizer_mon._get_single_state(torch_opt)
        assert optimizer_mon.state == {"b": 2}

        optimizer_mon = OptimizerMon(None)
        torch_opt = Mock()
        del torch_opt.param_to_cpu_states_map
        del torch_opt.state
        inner_opt = Mock()
        inner_opt.state = {"c": 3}
        torch_opt.optimizer = inner_opt
        optimizer_mon._get_single_state(torch_opt)
        assert optimizer_mon.state == {"c": 3}

    def test_get_state(self):
        optimizer_mon = OptimizerMon(None)
        torch_opt = SimpleNamespace()
        optimizer_mon.get_state(torch_opt)

        torch_opt = SimpleNamespace()
        opt1 = SimpleNamespace(state={"a": 1})
        opt2 = SimpleNamespace(state={"b": 2})
        torch_opt.chained_optimizers = [opt1, opt2]
        optimizer_mon.get_state(torch_opt)

    def test_fetch_grad(self):
        optimizer_mon = OptimizerMon(None)
        monitor = Mock()
        params2name = {}
        result = optimizer_mon.fetch_grad(monitor, params2name)
        self.assertEqual(result, {})

        def _mock_param(numel, grad_tensor):
            p = Mock()
            p.numel.return_value = numel
            p.grad = grad_tensor
            p.main_grad = grad_tensor
            return p

        monitor = Mock()
        p = _mock_param(3, torch.tensor([1., 2., 3.]))
        monitor.fp16_to_fp32_param = {p: p}
        monitor.duplicate_param = {"p": True}
        params2name = {p: "p"}
        result = optimizer_mon.fetch_grad(monitor, params2name)
        self.assertEqual(result, {})

        # param 映射 fp16->fp32
        monitor = Mock()
        monitor.duplicate_param = {"p1": False, "p2": False}
        p1 = _mock_param(4, torch.tensor([10., 20., 30., 40.]))
        p2 = _mock_param(6, torch.tensor([1., 2., 3., 4., 5., 6.]))
        optimizer_mon.fp16_to_fp32_param = {
            p1: p1,  # only p1 in this
        }
        params2name = {
            p1: "p1",
            p2: "p2"
        }
        result = optimizer_mon.fetch_grad(monitor, params2name)
        self.assertEqual(len(result), 1)
        value = next(iter(result.values()))
        self.assertTrue(torch.equal(value, p1.grad))

        p = _mock_param(3, torch.tensor([1., 2., 3.]))
        optimizer_mon.fp16_to_fp32_param = {p: p}
        monitor.duplicate_param = {"p": True}
        params2name = {p: "p"}
        result = optimizer_mon.fetch_grad(monitor, params2name)
        self.assertEqual(result, {})

        p = _mock_param(4, None)
        optimizer_mon.fp16_to_fp32_param = {p: p}
        params2name = {p: "p1"}
        result = optimizer_mon.fetch_grad(monitor, params2name)
        self.assertEqual(result, {})  # None → 跳过

    def test_map_fp16_to_fp32_param(self):
        optimizer_mon = OptimizerMon(None)
        torch_opt = Mock()
        optimizer_mon.map_fp16_to_fp32_param(torch_opt)

    def test_fetch_mv_blank(self):
        optimizer_mon = OptimizerMon(None)
        res = optimizer_mon.fetch_mv(None, {})
        self.assertEqual(res.exp_avg, {})

    def test_fetch_mv(self):
        self.torch_opt = Mock()
        self.torch_opt.state = {
            'param1': {'exp_avg': torch.tensor(0.1), 'exp_avg_sq': torch.tensor(0.2), 'step': torch.tensor(10)},
            'param2': {'exp_avg': torch.tensor(0.3), 'exp_avg_sq': torch.tensor(0.4), 'step': torch.tensor(20)}
        }
        self.torch_opt.param_groups = [{'step': 10}]
        self.torch_opt.defaults = {'betas': (0.9, 0.999), 'eps': 1e-8}
        self.params2name = {'param1': 'param1', 'param2': 'param2'}

        self.optimizer_mon = OptimizerMon(None)
        result = self.optimizer_mon.fetch_mv(self.monitor, self.params2name)
        self.assertIsInstance(result, MVResult)

    def test_patch_grad_sync(self):
        optimizer_mon = OptimizerMon(None)
        monitor = Mock()
        optimizer_mon.patch_grad_sync(monitor)

    def test_restore_grad_sync(self):
        optimizer_mon = OptimizerMon(None)
        monitor = Mock()
        monitor.enable_megatron = False
        optimizer_mon.patch_grad_sync(monitor)


class TestMixPrecisionOptimizerMon(unittest.TestCase):
    def test_fetch_mv_with_fp16_to_fp32_param_and_mix_prec_opt(self):
        # init monitor, torch_opt ...
        self.monitor = MagicMock()
        self.torch_opt = MagicMock()
        self.params2name = MagicMock()
        self.mix_prec_opt = MagicMock()
        self.mix_prec_opt.float16_groups = [MagicMock()]
        self.mix_prec_opt.fp32_from_float16_groups = [MagicMock()]
        self.optimizer = MegatronMixPrecisionOptimizerMon(self.torch_opt)
        self.optimizer.fp16_to_fp32_param = {}

        # Mock fetch_mv method and set a fixed return value
        mv_result = MVResult(exp_avg={}, exp_avg_sq={}, update={}, ratio={})
        self.mock_fetch_mv = MagicMock(return_value=mv_result)
        self.optimizer.fetch_mv = self.mock_fetch_mv

        res = self.optimizer.fetch_mv(self.monitor, self.params2name)
        self.mock_fetch_mv.assert_called_once_with(self.monitor, self.params2name)
        self.assertIsInstance(res, MVResult)


class TestChainedMixPrecisionOptimizerMon(unittest.TestCase):
    def test_fetch_mv_with_fp16_to_fp32_param_and_mix_prec_opt(self):
        # init monitor, torch_opt ...
        self.monitor = MagicMock()
        self.torch_opt = MagicMock()
        self.params2name = MagicMock()
        self.torch_opt.float16_groups = [MagicMock()]
        self.torch_opt.fp32_from_float16_groups = [MagicMock()]
        self.optimizer = MegatronChainedMixPrecisionOptimizerMon(self.torch_opt)
        self.optimizer.optimizer = [MagicMock(), MagicMock()]
        self.optimizer.fp16_to_fp32_param = {}

        # Mock fetch_mv method and set a fixed return value
        mv_result = MVResult(exp_avg={}, exp_avg_sq={}, update={}, ratio={})
        self.mock_fetch_mv = MagicMock(return_value=mv_result)
        self.optimizer.fetch_mv = self.mock_fetch_mv

        res = self.optimizer.fetch_mv(self.monitor, self.params2name)
        self.mock_fetch_mv.assert_called_once_with(self.monitor, self.params2name)
        self.assertIsInstance(res, MVResult)


class TestMegatronChainedDistributedOptimizerMon(unittest.TestCase):
    def setUp(self):
        self.monitor = MagicMock()
        self.torch_opt = MagicMock()
        self.params2name = MagicMock()
        self.torch_opt.chained_optimizers = [MagicMock(), MagicMock()]
        mv_result = MVResult(exp_avg={}, exp_avg_sq={}, update={}, ratio={})
        self.mock_fetch_mv = MagicMock(return_value=mv_result)
        self.optimizer = MegatronChainedDistributedOptimizerMon(self.torch_opt)

    def test_fetch_mv_with_valid_optimizer(self):
        for opt in self.torch_opt.chained_optimizers:
            opt.model_float16_groups = [MagicMock()]
            opt.shard_fp32_from_float16_groups = [MagicMock()]
        self.optimizer.fetch_mv = self.mock_fetch_mv

        res = self.optimizer.fetch_mv(self.monitor, self.params2name)
        self.assertIsInstance(res, MVResult)

    def test_fetch_mv_with_invalid_optimizer(self):
        for opt in self.torch_opt.chained_optimizers:
            del opt.model_float16_groups
            del opt.shard_fp32_from_float16_groups

        self.optimizer.fetch_mv(self.monitor, self.params2name)


class TestMegatronDistributedOptimizerMon(unittest.TestCase):
    def setUp(self):
        self.monitor = MagicMock()
        self.torch_opt = MagicMock()
        self.params2name = MagicMock()
        mv_result = MVResult(exp_avg={}, exp_avg_sq={}, update={}, ratio={})
        self.mock_fetch_mv = MagicMock(return_value=mv_result)
        self.optimizer = MegatronDistributedOptimizerMon(self.torch_opt)

    def test_fetch_mv_with_valid_optimizer(self):
        self.torch_opt.model_float16_groups = [MagicMock()]
        self.torch_opt.shard_fp32_from_float16_groups = [MagicMock()]
        self.optimizer.fetch_mv = self.mock_fetch_mv

        res = self.optimizer.fetch_mv(self.monitor, self.params2name)
        self.assertIsInstance(res, MVResult)

    def test_fetch_mv_with_invalid_optimizer(self):
        self.torch_opt.model_float16_groups = None
        self.torch_opt.shard_fp32_from_float16_groups = None

        with self.assertRaises(Exception):
            self.optimizer.fetch_mv(self.monitor, self.params2name)


class TestCommonFetchMv(unittest.TestCase):
    def setUp(self) -> None:
        self.monitor = MagicMock()
        self.torch_opt = MagicMock()
        self.params2name = MagicMock()

    def test_optimizer_mon(self):
        self.optimizer = OptimizerMon(None)
        res = self.optimizer.fetch_mv(self.monitor, self.params2name)
        self.assertIsInstance(res, MVResult)


class TestDeepSpeedZeroOptimizer(unittest.TestCase):
    def setUp(self):
        bit16_groups, param_names, param_slice_mappings, _ = setup_param_groups()

        mock_opt = MagicMock()
        mock_opt.state_dict.return_value = {
            'param_slice_mappings': param_slice_mappings
        }
        mock_opt.param_names = param_names
        mock_opt.bit16_groups = bit16_groups
        self.torch_opt = mock_opt
        self.mock_monitor = setup_mock_monitor()
        self.optimizer_mon = DeepSpeedZeroOptimizerMon(mock_opt)
        self.optimizer_mon.bit16_groups = mock_opt.bit16_groups
        self.optimizer_mon.param2group = self.optimizer_mon.get_group_index()

    def test_param_not_in_partition(self):
        param_in_partition = list(self.torch_opt.param_names.keys())[0]
        param_not_in_partition = torch.randn(2, 3)

        self.assertFalse(
            self.optimizer_mon.param_not_in_partition(param_in_partition, 0)
        )
        self.assertTrue(
            self.optimizer_mon.param_not_in_partition(param_not_in_partition, 0)
        )

    def test_get_position(self):
        param_in_partition = list(self.torch_opt.param_names.keys())[0]
        start, numel = self.optimizer_mon.get_position(param_in_partition, 0)
        self.assertEqual(start, 0)
        self.assertEqual(numel, 6)

    def test_get_group_index(self):
        param = list(self.torch_opt.param_names.keys())[6]
        self.assertEqual(self.optimizer_mon.param2group[param], 1)


class TestDeepSpeedZeroOptimizerStage0Mon(unittest.TestCase):
    def setUp(self):
        bit16_groups, param_names, param_slice_mappings, _ = setup_param_groups()

        mock_opt = MagicMock()
        mock_opt.state_dict.return_value = {
            'param_slice_mappings': param_slice_mappings
        }
        mock_opt.param_names = param_names
        mock_opt.bf16_groups = bit16_groups
        mock_opt.fp32_groups_flat_partition = [torch.stack(group, dim=0).flatten().float() \
                                               for group in bit16_groups]  # mock name 2 index in subgroup
        mock_opt.state = {
            flat_group: {
                'exp_avg': torch.ones_like(flat_group),
                'exp_avg_sq': torch.ones_like(flat_group)
            } for flat_group in mock_opt.fp32_groups_flat_partition
        }
        mock_opt.cpu_offload = False

        self.torch_opt = mock_opt
        self.mock_monitor = setup_mock_monitor()
        self.optimizer_mon = DeepSpeedZeroOptimizerStage0Mon(mock_opt)

    def test_get_grad_for_param(self):
        param = list(self.torch_opt.param_names.keys())[0]
        group_idx = 0
        param_id = 2
        grad_expected = torch.randn_like(param)
        self.torch_opt.fp32_groups_gradient_dict = [[0, 0, grad_expected, 0]]
        grad = self.optimizer_mon.get_grad_for_param(param, group_idx, param_id)

        self.assertTrue(torch.equal(grad_expected, grad))

    def test_fetch_grad(self):
        self.torch_opt.fp32_groups_gradient_dict = [[torch.randn_like(param) for param in group] for group in
                                                    self.optimizer_mon.bit16_groups]
        self.mock_monitor.name2tag = {name: {MonitorConst.POST_GRAD: name} for name in
                                      self.torch_opt.param_names.values()}
        result = self.optimizer_mon.fetch_grad(self.mock_monitor, self.torch_opt.param_names)
        for _, name in self.torch_opt.param_names.items():
            group_index, param_id = [int(i) for i in name.replace('param', '').split('_')]
            self.assertTrue(torch.equal(result[name], self.torch_opt.fp32_groups_gradient_dict[group_index][param_id]))

    def test_fetch_mv(self):
        del self.torch_opt.chained_optimizers
        del self.torch_opt.param_to_cpu_states_map
        result = self.optimizer_mon.fetch_mv(self.mock_monitor, self.torch_opt.param_names)
        for param, name in self.torch_opt.param_names.items():
            self.assertTrue(torch.equal(result.exp_avg[name], torch.ones_like(param).flatten()))
            self.assertTrue(torch.equal(result.exp_avg_sq[name], torch.ones_like(param).flatten()))


class TestDeepSpeedZeroOptimizerStage1or2Mon(unittest.TestCase):
    def setUp(self):
        bit16_groups, param_names, param_slice_mappings, _ = setup_param_groups()

        mock_opt = MagicMock()
        mock_opt.state_dict.return_value = {
            'param_slice_mappings': param_slice_mappings
        }
        mock_opt.param_names = param_names
        mock_opt.bit16_groups = bit16_groups
        mock_opt.single_partition_of_fp32_groups = [torch.stack(group, dim=0).flatten().float() \
                                                    for group in bit16_groups]
        mock_opt.averaged_gradients = {group_idx: [torch.randn_like(param) for param in group] for group_idx, group in
                                       enumerate(bit16_groups)}  # mock name 2 index in subgroup
        mock_opt.state = {
            flat_group: {
                'exp_avg': torch.ones_like(flat_group),
                'exp_avg_sq': torch.ones_like(flat_group)
            } for flat_group in mock_opt.single_partition_of_fp32_groups
        }
        mock_opt.cpu_offload = False

        self.torch_opt = mock_opt
        self.mock_monitor = setup_mock_monitor()
        self.optimizer_mon = DeepSpeedZeroOptimizerStage1or2Mon(mock_opt)

    def test_get_grad_for_param(self):
        param = list(self.torch_opt.param_names.keys())[0]
        group_idx = 0
        param_id = 2
        grad_expected = torch.randn_like(param)
        self.torch_opt.averaged_gradients = [[0, 0, grad_expected, 0]]
        grad = self.optimizer_mon.get_grad_for_param(param, group_idx, param_id)

        self.assertTrue(torch.equal(grad_expected, grad))

    def test_fetch_grad(self):
        self.mock_monitor.name2tag = {name: {MonitorConst.POST_GRAD: name} for name in
                                      self.torch_opt.param_names.values()}
        result = self.optimizer_mon.fetch_grad(self.mock_monitor, self.torch_opt.param_names)
        for param, name in self.torch_opt.param_names.items():
            group_index, param_id = [int(i) for i in name.replace('param', '').split('_')]
            self.assertTrue(torch.equal(result[name], self.torch_opt.averaged_gradients[group_index][param_id]))

    def test_fetch_mv(self):
        del self.torch_opt.chained_optimizers
        del self.torch_opt.param_to_cpu_states_map
        result = self.optimizer_mon.fetch_mv(self.mock_monitor, self.torch_opt.param_names)
        for param, name in self.torch_opt.param_names.items():
            self.assertTrue(torch.equal(result.exp_avg[name], torch.ones_like(param).flatten()))
            self.assertTrue(torch.equal(result.exp_avg_sq[name], torch.ones_like(param).flatten()))


class TestDeepSpeedZeroOptimizerStage3Mon(unittest.TestCase):
    def setUp(self):
        bit16_groups, param_names, _, grad_position = setup_param_groups()

        mock_opt = MagicMock()
        mock_opt.param_names = param_names
        mock_opt.fp16_groups = bit16_groups
        mock_opt.fp32_partitioned_groups_flat = [torch.stack(group, dim=0).flatten().float()
                                                 for group in bit16_groups]
        mock_opt.averaged_gradients = {group_idx: [torch.randn_like(param) for param in group]
                                       for group_idx, group in enumerate(bit16_groups)}
        mock_opt.grad_position = grad_position
        mock_opt.get_param_id = lambda x: int(param_names[x].split('_')[1])
        mock_opt.state = {
            flat_group: {
                'exp_avg': torch.ones_like(flat_group),
                'exp_avg_sq': torch.ones_like(flat_group)
            } for flat_group in mock_opt.fp32_partitioned_groups_flat
        }

        self.torch_opt = mock_opt
        self.optimizer_mon = DeepSpeedZeroOptimizerStage3Mon(mock_opt)
        self.mock_monitor = setup_mock_monitor()

    def test_fetch_grad(self):
        self.mock_monitor.name2tag = {name: {MonitorConst.POST_GRAD: name} for name in
                                      self.torch_opt.param_names.values()}
        result = self.optimizer_mon.fetch_grad(self.mock_monitor, self.torch_opt.param_names)
        for param, name in self.torch_opt.param_names.items():
            group_index, param_id = [int(i) for i in name.replace('param', '').split('_')]
            self.assertTrue(torch.equal(result[name], self.torch_opt.averaged_gradients[group_index][param_id]))

    def test_fetch_mv(self):
        del self.torch_opt.chained_optimizers
        del self.torch_opt.param_to_cpu_states_map
        result = self.optimizer_mon.fetch_mv(self.mock_monitor, self.torch_opt.param_names)
        for param, name in self.torch_opt.param_names.items():
            self.assertTrue(torch.equal(result.exp_avg[name], torch.ones_like(param).flatten()))
            self.assertTrue(torch.equal(result.exp_avg_sq[name], torch.ones_like(param).flatten()))


class TestOptimizerMonFactory(unittest.TestCase):

    def test_create_optimizer_mon(self):
        # 测试已知的优化器类型
        mix_optimizer = MagicMock()
        mix_optimizer_class = MagicMock()
        mix_optimizer_class.__name__ = "Float16OptimizerWithFloat16Params"
        mix_optimizer.__class__ = mix_optimizer_class
        self.assertIsInstance(OptimizerMonFactory.create_optimizer_mon(mix_optimizer),
                              MegatronMixPrecisionOptimizerMon)
        dis_optimizer = MagicMock()
        dis_optimizer_class = MagicMock()
        dis_optimizer_class.__name__ = "DistributedOptimizer"
        dis_optimizer.__class__ = dis_optimizer_class
        self.assertIsInstance(OptimizerMonFactory.create_optimizer_mon(dis_optimizer),
                              MegatronDistributedOptimizerMon)
        fp32_optimizer = MagicMock()
        fp32_optimizer_class = MagicMock()
        fp32_optimizer_class.__name__ = "FP32Optimizer"
        fp32_optimizer.__class__ = fp32_optimizer_class
        self.assertIsInstance(OptimizerMonFactory.create_optimizer_mon(fp32_optimizer),
                              OptimizerMon)
        chained_optimizer = MagicMock()
        chained_optimizer_class = MagicMock()
        chained_optimizer_class.__name__ = "ChainedOptimizer"
        chained_optimizer.__class__ = chained_optimizer_class
        chained_optimizer.chained_optimizers = [mix_optimizer, mix_optimizer]
        self.assertIsInstance(OptimizerMonFactory.create_optimizer_mon(chained_optimizer),
                              MegatronChainedMixPrecisionOptimizerMon)
        chained_optimizer.chained_optimizers = [dis_optimizer, dis_optimizer]
        self.assertIsInstance(OptimizerMonFactory.create_optimizer_mon(chained_optimizer),
                              MegatronChainedDistributedOptimizerMon)
        deepspeed_optimizer = MagicMock()
        deepspeed_optimizer_class = MagicMock()
        deepspeed_optimizer_class.__name__ = "BF16_Optimizer"
        deepspeed_optimizer.__class__ = deepspeed_optimizer_class
        self.assertIsInstance(OptimizerMonFactory.create_optimizer_mon(deepspeed_optimizer),
                              DeepSpeedZeroOptimizerStage0Mon)
        deepspeed_optimizer_class.__name__ = "DeepSpeedZeroOptimizer"
        self.assertIsInstance(OptimizerMonFactory.create_optimizer_mon(deepspeed_optimizer),
                              DeepSpeedZeroOptimizerStage1or2Mon)
        deepspeed_optimizer_class.__name__ = "DeepSpeedZeroOptimizer_Stage3"
        self.assertIsInstance(OptimizerMonFactory.create_optimizer_mon(deepspeed_optimizer),
                              DeepSpeedZeroOptimizerStage3Mon)
        # 测试未知的优化器类型，应该返回OptimizerMon
        unknown_optimizer = MagicMock()
        unknown_optimizer_class = MagicMock()
        unknown_optimizer_class.__name__ = "unknown"
        unknown_optimizer.__class__ = unknown_optimizer_class
        self.assertIsInstance(OptimizerMonFactory.create_optimizer_mon(unknown_optimizer), OptimizerMon)


if __name__ == '__main__':
    unittest.main()