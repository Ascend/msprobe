import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import mindspore as ms

from msprobe.mindspore.monitor.module_hook import (
    TrainerMon,
    ModuleHookContext,
    OptimizerContext,
    GradContext,
    FeatureHookContext,
    is_recording_module,
)
from msprobe.core.common.const import MonitorConst


class TestTrainerMon(unittest.TestCase):
    def setUp(self):
        base_dir = os.path.dirname(os.path.realpath(__file__))
        self.config_path = os.path.join(base_dir, "config/test_config.json")
        self.trainer = TrainerMon(self.config_path)

    def test_init_when_config_valid_then_pass(self):
        self.assertEqual(self.trainer.config_file_path, self.config_path)
        self.assertEqual(self.trainer.start_step, 0)
        self.assertEqual(self.trainer.collect_times, 2)
        self.assertTrue(self.trainer.monitoring)

    @patch("os.getenv", return_value="/custom/output")
    def test_get_output_base_dir_when_env_set_then_pass(self, mock_getenv):
        from msprobe.mindspore.monitor.module_hook import get_output_base_dir
        self.assertEqual(get_output_base_dir(), "/custom/output")

    @patch("os.path.getmtime", return_value=123456)
    @patch("json.load", return_value={})
    def test_dynamic_monitor_when_config_updated_then_pass(self, mock_load, mock_mtime):
        self.trainer.dynamic_enable = True
        self.trainer.config_timestamp = 0
        self.trainer.monitoring = False
        optimizer = MagicMock()
        self.trainer.optimizer_context[optimizer] = OptimizerContext()
        self.trainer.dynamic_monitor(optimizer)
        self.assertEqual(self.trainer.config_timestamp, 123456)

    def test_is_target_rank_when_rank_in_list_then_pass(self):
        self.trainer.module_rank_list = [0, 1]
        self.trainer.rank = 0
        self.assertTrue(self.trainer.is_target_rank())

    def test_is_target_rank_when_rank_not_in_list_then_pass(self):
        self.trainer.module_rank_list = [1, 2]
        self.trainer.rank = 0
        self.assertFalse(self.trainer.is_target_rank())

    def test_hook_optimizer_when_valid_optimizer_then_pass(self):
        optimizer = MagicMock()
        self.trainer.optimizer_mon = MagicMock(fetch_grad=MagicMock(return_value={}))
        self.trainer.hook_optimizer(optimizer)
        self.assertEqual(len(self.trainer.pre_step_hooks), 1)
        self.assertEqual(len(self.trainer.post_step_hooks), 1)

    def test_write_xy_tb_when_activation_present_then_pass(self):
        context = ModuleHookContext("test_module")
        context.actv = {"key": ms.Tensor(1.0)}
        self.trainer.module_fwd_hook_context_by_module[MagicMock()] = context
        self.trainer.summary_writer.write_metrics = MagicMock()
        self.trainer.write_xy_tb(1)
        self.trainer.summary_writer.write_metrics.assert_called()

    def test_write_grad_tb_when_grad_data_then_pass(self):
        self.trainer.grad_context.pre = {"grad1": ms.Tensor(0.5)}
        self.trainer.grad_context.post = {"grad2": ms.Tensor(0.8)}
        self.trainer.summary_writer.write_metrics = MagicMock()
        self.trainer.write_grad_tb(1)
        self.trainer.summary_writer.write_metrics.assert_called()

    def test_write_mv_tb_when_mv_data_then_pass(self):
        context = OptimizerContext()
        context.exp_avg_metric = {"m1": ms.Tensor(0.1)}
        context.exp_avg_sq_metric = {"v1": ms.Tensor(0.2)}
        self.trainer.summary_writer.write_metrics = MagicMock()
        self.trainer.write_mv_tb(context)
        self.trainer.summary_writer.write_metrics.assert_called()

    def test_write_param_tb_when_param_data_then_pass(self):
        context = OptimizerContext()
        context.param_metric = {"param_pre": ms.Tensor(1.0), "param_post": ms.Tensor(2.0)}
        self.trainer.summary_writer.write_metrics = MagicMock()
        self.trainer.write_param_tb(context)
        self.trainer.summary_writer.write_metrics.assert_called()

    def test_is_recording_module_when_target_matches_then_pass(self):
        result = is_recording_module("layer1.block", ["0:layer1.block"], "0:")
        self.assertEqual(result, "0:layer1.block")

    def test_is_recording_module_when_targets_empty_then_returns_empty(self):
        result = is_recording_module("layer", [], "0:")
        self.assertEqual(result, "")

    def test_feature_hook_context_reset_when_called_then_pass(self):
        ctx = FeatureHookContext("m")
        ctx.attention_feature["k"] = 1
        ctx.linear_feature["k"] = 2
        ctx.reset()
        self.assertEqual(len(ctx.attention_feature), 0)
        self.assertEqual(len(ctx.linear_feature), 0)

    @patch("msprobe.mindspore.monitor.module_hook.get_rank", return_value=0)
    @patch("msprobe.mindspore.monitor.module_hook.get_output_base_dir", return_value="/tmp/out")
    @patch("msprobe.mindspore.monitor.module_hook.validate_config", return_value=None)
    @patch("msprobe.mindspore.monitor.module_hook.load_json")
    @patch.dict("msprobe.mindspore.monitor.module_hook.FORMAT_MAPPING",
                {MonitorConst.CSV: MagicMock(return_value=MagicMock())})
    def test_init_when_append_output_set_then_pass(self, mock_load, *_):
        mock_load.return_value = {
            "start_step": 0,
            "collect_times": 1,
            "targets": {},
            "append_output": ["tag1", "tag2"],
        }
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            config_path = tmp.name
        with patch("msprobe.mindspore.monitor.module_hook.get_target_output_dir",
                   return_value={"0": "/tmp/out/rank0"}):
            trainer = TrainerMon(config_path)
            self.assertEqual(trainer.tensorboard_dir, "/tmp/out/rank0")

    @patch("msprobe.mindspore.monitor.module_hook.get_rank", side_effect=Exception("rank error"))
    @patch("msprobe.mindspore.monitor.module_hook.validate_config", return_value=None)
    @patch("msprobe.mindspore.monitor.module_hook.load_json", return_value={"start_step": 0, "collect_times": 1})
    def test_init_when_rank_raises_then_pass(self, *_):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            config_path = tmp.name
        trainer = TrainerMon(config_path)
        self.assertEqual(trainer.rank, 0)

    def test_get_linear_hook_target_when_weight_variants_then_pass(self):
        class WithWeight:
            def __init__(self):
                self.weight = ms.Tensor([[1, 2], [3, 4]])

        class WithWG:
            def __init__(self):
                self.wg = ms.Tensor([[1, 2], [3, 4]])

        emb = ms.nn.Embedding(2, 2)
        self.assertEqual(TrainerMon.get_linear_hook_target(emb), "")
        self.assertEqual(TrainerMon.get_linear_hook_target(WithWeight()), "weight")
        self.assertEqual(TrainerMon.get_linear_hook_target(WithWG()), "wg")

    @patch("msprobe.mindspore.monitor.module_hook.validate_config", return_value=None)
    @patch("msprobe.mindspore.monitor.module_hook.load_json")
    def test_set_config_when_cc_enabled_and_format_unknown_then_pass(self, mock_load, *_):
        mock_load.return_value = {
            "start_step": 0,
            "collect_times": 1,
            "targets": {},
            "format": "unknown",
            "cc_distribution": {"enable": True, "cc_codeline": ["l1"], "cc_log_only": True, "cc_pre_hook": True},
        }
        trainer = TrainerMon(self.config_path)
        self.assertTrue(trainer.cc_log_only)
        self.assertEqual(trainer.format, MonitorConst.CSV)

    def test_common_info_when_flags_disabled_then_pass(self):
        with patch.object(self.trainer, "xy_distribution", False), \
                patch.object(self.trainer, "mv_distribution", False), \
                patch("msprobe.mindspore.monitor.module_hook.logger.info") as mock_info:
            self.trainer.common_info()
            calls = [call for call in mock_info.mock_calls if "> module input/output" in str(call)]
            self.assertTrue(calls)

    def test_hook_step_final_when_stack_and_metrics_then_pass(self):
        optimizer = MagicMock()
        optimizer.__class__.construct = lambda self, *a, **k: None
        self.trainer.summary_writer.write_metrics = MagicMock()
        self.trainer.handles["stack"] = [MagicMock()]
        self.trainer.stack_info = True
        self.trainer.monitoring = True
        self.trainer.optimizer_context[optimizer] = OptimizerContext()
        self.trainer.optimizer_context[optimizer].metric_dict = {"k": "v"}
        self.trainer.hook_step_final(optimizer)
        optimizer.construct()
        self.trainer.summary_writer.write_metrics.assert_called()

    @patch("msprobe.mindspore.monitor.module_hook.api_register.initialize_hook", return_value=["h"])
    @patch("msprobe.mindspore.monitor.module_hook.api_register.redirect_api")
    @patch("msprobe.mindspore.monitor.module_hook.create_hooks", return_value=([], [], []))
    def test_register_hooks_when_cc_enabled_then_pass(self, mock_create, mock_redirect, *_):
        self.trainer.cc_distribution = {"enable": True}
        self.trainer.model = []
        self.trainer.targets = {}
        self.trainer.register_hooks(MagicMock())
        mock_create.assert_called()
        mock_redirect.assert_called()

    @patch.object(TrainerMon, "_save_module_struct")
    def test_hook_optimizer_when_print_struct_then_pass(self, mock_save_struct):
        optimizer = MagicMock()
        self.trainer.print_struct = True
        self.trainer.struct_printed = False
        self.trainer.cc_log_only = True
        self.trainer.module_struct = {"m": {"s": 1}}
        cc = MagicMock()
        cc.data = {"cc": {"op": 1}}
        self.trainer.cc_context["a"] = cc
        self.trainer.optimizer_mon = MagicMock(fetch_grad=MagicMock(return_value={}))
        self.trainer.hook_optimizer(optimizer)
        self.trainer.pre_step_hooks[0](optimizer)
        mock_save_struct.assert_called_once()
        cc.aggregate.assert_called_once()
        self.assertEqual(self.trainer.optimizer_context[optimizer].metric_dict, {"cc": {"op": 1}})

    def test_generate_param_map_when_param_present_then_pass(self):
        self.trainer.is_mindtorch = True
        self.trainer.param2name = {"p": "name1"}
        metrics = self.trainer.generate_param_map("tag", {"name1": 1})
        self.assertIn(f"name1/rank{self.trainer.rank}/tag", metrics)

    def test_generate_param_metrics_when_distribution_enabled_then_pass(self):
        param = MagicMock()
        param.numel.return_value = 1
        self.trainer.param_distribution = True
        self.trainer.name2param = {"p": param}
        self.trainer.name2tag = {"p": {MonitorConst.PRE_PARAM: "tag"}}
        with patch("msprobe.mindspore.monitor.module_hook.get_metrics") as mock_get:
            ctx = OptimizerContext()
            self.trainer.generate_param_metrics(ctx)
            mock_get.assert_called()

    @patch("msprobe.mindspore.monitor.module_hook.is_valid_instance", return_value=False)
    def test_get_mv_for_ms_when_optimizer_invalid_then_pass(self, *_):
        self.trainer.mv_distribution = True
        m, v = self.trainer.get_mv_for_ms(MagicMock())
        self.assertEqual(m, {})
        self.assertEqual(v, {})

    @patch("msprobe.mindspore.monitor.module_hook.pd.DataFrame")
    @patch("msprobe.mindspore.monitor.module_hook.write_df_to_csv")
    @patch("msprobe.mindspore.monitor.module_hook.os.path.exists", return_value=False)
    def test_write_stack_info_when_file_absent_then_pass(self, _, mock_write_csv, mock_df):
        ctx = ModuleHookContext("m")
        ctx.stack = "stack"
        self.trainer.module_fwd_hook_context_by_module = {MagicMock(): ctx}
        self.trainer.write_stack_info()
        mock_write_csv.assert_called()

    def test_write_metrics_if_not_empty_when_features_present_then_pass(self):
        self.trainer.summary_writer.write_metrics = MagicMock()
        features = {"k": {"entropy": 1}}
        self.trainer.write_metrics_if_not_empty(features, ["entropy"], 0, "attention_hook")
        self.trainer.summary_writer.write_metrics.assert_called()
        self.trainer.summary_writer.write_metrics.reset_mock()
        self.trainer.write_metrics_if_not_empty({}, ["entropy"], 0, "attention_hook")
        self.trainer.summary_writer.write_metrics.assert_not_called()

    def test_write_features_tb_when_features_exist_then_pass(self):
        self.trainer.recording_l2_features = True
        ctx = FeatureHookContext("m")
        ctx.attention_feature = {"k": {"entropy": 1}}
        self.trainer.feature_hook_context_by_module = {MagicMock(): ctx}
        self.trainer.summary_writer.write_metrics = MagicMock()
        self.trainer.write_features_tb(1)
        self.trainer.summary_writer.write_metrics.assert_called()

    @patch("msprobe.mindspore.monitor.module_hook.create_directory")
    @patch("msprobe.mindspore.monitor.module_hook.save_json")
    def test_save_module_struct_when_called_then_pass(self, mock_save, _):
        self.trainer._save_module_struct()
        mock_save.assert_called()

    @patch("msprobe.mindspore.monitor.module_hook.get_submodules")
    @patch("msprobe.mindspore.monitor.module_hook.is_valid_instance", return_value=True)
    def test_hook_module_when_targets_match_then_pass(self, _, mock_get_submodules):
        class DummyModule:
            def __init__(self):
                self.training = True
                self.last_hook = None

            def register_forward_hook(self, fn, with_kwargs=False):
                self.last_hook = fn
                handle = MagicMock()
                handle.remove = MagicMock()
                handle.fn = fn
                return handle

            def register_backward_hook(self, fn):
                self.last_bwd = fn
                handle = MagicMock()
                handle.fn = fn
                handle.remove = MagicMock()
                return handle

        dummy = DummyModule()
        mock_get_submodules.return_value = [("layer1", dummy)]
        self.trainer.xy_distribution = True
        self.trainer.targets = {"layer1": {}}
        hooked = self.trainer._hook_module(["layer1"], {}, dummy, "")
        self.assertGreaterEqual(hooked, 1)
        try:
            dummy.last_hook(dummy, (ms.Tensor(1.0),), {}, (ms.Tensor(2.0),))
        except TypeError:
            dummy.last_hook(dummy, (ms.Tensor(1.0),), (ms.Tensor(2.0),))
        dummy.last_bwd(dummy, (ms.Tensor(3.0),), (ms.Tensor(4.0),))

    @patch("msprobe.mindspore.monitor.module_hook.get_sr_metric")
    @patch("msprobe.mindspore.monitor.module_hook.get_entropy_metric")
    @patch("msprobe.mindspore.monitor.module_hook.cal_qkt", return_value=ms.Tensor([1.0]))
    @patch("msprobe.mindspore.monitor.module_hook.get_submodules")
    @patch("msprobe.mindspore.monitor.module_hook.is_valid_instance", return_value=True)
    @patch.object(TrainerMon, "get_linear_hook_target", return_value="weight")
    def test_hook_module_when_l2_features_enabled_then_pass(
            self, mock_get_linear_hook_target, mock_is_valid_instance, mock_get_submodules,
            mock_cal_qkt, mock_entropy, mock_sr):
        class DummyModule:
            def __init__(self):
                self.training = True
                # mimic a parameter with .data attribute used in hook
                weight_tensor = ms.Tensor([[1.0, 2.0], [3.0, 4.0]])
                self.weight = MagicMock()
                self.weight.data = weight_tensor
                self.forward_hooks = []

            def register_forward_hook(self, fn, with_kwargs=False):
                handle = MagicMock()
                handle.fn = fn
                handle.remove = MagicMock()
                self.forward_hooks.append(fn)
                return handle

        dummy = DummyModule()
        mock_get_submodules.return_value = [("layer1", dummy)]
        self.trainer.xy_distribution = False
        self.trainer.recording_l2_features = True
        self.trainer.targets = {"layer1": {}}
        l2_targets = {"attention_hook": ["layer1"], "linear_hook": ["layer1"]}
        hooked = self.trainer._hook_module(["layer1"], l2_targets, dummy, "")
        self.assertGreaterEqual(hooked, 2)
        for fn in dummy.forward_hooks:
            try:
                fn(dummy, (ms.Tensor([[1.0]]), ms.Tensor([[2.0]])), {}, ms.Tensor([[3.0]]))
            except TypeError:
                fn(dummy, (ms.Tensor([[1.0]]), ms.Tensor([[2.0]])), ms.Tensor([[3.0]]))
        mock_cal_qkt.assert_called()
        mock_entropy.assert_called()
        mock_sr.assert_called()

    def test_generate_mv_metrics_when_enabled_then_pass(self):
        ctx = OptimizerContext()
        self.trainer.mv_distribution = True
        self.trainer.is_mindtorch = True
        self.trainer.param2name = {"p": "name"}
        ctx.param_exp_avg = {"name": ms.Tensor(1.0)}
        ctx.param_exp_avg_sq = {"name": ms.Tensor(2.0)}
        with patch("msprobe.mindspore.monitor.module_hook.get_metrics") as mock_get:
            self.trainer.generate_mv_metrics(ctx)
            self.assertTrue(mock_get.called)

    def test_build_tbtag_tensor_map_when_single_tensor_then_pass(self):
        tensor = ms.Tensor(1.0)
        result = self.trainer.build_tbtag_tensor_map("module", "", "actv", tensor)
        expected_key = f"module/rank{self.trainer.rank}/actv"
        self.assertIn(expected_key, result)
        self.assertEqual(self.trainer.param_name_call_id[expected_key], 0)

    def test_build_tbtag_tensor_map_when_multiple_tensors_then_pass(self):
        tensors = [ms.Tensor(1.0), ms.Tensor(2.0)]
        result = self.trainer.build_tbtag_tensor_map("module", "", "actv", tensors)
        expected_keys = {
            f"module_0/rank{self.trainer.rank}/actv",
            f"module_1/rank{self.trainer.rank}/actv"
        }
        self.assertEqual(set(result.keys()), expected_keys)
        self.assertEqual(len(self.trainer.param_name_call_id), 2)

    def test_register_param_call_id_when_multiple_calls_then_pass(self):
        self.trainer.register_param_call_id("hook_a", "key_a")
        self.trainer.register_param_call_id("hook_b", "key_b")
        self.assertEqual(self.trainer.param_name_call_id["key_a"], 0)
        self.assertEqual(self.trainer.param_name_call_id["key_b"], 1)
        self.assertEqual(self.trainer.call_id, 2)

    def test_is_target_param_when_name_matches_then_pass(self):
        class DummyParam:
            def __init__(self):
                self.requires_grad = True

        param = DummyParam()
        self.trainer.targets = {"layer": {}}
        result = self.trainer._is_target_param("layer.weight", param, "")
        self.assertTrue(result)
        self.assertTrue(hasattr(param, "zero_out_wgrad"))
        self.assertTrue(param.zero_out_wgrad)

    def test_is_target_param_when_name_not_match_then_pass(self):
        class DummyParam:
            def __init__(self):
                self.requires_grad = True

        param = DummyParam()
        self.trainer.targets = {"other": {}}
        result = self.trainer._is_target_param("layer.weight", param, "")
        self.assertFalse(result)
        self.assertFalse(hasattr(param, "zero_out_wgrad"))


class TestModuleHookContext(unittest.TestCase):
    def test_reset_when_called_then_pass(self):
        context = ModuleHookContext("test")
        context.actv = {"data": ms.Tensor(1.0)}
        context.actvgrad = [ms.Tensor(2.0)]
        context.reset()
        self.assertEqual(len(context.actv), 0)
        self.assertEqual(len(context.actvgrad), 0)


class TestOptimizerContext(unittest.TestCase):
    def test_reset_when_called_then_pass(self):
        context = OptimizerContext()
        context.param_mg_direction = {"p1": 0.5}
        context.param_adam_update = {"p1": ms.Tensor(0.1)}
        context.reset()
        self.assertEqual(len(context.param_mg_direction), 0)
        self.assertEqual(len(context.param_adam_update), 0)


class TestGradContext(unittest.TestCase):
    def test_reset_when_called_then_pass(self):
        context = GradContext()
        context.pre = {"g1": ms.Tensor(0.1)}
        context.post = {"g2": ms.Tensor(0.2)}
        context.reset()
        self.assertEqual(len(context.pre), 0)
        self.assertEqual(len(context.post), 0)


if __name__ == "__main__":
    unittest.main()
