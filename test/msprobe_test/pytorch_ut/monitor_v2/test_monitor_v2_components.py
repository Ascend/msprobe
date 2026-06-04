import sys
import types
import unittest
from collections import defaultdict
from unittest.mock import MagicMock, patch

import torch

from msprobe.core.common.const import Const, MonitorConst
from msprobe.core.common.framework_adapter import FmkAdp
from msprobe.core.monitor_v2.cc import (
    _BaseCCMonitorV2,
    _MSCCContextV2,
    _PTCCContextV2,
)
from msprobe.core.monitor_v2.module import ModuleMonitorV2
from msprobe.core.monitor_v2.optimizer import OptimizerMonitorV2
from msprobe.core.monitor_v2.param import ParamMonitorV2


class DummyHandle:
    def __init__(self):
        self.removed = False

    def remove(self):
        self.removed = True


class DummyModuleMonitor(ModuleMonitorV2):
    def _register_backward_hook(self, module, module_name):
        return module.register_full_backward_hook(self._build_backward_hook(module_name))

    def _compute_metrics(self, tag2tensor):
        return {
            tag: {"mean": tensor.float().mean()}
            for tag, tensor in tag2tensor.items()
        }


class DummyOptimizerMonitor(OptimizerMonitorV2):
    def __init__(self):
        super().__init__()
        self.next_mv = ({}, {})

    def _build_backend(self, optimizer):
        return "backend"

    def _get_mv(self):
        return self.next_mv

    def _compute_metrics(self, tag2tensor):
        return {
            tag: {"norm": tensor.float().sum()}
            for tag, tensor in tag2tensor.items()
        }


class DummyParamMonitor(ParamMonitorV2):
    def _compute_metrics(self, tag2tensor):
        return {
            tag: {"max": tensor.float().max()}
            for tag, tensor in tag2tensor.items()
        }


class FakeApiRegister:
    def __init__(self, fail_restore=False):
        self.initialized = []
        self.redirected = False
        self.restored = False
        self.fail_restore = fail_restore

    def initialize_hook(self, pre_hooks, post_hooks):
        self.initialized.append((pre_hooks, post_hooks))
        return [DummyHandle()]

    def redirect_api(self):
        self.redirected = True

    def restore_api(self):
        if self.fail_restore:
            raise RuntimeError("restore failed")
        self.restored = True


class FakeCCContext:
    def __init__(self):
        self.data = {}
        self.aggregate_calls = 0
        self.reset_calls = 0

    def aggregate(self):
        self.aggregate_calls += 1

    def reset(self):
        self.reset_calls += 1
        self.data = {}


class DummyCCMonitor(_BaseCCMonitorV2):
    def __init__(self, env_ready=True, fail_restore=False):
        super().__init__()
        self.env_ready = env_ready
        self.api_register = FakeApiRegister(fail_restore=fail_restore)
        self.wrap_calls = 0

    def _create_context(self):
        return FakeCCContext()

    def _is_env_ready(self):
        return self.env_ready

    def _get_wrap_module(self):
        self.wrap_calls += 1

        def create_hooks(context, monitor):
            return ["pre"], ["post"]

        return self.api_register, create_hooks

    def _restore_api(self):
        self.api_register.restore_api()


class TestMonitorV2Components(unittest.TestCase):
    def setUp(self):
        self.prev_fmk = FmkAdp.fmk
        FmkAdp.set_fmk(Const.PT_FRAMEWORK)

    def tearDown(self):
        FmkAdp.set_fmk(self.prev_fmk)

    def test_module_monitor_when_forward_and_backward_hooks_run_then_pass(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2))
        mon = DummyModuleMonitor()
        mon.set_config({"targets": ["0"], "ops": ["mean"], "eps": 1e-6})
        mon.start(model=model)
        self.assertEqual(len(mon._hooks), 3)

        pre_hook = mon._build_forward_pre_hook("0")
        pre_hook(model[0], (torch.ones(2),), {"bias": torch.zeros(2)})
        fwd_hook = mon._build_forward_hook("0:layer")
        fwd_hook(model[0], None, torch.ones(2))
        bwd_hook = mon._build_backward_hook("0:layer")
        bwd_hook(model[0], (torch.ones(2),), (torch.zeros(2),))

        out = mon.collect()
        self.assertGreaterEqual(len(out["rows"]), 4)
        self.assertTrue(all("mean" in row["stats"] for row in out["rows"]))
        self.assertTrue(all("vpp_stage" in row for row in out["rows"]))

        mon.stop()
        self.assertEqual(mon._hooks, [])
        self.assertIsNone(mon._model)

    def test_module_monitor_when_helper_paths_and_invalid_model_are_used_then_pass(self):
        mon = DummyModuleMonitor()
        with self.assertRaises(ValueError):
            mon.start(model=None)

        register = MagicMock(side_effect=[TypeError("no kwargs"), DummyHandle()])
        handle = mon._try_register_hook(register, lambda *args: None)
        self.assertIsInstance(handle, DummyHandle)

        tagged, next_idx = mon._collect_tagged_tensors("layer", "input", torch.ones(1), start_idx=3)
        self.assertEqual(next_idx, 4)
        self.assertEqual(list(tagged.keys()), ["layer.input.3"])

        tagged, next_idx = mon._collect_tagged_tensors("layer", "input", [torch.ones(1), "skip"], start_idx=0)
        self.assertEqual(next_idx, 1)
        self.assertEqual(list(tagged.keys()), ["layer.input.0"])

    def test_optimizer_monitor_when_mv_data_exists_then_pass(self):
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        mon = DummyOptimizerMonitor()
        mon.set_config({"ops": ["norm"], "mv_distribution": True})
        mon.start(model=model, optimizer=optimizer)
        self.assertEqual(mon._backend, "backend")
        self.assertTrue(mon._param2name)

        mon.next_mv = (
            {"0:layer.weight": torch.tensor([1.0, 2.0])},
            {"layer.bias": torch.tensor([3.0])},
        )
        out = mon.collect()
        self.assertEqual(len(out["rows"]), 2)
        self.assertEqual({row["scope"] for row in out["rows"]}, {"exp_avg", "exp_avg_sq"})

        mon.stop()
        self.assertEqual(mon._param2name, {})
        self.assertEqual(mon._rows, [])

    def test_optimizer_monitor_when_distribution_is_disabled_or_start_is_invalid_then_pass(self):
        mon = DummyOptimizerMonitor()
        mon.set_config({"mv_distribution": False})
        self.assertIsNone(mon.collect())
        with self.assertRaises(ValueError):
            mon.start(model=None, optimizer=None)

    def test_param_monitor_when_step_hooks_are_registered_then_pass(self):
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        mon = DummyParamMonitor()
        mon.set_config({"param_distribution": True, "ops": ["max"]})
        mon.start(model=model, optimizer=optimizer)
        self.assertEqual(len(mon._step_hook_handles), 2)

        mon._on_pre_step(optimizer, (), {})
        mon._on_post_step(optimizer, (), {})
        out = mon.collect()
        self.assertEqual(len(out["rows"]), 4)
        self.assertEqual({row["scope"] for row in out["rows"]}, {MonitorConst.PRE_PARAM, MonitorConst.POST_PARAM})

        mon.stop()
        self.assertEqual(mon._step_hook_handles, [])
        self.assertEqual(mon._param2name, {})
        self.assertFalse(mon._step_patched)

    def test_param_monitor_when_optimizer_lacks_step_hooks_then_pass(self):
        class BareOptimizer:
            def __init__(self):
                self.step_calls = 0

            def step(self):
                self.step_calls += 1
                return "ok"

        param = torch.nn.Parameter(torch.tensor([1.0]))
        model = torch.nn.Linear(1, 1)
        mon = DummyParamMonitor()
        mon.set_config({"param_distribution": True})

        optimizer = BareOptimizer()
        with patch("msprobe.core.monitor_v2.param.build_param2name", return_value={param: "0:weight"}):
            mon.start(model=model, optimizer=optimizer)

        result = optimizer.step()
        self.assertEqual(result, "ok")
        out = mon.collect()
        self.assertEqual(len(out["rows"]), 2)
        mon.stop()

    def test_cc_monitor_when_context_and_lifecycle_flow_run_then_pass(self):
        pt_wrap = types.SimpleNamespace(op_aggregate=lambda op, tensorlist: f"{op}:{len(tensorlist)}")
        ms_wrap = types.SimpleNamespace(op_aggregate=lambda op, tensorlist: f"ms-{op}:{len(tensorlist)}")
        with patch.dict(sys.modules, {
            "msprobe.pytorch.monitor.distributed.wrap_distributed": pt_wrap,
            "msprobe.mindspore.monitor.distributed.wrap_distributed": ms_wrap,
        }):
            pt_ctx = _PTCCContextV2()
            pt_ctx.data = {"tag": {"min": [1, 2]}}
            pt_ctx.aggregate()
            self.assertEqual(pt_ctx.data["tag"]["min"], "min:2")
            pt_ctx.reset()
            self.assertEqual(pt_ctx.data, {})

            ms_ctx = _MSCCContextV2()
            ms_ctx.data = {"tag": {"max": [1]}}
            ms_ctx.aggregate()
            self.assertEqual(ms_ctx.data["tag"]["max"], "ms-max:1")

        mon = DummyCCMonitor(env_ready=False)
        mon.start()
        self.assertFalse(mon._patched)

        mon = DummyCCMonitor(env_ready=True)
        mon.set_config(
            {
                "cc_distribution": {
                    "cc_codeline": ["l1"],
                    "cc_log_only": True,
                    "cc_pre_hook": True,
                    "module_ranks": [0, 1],
                }
            }
        )
        mon.start()
        self.assertTrue(mon._patched)
        self.assertTrue(mon.api_register.redirected)
        self.assertEqual(mon.cc_codeline, ["l1"])
        self.assertTrue(mon.cc_log_only)
        self.assertTrue(mon.cc_pre_hook)
        self.assertEqual(mon.module_rank_list, [0, 1])

        ctx = mon.cc_context["rank0"]
        ctx.data = {"tag": {"min": 1.0}}
        out = mon.collect()
        self.assertEqual(out["rows"][0]["module_name"], "tag")
        self.assertEqual(out["rows"][0]["scope"], "comm")

        mon.stop()
        self.assertFalse(mon._patched)
        self.assertEqual(mon._rows, [])
        self.assertTrue(mon.api_register.restored)

    @patch("msprobe.core.monitor_v2.cc.logger.warning")
    def test_stop_when_cc_monitor_restore_fails_then_pass(self, mock_warning):
        mon = DummyCCMonitor(env_ready=True, fail_restore=True)
        mon.start()
        mon.stop()
        self.assertTrue(mock_warning.called)
