import unittest
import sys
import types
from unittest.mock import MagicMock, patch

import torch

from msprobe.core.common.const import Const
from msprobe.core.common.framework_adapter import FmkAdp
from msprobe.core.monitor_v2.weight_grad import (
    PyTorchWeightGradMonitorV2,
    WeightGradMonitorV2,
)


class DummyWeightGradMonitor(WeightGradMonitorV2):
    def _compute_metrics(self, tag2tensor):
        return {
            tag: {"mean": tensor.float().mean()}
            for tag, tensor in tag2tensor.items()
        }


class OptimizerWithStep:
    def __init__(self):
        self.step_calls = 0

    def step(self):
        self.step_calls += 1
        return "ok"


class TestWeightGradMonitorV2(unittest.TestCase):
    def setUp(self):
        self.prev_fmk = FmkAdp.fmk
        FmkAdp.set_fmk(Const.PT_FRAMEWORK)

    def tearDown(self):
        FmkAdp.set_fmk(self.prev_fmk)

    def test_start_when_weight_grad_monitor_records_micro_steps_and_reduced_grads_then_pass(self):
        model = torch.nn.Linear(2, 2)
        optimizer = OptimizerWithStep()
        mon = DummyWeightGradMonitor()
        mon.set_config({"ops": ["mean"], "micro_batch_number": 2, "monitor_mbs_grad": False})
        mon.start(model=model, optimizer=optimizer)
        self.assertTrue(mon._patched)
        self.assertEqual(len(mon._grad_hooks), len(mon._param2name))

        param, name = next(iter(mon._param2name.items()))
        grad = torch.ones_like(param)
        mon._on_param_grad(param, name, grad)
        self.assertEqual(mon._rows, [])

        mon._on_param_grad(param, name, grad * 2)
        out = mon.collect()
        self.assertEqual(len(out["rows"]), 1)
        self.assertEqual(out["rows"][0]["micro_step"], 2)

        param.grad = grad * 3
        result = optimizer.step()
        self.assertEqual(result, "ok")
        reduced = mon.collect()
        self.assertEqual(len(reduced["rows"]), 1)
        self.assertEqual(reduced["rows"][0]["scope"], "reduced")

        mon.stop()
        self.assertFalse(mon._patched)
        self.assertEqual(mon._grad_hooks, [])

    def test_set_config_when_weight_grad_monitor_parses_monitor_mbs_grad_then_pass(self):
        model = torch.nn.Linear(1, 1)
        optimizer = OptimizerWithStep()
        mon = DummyWeightGradMonitor()
        mon.set_config({"eps": "bad", "monitor_mbs_grad": "bad"})
        self.assertEqual(mon._eps, 1e-8)
        self.assertFalse(mon.monitor_mbs_grad)

        mon.set_config({"micro_batch_number": 3, "monitor_mbs_grad": True})
        mon.start(model=model, optimizer=optimizer, grad_acc_steps=3)
        param, name = next(iter(mon._param2name.items()))
        mon._on_param_grad(param, name, torch.ones_like(param))
        mon._on_param_grad(param, name, torch.ones_like(param) * 2)
        rows = mon.collect()["rows"]
        self.assertEqual([row["micro_step"] for row in rows], [1, 2])
        mon.stop()

    def test_start_when_weight_grad_monitor_inputs_are_invalid_then_fail(self):
        mon = DummyWeightGradMonitor()
        with self.assertRaises(ValueError):
            mon.start(model=None, optimizer=None)

        self.assertEqual(mon._resolve_micro_batch_number(), 1)
        mon.set_context(micro_batch_number="x", grad_acc_steps=4)
        self.assertEqual(mon._resolve_micro_batch_number(), 4)


class TestPyTorchWeightGradMonitorV2(unittest.TestCase):
    def setUp(self):
        self.prev_fmk = FmkAdp.fmk
        FmkAdp.set_fmk(Const.PT_FRAMEWORK)

    def tearDown(self):
        FmkAdp.set_fmk(self.prev_fmk)

    def test_detect_fsdp_when_grad_sources_and_micro_steps_are_valid_then_pass(self):
        mon = PyTorchWeightGradMonitorV2()

        fake_dtensor_param = type("DTensor", (), {})()
        fake_model = type(
            "FakeModel",
            (),
            {"named_parameters": lambda self: [("w", fake_dtensor_param)]},
        )()
        self.assertTrue(mon._detect_fsdp(fake_model))
        self.assertFalse(mon._detect_fsdp(None))

        param_holder = type("ParamHolder", (), {})()
        param_holder.main_grad = torch.tensor([1.0], requires_grad=True)
        with patch.dict(
            sys.modules,
            {"msprobe.pytorch.common.utils": types.SimpleNamespace(is_float8_tensor=lambda tensor: False)},
        ):
            grad = mon._fetch_param_grad(param_holder)
        self.assertTrue(torch.equal(grad, torch.tensor([1.0])))
        self.assertIsNot(grad, param_holder.main_grad)

        p = torch.nn.Parameter(torch.tensor([2.0]))
        p.grad = torch.tensor([4.0])
        mon._model = object()
        mon._param2name = {p: "_fsdp_wrapped_module.layer.weight"}
        mon._micro_batch_number = 2
        mon.monitor_mbs_grad = False

        mon._capture_fsdp_pre_grads()
        self.assertEqual(mon._rows, [])
        mon._capture_fsdp_pre_grads()
        rows = mon.collect()["rows"]
        self.assertEqual(rows[0]["module_name"], "layer.weight")
        self.assertEqual(rows[0]["micro_step"], 2)

    def test_patch_autograd_backward_when_backward_is_wrapped_then_pass(self):
        mon = PyTorchWeightGradMonitorV2()
        mon._capture_fsdp_pre_grads = MagicMock()
        mon._patch_autograd_backward()

        x = torch.tensor(2.0, requires_grad=True)
        y = x * 3
        torch.autograd.backward(y)
        self.assertTrue(mon._capture_fsdp_pre_grads.called)

        mon._restore_autograd_backward()
        self.assertFalse(mon._fsdp_backward_patched)
