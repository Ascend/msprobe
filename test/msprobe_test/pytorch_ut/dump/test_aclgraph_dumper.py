# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
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

import importlib
import importlib.util
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import torch


def _build_aclgraph_dumper_import_env():
    import msprobe

    fake_aclgraph_dump = types.ModuleType("msprobe.pytorch.aclgraph_dump")
    fake_aclgraph_dump.acl_save = MagicMock(side_effect=lambda tensor, path: tensor)
    fake_aclgraph_dump.acl_stat = MagicMock(side_effect=lambda tensor, tag: tensor)
    fake_aclgraph_dump.get_acl_stat_dict = MagicMock(return_value={})

    fake_torch_npu = types.ModuleType("torch_npu")
    fake_torch_npu.npu = types.SimpleNamespace(synchronize=MagicMock())

    pytorch_pkg_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "python", "msprobe", "pytorch")
    )
    fake_pytorch_pkg = types.ModuleType("msprobe.pytorch")
    fake_pytorch_pkg.__path__ = [pytorch_pkg_dir]
    modules_patcher = patch.dict(
        sys.modules,
        {
            "msprobe.pytorch": fake_pytorch_pkg,
            "msprobe.pytorch.aclgraph_dump": fake_aclgraph_dump,
            "torch_npu": fake_torch_npu,
        },
    )
    pytorch_attr_patcher = patch.object(msprobe, "pytorch", fake_pytorch_pkg)
    return modules_patcher, pytorch_attr_patcher, pytorch_pkg_dir, fake_aclgraph_dump, fake_torch_npu


def _load_aclgraph_dumper_module(pytorch_pkg_dir):
    sys.modules.pop("msprobe.pytorch.aclgraph_dumper", None)
    importlib.invalidate_caches()

    module_name = "msprobe.pytorch.aclgraph_dumper"
    module_path = os.path.join(pytorch_pkg_dir, "aclgraph_dumper.py")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 4)

    def forward(self, x):
        return self.linear(x)


class KwModel(torch.nn.Module):
    def forward(self, x, bias=None):
        if bias is None:
            return x + 1
        return x + bias


class OnlyRootModel(torch.nn.Module):
    def forward(self, x):
        return x.relu()


class SimpleIterable:
    def __init__(self, *values):
        self._values = values

    def __iter__(self):
        return iter(self._values)


class FakeSchema:
    def __init__(self, name):
        self.name = name


class FakeFunc:
    def __init__(self, func_text, result=None, schema_name=None, overloadname="default"):
        self._func_text = func_text
        self._result = result
        self._schema = FakeSchema(schema_name) if schema_name is not None else None
        self.overloadname = overloadname
        self.calls = []

    def __str__(self):
        return self._func_text

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self._result is not None:
            return self._result
        return args[0] if args else None


class TestAclGraphDumper(unittest.TestCase):
    def setUp(self):
        modules_patcher, pytorch_attr_patcher, pytorch_pkg_dir, aclgraph_dump_stub, torch_npu_stub = (
            _build_aclgraph_dumper_import_env()
        )
        self._modules_patcher = modules_patcher
        self._pytorch_attr_patcher = pytorch_attr_patcher
        self._modules_patcher.start()
        self._pytorch_attr_patcher.start()
        self.module = _load_aclgraph_dumper_module(pytorch_pkg_dir)
        self.AclGraphDumper = self.module.AclGraphDumper
        self.aclgraph_dump_stub = aclgraph_dump_stub
        self.torch_npu_stub = torch_npu_stub
        self.aclgraph_dump_stub.acl_stat.reset_mock(side_effect=False)
        self.aclgraph_dump_stub.acl_stat.side_effect = lambda tensor, tag: tensor
        self.aclgraph_dump_stub.get_acl_stat_dict.reset_mock(side_effect=False)
        self.aclgraph_dump_stub.get_acl_stat_dict.return_value = {}
        self.torch_npu_stub.npu.synchronize.reset_mock()

    def tearDown(self):
        sys.modules.pop("msprobe.pytorch.aclgraph_dumper", None)
        importlib.invalidate_caches()
        self._pytorch_attr_patcher.stop()
        self._modules_patcher.stop()

    def make_dumper(self, dump_path="./dump", keywords=None, level="mix", rank=None, rank_id=0):
        with patch.object(
            self.AclGraphDumper,
            "_load_msprobe_config",
            return_value=(dump_path, keywords or [], level, rank, 0),
        ), \
                patch.object(self.AclGraphDumper, "_validate_dump_path", return_value=dump_path), \
                patch.object(self.AclGraphDumper, "_resolve_rank_id", return_value=rank_id):
            return self.AclGraphDumper(config_path="./config.json")

    def test_iter_tensors_if_nested_values_then_pass(self):
        tensor_a = torch.randn(1)
        tensor_b = torch.randn(1)
        tensor_c = torch.randn(1)
        tensor_d = torch.randn(1)
        value = {
            "a": tensor_a,
            "b": (tensor_b, [tensor_c]),
            "c": SimpleIterable(tensor_d),
            "ignored": "text",
        }

        result = list(self.module._iter_tensors(value))

        self.assertEqual(
            [prefix for prefix, _ in result],
            ["a", "b.0", "b.1.0", "c.0"],
        )
        self.assertEqual([tensor for _, tensor in result], [tensor_a, tensor_b, tensor_c, tensor_d])
        self.assertEqual(list(self.module._iter_tensors("abc")), [])

    def test_is_collectable_tensor_if_tensor_variants_then_pass(self):
        self.assertTrue(self.module._is_collectable_tensor(torch.randn(2, 3)))
        self.assertFalse(self.module._is_collectable_tensor(torch.empty(2, device="meta")))
        self.assertFalse(self.module._is_collectable_tensor("not_a_tensor"))

    def test_load_msprobe_config_if_config_and_validations_then_pass(self):
        default_path = os.path.normpath(self.AclGraphDumper._default_config_path())
        self.assertTrue(default_path.endswith(os.path.join("msprobe", "config.json")))

        config = {
            "task": "statistics",
            "dump_path": "./dump_dir",
            "level": "L1",
            "statistics": {"list": ["linear"], "level": "mix"},
            "rank": [0],
        }
        with patch.object(self.AclGraphDumper, "_default_config_path", return_value="/tmp/default.json"), \
                patch.object(self.module, "check_and_get_real_path", return_value="/tmp/default.json") as mock_real_path, \
                patch.object(self.module, "load_json", return_value=config):
            dump_path, module_list, level, rank, seq_len = self.AclGraphDumper._load_msprobe_config(None)

        self.assertEqual((dump_path, module_list, level, rank, seq_len), ("./dump_dir", ["linear"], "mix", [0], 0))
        mock_real_path.assert_called_once()

        with self.assertRaises(TypeError):
            self.AclGraphDumper._load_msprobe_config(123)

        with patch.object(self.module, "check_and_get_real_path", return_value="/tmp/config.json"), \
                patch.object(self.module, "load_json", return_value=[]):
            with self.assertRaises(TypeError):
                self.AclGraphDumper._load_msprobe_config("./config.json")

        with patch.object(self.module, "check_and_get_real_path", return_value="/tmp/config.json"), \
                patch.object(self.module, "load_json", return_value={"task": "statistics", "statistics": []}):
            with self.assertRaises(TypeError):
                self.AclGraphDumper._load_msprobe_config("./config.json")

        with patch.object(self.module, "check_and_get_real_path", return_value="/tmp/config.json"), \
                patch.object(self.module, "load_json", return_value={"task": 1, "dump_path": "./x", "level": "L0"}):
            dump_path, module_list, level, rank, seq_len = self.AclGraphDumper._load_msprobe_config("./config.json")
        self.assertEqual((dump_path, module_list, level, rank, seq_len), ("./x", [], "L0", None, 0))

        with self.assertRaises(TypeError):
            self.AclGraphDumper._validate_dump_path(1)

        with patch.object(self.module, "check_and_get_real_path", return_value="/tmp/dump") as mock_real_path, \
                patch.object(self.module, "create_directory") as mock_create_dir:
            self.assertEqual(self.AclGraphDumper._validate_dump_path("./dump"), "/tmp/dump")
        mock_real_path.assert_called_once()
        mock_create_dir.assert_called_once_with("/tmp/dump")

        self.assertEqual(self.AclGraphDumper._validate_list(None), [])
        with self.assertRaises(TypeError):
            self.AclGraphDumper._validate_list("linear")
        with self.assertRaises(TypeError):
            self.AclGraphDumper._validate_list(["linear", 1])
        self.assertEqual(self.AclGraphDumper._validate_list(["linear"]), ["linear"])

        with self.assertRaises(TypeError):
            self.AclGraphDumper._validate_level(1)
        with self.assertRaises(ValueError):
            self.AclGraphDumper._validate_level("L2")
        self.assertEqual(self.AclGraphDumper._validate_level("mix"), "mix")

    def test_resolve_rank_id_if_distributed_paths_then_pass(self):
        with patch.object(self.module.torch, "distributed", None):
            self.assertIsNone(self.AclGraphDumper._resolve_rank_id())

        dist_unavailable = types.SimpleNamespace(is_available=lambda: False, is_initialized=lambda: True)
        with patch.object(self.module.torch, "distributed", dist_unavailable):
            self.assertIsNone(self.AclGraphDumper._resolve_rank_id())

        dist_uninitialized = types.SimpleNamespace(is_available=lambda: True, is_initialized=lambda: False)
        with patch.object(self.module.torch, "distributed", dist_uninitialized):
            self.assertIsNone(self.AclGraphDumper._resolve_rank_id())

        dist_failed = types.SimpleNamespace(
            is_available=lambda: True,
            is_initialized=lambda: True,
            get_rank=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        with patch.object(self.module.torch, "distributed", dist_failed):
            self.assertIsNone(self.AclGraphDumper._resolve_rank_id())

        dist_ok = types.SimpleNamespace(is_available=lambda: True, is_initialized=lambda: True, get_rank=lambda: "3")
        with patch.object(self.module.torch, "distributed", dist_ok):
            self.assertEqual(self.AclGraphDumper._resolve_rank_id(), 3)

    def test_scope_keyword_and_rank_helpers_if_helper_inputs_then_pass(self):
        dumper = self.make_dumper(keywords=["Linear"], level="mix", rank=[0], rank_id=0)
        self.assertEqual(dumper._module_scope(""), "__root__")
        self.assertEqual(dumper._module_scope_name("", "Model"), "Module.__root__.Model")
        self.assertTrue(dumper._match_list_keywords("module.linear", "other"))
        self.assertTrue(dumper._should_collect_module("linear", "Linear"))
        self.assertTrue(dumper._should_dump_current_rank())
        self.assertTrue(dumper._collect_module_enabled())
        self.assertTrue(dumper._collect_api_enabled())

        dumper.rank = [2]
        self.assertFalse(dumper._should_dump_current_rank())
        dumper.rank = []
        dumper.rank_id = None
        self.assertTrue(dumper._should_dump_current_rank())

        api_func = FakeFunc("aten.add.Tensor")
        dumper._push_scope("encoder.block")
        self.assertEqual(
            dumper._next_api_scope(api_func),
            f"encoder.block.{self.module.Const.ATEN_API_TYPE_PREFIX}.add",
        )
        self.assertFalse(dumper._should_collect_api("encoder.block", api_func))
        dumper.list = ["add"]
        self.assertTrue(dumper._should_collect_api("encoder.block", api_func))
        dumper.list = ["matmul"]
        self.assertFalse(dumper._should_collect_api("encoder.block", api_func))

    def test_op_name_helpers_if_dispatch_funcs_then_pass(self):
        aten_func = FakeFunc("ignored", schema_name="aten::add")
        self.assertEqual(
            self.AclGraphDumper._op_name_from_dispatch_func(aten_func),
            f"{self.module.Const.ATEN_API_TYPE_PREFIX}.add",
        )

        npu_func = FakeFunc("ignored", schema_name="npu::rotary_mul", overloadname="special")
        self.assertEqual(
            self.AclGraphDumper._op_name_from_dispatch_func(npu_func),
            f"{self.module.Const.NPU_API_TYPE_PREFIX}.rotary_mul.special",
        )

        other_func = FakeFunc("custom.op.Tensor")
        self.assertEqual(
            self.AclGraphDumper._op_name_from_dispatch_func(other_func),
            f"{self.module.Const.TORCH_API_TYPE_PREFIX}.custom.op",
        )

        unknown_func = FakeFunc("justone")
        self.assertEqual(
            self.AclGraphDumper._op_name_from_dispatch_func(unknown_func),
            f"{self.module.Const.TORCH_API_TYPE_PREFIX}.unknown.justone",
        )

        self.assertTrue(self.AclGraphDumper._should_skip_dispatch_func(FakeFunc("aten.acl_stat.default")))
        self.assertTrue(self.AclGraphDumper._should_skip_dispatch_func(FakeFunc("aten.acl_save.default")))
        self.assertFalse(self.AclGraphDumper._should_skip_dispatch_func(FakeFunc("aten.add.Tensor")))

    def test_tls_and_scope_stack_helpers_if_tls_operations_then_pass(self):
        dumper = self.make_dumper()
        self.assertEqual(dumper._dispatch_depth(), 0)
        dumper._set_dispatch_depth(2)
        self.assertEqual(dumper._dispatch_depth(), 2)
        self.assertFalse(dumper._is_dispatch_collecting())

        dumper._push_scope("outer")
        dumper._push_scope("inner")
        self.assertEqual(dumper._current_scope(), "inner")
        dumper._pop_scope()
        self.assertEqual(dumper._current_scope(), "outer")
        dumper._pop_scope()
        dumper._pop_scope()
        self.assertEqual(dumper._current_scope(), "")

        def raise_inside():
            self.assertTrue(dumper._is_dispatch_collecting())
            raise RuntimeError("collect failed")

        with self.assertRaises(RuntimeError):
            dumper._dc(raise_inside)
        self.assertFalse(dumper._is_dispatch_collecting())

    def test_step_rank_dir_and_dtype_helpers_if_helper_inputs_then_pass(self):
        dumper = self.make_dumper(dump_path="./dump", rank_id=None)
        with patch.object(self.module.os, "getpid", return_value=123), \
                patch.object(self.module, "create_directory") as mock_create_dir:
            rank_dir = dumper._step_rank_dir()
        self.assertEqual(rank_dir, os.path.join("./dump", "step0", "pid123"))
        mock_create_dir.assert_called_once_with(rank_dir)

        self.assertEqual(self.AclGraphDumper._normalize_dtype("Float"), "torch.float32")
        self.assertEqual(self.AclGraphDumper._normalize_dtype("CustomType"), "CustomType")

    def test_parse_and_convert_stats_to_dump_data_if_stats_inputs_then_pass(self):
        self.assertEqual(self.AclGraphDumper._normalize_l0_op_name("Module.layer.0.forward.1"), "Module.layer.0.forward.1")
        self.assertEqual(
            self.AclGraphDumper._normalize_l0_op_name("Module.layer.Linear.0.forward"),
            "Module.layer.Linear.forward.0",
        )
        self.assertEqual(
            self.AclGraphDumper._normalize_l0_op_name("Module.block.forward.0.forward"),
            "Module.block.forward.forward.0",
        )

        self.assertIsNone(self.AclGraphDumper._parse_stat_key("invalid.key"))
        self.assertEqual(
            self.AclGraphDumper._parse_stat_key("toy.forward.input"),
            ("toy.forward", "input", []),
        )
        self.assertEqual(
            self.AclGraphDumper._parse_stat_key("toy.forward.input_kwargs.kw.arg"),
            ("toy.forward", "input_kwargs", ["kw", "arg"]),
        )
        self.assertIsNone(self.AclGraphDumper._parse_stat_key("toy.forward.unknown"))

        stats = {
            "Module.linear.Linear.forward.input.0": {
                "dtype": "Float",
                "shape": [2, 8],
                "max": 1.0,
                "min": -1.0,
                "mean": 0.0,
                "norm": 2.0,
            },
            "Module.linear.Linear.forward.input_kwargs.bias.0": {
                "dtype": "Half",
                "shape": [2, 8],
                "max": 2.0,
                "min": -2.0,
                "mean": 0.5,
                "norm": 3.0,
            },
            "Module.linear.Linear.forward.output.0": {
                "dtype": "Double",
                "shape": [2, 4],
                "max": 3.0,
                "min": -3.0,
                "mean": 1.5,
                "norm": 4.0,
            },
            "ignored": {"dtype": "Float"},
        }
        dump_data = self.AclGraphDumper._convert_stats_to_dump_data(stats)
        op_entry = dump_data["Module.linear.Linear.forward"]
        self.assertEqual(op_entry[self.module.Const.INPUT_ARGS][0][self.module.Const.DTYPE], "torch.float32")
        self.assertEqual(op_entry[self.module.Const.INPUT_KWARGS]["bias"]["0"][self.module.Const.DTYPE], "torch.float16")
        self.assertEqual(op_entry[self.module.Const.OUTPUT][0][self.module.Const.DTYPE], "torch.float64")

        with self.assertRaises(TypeError):
            self.AclGraphDumper._assign_nested_value({"a": 1}, ["a", "b"], {})

        compressed = self.AclGraphDumper._compress_numeric_tree_to_list({"1": {"0": "x"}, "0": {"0": "y"}})
        self.assertEqual(compressed, [["y"], ["x"]])

    def test_collect_if_forward_start_and_invalid_values_then_pass(self):
        dumper = self.make_dumper()
        valid_tensor = torch.randn(2, 3)
        meta_tensor = torch.empty(1, device="meta")

        collected = dumper._collect("scope", "input", [valid_tensor, meta_tensor, "bad"], mark_forward_start=True)

        self.assertTrue(collected)
        self.assertEqual(self.aclgraph_dump_stub.acl_stat.call_count, 1)
        tag = self.aclgraph_dump_stub.acl_stat.call_args[0][1]
        self.assertEqual(tag, f"scope.input.{self.module.FORWARD_START_MARKER}.0")

        self.aclgraph_dump_stub.acl_stat.reset_mock(side_effect=False)
        self.aclgraph_dump_stub.acl_stat.side_effect = lambda tensor, tag: tensor
        collected = dumper._collect("scope", "output", ["bad", meta_tensor], mark_forward_start=False)
        self.assertFalse(collected)
        self.aclgraph_dump_stub.acl_stat.assert_not_called()

    def test_patch_if_module_inputs_kwargs_outputs_then_pass(self):
        model = KwModel()
        dumper = self.make_dumper(keywords=[], level="L0")
        dumper.start(model)
        x = torch.randn(2, 8)
        bias = torch.randn(2, 8)

        output = model(x, bias=bias)

        self.assertTrue(torch.equal(output, x + bias))
        tags = [call.args[1] for call in self.aclgraph_dump_stub.acl_stat.call_args_list]
        self.assertTrue(any(".input." in tag for tag in tags))
        self.assertTrue(any("input_kwargs" in tag for tag in tags))
        self.assertTrue(any(tag.endswith(".output") or ".output." in tag for tag in tags))
        self.assertTrue(hasattr(model, "_msprobe_aclgraph_origin_forward"))

        origin_forward = model._msprobe_aclgraph_origin_forward
        dumper._patch(model)
        self.assertIs(model._msprobe_aclgraph_origin_forward, origin_forward)

    def test_patch_if_unmatched_modules_then_pass(self):
        model = OnlyRootModel()
        dumper = self.make_dumper(keywords=["linear"], level="L0")

        dumper._patch(model)

        self.assertFalse(hasattr(model, "_msprobe_aclgraph_origin_forward"))

    def test_dispatch_mode_if_skip_and_collecting_guard_then_pass(self):
        if self.module.TorchDispatchMode is None:
            self.skipTest("TorchDispatchMode unavailable")
        dumper = self.make_dumper(level="L1")
        mode = self.module._AclTorchDispatchMode(dumper)
        tensor = torch.randn(2, 3)

        skip_func = FakeFunc("aten.acl_stat.default", result=tensor)
        result = mode.__torch_dispatch__(skip_func, (), args=(tensor,), kwargs={})
        self.assertIs(result, tensor)
        self.assertEqual(len(skip_func.calls), 1)

        dumper._push_scope("scope")
        setattr(dumper._tls, "dispatch_collecting", True)
        guarded_func = FakeFunc("aten.add.Tensor", result=tensor)
        with patch.object(dumper, "_collect") as mock_collect:
            result = mode.__torch_dispatch__(guarded_func, (), args=(tensor,), kwargs={})
        self.assertIs(result, tensor)
        mock_collect.assert_not_called()

    def test_dispatch_mode_if_collects_inputs_kwargs_outputs_then_pass(self):
        if self.module.TorchDispatchMode is None:
            self.skipTest("TorchDispatchMode unavailable")
        dumper = self.make_dumper(level="L1")
        dumper._push_scope("module")
        mode = self.module._AclTorchDispatchMode(dumper)
        tensor = torch.randn(2, 3)
        func = FakeFunc("aten.add.Tensor", result=tensor)
        collected_calls = []

        def fake_collect(scope, io_name, value, mark_forward_start=False):
            collected_calls.append((scope, io_name, mark_forward_start, value))
            return io_name != "output"

        with patch.object(dumper, "_collect", side_effect=fake_collect):
            result = mode.__torch_dispatch__(func, (), args=(tensor,), kwargs={"alpha": tensor})

        self.assertIs(result, tensor)
        self.assertEqual([call[1] for call in collected_calls], ["input", "input_kwargs", "output"])
        self.assertTrue(collected_calls[0][2])
        self.assertFalse(collected_calls[1][2])

    def test_collect_if_dispatch_on_nested_model_then_pass(self):
        model = ToyModel()
        dumper = self.make_dumper(keywords=[], level="mix")
        dumper.start(model)

        _ = model(torch.randn(2, 8))

        tags = [call.args[1] for call in self.aclgraph_dump_stub.acl_stat.call_args_list]
        self.assertTrue(any(tag.startswith("Module.linear.Linear.") for tag in tags))
        if self.module.TorchDispatchMode is not None:
            self.assertTrue(any(".Aten." in tag for tag in tags))

    def test_synchronize_if_sync_paths_then_pass(self):
        dumper = self.make_dumper()
        with patch.object(self.module, "torch_npu", self.torch_npu_stub), \
                patch.object(self.module.torch.cuda, "is_available", return_value=False), \
                patch.object(self.module.torch.cuda, "synchronize") as mock_cuda_sync:
            dumper._synchronize()
        self.torch_npu_stub.npu.synchronize.assert_called_once()
        mock_cuda_sync.assert_not_called()

        self.torch_npu_stub.npu.synchronize.reset_mock(side_effect=False)
        self.torch_npu_stub.npu.synchronize.side_effect = RuntimeError("npu fail")
        with patch.object(self.module, "torch_npu", self.torch_npu_stub), \
                patch.object(self.module.torch.cuda, "is_available", return_value=True), \
                patch.object(self.module.torch.cuda, "synchronize") as mock_cuda_sync:
            dumper._synchronize()
        mock_cuda_sync.assert_called_once()
        self.torch_npu_stub.npu.synchronize.side_effect = None

        with patch.object(self.module, "torch_npu", None), \
                patch.object(self.module.torch.cuda, "is_available", return_value=False), \
                patch.object(self.module.torch.cuda, "synchronize") as mock_cuda_sync:
            dumper._synchronize()
        mock_cuda_sync.assert_not_called()

    def test_start_and_step_if_runtime_paths_then_pass(self):
        dumper = self.make_dumper(level="mix", rank=[1], rank_id=0)
        with patch.object(dumper, "_resolve_rank_id", return_value=0), \
                patch.object(dumper, "_patch") as mock_patch:
            dumper.start(MagicMock())
        self.assertFalse(dumper._running)
        mock_patch.assert_not_called()
        dumper.step()
        self.aclgraph_dump_stub.get_acl_stat_dict.assert_not_called()

        dumper = self.make_dumper(level="mix", rank=[], rank_id=0)
        with patch.object(dumper, "_resolve_rank_id", return_value=0), \
                patch.object(dumper, "_patch") as mock_patch:
            dumper.start(MagicMock())
        self.assertTrue(dumper._running)
        mock_patch.assert_called_once()

        self.aclgraph_dump_stub.get_acl_stat_dict.return_value = {"toy.forward.input.0": {"dtype": "Float", "shape": []}}
        with patch.object(dumper, "_synchronize") as mock_sync, \
                patch.object(dumper, "_step_rank_dir", return_value="./dump/step0/rank0"), \
                patch.object(self.module, "save_json") as mock_save_json:
            dumper.step(dump=False)
            mock_sync.assert_called_once()
            mock_save_json.assert_not_called()
            self.assertEqual(dumper.step_id, 0)

        stats = {
            "toy.forward.input.0": {
                "dtype": "Float",
                "shape": [2, 8],
                "max": 1.0,
                "min": -1.0,
                "mean": 0.0,
                "norm": 2.0,
            },
            "toy.forward.output.0": {
                "dtype": "Float",
                "shape": [2, 4],
                "max": 2.0,
                "min": -2.0,
                "mean": 0.1,
                "norm": 3.0,
            },
        }
        self.aclgraph_dump_stub.get_acl_stat_dict.return_value = stats
        with patch.object(dumper, "_synchronize"), \
                patch.object(dumper, "_step_rank_dir", return_value="./dump/step0/rank0"), \
                patch.object(self.module, "save_json") as mock_save_json:
            dumper.step()

        save_path, dump_json = mock_save_json.call_args[0][0], mock_save_json.call_args[0][1]
        self.assertEqual(save_path, os.path.join("./dump/step0/rank0", "dump.json"))
        self.assertEqual(dump_json["task"], self.module.Const.STATISTICS)
        self.assertEqual(dump_json["level"], self.module.Const.LEVEL_MIX)
        self.assertEqual(dump_json["framework"], self.module.Const.PT_FRAMEWORK)
        self.assertIn("toy.forward", dump_json["data"])
        self.assertEqual(mock_save_json.call_args.kwargs["indent"], 2)
        self.assertEqual(dumper.step_id, 1)


if __name__ == "__main__":
    unittest.main()
