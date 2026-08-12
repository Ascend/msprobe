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
from unittest.mock import MagicMock, PropertyMock, patch

import torch


def _build_aclgraph_dumper_import_env():
    import msprobe

    fake_aclgraph_dump = types.ModuleType("msprobe.pytorch.aclgraph_dump")
    fake_aclgraph_dump.acl_save = MagicMock(side_effect=lambda tensor, path: tensor)
    fake_aclgraph_dump.acl_tensor_save = MagicMock(
        side_effect=lambda tensor, path, api_name, is_call_start=False, switch=None: tensor
    )
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
    pytorch_attr_patcher = patch.object(msprobe, "pytorch", fake_pytorch_pkg, create=True)
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


def _load_aclgraph_dump_api_module(pytorch_pkg_dir):
    module_name = "msprobe.pytorch.aclgraph_dump"
    module_path = os.path.join(pytorch_pkg_dir, "aclgraph_dump", "__init__.py")

    fake_msprobe = types.ModuleType("msprobe")
    fake_msprobe.__path__ = []
    fake_pytorch = types.ModuleType("msprobe.pytorch")
    fake_pytorch.__path__ = []
    fake_lib = types.ModuleType("msprobe.lib")
    fake_extension = types.ModuleType("msprobe.lib.aclgraph_dump_ext")
    fake_lib.aclgraph_dump_ext = fake_extension
    fake_meta = types.ModuleType(f"{module_name}._meta")
    fake_meta._register_meta = MagicMock()
    fake_torch_npu = types.ModuleType("torch_npu")

    acl_save_op = MagicMock(side_effect=lambda tensor, path: tensor)
    acl_save_op.default = MagicMock()
    acl_tensor_save_op = MagicMock(
        side_effect=lambda tensor, path, api_name, is_call_start=False, switch=None: tensor
    )
    acl_tensor_save_op.default = MagicMock()
    acl_stat_op = MagicMock(side_effect=lambda tensor, tag, switch=None: tensor)
    acl_stat_op.default = MagicMock()
    fake_ops = types.SimpleNamespace(
        acl_save=acl_save_op,
        acl_tensor_save=acl_tensor_save_op,
        acl_stat=acl_stat_op,
    )

    modules_patcher = patch.dict(
        sys.modules,
        {
            "msprobe": fake_msprobe,
            "msprobe.pytorch": fake_pytorch,
            "msprobe.lib": fake_lib,
            "msprobe.lib.aclgraph_dump_ext": fake_extension,
            f"{module_name}._meta": fake_meta,
            "torch_npu": fake_torch_npu,
        },
    )
    ops_patcher = patch.object(torch.ops, "my_ns", fake_ops)
    modules_patcher.start()
    ops_patcher.start()

    spec = importlib.util.spec_from_file_location(
        module_name, module_path, submodule_search_locations=[os.path.dirname(module_path)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, fake_ops, modules_patcher, ops_patcher


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
        self.aclgraph_dump_stub.acl_stat.side_effect = lambda tensor, tag, switch=None: tensor
        self.aclgraph_dump_stub.get_acl_stat_dict.reset_mock(side_effect=False)
        self.aclgraph_dump_stub.get_acl_stat_dict.return_value = {}
        self.torch_npu_stub.npu.synchronize.reset_mock()

    def tearDown(self):
        sys.modules.pop("msprobe.pytorch.aclgraph_dumper", None)
        importlib.invalidate_caches()
        self._pytorch_attr_patcher.stop()
        self._modules_patcher.stop()

    def make_dumper(self, dump_path="./dump", keywords=None, level="mix", rank=None, rank_id=0,
                    task="statistics", slice_info=None):
        with patch.object(
            self.AclGraphDumper,
            "_load_msprobe_config",
            return_value=(task, dump_path, keywords or [], [], level, rank, slice_info or [], True),
        ), \
                patch.object(self.AclGraphDumper, "_resolve_config_path", return_value="./config.json"), \
                patch.object(self.AclGraphDumper, "_get_config_signature", return_value=(1, 1)), \
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
        self.assertFalse(self.module._is_collectable_tensor(torch.empty(2, device="meta")))
        with patch.object(torch.Tensor, "device", new_callable=PropertyMock) as mock_device:
            mock_device.side_effect = RuntimeError("device access failed")
            self.assertFalse(self.module._is_collectable_tensor(torch.randn(2, 3)))
        self.assertTrue(self.module._is_collectable_tensor(torch.ones(2, 3, device="cpu")))
        self.assertFalse(self.module._is_collectable_tensor("not_a_tensor"))

    def test_is_collectable_tensor_if_fake_mode_detected_then_still_checks_tensor_only(self):
        tensor = torch.randn(2, 3)

        with patch.object(self.module, "_detect_fake_mode", return_value=object()) as mock_detect_fake_mode:
            self.assertTrue(self.module._is_collectable_tensor(tensor))

        mock_detect_fake_mode.assert_not_called()

    def test_load_msprobe_config_if_config_and_validations_then_pass(self):
        default_path = os.path.normpath(self.AclGraphDumper._default_config_path())
        self.assertTrue(default_path.endswith(os.path.join("msprobe", "config.json")))

        config = {
            "task": "statistics",
            "dump_path": "./dump_dir",
            "level": "L1",
            "statistics": {"list": ["linear"]},
            "rank": [0],
        }
        with patch.object(self.AclGraphDumper, "_default_config_path", return_value="/tmp/default.json"), \
                patch.object(self.module, "check_and_get_real_path", return_value="/tmp/default.json") as mock_real_path, \
                patch.object(self.module, "load_json", return_value=config):
            task, dump_path, module_list, custom_api, level, rank, slice_info, dump_enable = self.AclGraphDumper._load_msprobe_config(None)

        self.assertEqual((task, dump_path, module_list, custom_api, level, rank, slice_info, dump_enable),
                         ("statistics", "./dump_dir", ["linear"], [], "L1", [0], [], True))
        mock_real_path.assert_called_once()

        config = {
            "task": "statistics",
            "dump_path": "./dump_dir",
            "statistics": {"level": "mix"},
        }
        with patch.object(self.module, "check_and_get_real_path", return_value="/tmp/config.json"), \
                patch.object(self.module, "load_json", return_value=config):
            _, _, _, _, level, _, _, _ = self.AclGraphDumper._load_msprobe_config("./config.json")
        self.assertEqual(level, "L0")

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
            task, dump_path, module_list, custom_api, level, rank, slice_info, dump_enable = self.AclGraphDumper._load_msprobe_config("./config.json")
        self.assertEqual((task, dump_path, module_list, custom_api, level, rank, slice_info, dump_enable),
                         (1, "./x", [], [], "L0", None, [], True))

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

        self.assertEqual(self.AclGraphDumper._validate_custom_api(None), [])
        self.assertEqual(self.AclGraphDumper._validate_custom_api(["pkg.api"]), ["pkg.api"])
        with self.assertRaises(TypeError):
            self.AclGraphDumper._validate_custom_api("pkg.api")
        with self.assertRaises(TypeError):
            self.AclGraphDumper._validate_custom_api(["pkg.api", 1])

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

        op_name = f"{self.module.Const.ATEN_API_TYPE_PREFIX}.add"
        self.assertFalse(dumper._should_collect_api(op_name))
        dumper.list = ["add"]
        self.assertTrue(dumper._should_collect_api(op_name))
        dumper.list = ["matmul"]
        self.assertFalse(dumper._should_collect_api(op_name))

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
        self.assertTrue(
            self.AclGraphDumper._should_skip_dispatch_func(FakeFunc("torch.ops.higher_order.auto_functionalized_v2"))
        )
        self.assertFalse(self.AclGraphDumper._should_skip_dispatch_func(FakeFunc("aten.add.Tensor")))

    def test_dispatch_collecting_guard_if_collection_raises_then_restores_state(self):
        dumper = self.make_dumper()
        self.assertFalse(dumper._is_dispatch_collecting())

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

        with patch.object(self.module, "_is_collectable_tensor", side_effect=lambda tensor: tensor is valid_tensor):
            collected = dumper._collect("scope", "input", [valid_tensor, meta_tensor, "bad"], mark_forward_start=True)

        self.assertTrue(collected)
        self.assertEqual(self.aclgraph_dump_stub.acl_stat.call_count, 1)
        tag = self.aclgraph_dump_stub.acl_stat.call_args[0][1]
        self.assertEqual(tag, f"scope.input.{self.module.FORWARD_START_MARKER}.0")

        self.aclgraph_dump_stub.acl_stat.reset_mock(side_effect=False)
        self.aclgraph_dump_stub.acl_stat.side_effect = lambda tensor, tag: tensor
        with patch.object(self.module, "_is_collectable_tensor", return_value=False):
            collected = dumper._collect("scope", "output", ["bad", meta_tensor], mark_forward_start=False)
        self.assertFalse(collected)
        self.aclgraph_dump_stub.acl_stat.assert_not_called()

    def test_patch_if_module_inputs_kwargs_outputs_then_pass(self):
        model = KwModel()
        dumper = self.make_dumper(keywords=[], level="L0")
        dumper.start(model)
        x = torch.randn(2, 8)
        bias = torch.randn(2, 8)

        with patch.object(self.module, "_is_collectable_tensor", return_value=True):
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

    def test_patch_if_l0_and_fake_mode_detected_then_module_collect_still_runs(self):
        model = KwModel()
        dumper = self.make_dumper(keywords=[], level="L0")
        dumper.start(model)
        x = torch.randn(2, 8)
        bias = torch.randn(2, 8)

        with patch.object(self.module, "_detect_fake_mode", return_value=object()) as mock_detect_fake_mode:
            output = model(x, bias=bias)

        self.assertTrue(torch.equal(output, x + bias))
        mock_detect_fake_mode.assert_not_called()
        tags = [call.args[1] for call in self.aclgraph_dump_stub.acl_stat.call_args_list]
        self.assertTrue(any(".input." in tag for tag in tags))
        self.assertTrue(any("input_kwargs" in tag for tag in tags))
        self.assertTrue(any(tag.endswith(".output") or ".output." in tag for tag in tags))

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
        self.assertTrue(mode.supports_higher_order_operators)

        skip_func = FakeFunc("aten.acl_stat.default", result=tensor)
        result = mode.__torch_dispatch__(skip_func, (), args=(tensor,), kwargs={})
        self.assertIs(result, tensor)
        self.assertEqual(len(skip_func.calls), 1)

        higher_order_func = FakeFunc("torch.ops.higher_order.auto_functionalized_v2", result=tensor)
        with patch.object(dumper, "_collect") as mock_collect:
            result = mode.__torch_dispatch__(higher_order_func, (), args=(tensor,), kwargs={})
        self.assertIs(result, tensor)
        mock_collect.assert_not_called()
        self.assertEqual(len(higher_order_func.calls), 1)

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
        mode = self.module._AclTorchDispatchMode(dumper)
        tensor = torch.randn(2, 3)
        func = FakeFunc("aten.add.Tensor", result=tensor)
        collected_calls = []

        def fake_collect(scope, io_name, value, mark_forward_start=False, call_started=None):
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

        with patch.object(self.module, "_is_collectable_tensor", return_value=True):
            _ = model(torch.randn(2, 8))

        tags = [call.args[1] for call in self.aclgraph_dump_stub.acl_stat.call_args_list]
        self.assertTrue(any(tag.startswith("Module.linear.Linear.") for tag in tags))
        if self.module.TorchDispatchMode is not None:
            self.assertTrue(any(tag.startswith(f"{self.module.Const.ATEN_API_TYPE_PREFIX}.") for tag in tags))

    def test_patch_if_l1_then_only_root_module_is_wrapped(self):
        model = ToyModel()
        dumper = self.make_dumper(keywords=[], level="L1")

        dumper._patch(model)

        self.assertTrue(hasattr(model, "_msprobe_aclgraph_origin_forward"))
        for module_name, module in model.named_modules():
            if module_name:
                self.assertFalse(hasattr(module, "_msprobe_aclgraph_origin_forward"))

    def test_patch_custom_api_if_configured_then_collects_indexed_inputs_and_outputs(self):
        target_module_name = "msprobe_custom_api_test_module"
        target_module = types.ModuleType(target_module_name)

        def reshape_and_cache(key, *, value_cache):
            return key + value_cache

        target_module.reshape_and_cache = reshape_and_cache
        dumper = self.make_dumper(task="tensor")
        dumper.custom_api = [f"{target_module_name}.reshape_and_cache"]
        dumper._tensor_data_dir_path = "./tensor_data"
        dumper._running = True

        with patch.dict(sys.modules, {target_module_name: target_module}):
            dumper._patch_custom_api()
            output = target_module.reshape_and_cache(torch.ones(1), value_cache=torch.ones(1))

        self.assertTrue(torch.equal(output, torch.full((1,), 2.0)))
        self.assertEqual(self.aclgraph_dump_stub.acl_tensor_save.call_count, 3)
        calls = self.aclgraph_dump_stub.acl_tensor_save.call_args_list
        self.assertEqual([call.args[2] for call in calls], ["reshape_and_cache"] * 3)
        self.assertEqual([call.args[3] for call in calls], [True, False, False])
        self.assertIn("reshape_and_cache.input.0", calls[0].args[1])
        self.assertIn("reshape_and_cache.input_kwargs.value_cache", calls[1].args[1])
        self.assertIn("reshape_and_cache.output", calls[2].args[1])

    def test_start_if_statistics_then_does_not_patch_custom_api(self):
        dumper = self.make_dumper(task="statistics")
        dumper.custom_api = ["missing.module.api"]

        with patch.object(dumper, "_resolve_rank_id", return_value=0), \
                patch.object(dumper, "_patch"), \
                patch.object(dumper, "_patch_custom_api") as mock_patch_custom_api:
            dumper.start(MagicMock())

        self.assertTrue(dumper._running)
        mock_patch_custom_api.assert_not_called()

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

    def test_step_if_statistics_are_invalid_then_warn_and_save(self):
        dumper = self.make_dumper(level="L1", rank=[], rank_id=0)
        dumper._running = True
        self.aclgraph_dump_stub.get_acl_stat_dict.return_value = {
            "Torch.npu_quant_matmul.forward.input.0": {
                "dtype": "Char",
                "shape": [896, 1152],
                "max": None,
                "min": None,
                "mean": None,
                "norm": None,
            }
        }

        with patch.object(dumper, "_synchronize"), \
                patch.object(dumper, "_step_rank_dir", return_value="./dump/step0/rank0"), \
                patch.object(self.module, "save_json") as mock_save_json, \
                patch.object(self.module.logger, "warning") as mock_warning:
            dumper.step()

        mock_warning.assert_called_once_with(
            "Invalid statistics detected. Please use tensor mode to collect the affected data."
        )
        saved_record = mock_save_json.call_args[0][1]["data"]["Torch.npu_quant_matmul.forward"][
            self.module.Const.INPUT_ARGS
        ][0]
        self.assertIsNone(saved_record[self.module.Const.MIN])
        self.assertIsNone(saved_record[self.module.Const.MAX])
        self.assertEqual(saved_record[self.module.Const.DTYPE], "torch.int8")

    def test_collect_with_slice_matched(self):
        """TC-201: tensor 第0维匹配 total 时切片，acl_stat 接收切片后数据"""
        slice_info = [{"dim": 0, "size": 100, "begin": 0, "end": 50}]
        dumper = self.make_dumper(slice_info=slice_info)
        tensor = torch.arange(300).reshape(100, 3)
        with patch.object(self.module, "_is_collectable_tensor", return_value=True):
            dumper._collect("scope", "input", [tensor])
        self.assertEqual(self.aclgraph_dump_stub.acl_stat.call_count, 1)
        stat_tensor = self.aclgraph_dump_stub.acl_stat.call_args[0][0]
        self.assertEqual(stat_tensor.shape, (50, 3))
        torch.testing.assert_close(stat_tensor, tensor[0:50])

    def test_collect_with_slice_not_matched(self):
        """TC-202: tensor 第0维不匹配 total 时不切片"""
        slice_info = [{"dim": 0, "size": 100, "begin": 0, "end": 50}]
        dumper = self.make_dumper(slice_info=slice_info)
        tensor = torch.randn(200, 3)
        with patch.object(self.module, "_is_collectable_tensor", return_value=True):
            dumper._collect("scope", "input", [tensor])
        self.assertEqual(self.aclgraph_dump_stub.acl_stat.call_count, 1)
        stat_tensor = self.aclgraph_dump_stub.acl_stat.call_args[0][0]
        self.assertEqual(stat_tensor.shape, (200, 3))
        torch.testing.assert_close(stat_tensor, tensor)

    def test_collect_slice_with_dim1(self):
        """TC-203: 在 dim1 上切片"""
        slice_info = [{"dim": 1, "size": 3, "begin": 0, "end": 2}]
        dumper = self.make_dumper(slice_info=slice_info)
        tensor = torch.arange(300).reshape(100, 3)
        with patch.object(self.module, "_is_collectable_tensor", return_value=True):
            dumper._collect("scope", "input", [tensor])
        self.assertEqual(self.aclgraph_dump_stub.acl_stat.call_count, 1)
        stat_tensor = self.aclgraph_dump_stub.acl_stat.call_args[0][0]
        self.assertEqual(stat_tensor.shape, (100, 2))
        torch.testing.assert_close(stat_tensor, tensor[:, 0:2])

    def test_collect_slice_multi_dim(self):
        """TC-204: 多维度同时切片"""
        slice_info = [
            {"dim": 0, "size": 100, "begin": 0, "end": 50},
            {"dim": 1, "size": 3, "begin": 0, "end": 2},
        ]
        dumper = self.make_dumper(slice_info=slice_info)
        tensor = torch.arange(600).reshape(100, 3, 2)
        with patch.object(self.module, "_is_collectable_tensor", return_value=True):
            dumper._collect("scope", "input", [tensor])
        self.assertEqual(self.aclgraph_dump_stub.acl_stat.call_count, 1)
        stat_tensor = self.aclgraph_dump_stub.acl_stat.call_args[0][0]
        self.assertEqual(stat_tensor.shape, (50, 2, 2))
        torch.testing.assert_close(stat_tensor, tensor[0:50, 0:2])

    def test_collect_slice_with_partial_range(self):
        """TC-205: 部分范围切片 [10:60]"""
        slice_info = [{"dim": 0, "size": 100, "begin": 10, "end": 60}]
        dumper = self.make_dumper(slice_info=slice_info)
        tensor = torch.arange(300).reshape(100, 3)
        with patch.object(self.module, "_is_collectable_tensor", return_value=True):
            dumper._collect("scope", "input", [tensor])
        self.assertEqual(self.aclgraph_dump_stub.acl_stat.call_count, 1)
        stat_tensor = self.aclgraph_dump_stub.acl_stat.call_args[0][0]
        self.assertEqual(stat_tensor.shape, (50, 3))
        torch.testing.assert_close(stat_tensor, tensor[10:60])

    def test_collect_slice_with_zero_dim_tensor(self):
        """TC-206: 0维 tensor 不切片"""
        slice_info = [{"dim": 0, "size": 100, "begin": 0, "end": 50}]
        dumper = self.make_dumper(slice_info=slice_info)
        tensor = torch.tensor(42)
        with patch.object(self.module, "_is_collectable_tensor", return_value=True):
            dumper._collect("scope", "input", [tensor])
        self.assertEqual(self.aclgraph_dump_stub.acl_stat.call_count, 1)
        stat_tensor = self.aclgraph_dump_stub.acl_stat.call_args[0][0]
        self.assertEqual(stat_tensor.shape, ())
        self.assertEqual(stat_tensor.item(), 42)

    def test_collect_without_slice_info(self):
        """未配置 slice_info 时正常采集不切片"""
        dumper = self.make_dumper(slice_info=None)
        self.assertEqual(dumper.slice_info, [])
        tensor = torch.randn(100, 3)
        with patch.object(self.module, "_is_collectable_tensor", return_value=True):
            dumper._collect("scope", "input", [tensor])
        self.assertEqual(self.aclgraph_dump_stub.acl_stat.call_count, 1)
        stat_tensor = self.aclgraph_dump_stub.acl_stat.call_args[0][0]
        self.assertEqual(stat_tensor.shape, (100, 3))

    def test_load_msprobe_config_returns_slice_info(self):
        """EC-201: _load_msprobe_config 正确返回 slice 配置"""
        config = {
            "task": "statistics",
            "dump_path": "./dump_dir",
            "level": "L1",
            "statistics": {
                "list": ["linear"],
                "slice": [{"dim": 0, "size": 100, "begin": 0, "end": 50}],
            },
            "rank": [0],
        }
        with patch.object(self.AclGraphDumper, "_default_config_path", return_value="/tmp/default.json"), \
                patch.object(self.module, "check_and_get_real_path", return_value="/tmp/default.json"), \
                patch.object(self.module, "load_json", return_value=config):
            _, _, _, _, _, _, slice_info, _ = self.AclGraphDumper._load_msprobe_config(None)
        self.assertEqual(slice_info, [{"dim": 0, "size": 100, "begin": 0, "end": 50}])

    def test_load_msprobe_config_without_slice_info(self):
        """EC-202: _load_msprobe_config 无 slice 配置时返回空 list"""
        config = {
            "task": "statistics",
            "dump_path": "./dump_dir",
            "level": "L1",
            "statistics": {"list": ["linear"]},
            "rank": [0],
        }
        with patch.object(self.AclGraphDumper, "_default_config_path", return_value="/tmp/default.json"), \
                patch.object(self.module, "check_and_get_real_path", return_value="/tmp/default.json"), \
                patch.object(self.module, "load_json", return_value=config):
            _, _, _, _, _, _, slice_info, _ = self.AclGraphDumper._load_msprobe_config(None)
        self.assertEqual(slice_info, [])

    def test_load_msprobe_config_with_invalid_slice_info(self):
        """EC-203: _load_msprobe_config 遇到非法 slice 时抛出异常"""
        config = {
            "task": "statistics",
            "dump_path": "./dump_dir",
            "level": "L1",
            "statistics": {
                "list": ["linear"],
                "slice": [{"dim": 0, "size": -1, "begin": 0, "end": 0}],
            },
            "rank": [0],
        }
        with patch.object(self.AclGraphDumper, "_default_config_path", return_value="/tmp/default.json"), \
                patch.object(self.module, "check_and_get_real_path", return_value="/tmp/default.json"), \
                patch.object(self.module, "load_json", return_value=config):
            with self.assertRaises(Exception):
                self.AclGraphDumper._load_msprobe_config(None)


class TestAclGraphDumpApi(unittest.TestCase):
    def setUp(self):
        pytorch_pkg_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "python", "msprobe", "pytorch")
        )
        self.module, self.fake_ops, self.modules_patcher, self.ops_patcher = _load_aclgraph_dump_api_module(
            pytorch_pkg_dir
        )

    def tearDown(self):
        sys.modules.pop("msprobe.pytorch.aclgraph_dump", None)
        self.ops_patcher.stop()
        self.modules_patcher.stop()

    def test_acl_save_makes_mixed_qkv_split_contiguous(self):
        qkv_size, z_size = 8, 2
        mixed_qkvz = torch.arange(2 * (qkv_size + z_size), dtype=torch.bfloat16).reshape(
            2, qkv_size + z_size
        )
        mixed_qkv, _ = mixed_qkvz.split([qkv_size, z_size], dim=-1)
        self.assertFalse(mixed_qkv.is_contiguous())

        self.module.acl_save(mixed_qkv, "mixed_qkv.pt")

        call = self.fake_ops.acl_save.call_args
        saved_tensor = call.args[0]
        self.assertTrue(saved_tensor.is_contiguous())
        self.assertTrue(torch.equal(saved_tensor, mixed_qkv))
        self.assertEqual(call.args[1], "mixed_qkv.pt")

    def test_acl_save_reuses_contiguous_tensor(self):
        tensor = torch.arange(12).reshape(3, 4)

        result = self.module.acl_save(tensor, "tensor.pt")

        self.assertIs(self.fake_ops.acl_save.call_args.args[0], tensor)
        self.assertIs(result, tensor)

    def test_acl_tensor_save_makes_tensor_contiguous_and_forwards_arguments(self):
        tensor = torch.arange(24).reshape(4, 6)[:, :4]
        switch = torch.tensor(True)

        result = self.module.acl_tensor_save(tensor, "tensor.pt", "linear", True, switch)

        call = self.fake_ops.acl_tensor_save.call_args
        saved_tensor = call.args[0]
        self.assertTrue(saved_tensor.is_contiguous())
        self.assertTrue(torch.equal(saved_tensor, tensor))
        self.assertEqual(call.args[1:], ("tensor.pt", "linear", True, switch))
        self.assertIs(result, saved_tensor)


if __name__ == "__main__":
    unittest.main()
