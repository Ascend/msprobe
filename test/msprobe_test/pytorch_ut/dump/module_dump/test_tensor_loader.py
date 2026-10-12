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
import os
import tempfile
import unittest
from collections import namedtuple
from dataclasses import dataclass, field

import torch

from msprobe.core.common.const import Const
from msprobe.pytorch.dump.module_load.tensor_loader import TensorLoader


class FakeLoadConfig:
    """Minimal stand-in for LoadConfig to avoid filesystem validation in setUp."""

    def __init__(self, path, modules, step=None, rank=None, dump_after_load=False):
        self.path = path
        self.modules = modules
        self.step = step if step is not None else []
        self.rank = rank if rank is not None else []
        self.dump_after_load = dump_after_load

    @property
    def is_enabled(self):
        return bool(self.modules)


class TestTensorLoaderShouldOverride(unittest.TestCase):
    """Tests for should_override with exact full_forward_name matching."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        cfg = FakeLoadConfig(self.tmpdir, [
            "Module.blocks.0.attn.MultiHeadSelfAttention.forward.0",
            "Module.blocks.1.mlp.MLP.forward.0",
        ])
        self.loader = TensorLoader(cfg)

    def test_exact_match(self):
        self.assertTrue(
            self.loader.should_override("Module.blocks.0.attn.MultiHeadSelfAttention.forward.0")
        )
        self.assertTrue(
            self.loader.should_override("Module.blocks.1.mlp.MLP.forward.0")
        )

    def test_no_match_different_call_index(self):
        """forward.0 vs forward.1 must not match."""
        self.assertFalse(
            self.loader.should_override("Module.blocks.0.attn.MultiHeadSelfAttention.forward.1")
        )

    def test_no_match_different_module(self):
        """Different submodule must not match."""
        self.assertFalse(
            self.loader.should_override("Module.blocks.0.attn.proj.Linear.forward.0")
        )

    def test_no_match_substring_only(self):
        """Substring match must not work — exact full_forward_name required."""
        self.assertFalse(self.loader.should_override("blocks.0.attn"))
        self.assertFalse(self.loader.should_override("forward.0"))

    def test_empty_modules(self):
        cfg = FakeLoadConfig(self.tmpdir, [])
        loader = TensorLoader(cfg)
        self.assertFalse(loader.should_override("Module.any.Thing.forward.0"))


class TestTensorLoaderBuildPath(unittest.TestCase):
    """Tests for _build_pt_path naming consistency with dump side."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        cfg = FakeLoadConfig(self.tmpdir, ["m"], step=[0], rank=[0])
        self.loader = TensorLoader(cfg)
        self.loader.update_step_rank(0, 0)

    def test_args_path(self):
        ffn = "Module.blocks.0.attn.MultiHeadSelfAttention.forward.0"
        path = self.loader._build_pt_path(ffn, Const.INPUT, 0)
        expected = os.path.join(
            self.tmpdir, "step0", "rank0", "dump_tensor_data",
            "Module.blocks.0.attn.MultiHeadSelfAttention.forward.0.input.0.pt",
        )
        self.assertEqual(path, expected)

    def test_kwargs_path(self):
        ffn = "Module.blocks.0.attn.MultiHeadSelfAttention.forward.0"
        path = self.loader._build_pt_path(ffn, Const.KWARGS, "mask")
        expected = os.path.join(
            self.tmpdir, "step0", "rank0", "dump_tensor_data",
            "Module.blocks.0.attn.MultiHeadSelfAttention.forward.0.kwargs.mask.pt",
        )
        self.assertEqual(path, expected)

    def test_auto_step_rank(self):
        """step/rank always use current_step/current_rank (auto-align to current step/rank)."""
        cfg = FakeLoadConfig(self.tmpdir, ["m"])
        loader = TensorLoader(cfg)
        loader.update_step_rank(3, 2)
        ffn = "Module.head.Linear.forward.1"
        path = loader._build_pt_path(ffn, Const.INPUT, 0)
        self.assertIn("step3", path)
        self.assertIn("rank2", path)

    def test_step_rank_always_auto_align(self):
        """Even when load.step/rank are non-empty, _build_pt_path uses current step/rank.
        load.step/rank only controls which steps/ranks are active, not the source step/rank."""
        cfg = FakeLoadConfig(self.tmpdir, ["m"], step=[5], rank=[7])
        loader = TensorLoader(cfg)
        loader.update_step_rank(3, 2)
        ffn = "Module.head.Linear.forward.0"
        path = loader._build_pt_path(ffn, Const.INPUT, 0)
        self.assertIn("step3", path)
        self.assertIn("rank2", path)

    def test_active_when_step_in_range(self):
        """load.step=[0, 1]: current step 0 is in range, active should be True."""
        cfg = FakeLoadConfig(self.tmpdir, ["m"], step=[0, 1])
        loader = TensorLoader(cfg)
        loader.update_step_rank(0, 0)
        self.assertTrue(loader.active)

    def test_inactive_when_step_not_in_range(self):
        """load.step=[0]: current step 1 is not in range, active should be False."""
        cfg = FakeLoadConfig(self.tmpdir, ["m"], step=[0])
        loader = TensorLoader(cfg)
        loader.update_step_rank(1, 0)
        self.assertFalse(loader.active)

    def test_active_when_step_empty(self):
        """load.step=[] (default): active should be True for any step."""
        cfg = FakeLoadConfig(self.tmpdir, ["m"])
        loader = TensorLoader(cfg)
        loader.update_step_rank(99, 0)
        self.assertTrue(loader.active)

    def test_active_when_rank_in_range(self):
        """load.rank=[0]: current rank 0 is in range, active should be True."""
        cfg = FakeLoadConfig(self.tmpdir, ["m"], rank=[0])
        loader = TensorLoader(cfg)
        loader.update_step_rank(0, 0)
        self.assertTrue(loader.active)

    def test_inactive_when_rank_not_in_range(self):
        """load.rank=[0]: current rank 1 is not in range, active should be False."""
        cfg = FakeLoadConfig(self.tmpdir, ["m"], rank=[0])
        loader = TensorLoader(cfg)
        loader.update_step_rank(0, 1)
        self.assertFalse(loader.active)

    def test_single_card_proc_dir(self):
        """Single-card: rank=None, source dump uses proc{pid} dir, load auto-discovers it."""
        import os
        # create source dump with proc{pid} dir
        pid = 99999
        proc_dir = os.path.join(self.tmpdir, "step0", f"proc{pid}", "dump_tensor_data")
        os.makedirs(proc_dir)
        ffn = "Module.head.Linear.forward.0"
        tensor = torch.randn(2, 3)
        torch.save(tensor, os.path.join(proc_dir, f"{ffn}.input.0.pt"))

        cfg = FakeLoadConfig(self.tmpdir, ["m"])  # rank=[] -> auto-align
        loader = TensorLoader(cfg)
        loader.update_step_rank(0, None)  # rank=None (single-card)
        path = loader._build_pt_path(ffn, Const.INPUT, 0)
        self.assertIn("proc", path)
        self.assertIn(str(pid), path)
        self.assertNotIn("rank", path)


class TestTensorLoaderOverrideArgs(unittest.TestCase):
    """Tests for override_args covering args, kwargs, non-tensor, missing files."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.tmpdir, "step0", "rank0", "dump_tensor_data")
        os.makedirs(self.data_dir)

        self.ffn = "Module.blocks.0.attn.MultiHeadSelfAttention.forward.0"

        # prepare source tensors
        self.src_x = torch.randn(2, 3)
        self.src_mask = torch.ones(1, 1, 3, 3)
        torch.save(self.src_x, os.path.join(self.data_dir, f"{self.ffn}.input.0.pt"))
        torch.save(self.src_mask, os.path.join(self.data_dir, f"{self.ffn}.kwargs.mask.pt"))

        cfg = FakeLoadConfig(self.tmpdir, [self.ffn])
        self.loader = TensorLoader(cfg)
        self.loader.update_step_rank(0, 0)

    def test_override_tensor_arg(self):
        orig_x = torch.zeros(2, 3)
        args, kwargs = self.loader.override_args(self.ffn, (orig_x,), {})
        self.assertTrue(torch.equal(args[0], self.src_x))
        self.assertFalse(torch.equal(args[0], orig_x))

    def test_override_tensor_kwarg(self):
        orig_mask = torch.zeros(1, 1, 3, 3)
        args, kwargs = self.loader.override_args(self.ffn, (), {"mask": orig_mask})
        self.assertTrue(torch.equal(kwargs["mask"], self.src_mask))
        self.assertFalse(torch.equal(kwargs["mask"], orig_mask))

    def test_non_tensor_arg_skipped(self):
        args, kwargs = self.loader.override_args(self.ffn, (42, "hello"), {})
        self.assertEqual(args[0], 42)
        self.assertEqual(args[1], "hello")

    def test_non_tensor_kwarg_skipped(self):
        args, kwargs = self.loader.override_args(self.ffn, (), {"scale": 0.5, "name": "test"})
        self.assertEqual(kwargs["scale"], 0.5)
        self.assertEqual(kwargs["name"], "test")

    def test_mixed_args(self):
        orig_x = torch.zeros(2, 3)
        args, kwargs = self.loader.override_args(self.ffn, (orig_x, 42), {})
        self.assertTrue(torch.equal(args[0], self.src_x))
        self.assertEqual(args[1], 42)

    def test_mixed_kwargs(self):
        orig_mask = torch.zeros(1, 1, 3, 3)
        args, kwargs = self.loader.override_args(
            self.ffn, (), {"mask": orig_mask, "training": True}
        )
        self.assertTrue(torch.equal(kwargs["mask"], self.src_mask))
        self.assertTrue(kwargs["training"])

    def test_missing_source_file_keeps_original(self):
        """When source .pt does not exist, keep original tensor."""
        orig = torch.randn(4, 5)
        # use a full_forward_name that has no source file
        missing_ffn = "Module.nonexistent.Module.forward.99"
        args, kwargs = self.loader.override_args(missing_ffn, (orig,), {})
        self.assertTrue(torch.equal(args[0], orig))

    def test_cache_hit(self):
        """Second call with same key uses cached tensor (no re-load)."""
        orig_x = torch.zeros(2, 3)
        args1, _ = self.loader.override_args(self.ffn, (orig_x,), {})
        # corrupt the source file to prove cache is used
        os.remove(os.path.join(self.data_dir, f"{self.ffn}.input.0.pt"))
        args2, _ = self.loader.override_args(self.ffn, (orig_x,), {})
        self.assertTrue(torch.equal(args1[0], args2[0]))

    def test_cache_cleared_on_update_step_rank(self):
        """update_step_rank clears cache so next call re-loads from disk."""
        orig_x = torch.zeros(2, 3)
        self.loader.override_args(self.ffn, (orig_x,), {})
        os.remove(os.path.join(self.data_dir, f"{self.ffn}.input.0.pt"))
        self.loader.update_step_rank(0, 0)
        args, kwargs = self.loader.override_args(self.ffn, (orig_x,), {})
        self.assertTrue(torch.equal(args[0], orig_x))

    def test_override_preserves_device(self):
        """Loaded tensor is mapped to the original arg's device (cpu in test)."""
        orig_x = torch.zeros(2, 3)
        args, _ = self.loader.override_args(self.ffn, (orig_x,), {})
        self.assertEqual(args[0].device, orig_x.device)

    def test_empty_args_kwargs(self):
        args, kwargs = self.loader.override_args(self.ffn, (), {})
        self.assertEqual(args, ())
        self.assertEqual(kwargs, {})


SampleNT = namedtuple("SampleNT", ["proprio", "action"])


@dataclass
class SensorDataDC:
    exterior: torch.Tensor
    wrist_right: torch.Tensor


@dataclass(frozen=True)
class FrozenDC:
    x: torch.Tensor
    y: torch.Tensor


@dataclass
class InitFalseDC:
    x: torch.Tensor
    y: torch.Tensor = field(default=None, init=False)


class TestTensorLoaderOverrideNested(unittest.TestCase):
    """Tests for override_args recursively overriding tensors in nested containers."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.tmpdir, "step0", "rank0", "dump_tensor_data")
        os.makedirs(self.data_dir)
        self.ffn = "Module.nested.NestedModule.forward.0"

        def mk(v):
            return torch.full((4,), float(v))

        self.src = {
            f"{Const.INPUT}.0.action": mk(1),
            f"{Const.INPUT}.0.proprio": mk(2),
            f"{Const.INPUT}.0.pixel_values.exterior": mk(3),
            f"{Const.INPUT}.0.pixel_values.wrist_right": mk(4),
            f"{Const.INPUT}.0.samples.0.proprio.value": mk(5),
            f"{Const.INPUT}.0.samples.0.action.value": mk(6),
            f"{Const.KWARGS}.pixel_values.exterior": mk(7),
            f"{Const.KWARGS}.pixel_values.wrist_right": mk(8),
            f"{Const.INPUT}.1.0": mk(9),
            f"{Const.INPUT}.1.1": mk(10),
            f"{Const.INPUT}.2.0": mk(11),
            f"{Const.INPUT}.2.1": mk(12),
            f"{Const.INPUT}.3.proprio": mk(13),
            f"{Const.INPUT}.3.action": mk(14),
            f"{Const.INPUT}.4.exterior": mk(15),
            f"{Const.INPUT}.4.wrist_right": mk(16),
            f"{Const.INPUT}.5.x": mk(17),
            f"{Const.INPUT}.5.y": mk(18),
            f"{Const.INPUT}.6.x": mk(19),
            f"{Const.INPUT}.6.y": mk(20),
        }
        for suffix, tensor in self.src.items():
            torch.save(tensor, os.path.join(self.data_dir, f"{self.ffn}.{suffix}.pt"))

        cfg = FakeLoadConfig(self.tmpdir, [self.ffn])
        self.loader = TensorLoader(cfg)
        self.loader.update_step_rank(0, 0)

    def test_override_nested_dict_arg(self):
        data = {
            "action": torch.zeros(4),
            "proprio": torch.zeros(4),
            "pixel_values": {"exterior": torch.zeros(4), "wrist_right": torch.zeros(4)},
            "samples": [
                {"proprio": {"value": torch.zeros(4)}, "action": {"value": torch.zeros(4)}},
            ],
        }
        args, _ = self.loader.override_args(self.ffn, (data,), {})
        out = args[0]
        self.assertIsInstance(out, dict)
        self.assertTrue(torch.equal(out["action"], self.src["input.0.action"]))
        self.assertTrue(torch.equal(out["proprio"], self.src["input.0.proprio"]))
        self.assertTrue(torch.equal(out["pixel_values"]["exterior"], self.src["input.0.pixel_values.exterior"]))
        self.assertTrue(torch.equal(out["pixel_values"]["wrist_right"], self.src["input.0.pixel_values.wrist_right"]))
        self.assertTrue(
            torch.equal(out["samples"][0]["proprio"]["value"], self.src["input.0.samples.0.proprio.value"])
        )
        self.assertTrue(
            torch.equal(out["samples"][0]["action"]["value"], self.src["input.0.samples.0.action.value"])
        )
        # original structure is rebuilt, not mutated in place
        self.assertTrue(torch.equal(data["action"], torch.zeros(4)))

    def test_override_nested_dict_kwarg(self):
        pixel_values = {"exterior": torch.zeros(4), "wrist_right": torch.zeros(4)}
        _, kwargs = self.loader.override_args(self.ffn, (), {"pixel_values": pixel_values})
        self.assertTrue(torch.equal(kwargs["pixel_values"]["exterior"], self.src["kwargs.pixel_values.exterior"]))
        self.assertTrue(torch.equal(kwargs["pixel_values"]["wrist_right"], self.src["kwargs.pixel_values.wrist_right"]))

    def test_override_list_arg(self):
        lst = [torch.zeros(4), torch.zeros(4)]
        args, _ = self.loader.override_args(self.ffn, (None, lst), {})
        self.assertIsInstance(args[1], list)
        self.assertTrue(torch.equal(args[1][0], self.src["input.1.0"]))
        self.assertTrue(torch.equal(args[1][1], self.src["input.1.1"]))

    def test_override_tuple_arg_preserves_type(self):
        tup = (torch.zeros(4), torch.zeros(4))
        args, _ = self.loader.override_args(self.ffn, (None, None, tup), {})
        self.assertIsInstance(args[2], tuple)
        self.assertTrue(torch.equal(args[2][0], self.src["input.2.0"]))
        self.assertTrue(torch.equal(args[2][1], self.src["input.2.1"]))

    def test_override_torch_size_preserves_type(self):
        size = torch.Size([2, 3])
        args, _ = self.loader.override_args(self.ffn, (size,), {})
        self.assertIsInstance(args[0], torch.Size)
        self.assertEqual(tuple(args[0]), (2, 3))

    def test_override_namedtuple_arg_preserves_type(self):
        nt = SampleNT(proprio=torch.zeros(4), action=torch.zeros(4))
        args, _ = self.loader.override_args(self.ffn, (None, None, None, nt), {})
        self.assertIsInstance(args[3], SampleNT)
        self.assertTrue(torch.equal(args[3].proprio, self.src["input.3.proprio"]))
        self.assertTrue(torch.equal(args[3].action, self.src["input.3.action"]))

    def test_override_dataclass_arg_preserves_type(self):
        dc = SensorDataDC(exterior=torch.zeros(4), wrist_right=torch.zeros(4))
        args, _ = self.loader.override_args(self.ffn, (None, None, None, None, dc), {})
        self.assertIsInstance(args[4], SensorDataDC)
        self.assertTrue(torch.equal(args[4].exterior, self.src["input.4.exterior"]))
        self.assertTrue(torch.equal(args[4].wrist_right, self.src["input.4.wrist_right"]))

    def test_override_frozen_dataclass(self):
        dc = FrozenDC(x=torch.zeros(4), y=torch.zeros(4))
        args, _ = self.loader.override_args(self.ffn, (None, None, None, None, None, dc), {})
        self.assertIsInstance(args[5], FrozenDC)
        self.assertTrue(torch.equal(args[5].x, self.src["input.5.x"]))
        self.assertTrue(torch.equal(args[5].y, self.src["input.5.y"]))

    def test_override_init_false_dataclass(self):
        dc = InitFalseDC(x=torch.zeros(4))
        dc.y = torch.zeros(4)
        args, _ = self.loader.override_args(self.ffn, (None, None, None, None, None, None, dc), {})
        self.assertIsInstance(args[6], InitFalseDC)
        self.assertTrue(torch.equal(args[6].x, self.src["input.6.x"]))
        self.assertTrue(torch.equal(args[6].y, self.src["input.6.y"]))

    def test_non_tensor_leaf_preserved_in_nested(self):
        data = {"meta": {"n": 5, "name": "x"}, "flags": [True, None]}
        args, _ = self.loader.override_args(self.ffn, (data,), {})
        self.assertEqual(args[0]["meta"]["n"], 5)
        self.assertEqual(args[0]["meta"]["name"], "x")
        self.assertEqual(args[0]["flags"], [True, None])

    def test_missing_source_file_keeps_nested_original(self):
        data = {"missing_key": torch.zeros(4)}
        args, _ = self.loader.override_args(self.ffn, (data,), {})
        self.assertTrue(torch.equal(args[0]["missing_key"], torch.zeros(4)))


class TestTensorLoaderDisabled(unittest.TestCase):
    """Tests for disabled loader (empty modules)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_disabled_loader_should_not_override(self):
        cfg = FakeLoadConfig(self.tmpdir, [])
        loader = TensorLoader(cfg)
        self.assertFalse(loader.should_override("Module.any.Thing.forward.0"))

    def test_disabled_loader_override_args_passthrough(self):
        cfg = FakeLoadConfig(self.tmpdir, [])
        loader = TensorLoader(cfg)
        orig = torch.randn(2, 2)
        args, kwargs = loader.override_args("Module.any.Thing.forward.0", (orig,), {"k": orig})
        self.assertTrue(torch.equal(args[0], orig))
        self.assertTrue(torch.equal(kwargs["k"], orig))


if __name__ == "__main__":
    unittest.main()
