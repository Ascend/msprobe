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
        """When src_step/src_rank are empty list, uses current_step/current_rank."""
        cfg = FakeLoadConfig(self.tmpdir, ["m"])
        loader = TensorLoader(cfg)
        loader.update_step_rank(3, 2)
        ffn = "Module.head.Linear.forward.1"
        path = loader._build_pt_path(ffn, Const.INPUT, 0)
        self.assertIn("step3", path)
        self.assertIn("rank2", path)

    def test_explicit_step_rank_overrides_current(self):
        """src_step/src_rank (non-empty list) take priority over current_step/current_rank."""
        cfg = FakeLoadConfig(self.tmpdir, ["m"], step=[5], rank=[7])
        loader = TensorLoader(cfg)
        loader.update_step_rank(3, 2)
        ffn = "Module.head.Linear.forward.0"
        path = loader._build_pt_path(ffn, Const.INPUT, 0)
        self.assertIn("step5", path)
        self.assertIn("rank7", path)

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
