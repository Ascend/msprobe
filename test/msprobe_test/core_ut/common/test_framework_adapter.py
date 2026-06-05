"""FmkAdp 单元测试 — 测试框架适配器在 PyTorch 模式下的行为。"""
import os
import tempfile
import unittest
import numpy as np
from unittest.mock import patch, MagicMock, PropertyMock

from msprobe.core.common.const import Const
from msprobe.core.common.framework_adapter import FmkAdp


class TestFmkAdp(unittest.TestCase):

    def setUp(self):
        # Reset to PT mode before each test
        FmkAdp.set_fmk(Const.PT_FRAMEWORK)
        FmkAdp._framework = None

    def tearDown(self):
        FmkAdp.set_fmk(Const.PT_FRAMEWORK)
        FmkAdp._framework = None

    # ── set_fmk ──

    def test_set_fmk_valid(self):
        FmkAdp.set_fmk(Const.PT_FRAMEWORK)
        self.assertEqual(FmkAdp.fmk, Const.PT_FRAMEWORK)
        self.assertIsNone(FmkAdp._framework)

    def test_set_fmk_invalid_raises(self):
        with self.assertRaises(Exception):
            FmkAdp.set_fmk("unknown_framework")

    # ── framework property & import_framework ──

    def test_framework_property_imports_torch(self):
        FmkAdp._framework = None
        fmk = FmkAdp.framework
        self.assertEqual(fmk.__name__, "torch")
        self.assertIsNotNone(FmkAdp._framework)

    # ── get_rank_id ──

    @patch("msprobe.core.common.framework_adapter.FmkAdp.is_initialized", return_value=False)
    def test_get_rank_id_not_initialized(self, mock_init):
        FmkAdp._framework = MagicMock()
        rank = FmkAdp.get_rank_id()
        self.assertEqual(rank, 0)

    @patch("msprobe.core.common.framework_adapter.FmkAdp.is_initialized", return_value=True)
    @patch("msprobe.core.common.framework_adapter.FmkAdp.get_rank", return_value=3)
    def test_get_rank_id_initialized(self, mock_get_rank, mock_init):
        rank = FmkAdp.get_rank_id()
        self.assertEqual(rank, 3)

    # ── is_initialized (PT mode) ──

    @patch("msprobe.core.common.framework_adapter.FmkAdp.framework", new_callable=PropertyMock)
    def test_is_initialized_pt(self, mock_framework):
        mock_dist = MagicMock()
        mock_dist.is_initialized.return_value = True
        mock_framework.return_value.distributed = mock_dist
        self.assertTrue(FmkAdp.is_initialized())

    # ── is_tensor ──

    def test_is_tensor_with_torch_tensor(self):
        import torch
        t = torch.tensor([1.0])
        self.assertTrue(FmkAdp.is_tensor(t))

    def test_is_tensor_with_non_tensor(self):
        self.assertFalse(FmkAdp.is_tensor("not a tensor"))

    # ── is_nn_module ──

    def test_is_nn_module_with_module(self):
        import torch
        m = torch.nn.Linear(2, 2)
        self.assertTrue(FmkAdp.is_nn_module(m))

    def test_is_nn_module_with_non_module(self):
        self.assertFalse(FmkAdp.is_nn_module("not a module"))

    # ── dtype ──

    def test_dtype_valid(self):
        import torch
        result = FmkAdp.dtype("float32")
        self.assertEqual(result, torch.float32)

    def test_dtype_invalid_raises(self):
        with self.assertRaises(Exception):
            FmkAdp.dtype("unsupported_dtype")

    # ── process_tensor ──

    def test_process_tensor_float(self):
        import torch
        t = torch.tensor([1.0, 2.0, 3.0])
        result = FmkAdp.process_tensor(t, lambda x: x.max())
        self.assertAlmostEqual(result, 3.0)

    def test_process_tensor_int64_converts_to_float(self):
        import torch
        t = torch.tensor([1, 2, 3], dtype=torch.int64)
        result = FmkAdp.process_tensor(t, lambda x: x.max())
        self.assertAlmostEqual(result, 3.0)

    # ── tensor_max / min / mean / norm ──

    def test_tensor_max(self):
        import torch
        t = torch.tensor([1.0, 5.0, 3.0])
        self.assertAlmostEqual(FmkAdp.tensor_max(t), 5.0)

    def test_tensor_min(self):
        import torch
        t = torch.tensor([1.0, 5.0, 3.0])
        self.assertAlmostEqual(FmkAdp.tensor_min(t), 1.0)

    def test_tensor_mean(self):
        import torch
        t = torch.tensor([1.0, 2.0, 3.0])
        self.assertAlmostEqual(FmkAdp.tensor_mean(t), 2.0)

    def test_tensor_norm(self):
        import torch
        t = torch.tensor([3.0, 4.0])
        self.assertAlmostEqual(FmkAdp.tensor_norm(t), 5.0)

    # ── save_tensor ──

    @patch("msprobe.core.common.framework_adapter.save_npy")
    def test_save_tensor_pt(self, mock_save_npy):
        import torch
        t = torch.tensor([1.0, 2.0])
        tmpdir = tempfile.mkdtemp()
        filepath = os.path.join(tmpdir, "test.npy")
        try:
            FmkAdp.save_tensor(t, filepath)
            mock_save_npy.assert_called_once()
            args, _ = mock_save_npy.call_args
            self.assertIsInstance(args[0], np.ndarray)
            self.assertEqual(args[1], filepath)
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
            os.rmdir(tmpdir)

    # ── asnumpy ──

    def test_asnumpy(self):
        import torch
        t = torch.tensor([1.0, 2.0])
        result = FmkAdp.asnumpy(t)
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([1.0, 2.0]))

    # ── named_parameters ──

    def test_named_parameters_valid_module(self):
        import torch
        m = torch.nn.Linear(2, 2)
        params = list(FmkAdp.named_parameters(m))
        self.assertTrue(len(params) > 0)
        for name, param in params:
            self.assertIsInstance(name, str)

    def test_named_parameters_invalid(self):
        with self.assertRaises(Exception):
            FmkAdp.named_parameters("not a module")

    # ── iter_named_modules ──

    def test_iter_named_modules(self):
        import torch
        m = torch.nn.Linear(2, 2)
        modules = list(FmkAdp.iter_named_modules(m))
        self.assertTrue(len(modules) > 0)

    def test_iter_named_modules_invalid(self):
        with self.assertRaises(Exception):
            FmkAdp.iter_named_modules("not a model")

    # ── register_forward_pre_hook ──

    def test_register_forward_pre_hook(self):
        import torch
        m = torch.nn.Linear(2, 2)
        hook_called = False

        def hook(module, inputs):
            nonlocal hook_called
            hook_called = True

        FmkAdp.register_forward_pre_hook(m, hook)
        dummy = torch.randn(1, 2)
        m(dummy)
        self.assertTrue(hook_called)

    def test_register_forward_pre_hook_invalid(self):
        with self.assertRaises(Exception):
            FmkAdp.register_forward_pre_hook("not a module", lambda m, i: None)

    # ── load_checkpoint ──

    @patch("msprobe.core.common.framework_adapter.check_file_or_directory_path")
    def test_load_checkpoint_to_cpu(self, mock_check):
        import torch
        fd, path = tempfile.mkstemp(suffix=".pt", text=False)
        os.close(fd)
        try:
            # Save a small model to create a valid checkpoint
            m = torch.nn.Linear(2, 2)
            torch.save(m.state_dict(), path)

            state = FmkAdp.load_checkpoint(path, to_cpu=True, weights_only=True)
            self.assertIsInstance(state, dict)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
