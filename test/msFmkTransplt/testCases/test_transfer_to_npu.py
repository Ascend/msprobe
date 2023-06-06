#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

import os
import sys
import unittest

import torch

sys.path.append(os.path.abspath("../../../"))
sys.path.append(os.path.abspath("../../../src/ms_fmk_transplt"))

try:
    import torch_npu

    TORCH_NPU_AVAILABLE = True
except ImportError:
    TORCH_NPU_AVAILABLE = False


@unittest.skipIf(not TORCH_NPU_AVAILABLE, reason='torch_npu is not available')
class TestTransferToNpu(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from src.ms_fmk_transplt.torch_npu_bridge import transfer_to_npu

    def test_wrap_isinstance(self):
        # check builtins isinstance grammar
        self.assertTrue(isinstance(1, int))
        self.assertTrue(isinstance(1, (int, str)))
        self.assertFalse(isinstance(1, str))
        with self.assertRaises(TypeError):
            isinstance(1, [str, int])

        # check torch.device
        self.assertFalse(isinstance(1, torch.device))

        # check torch.cuda.device
        device = -1
        torch.cuda.device(device)

        # test multi imports
        import torch_npu
        from src.ms_fmk_transplt.torch_npu_bridge import transfer_to_npu
        import torch_npu
        self.assertFalse(isinstance(1, torch.device))

    def test_amp_function(self):
        self.assertEqual(torch.cuda.amp.autocast_mode, torch_npu.npu.amp.autocast_mode)
        self.assertEqual(torch.cuda.amp.common, torch_npu.npu.amp.common)
        self.assertEqual(torch.cuda.amp.grad_scaler, torch_npu.npu.amp.grad_scaler)

    def test_wrap_device(self):
        device = torch.device(f"cuda:{0}")
        torch.cuda.set_device(device)
        a = torch.randint(1, 5, (2, 3), device=device)
        self.assertEqual(a.device.type, 'npu')

    def test_patch_profiler(self):
        self.assertEqual(torch.profiler.profile.export_chrome_trace, torch_npu.profiler.profile.export_chrome_trace)
        self.assertEqual(torch.profiler.profile.step, torch_npu.profiler.profile.step)
        self.assertEqual(torch.profiler.ProfilerAction, torch_npu.profiler.ProfilerAction)
        self.assertEqual(torch.profiler.schedule, torch_npu.profiler.schedule)
        self.assertEqual(torch.profiler.tensorboard_trace_handler, torch_npu.profiler.tensorboard_trace_handler)
        self.assertEqual(torch.profiler.ProfilerActivity.CUDA, torch_npu.profiler.ProfilerActivity.NPU)
        self.assertEqual(torch.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.CPU)
        self.assertIsInstance(torch.profiler.profile(experimental_config=1), torch_npu.profiler.profile)
