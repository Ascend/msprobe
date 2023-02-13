#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

import os
import sys
import unittest

import torch

sys.path.append(os.path.abspath("../../../"))
sys.path.append(os.path.abspath("../../../src/ms_fmk_transplt"))


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
