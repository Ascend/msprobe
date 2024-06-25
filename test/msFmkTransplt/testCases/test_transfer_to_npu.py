#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

import os
import sys
import unittest
from unittest import mock

import torch

sys.path.append(os.path.abspath("../../../"))
sys.path.append(os.path.abspath("../../../src/ms_fmk_transplt"))

npu_mock = mock.Mock()
ori_import = __import__
CUDA = "cuda"
NPU = "npu"


def import_mock(name, *args):
    if name == "torch_npu" or name == "torchair":
        return npu_mock
    return ori_import(name, *args)


def device_func(device):
    return True if device == NPU else False


def hccl_func(arg):
    return True if arg == "hccl" else False


def data_loader_func(arg, pin_memory=False, pin_memory_device=None):
    if pin_memory and pin_memory_device == NPU:
        return True
    elif not pin_memory:
        return True
    else:
        return False


class TestTransferToNpu(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with mock.patch('builtins.__import__', side_effect=import_mock):
            torch.Tensor.npu = torch.Tensor.cuda
            torch.Tensor.is_npu = torch.Tensor.is_cuda
            torch.npu = torch.cuda
            torch.nn.Module.npu = torch.nn.Module.cuda
            torch.distributed.is_hccl_available = torch.distributed.is_nccl_available
            torch.npu.amp.autocast_mode.npu_autocast = torch.cuda.amp.autocast_mode.autocast
            torch.distributed.ProcessGroup._get_backend = torch.distributed.ProcessGroup
            cls.rand = torch.rand
            cls.profile = torch.profiler.profile
            cls.jit = torch.jit.script
            from src.ms_fmk_transplt.torch_npu_bridge import transfer_to_npu
            cls.transfer_to_npu = transfer_to_npu

    def test_is_torch_version_greater_than_2_1(self):
        result = self.transfer_to_npu._is_torch_version_greater_than_2_1()
        version = torch.__version__
        if '1.11' in version or '2.0' in version:
            self.assertFalse(result)
        elif '2.1' in version:
            self.assertTrue(result)

    def test_wrapper_cuda(self):
        func = self.transfer_to_npu._wrapper_cuda(device_func)
        self.assertTrue(func(CUDA))

    def test_wrapper_hccl(self):
        func = self.transfer_to_npu._wrapper_hccl(hccl_func)
        self.assertTrue(func("nccl"))

    def test_replace_cuda_to_npu_in_kwargs(self):
        device = "device"
        device_type = "device_type"
        map_location = "map_location"
        device_kwargs_list = [device, device_type, map_location, device + "0"]
        kwargs = {device_type: CUDA, device: "cuda:0", map_location: 0, device + "0": {CUDA: NPU}}
        for item in device_kwargs_list:
            self.transfer_to_npu._replace_cuda_to_npu_in_kwargs(kwargs, item, kwargs.get(item, device))
        self.assertEqual(kwargs.get(device_type), NPU)
        self.assertEqual(kwargs.get(device), "npu:0")
        self.assertEqual(kwargs.get(map_location), "npu:0")
        self.assertEqual(kwargs.get(device + "0"), {NPU: NPU})

    def test_replace_cuda_to_npu_in_list(self):
        args_list = [CUDA, 0]
        self.transfer_to_npu._replace_cuda_to_npu_in_list(args_list, True)
        self.assertEqual(args_list[0], NPU)
        self.assertEqual(args_list[1], "npu:0")

    def test_replace_cuda_to_npu_in_dict(self):
        device_dict = {CUDA: NPU}
        new_dict = self.transfer_to_npu._replace_cuda_to_npu_in_dict(device_dict)
        self.assertEqual(new_dict, {NPU: NPU})

    def test_wrapper_profiler(self):
        self.profile = self.transfer_to_npu._wrapper_profiler(self.profile)
        self.profile(experimental_config=1)

    def test_jit_script(self):
        self.jit = self.transfer_to_npu._jit_script
        self.jit("test")

    def test_wrapper_data_loader(self):
        func = self.transfer_to_npu._wrapper_data_loader(data_loader_func)
        self.assertTrue(func(None, pin_memory=False, pin_memory_device=None))
        self.assertTrue(func(None, pin_memory=True, pin_memory_device=None))
        self.assertTrue(func(None, pin_memory=True, pin_memory_device=CUDA))

    def test_device_wrapper(self):
        self.transfer_to_npu._device_wrapper(self.rand, ["rand"])

    def test_warning_fn(self):
        self.transfer_to_npu._warning_fn("warning_fn success")

    def test_patch(self):
        self.transfer_to_npu._patch_cuda()
        self.transfer_to_npu._patch_profiler()
