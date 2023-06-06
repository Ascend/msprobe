#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

import builtins
import os
import warnings
from functools import wraps
import torch
import torch_npu

warnings.filterwarnings(action='once')

torch_fn_white_list = [
    'logspace', 'randint', 'hann_window', 'rand', 'full_like', 'ones_like', 'rand_like', 'randperm',
    'arange', 'frombuffer', 'normal', '_empty_per_channel_affine_quantized', 'empty_strided',
    'empty_like', 'scalar_tensor', 'tril_indices', 'bartlett_window', 'ones', 'sparse_coo_tensor',
    'randn', 'kaiser_window', 'tensor', 'triu_indices', 'as_tensor', 'zeros', 'randint_like', 'full',
    'eye', '_sparse_csr_tensor_unsafe', 'empty', '_sparse_coo_tensor_unsafe', 'blackman_window',
    'zeros_like', 'range', 'sparse_csr_tensor', 'randn_like', 'from_file',
    '_cudnn_init_dropout_state', '_empty_affine_quantized', 'linspace', 'hamming_window',
    'empty_quantized', '_pin_memory'
]
torch_tensor_fn_white_list = ['new_empty', 'new_empty_strided', 'new_full', 'new_ones', 'new_tensor', 'new_zeros', 'to']
torch_module_fn_white_list = ['to', 'to_empty']
torch_cuda_fn_white_list = [
    'get_device_properties', 'get_device_name', 'get_device_capability', 'list_gpu_processes', 'set_device',
    'synchronize', 'mem_get_info', 'memory_stats', 'memory_summary', 'memory_allocated', 'max_memory_allocated',
    'reset_max_memory_allocated', 'memory_reserved', 'max_memory_reserved', 'reset_max_memory_cached'
]
torch_profiler_fn_white_list = ['profile']


def wrapper_cuda(func):
    @wraps(func)
    def decorated(*args, **kwargs):
        if args:
            args_new = list(args)
            for idx, arg in enumerate(args_new):
                if isinstance(arg, str) and 'cuda' in arg:
                    args_new[idx] = arg.replace('cuda', 'npu')
                if isinstance(arg, torch.device) and 'cuda' in arg.type:
                    device_info = 'npu:{}'.format(arg.index) if arg.index is not None else 'npu'
                    args_new[idx] = torch.device(device_info)
            args = args_new
        if kwargs:
            if isinstance(kwargs.get('device', None), str) and 'cuda' in kwargs.get('device', ''):
                kwargs['device'] = kwargs.get('device').replace('cuda', 'npu')
            device = kwargs.get('device', None)
            if isinstance(device, torch.device) and 'cuda' in device.type:
                device_info = 'npu:{}'.format(device.index) if device.index is not None else 'npu'
                kwargs['device'] = torch.device(device_info)
            # delete the experimental_config parameter of torch.profiler.profile
            if 'experimental_config' in kwargs.keys():
                del kwargs['experimental_config']
        return func(*args, **kwargs)

    return decorated


def device_wrapper(enter_fn, white_list):
    for fn_name in white_list:
        func = getattr(enter_fn, fn_name, None)
        if func:
            setattr(enter_fn, fn_name, wrapper_cuda(func))


def wrapper_hccl(func):
    @wraps(func)
    def decorated(*args, **kwargs):
        if args:
            args_new = list(args)
            for idx, arg in enumerate(args_new):
                if isinstance(arg, str) and 'nccl' in arg:
                    args_new[idx] = arg.replace('nccl', 'hccl')
            args = args_new
        if kwargs:
            if isinstance(kwargs.get('backend', None), str):
                kwargs['backend'] = 'hccl'
        return func(*args, **kwargs)

    return decorated


def patch_cuda():
    patchs = [
        ['cuda', torch_npu.npu], ['cuda.amp', torch_npu.npu.amp],
        ['cuda.random', torch_npu.npu.random],
        ['cuda.amp.autocast_mode', torch_npu.npu.amp.autocast_mode],
        ['cuda.amp.common', torch_npu.npu.amp.common],
        ['cuda.amp.grad_scaler', torch_npu.npu.amp.grad_scaler]
    ]
    torch_npu._apply_patches(patchs)


def patch_profiler():
    patchs = [
        ['profiler.profile', torch_npu.profiler.profile],
        ['profiler.schedule', torch_npu.profiler.schedule],
        ['profiler.tensorboard_trace_handler', torch_npu.profiler.tensorboard_trace_handler],
        ['profiler.ProfilerAction', torch_npu.profiler.ProfilerAction],
        ['profiler.ProfilerActivity.CUDA', torch_npu.profiler.ProfilerActivity.NPU],
        ['profiler.ProfilerActivity.CPU', torch_npu.profiler.ProfilerActivity.CPU]
    ]
    torch_npu._apply_patches(patchs)


def warning_fn(msg, rank0=True):
    is_distributed = torch.distributed.is_available() and \
                     torch.distributed.is_initialized() and \
                     torch.distributed.get_world_size() > 1
    env_rank = os.getenv('RANK', None)

    if rank0 and is_distributed:
        if torch.distributed.get_rank() == 0:
            warnings.warn(msg, ImportWarning)
    elif rank0 and env_rank:
        if env_rank == '0':
            warnings.warn(msg, ImportWarning)
    else:
        warnings.warn(msg, ImportWarning)


def wrapped_isinstance(obj, class_or_tuple):
    try:
        return torch_npu._isinstance(obj, class_or_tuple)
    except TypeError as exp:
        class_tuple = (class_or_tuple,) if type(class_or_tuple) != tuple else class_or_tuple
        if torch.device not in class_tuple:
            raise exp
        class_list = []
        for type_item in class_tuple:
            if type_item is torch.device:
                class_list.append(torch_npu._C.device)
            else:
                class_list.append(type_item)
        return torch_npu._isinstance(obj, tuple(class_list))


def init():
    warning_fn('''
    *************************************************************************************************************
    The torch.Tensor.cuda and torch.nn.Module.cuda are replaced with torch.Tensor.npu and torch.nn.Module.npu now..
    The torch.cuda.DoubleTensor is replaced with torch.npu.FloatTensor cause the double type is not supported now..
    The backend in torch.distributed.init_process_group set to hccl now..
    The torch.cuda.* and torch.cuda.amp.* are replaced with torch.npu.* and torch.npu.amp.* now..
    The device parameters have been replaced with npu in the function below:
    {}
    *************************************************************************************************************
    '''.format(', '.join(
        ['torch.' + i for i in torch_fn_white_list] + ['torch.Tensor.' + i for i in torch_tensor_fn_white_list] +
        ['torch.nn.Module.' + i for i in torch_module_fn_white_list]))
    )

    # torch.cuda.*
    patch_cuda()
    device_wrapper(torch.cuda, torch_cuda_fn_white_list)

    # torch.profiler.*
    patch_profiler()
    device_wrapper(torch.profiler, torch_profiler_fn_white_list)

    # torch.*
    device_wrapper(torch, torch_fn_white_list)

    # wrap isinstance for torch.device
    builtins.isinstance = wrapped_isinstance

    # torch.Tensor.*
    device_wrapper(torch.Tensor, torch_tensor_fn_white_list)
    torch.Tensor.cuda = torch.Tensor.npu
    torch.Tensor.is_cuda = torch.Tensor.is_npu
    torch.cuda.DoubleTensor = torch.npu.FloatTensor

    # torch.nn.Module.*
    device_wrapper(torch.nn.Module, torch_module_fn_white_list)
    torch.nn.Module.cuda = torch.nn.Module.npu

    # torch.distributed.init_process_group
    torch.distributed.init_process_group = wrapper_hccl(torch.distributed.init_process_group)


init()
