#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

import os
import warnings
import logging as logger
import functools
from functools import wraps
import torch
import torch_npu


warnings.filterwarnings(action='once')
LOG_FORMAT = '%(asctime)s [%(levelname)s] %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
logger.basicConfig(format=LOG_FORMAT, datefmt=DATE_FORMAT)

torch_fn_white_list = [
    'logspace', 'randint', 'hann_window', 'rand', 'full_like', 'ones_like', 'rand_like', 'randperm',
    'arange', 'frombuffer', 'normal', '_empty_per_channel_affine_quantized', 'empty_strided',
    'empty_like', 'scalar_tensor', 'tril_indices', 'bartlett_window', 'ones', 'sparse_coo_tensor',
    'randn', 'kaiser_window', 'tensor', 'triu_indices', 'as_tensor', 'zeros', 'randint_like', 'full',
    'eye', '_sparse_csr_tensor_unsafe', 'empty', '_sparse_coo_tensor_unsafe', 'blackman_window',
    'zeros_like', 'range', 'sparse_csr_tensor', 'randn_like', 'from_file',
    '_cudnn_init_dropout_state', '_empty_affine_quantized', 'linspace', 'hamming_window',
    'empty_quantized', '_pin_memory', 'autocast', 'load'
]
torch_tensor_fn_white_list = ['new_empty', 'new_empty_strided', 'new_full', 'new_ones', 'new_tensor', 'new_zeros', 'to']
torch_module_fn_white_list = ['to', 'to_empty']
torch_cuda_fn_white_list = [
    'get_device_properties', 'get_device_name', 'get_device_capability', 'list_gpu_processes', 'set_device',
    'synchronize', 'mem_get_info', 'memory_stats', 'memory_summary', 'memory_allocated', 'max_memory_allocated',
    'reset_max_memory_allocated', 'memory_reserved', 'max_memory_reserved', 'reset_max_memory_cached',
    'reset_peak_memory_stats', 'device'
]
torch_profiler_fn_white_list = ['profile']
torch_distributed_fn_white_list = ['__init__']
device_kwargs_list = ['device', 'device_type', 'map_location']
CUDA = 'cuda'
NPU = 'npu'


def is_torch_version_greater_than_2_1():
    major, minor = torch.__version__.split('.')[:2]
    if int(major) > 2 or (int(major) == 2 and int(minor) >= 1):
        return True
    else:
        return False


if is_torch_version_greater_than_2_1():
    import torchair


def wrapper_cuda(func):
    @wraps(func)
    def decorated(*args, **kwargs):
        replace_int = func.__name__ in ['to', 'to_empty']
        if args:
            args_new = list(args)
            args = replace_cuda_to_npu_in_list(args_new, replace_int)
        if kwargs:
            for device_arg in device_kwargs_list:
                device = kwargs.get(device_arg, None)
                if device is not None:
                    replace_cuda_to_npu_in_kwargs(kwargs, device_arg, device)
            device_ids = kwargs.get('device_ids', None)
            if type(device_ids) == list:
                replace_cuda_to_npu_in_list(device_ids, replace_int)
        return func(*args, **kwargs)

    return decorated


def replace_cuda_to_npu_in_kwargs(kwargs, device_arg, device):
    if type(device) == str and CUDA in device:
        kwargs[device_arg] = device.replace(CUDA, NPU)
    elif (type(device) == torch.device or str(type(device)) == "<class 'torch.device'>") and CUDA in device.type:
        device_info = 'npu:{}'.format(device.index) if device.index is not None else NPU
        kwargs[device_arg] = torch.device(device_info)
    elif type(device) == int:
        kwargs[device_arg] = f'npu:{device}'
    elif type(device) == dict:
        kwargs[device_arg] = replace_cuda_to_npu_in_dict(device)


def replace_cuda_to_npu_in_list(args_list, replace_int):
    for idx, arg in enumerate(args_list):
        if isinstance(arg, str) and CUDA in arg:
            args_list[idx] = arg.replace(CUDA, NPU)
        elif (isinstance(arg, torch.device) or str(type(arg)) == "<class 'torch.device'>") and CUDA in arg.type:
            device_info = 'npu:{}'.format(arg.index) if arg.index is not None else NPU
            args_list[idx] = torch.device(device_info)
        elif replace_int and not isinstance(arg, bool) and isinstance(arg, int):
            args_list[idx] = f'npu:{arg}'
        elif isinstance(arg, dict):
            args_list[idx] = replace_cuda_to_npu_in_dict(arg)
    return args_list


def replace_cuda_to_npu_in_dict(device_dict):
    new_dict = {}
    for key, value in device_dict.items():
        if isinstance(key, str):
            key = key.replace('cuda', 'npu')
        if isinstance(value, str):
            value = value.replace('cuda', 'npu')
        new_dict[key] = value
    return new_dict


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


def wrapper_data_loader(func):
    @wraps(func)
    def decorated(*args, **kwargs):
        if kwargs:
            pin_memory_device_key = 'pin_memory_device'
            pin_memory = kwargs.get('pin_memory', False)
            pin_memory_device = kwargs.get(pin_memory_device_key, None)
            if pin_memory and not pin_memory_device:
                kwargs[pin_memory_device_key] = 'npu'
            if pin_memory and isinstance(pin_memory_device, str) and 'cuda' in pin_memory_device:
                kwargs[pin_memory_device_key] = pin_memory_device.replace('cuda', 'npu')
        return func(*args, **kwargs)

    return decorated


def wrapper_profiler(fn):
    @wraps(fn)
    def decorated(*args, **kwargs):
        if kwargs:
            key = 'experimental_config'
            if key in kwargs.keys() and \
                    type(kwargs.get(key)) != torch_npu.profiler._ExperimentalConfig:
                logger.warning(
                    'The parameter experimental_config of torch.profiler.profile has been deleted by the tool '
                    'because it can only be used in cuda, please manually modify the code '
                    'and use the experimental_config parameter adapted to npu.')
                del kwargs[key]
        return fn(*args, **kwargs)

    return decorated


def wrapper_compile(fn):
    @wraps(fn)
    def decorated(*args, **kwargs):
        npu_backend = torchair.get_npu_backend()
        key = 'backend'
        if kwargs:
            backend = kwargs.get(key, None)
            if not backend or not isinstance(backend, functools.partial) or not isinstance(backend.func,
                                                                                           type(npu_backend.func)):
                kwargs[key] = npu_backend
        else:
            kwargs[key] = npu_backend
        return fn(*args, **kwargs)

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


def jit_script(obj, optimize=None, _frames_up=0, _rcb=None, example_inputs=None):
    msg = 'torch.jit.script will be disabled by transfer_to_npu, which currently does not support it.'
    warnings.warn(msg, RuntimeWarning)
    return obj


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
    torch.profiler.profile = wrapper_profiler(torch.profiler.profile)

    # torch.*
    device_wrapper(torch, torch_fn_white_list)

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
    torch.distributed.is_nccl_available = torch.distributed.is_hccl_available

    # torch.nn.parallel.DistributedDataParallel
    device_wrapper(torch.nn.parallel.DistributedDataParallel, torch_distributed_fn_white_list)
    # torch.utils.data.DataLoader
    if is_torch_version_greater_than_2_1():
        torch.utils.data.DataLoader.__init__ = wrapper_data_loader(torch.utils.data.DataLoader.__init__)
        torch.jit.script = jit_script
        torch.compile = wrapper_compile(torch.compile)
        torch._dynamo.allowed_functions._disallowed_function_ids.function_ids = None

    # torch version < 2.1 needs to be adapted
    if not is_torch_version_greater_than_2_1():
        torch.distributed.distributed_c10d.broadcast_object_list \
            = torch_npu.distributed.distributed_c10d.broadcast_object_list
        torch.distributed.distributed_c10d.all_gather_object \
            = torch_npu.distributed.distributed_c10d.all_gather_object
        torch.npu.amp.autocast_mode.npu_autocast.__init__ = wrapper_cuda(
            torch.npu.amp.autocast_mode.npu_autocast.__init__)
    # torch 1.11.0 does not have this API
    if '1.11.0' not in torch.__version__:
        torch.distributed.ProcessGroup._get_backend = wrapper_cuda(torch.distributed.ProcessGroup._get_backend)

    if '2.2.0' in torch.__version__:
        from torch.utils._triton import has_triton

        def patch_has_triton():
            return False

        setattr(torch.utils._triton, 'has_triton', patch_has_triton)


init()
