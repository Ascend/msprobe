#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
import os
import sys
import warnings
import json
import logging as logger
import functools
from functools import wraps
from enum import Enum
import torch
from torch.nn.parameter import UninitializedTensorMixin
import torch_npu

try:
    from packaging.version import Version as Version
except ImportError:
    from distutils.version import LooseVersion as Version
try:
    from torch.utils._device import _device_constructors
except ImportError:
    DO_DEVICE_CONSTRUCTORS = False
else:
    DO_DEVICE_CONSTRUCTORS = True
try:
    from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel
except ImportError:
    DO_FSDP_WRAP = False
else:
    DO_FSDP_WRAP = True

if DO_DEVICE_CONSTRUCTORS:
    _device_constructors()

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
    'empty_quantized', '_pin_memory', 'autocast', 'load', "Generator", 'set_default_device'
]
torch_tensor_fn_white_list = ['new_empty', 'new_empty_strided', 'new_full', 'new_ones', 'new_tensor', 'new_zeros', 'to']
torch_module_fn_white_list = ['to', 'to_empty']
torch_cuda_fn_white_list = [
    'get_device_properties', 'get_device_name', 'get_device_capability', 'list_gpu_processes', 'set_device',
    'synchronize', 'mem_get_info', 'memory_stats', 'memory_summary', 'memory_allocated', 'max_memory_allocated',
    'reset_max_memory_allocated', 'memory_reserved', 'max_memory_reserved', 'reset_max_memory_cached',
    'reset_peak_memory_stats'
]
torch_profiler_fn_white_list = ['profile']
torch_distributed_fn_white_list = ['__init__']
device_kwargs_list = ['device', 'device_type', 'map_location', 'device_id']
CUDA = 'cuda'
NPU = 'npu'
is_available = torch.cuda.is_available
cur_path = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(cur_path, 'apis_config.json')
python_version = sys.version_info
NCCL = 'nccl'
HCCL = 'hccl'


class ApiType(Enum):
    METHOD = 'method'
    FUNCTION = 'function'


def _is_torch_version_greater_than_2_x(x: int):
    major, minor = torch.__version__.split('.')[:2]
    if int(major) > 2 or (int(major) == 2 and int(minor) > x):
        return True
    else:
        return False


if python_version >= (3, 8):
    import importlib.metadata
else:
    import importlib
    import pkg_resources


def _get_function_from_string(attribute_string):
    try:
        module_path, _, attr_name = attribute_string.rpartition('.')
        module = importlib.import_module(module_path)
        return [module, attr_name]
    except BaseException:
        return []


def _get_method_from_string(attribute_string):
    try:
        parts = attribute_string.split('.')
        module_path = '.'.join(parts[:-2])
        class_name = parts[-2]
        attr_name = parts[-1]
        module = getattr(importlib.import_module(module_path), class_name)
        return [module, attr_name]
    except BaseException:
        return []


def _get_package_version(package_name):
    if python_version >= (3, 8):
        try:
            return importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            return ""
    else:
        try:
            return pkg_resources.get_distribution(package_name).version
        except pkg_resources.DistributionNotFound:
            return ""


def _compare_versions(current_version, version):
    return Version(current_version) >= Version(version)


def _check_input_file_valid(file_path, max_file_size=10 * 1024 ** 2):
    if os.path.islink(os.path.abspath(file_path)):
        return False
    input_path = os.path.realpath(file_path)
    if not os.path.exists(input_path):
        return False
    if not os.access(input_path, os.R_OK):
        return False
    if not len(os.path.basename(input_path)) <= 200:
        return False
    if os.path.getsize(input_path) > max_file_size:
        return False
    return True


def _load_json_file(file_path):
    if not _check_input_file_valid(file_path):
        return {}
    try:
        with open(file_path, 'r') as file:
            file_dict = json.load(file)
            if not isinstance(file_dict, dict):
                return {}
            return file_dict
    except json.JSONDecodeError:
        return {}


def _wrapper_libraries_func(fn):
    @wraps(fn)
    def decorated(*args, **kwargs):
        patched_is_available = torch.cuda.is_available
        torch.cuda.is_available = is_available
        result = fn(*args, **kwargs)
        torch.cuda.is_available = patched_is_available
        return result

    return decorated


def _set_attr_wrapper_func(apis_dict):
    for full_name, api_type in apis_dict.items():
        modules = None
        if api_type == ApiType.METHOD.value:
            modules = _get_method_from_string(full_name)
        elif api_type == ApiType.FUNCTION.value:
            modules = _get_function_from_string(full_name)
        if modules and getattr(modules[0], modules[1], None):
            setattr(modules[0], modules[1], _wrapper_libraries_func(getattr(modules[0], modules[1])))


def _do_wrapper_libraries_func(json_dict):
    for key, value in json_dict.items():
        current_version = _get_package_version(key)
        if not current_version:
            continue
        version = value.get('version')
        apis: dict = value.get('apis')
        if version and apis and _compare_versions(current_version, version):
            _set_attr_wrapper_func(apis)


def _wrapper_cuda(func):
    @wraps(func)
    def decorated(*args, **kwargs):
        replace_int = func.__name__ in ['to', 'to_empty']
        if args:
            args_new = list(args)
            args = _replace_cuda_to_npu_in_list(args_new, replace_int)
        if kwargs:
            for device_arg in device_kwargs_list:
                device = kwargs.get(device_arg, None)
                if device is not None:
                    _replace_cuda_to_npu_in_kwargs(kwargs, device_arg, device)
            device_ids = kwargs.get('device_ids', None)
            if type(device_ids) == list:
                _replace_cuda_to_npu_in_list(device_ids, replace_int)
        return func(*args, **kwargs)

    return decorated


def _replace_cuda_to_npu_in_kwargs(kwargs, device_arg, device):
    if type(device) == str and CUDA in device:
        kwargs[device_arg] = device.replace(CUDA, NPU)
    elif (type(device) == torch.device or str(type(device)) == "<class 'torch.device'>") and CUDA in device.type:
        device_info = 'npu:{}'.format(device.index) if device.index is not None else NPU
        kwargs[device_arg] = torch.device(device_info)
    elif type(device) == int:
        kwargs[device_arg] = f'npu:{device}'
    elif type(device) == dict:
        kwargs[device_arg] = _replace_cuda_to_npu_in_dict(device)


def _replace_cuda_to_npu_in_list(args_list, replace_int):
    for idx, arg in enumerate(args_list):
        if isinstance(arg, str) and CUDA in arg:
            args_list[idx] = arg.replace(CUDA, NPU)
        elif (isinstance(arg, torch.device) or str(type(arg)) == "<class 'torch.device'>") and CUDA in arg.type:
            device_info = 'npu:{}'.format(arg.index) if arg.index is not None else NPU
            args_list[idx] = torch.device(device_info)
        elif replace_int and not isinstance(arg, bool) and isinstance(arg, int):
            args_list[idx] = f'npu:{arg}'
        elif isinstance(arg, dict):
            args_list[idx] = _replace_cuda_to_npu_in_dict(arg)
    return args_list


def _replace_cuda_to_npu_in_dict(device_dict):
    new_dict = {}
    for key, value in device_dict.items():
        if isinstance(key, str):
            key = key.replace('cuda', 'npu')
        if isinstance(value, str):
            value = value.replace('cuda', 'npu')
        new_dict[key] = value
    return new_dict


def _device_wrapper(enter_fn, white_list):
    for fn_name in white_list:
        func = getattr(enter_fn, fn_name, None)
        if func:
            setattr(enter_fn, fn_name, _wrapper_cuda(func))


def _wrapper_hccl(func):
    @wraps(func)
    def decorated(*args, **kwargs):
        if args:
            args_new = list(args)
            for idx, arg in enumerate(args_new):
                if isinstance(arg, str) and NCCL in arg:
                    args_new[idx] = arg.replace(NCCL, HCCL)
            args = args_new
        if kwargs:
            backend = kwargs.get('backend', None)
            if isinstance(backend, str) and NCCL in backend:
                kwargs['backend'] = backend.replace(NCCL, HCCL)
        return func(*args, **kwargs)

    return decorated


def _wrapper_data_loader(func):
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


def _wrapper_profiler(fn):
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


def _patch_cuda():
    patchs = [
        ['cuda', torch_npu.npu], ['cuda.amp', torch_npu.npu.amp],
        ['cuda.random', torch_npu.npu.random],
        ['cuda.amp.autocast_mode', torch_npu.npu.amp.autocast_mode],
        ['cuda.amp.common', torch_npu.npu.amp.common],
        ['cuda.amp.grad_scaler', torch_npu.npu.amp.grad_scaler]
    ]
    torch_npu._apply_patches(patchs)


def _patch_profiler():
    patchs = [
        ['profiler.profile', torch_npu.profiler.profile],
        ['profiler.schedule', torch_npu.profiler.schedule],
        ['profiler.tensorboard_trace_handler', torch_npu.profiler.tensorboard_trace_handler],
        ['profiler.ProfilerAction', torch_npu.profiler.ProfilerAction],
        ['profiler.ProfilerActivity.CUDA', torch_npu.profiler.ProfilerActivity.NPU],
        ['profiler.ProfilerActivity.CPU', torch_npu.profiler.ProfilerActivity.CPU]
    ]
    torch_npu._apply_patches(patchs)


def _warning_fn(msg, rank0=True):
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


def _jit_script(obj, optimize=None, _frames_up=0, _rcb=None, example_inputs=None):
    return obj


def _jit_script_method(fn):
    return fn


def _patch_jit_script():
    msg = ('torch.jit.script and torch.jit.script_method will be disabled by transfer_to_npu, '
           'which currently does not support them, if you need to enable them, please do not use transfer_to_npu.')
    warnings.warn(msg, RuntimeWarning)
    torch.jit.script = _jit_script
    torch.jit.script_method = _jit_script_method


def _disable_torch_triton():
    from torch.utils._triton import has_triton

    def patch_has_triton():
        return False

    setattr(torch.utils._triton, 'has_triton', patch_has_triton)


def _replace_to_method_in_allowed_methods():
    for i, method in enumerate(UninitializedTensorMixin._allowed_methods):
        if method.__name__ == "to":
            UninitializedTensorMixin._allowed_methods[i] = torch.Tensor.to
            break


def _torch_version_less_than_2_1_adapt():
    if not _is_torch_version_greater_than_2_x(0):
        torch.distributed.distributed_c10d.broadcast_object_list \
            = torch_npu.distributed.distributed_c10d.broadcast_object_list
        torch.distributed.distributed_c10d.all_gather_object \
            = torch_npu.distributed.distributed_c10d.all_gather_object
        torch.npu.amp.autocast_mode.npu_autocast.__init__ = _wrapper_cuda(
            torch.npu.amp.autocast_mode.npu_autocast.__init__)


def _del_nccl_device_backend_map():
    if hasattr(torch.distributed.Backend, 'default_device_backend_map'):
        if 'cuda' in torch.distributed.Backend.default_device_backend_map:
            del torch.distributed.Backend.default_device_backend_map['cuda']


def _init():
    _warning_fn('''
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
    _patch_cuda()
    _device_wrapper(torch.cuda, torch_cuda_fn_white_list)
    torch.cuda.device.__init__ = _wrapper_cuda(torch.cuda.device.__init__)

    # torch.profiler.*
    _patch_profiler()
    torch.profiler.profile = _wrapper_profiler(torch.profiler.profile)

    # torch.*
    _device_wrapper(torch, torch_fn_white_list)

    # torch.Tensor.*
    _device_wrapper(torch.Tensor, torch_tensor_fn_white_list)
    torch.Tensor.cuda = torch.Tensor.npu
    torch.Tensor.is_cuda = torch.Tensor.is_npu
    torch.cuda.DoubleTensor = torch.npu.FloatTensor

    # torch.nn.Module.*
    _device_wrapper(torch.nn.Module, torch_module_fn_white_list)
    torch.nn.Module.cuda = torch.nn.Module.npu

    # torch.distributed
    torch.distributed.init_process_group = _wrapper_hccl(torch.distributed.init_process_group)
    torch.distributed.is_nccl_available = torch.distributed.is_hccl_available
    if DO_FSDP_WRAP:
        torch.distributed.fsdp.fully_sharded_data_parallel.FullyShardedDataParallel.__init__ = \
            _wrapper_cuda(torch.distributed.fsdp.fully_sharded_data_parallel.FullyShardedDataParallel.__init__)
    if hasattr(torch.distributed, 'init_device_mesh'):
        _del_nccl_device_backend_map()
        torch.distributed.init_device_mesh = _wrapper_cuda(torch.distributed.init_device_mesh)

    # torch.nn.parallel.DistributedDataParallel
    _device_wrapper(torch.nn.parallel.DistributedDataParallel, torch_distributed_fn_white_list)
    # torch.utils.data.DataLoader
    if _is_torch_version_greater_than_2_x(0):
        torch.utils.data.DataLoader.__init__ = _wrapper_data_loader(torch.utils.data.DataLoader.__init__)
        _patch_jit_script()
        if _is_torch_version_greater_than_2_x(2):
            torch._dynamo.trace_rules._disallowed_callable_ids.function_ids = None
        else:
            torch._dynamo.allowed_functions._disallowed_function_ids.function_ids = None
        torch.UntypedStorage.__new__ = _wrapper_cuda(torch.UntypedStorage.__new__)
        torch.utils._device.DeviceContext.__init__ = _wrapper_cuda(torch.utils._device.DeviceContext.__init__)

    # torch version < 2.1 needs to be adapted
    _torch_version_less_than_2_1_adapt()

    # torch 1.11.0 does not have this API
    if '1.11.0' not in torch.__version__:
        torch.distributed.ProcessGroup._get_backend = _wrapper_cuda(torch.distributed.ProcessGroup._get_backend)

    if _is_torch_version_greater_than_2_x(1):
        _disable_torch_triton()

    _do_wrapper_libraries_func(_load_json_file(config_path))

    _replace_to_method_in_allowed_methods()


_init()
