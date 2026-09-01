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

import functools
import inspect
import os

import torch
import torch.distributed as dist

from msprobe.core.common.const import Const
from msprobe.core.common.file_utils import load_yaml
from msprobe.core.common.runtime import Runtime
from msprobe.core.dump.api_dump.api_registry import ApiRegistry
from msprobe.pytorch.common.log import logger
from msprobe.pytorch.common.utils import torch_without_guard_version, is_gpu, torch_device_guard, parameter_adapter
from msprobe.pytorch.dump.function_factory import get_npu_custom_functions
from msprobe.pytorch.dump.api_dump.hook_module import HOOKModule
from msprobe.pytorch.dump.api_dump.utils import dynamic_import_op

try:
    import mindspeed.ops
except ImportError:
    mindspeed_enable = False
else:
    mindspeed_enable = True

torch_version_above_2 = torch.__version__.split('+')[0] > '2.0'

_supported_api_list_path = (os.path.join(os.path.dirname(os.path.realpath(__file__)), Const.SUPPORT_API_FILE_NAME),)
_cuda_func_mapping = {"npu_fusion_attention": "gpu_fusion_attention"}
dist_data_collect_func = {}
dist_batch_data_collect_func = []

_api_types = {
    Const.PT_FRAMEWORK: {
        Const.PT_API_TYPE_FUNCTIONAL: ((torch.nn.functional,), (torch.nn.functional,)),
        Const.PT_API_TYPE_TENSOR: ((torch.Tensor,), (torch.Tensor,)),
        Const.PT_API_TYPE_TORCH: ((torch,), (torch,)),
        Const.PT_API_TYPE_VF: ((torch._C._VariableFunctionsClass,), (torch._VF,)),
        Const.PT_API_TYPE_DIST: ((dist, dist.distributed_c10d), (dist, dist.distributed_c10d)),
    }
}
if not is_gpu:
    import torch_npu

    if torch_without_guard_version:
        _api_types.get(Const.PT_FRAMEWORK).update(
            {
                Const.PT_API_TYPE_NPU: ((torch.ops.npu, torch_npu), (torch_npu, torch.ops.npu)),
            }
        )
    else:
        _api_types.get(Const.PT_FRAMEWORK).update(
            {Const.PT_API_TYPE_NPU: ((torch_npu._C._VariableFunctionsClass,), (torch_npu,))}
        )
        _api_types.get(Const.PT_FRAMEWORK).update(
            {
                Const.PT_API_TYPE_NPU_DIST: (
                    (torch_npu.distributed,),
                    (torch_npu.distributed, torch_npu.distributed.distributed_c10d),
                )
            }
        )
    if mindspeed_enable:
        _api_types.get(Const.PT_FRAMEWORK).update({Const.PT_API_TYPE_MINDSPEED: ((mindspeed.ops,), (mindspeed.ops,))})
        mindspeed_op_list = load_yaml(_supported_api_list_path[0]).get(Const.PT_API_TYPE_MINDSPEED)
        mindspeed_op_file_list = [op.split(Const.SEP)[0] + Const.PY_SUFFIX for op in mindspeed_op_list]
        dynamic_import_op(mindspeed.ops, mindspeed_op_file_list)

_inner_used_api = {Const.PT_FRAMEWORK + Const.SEP + Const.PT_API_TYPE_TENSOR: (torch.Tensor, "view_as")}


def reset_dist_collect_func():
    dist_data_collect_func.clear()
    dist_batch_data_collect_func.clear()


@parameter_adapter
def tensor_module_forward(module, *args, **kwargs):
    return module.api_func(*args, **kwargs)


def dist_module_forward(module, *args, **kwargs):
    handle = module.api_func(*args, **kwargs)
    try:
        bound = inspect.signature(module.api_func).bind(*args, **kwargs)
        bound.apply_defaults()
        use_async_op_flag = bound.arguments.get("async_op", False)
    except Exception as e:
        use_async_op_flag = False
        logger.warning(f"fail to get dist api's func signature because {e}, no wait")

    def create_async_callback_func(catch_func):
        full_name = module.full_forward_name if hasattr(module, "full_forward_name") else None

        def store_data():
            catch_func(module, full_name, args, kwargs, handle)

        return store_data

    if Runtime.is_running and (use_async_op_flag or module.api_name in ['isend', 'irecv']):
        dist_data_collect_func[handle] = create_async_callback_func(module.distributed_forward_hook)
    if Runtime.is_running and module.api_name == 'batch_isend_irecv':
        dist_batch_data_collect_func.append([handle, create_async_callback_func(module.distributed_forward_hook)])
    return handle


def redirect_wait():
    if hasattr(dist, "Work"):
        from torch.distributed import Work
    else:
        from torch._C._distributed_c10d import Work
    origin_wait = Work.wait

    def wrapped_wait(work):
        def wrapped_wait(*args, **kwargs):
            origin_wait(*args, **kwargs)
            if args[0] in dist_data_collect_func:
                store_func = dist_data_collect_func.pop(args[0])
                store_func()
                return
            remove_value = None
            for value in dist_batch_data_collect_func:
                if args[0] in value[0]:
                    value[0].remove(args[0])
                    if len(value[0]) == 0:
                        store_func = value[1]
                        store_func()
                        remove_value = value
                    break
            if remove_value:
                dist_batch_data_collect_func.remove(remove_value)

        return wrapped_wait

    Work.wait = wrapped_wait(Work)


def npu_module_forward(module, *args, **kwargs):
    if not module.need_hook:
        npu_custom_functions = get_npu_custom_functions()
        if module.api_name not in npu_custom_functions:
            raise KeyError(f'There is not bench function {module.api_name}')
        if module.device == Const.CUDA_LOWERCASE:
            module.api_name = _cuda_func_mapping.get(module.api_name, module.api_name)
        if module.device in [Const.CUDA_LOWERCASE, Const.CPU_LOWERCASE]:
            return npu_custom_functions[module.api_name](*args, **kwargs)
    return module.api_func(*args, **kwargs)


forward_methods = {"Tensor": tensor_module_forward, "Distributed": dist_module_forward, "NPU": npu_module_forward}


class ApiTemplate(HOOKModule):
    def __init__(self, api_name, api_func, prefix, hook_build_func, need_hook=True, device=Const.CPU_LOWERCASE):
        self.api_name = api_name
        self.prefix = prefix
        self.prefix_api_name = prefix + Const.SEP + str(api_name.split(Const.SEP)[-1]) + Const.SEP
        self.need_hook = need_hook
        self.device = device
        self.op_is_distributed = prefix == Const.DIST_API_TYPE_PREFIX
        if self.need_hook:
            super().__init__(hook_build_func)
        self.api_func = api_func

    @torch_device_guard
    def forward(self, *args, **kwargs):
        exec_func = forward_methods.get(self.prefix)
        exec_func = functools.partial(exec_func, self) if exec_func else self.api_func
        return exec_func(*args, **kwargs)


api_register = None


def get_api_register(return_new=False):
    if return_new:
        return ApiRegistry(_api_types, _inner_used_api, _supported_api_list_path, ApiTemplate)

    global api_register
    if api_register is None:
        api_register = ApiRegistry(_api_types, _inner_used_api, _supported_api_list_path, ApiTemplate)
    return api_register
