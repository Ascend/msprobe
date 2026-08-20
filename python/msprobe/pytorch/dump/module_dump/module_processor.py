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

# pylint: disable=duplicate-code

import sys
import threading
from collections import OrderedDict

import torch
from torch.utils.hooks import BackwardHook, RemovableHandle

from msprobe.core.common.const import Const
from msprobe.core.common.megatron_utils import wrap_megatron_step, get_micro_step, is_megatron
from msprobe.core.common.runtime import Runtime
from msprobe.core.common.utils import ModuleQueue, ThreadSafe
from msprobe.core.dump.data_dump.scope import BaseScope, ModuleRangeScope, MixRangeScope
from msprobe.pytorch.common.log import logger
from msprobe.pytorch.common.utils import is_torch_nn_module, register_forward_pre_hook
from msprobe.pytorch.dump.module_dump.hook_wrapper import (
    wrap_setup_input_output_hook,
    wrap_backward_hook_function_apply,
)
from msprobe.pytorch.dump.module_load.tensor_loader import TensorLoader


torch_version_above_or_equal_2 = torch.__version__.split('+')[0] >= '2.0'
torch_version_above_or_equal_21 = torch.__version__.split('+')[0] >= '2.1'
if torch_version_above_or_equal_21:
    from torch.utils.checkpoint import (  # pylint: disable=ungrouped-imports
        _StopRecomputationError,
    )
else:
    _StopRecomputationError = type('_StopRecomputationError', (Exception,), {})


def wrap_megatron_deallocate(func):
    def wrapper_func(out, deallocate_pipeline_outputs=False):
        if deallocate_pipeline_outputs and isinstance(out, torch.Tensor) and getattr(out, "_base") is not None:
            out_clone = out.clone()
            out.data = torch.empty(
                (1,),
                device=out.device,
                dtype=out.dtype,
            )
            return func(out_clone, deallocate_pipeline_outputs)
        return func(out, deallocate_pipeline_outputs)

    return wrapper_func


def wrap_forward_with_hook_safety(module):
    """
    包装模块的forward方法，确保异常时也执行forward_hook。
    """
    original_forward = module.forward

    def wrapped_forward(*args, **kwargs):
        try:
            output = original_forward(*args, **kwargs)
            return output
        except _StopRecomputationError as e:
            exception_output = None
            if len(module._forward_hooks.values()) > 0:
                # 仅执行msprobe的forward_hook, hook名称必然包含'ModuleProcessor.'
                for hook_fn in module._forward_hooks.values():
                    if 'ModuleProcessor' in str(hook_fn):
                        hook_fn(module, args, kwargs, exception_output)
                        break
            raise e

    if torch_version_above_or_equal_21:
        module.forward = wrapped_forward


class ModuleProcessor:
    module_queue = ModuleQueue()
    module_count = {}
    module_stack = {}
    api_parent_node = {}
    module_node = {}
    module_bw_hook_kernels = {}
    module_with_backward_hook = {}
    enable_module_dump = False
    is_megatron_module = False
    module_dump_level = None  # level from config, used to decide if module-level dump should run

    def __init__(self, scope, load_config=None):
        self.scope = scope if isinstance(scope, (ModuleRangeScope, MixRangeScope)) else None
        self.tensor_loader = None
        if load_config and load_config.is_enabled:
            self.tensor_loader = TensorLoader(load_config)
        wrap_setup_input_output_hook()
        wrap_backward_hook_function_apply()
        try:
            from megatron.core.pipeline_parallel import schedules

            origin_func_id = id(schedules.deallocate_output_tensor)
            schedules.deallocate_output_tensor = wrap_megatron_deallocate(schedules.deallocate_output_tensor)
            schedules.forward_step = wrap_megatron_step(schedules.forward_step)
            schedules.backward_step = wrap_megatron_step(schedules.backward_step, is_forward=False)
            for module in list(sys.modules.values()):
                # 1. 安全获取 __name__，如果不存在则设为 None
                module_name = getattr(module, '__name__', None)

                # 2. 如果名字为 None 或者就是 schedules 本身，跳过
                if module_name is None or module_name == 'schedules':
                    continue

                # 3. 安全检查是否有 __dict__ (某些 C 扩展可能没有标准的 __dict__)
                if not hasattr(module, '__dict__'):
                    continue

                for func in module.__dict__:
                    if id(module.__dict__[func]) == origin_func_id:
                        module.__setattr__(func, schedules.deallocate_output_tensor)
                        logger.debug(f'patch {module.__name__}.{func}.')
            logger.info_on_rank_0("Patch megatron method success.")
        except ImportError:
            logger.info_on_rank_0("No megatron find.")
        except Exception as e:
            logger.info_on_rank_0(f"Patch megatron method failed, detail:{str(e)}")

    @staticmethod
    def set_and_get_calls_number(module_name):
        if module_name not in ModuleProcessor.module_count:
            ModuleProcessor.module_count[module_name] = 0
        else:
            ModuleProcessor.module_count[module_name] += 1
        return ModuleProcessor.module_count[module_name]

    @staticmethod
    def has_register_backward_hook(module):
        return (
            hasattr(module, '_backward_hooks')
            and len(module._backward_hooks) > 0
            and module._is_full_backward_hook is False
        )

    @staticmethod
    def get_modules_and_names(models, recursive, module_names):
        modules_and_names_with_index = {}
        if isinstance(models, (list, tuple)):
            if not recursive and len(module_names) != len(models):
                return modules_and_names_with_index
            for index, model in enumerate(models):
                modules_and_names_with_index[str(index)] = (
                    model.named_modules() if recursive else [(module_names[index], model)]
                )
        else:
            if not recursive and len(module_names) != 1:
                return modules_and_names_with_index
            modules_and_names_with_index["-1"] = models.named_modules() if recursive else [(module_names[0], models)]
        return modules_and_names_with_index

    @staticmethod
    def check_is_megatron_module(models):
        if isinstance(models, (list, tuple)):
            for model in models:
                if "megatron" in type(model).__module__:
                    return True
            return False
        return "megatron" in type(models).__module__

    @classmethod
    def reset_module_stats(cls):
        cls.module_queue = ModuleQueue()
        cls.module_count = {}
        cls.module_stack = {}
        cls.api_parent_node = {}
        cls.module_node = {}
        cls.module_bw_hook_kernels = {}
        cls.enable_module_dump = False

    def register_module_hook(self, models, build_hook, recursive=True, module_names=None):
        if module_names is None:
            module_names = []

        ModuleProcessor.is_megatron_module = self.check_is_megatron_module(models)
        modules_and_names_with_index = self.get_modules_and_names(models, recursive, module_names)
        for index, modules_and_names in modules_and_names_with_index.items():
            model = models if index == "-1" else models[int(index)]

            model_list = []
            for name, module in modules_and_names:
                model_list.append((name, module))

            is_verl = "verl" in sys.modules
            for idx, (name, module) in enumerate(model_list):
                if recursive and module == model:
                    continue
                if not is_torch_nn_module(module):
                    logger.warning(
                        f"The module dump does not support {type(module)} type. "
                        f"The data dump for this module will be skipped."
                    )
                    continue
                if module.__class__.__name__ == "FullyShardedDataParallel":
                    continue

                # verl 场景下跳过第一层和最后一层
                if is_verl and idx in (1, len(model_list) - 1):
                    logger.warning(
                        f"The module {name} is the first or last layer in verl scenario, "
                        f"the data dump for this module will be skipped."
                    )
                    continue

                setattr(module, 'msprobe_hook', True)
                module_index = (index + Const.SEP) if index != "-1" else ""
                prefix_name = (
                    f'{BaseScope.Module_Type_Module}{Const.SEP}{module_index}{name}{Const.SEP}'
                    + f'{module.__class__.__name__}{Const.SEP}'
                )

                forward_pre_hook = self.build_module_hook(prefix_name, build_hook)

                if self.has_register_backward_hook(module):
                    logger.warning(
                        f"The {prefix_name[:-1]} has registered deprecated register_backward_hook,"
                        f"which may cause abnormal data dump. The backward data dump for this module will be skipped."
                    )
                    ModuleProcessor.module_with_backward_hook[prefix_name] = True
                wrap_forward_with_hook_safety(module)
                register_forward_pre_hook(module, forward_pre_hook)

    def build_module_hook(self, module_name, build_data_hook):
        @ThreadSafe.synchronized
        def forward_pre_hook(module, args, kwargs=None):
            if kwargs is None:
                kwargs = {}

            # === load override: executed before Runtime.is_running check ===
            # so that override takes effect even when dump is off (dump_after_load=false).
            # When tensor_loader is active, index must be incremented here (before the
            # Runtime check) to keep call_index aligned with source dump filenames.
            # Note: in load-only mode (Runtime.is_running=False), index is incremented but
            # the function returns immediately without dump. This is harmless because step()
            # calls _reset_status which resets module_count, so index does not accumulate
            # across steps.
            if self.tensor_loader is not None:
                # If loader is not active for current step (load.step out of range),
                # skip load logic entirely — no override, no warning, no overhead.
                if not self.tensor_loader.active:
                    return (args, kwargs) if torch_version_above_or_equal_2 else args
                index = ModuleProcessor.set_and_get_calls_number(module_name)
                full_forward_name = f'{module_name}{Const.FORWARD}{Const.SEP}{index}'
                if self.tensor_loader.should_override(full_forward_name):
                    args, kwargs = self.tensor_loader.override_args(full_forward_name, args, kwargs)
                # tensor_loader active but dump not running: return after override
                if not Runtime.is_running:
                    return (args, kwargs) if torch_version_above_or_equal_2 else args
                # load only needs forward_pre_hook for override, not for module-level dump.
                # If level is not L0/mix, skip module-level dump logic to avoid collecting
                # module-level data in L1 scenarios.
                if ModuleProcessor.module_dump_level not in [Const.LEVEL_L0, Const.LEVEL_MIX]:
                    return (args, kwargs) if torch_version_above_or_equal_2 else args
            else:
                # original path: no override, check Runtime first
                if not Runtime.is_running:
                    return (args, kwargs) if torch_version_above_or_equal_2 else args

            # === dump logic below (shared by both paths) ===
            if hasattr(module, 'msprobe_module_dump') and not self.enable_module_dump:
                return (args, kwargs) if torch_version_above_or_equal_2 else args

            # original path increments index here; load path already did it above
            if self.tensor_loader is None:
                index = ModuleProcessor.set_and_get_calls_number(module_name)
                full_forward_name = f'{module_name}{Const.FORWARD}{Const.SEP}{index}'
            full_backward_name = f'{module_name}{Const.BACKWARD}{Const.SEP}{index}'

            logger.debug(f"module forward_pre_hook executed for {module_name}")
            self.set_construct_info_in_pre_hook(full_forward_name)

            if not hasattr(module, 'msprobe_forward_hook'):
                forward_hooks_dict = getattr(module, '_forward_hooks', OrderedDict())
                handle = RemovableHandle(forward_hooks_dict)
                forward_hooks_dict[handle.id] = forward_hook
                forward_hooks_dict.move_to_end(handle.id, last=False)
                if torch_version_above_or_equal_2:
                    forward_hooks_with_kwargs_dict = getattr(module, '_forward_hooks_with_kwargs', OrderedDict())
                    forward_hooks_with_kwargs_dict[handle.id] = True

                setattr(module, 'msprobe_forward_hook', True)

            hook_set = build_data_hook(BaseScope.Module_Type_Module, full_forward_name)

            if hook_set.module_forward_pre_hook:
                hook_set.module_forward_pre_hook(module, args, kwargs)

            def get_backward_pre_hook(full_backward_name):
                @ThreadSafe.synchronized
                def backward_pre_hook_fn(module, grad_output):
                    self.set_construct_info_in_pre_hook(full_backward_name)

                return backward_pre_hook_fn

            def get_backward_hook(backward_data_hook, full_backward_name):
                @ThreadSafe.synchronized
                def backward_hook_fn(module, grad_input, grad_output):
                    logger.debug(f"module backward_hook executed for {full_backward_name}")
                    new_output = backward_data_hook(module, grad_input, grad_output)
                    self.set_construct_info_in_hook(full_backward_name, is_forward=False)
                    return new_output

                return backward_hook_fn

            if not ModuleProcessor.module_with_backward_hook.get(module_name):
                backward_pre_hook = get_backward_pre_hook(full_backward_name)
                backward_hook = get_backward_hook(hook_set.backward_hook, full_backward_name)
                if torch_version_above_or_equal_2:
                    bw_hook = BackwardHook(module, [backward_hook], [backward_pre_hook])
                else:
                    bw_hook = BackwardHook(module, [backward_hook])
                ModuleProcessor.module_bw_hook_kernels[full_forward_name] = bw_hook
                args = bw_hook.setup_input_hook(args)
            return (args, kwargs) if torch_version_above_or_equal_2 else args

        @ThreadSafe.synchronized
        def forward_hook(module, args, kwargs_or_output, output_or_kwargs=None):
            if hasattr(module, 'msprobe_module_dump') and not self.enable_module_dump:
                return output_or_kwargs if torch_version_above_or_equal_2 else kwargs_or_output

            index = ModuleProcessor.module_count.get(module_name)
            full_name = f'{module_name}{Const.FORWARD}{Const.SEP}{index}'

            logger.debug(f"module forward_hook executed for {module_name}")
            hook_set = build_data_hook(BaseScope.Module_Type_Module, full_name)
            hook_result = hook_set.forward_hook(module, args, kwargs_or_output, output_or_kwargs)
            self.set_construct_info_in_hook(full_name)

            if hook_result is not None:
                result = hook_result
            else:
                result = output_or_kwargs if torch_version_above_or_equal_2 else kwargs_or_output

            bw_hook = ModuleProcessor.module_bw_hook_kernels.get(full_name)
            if bw_hook:
                result = bw_hook.setup_output_hook(result)

            return result

        return forward_pre_hook

    def set_construct_info_in_pre_hook(self, full_name):
        tid = threading.get_ident()
        if tid not in self.module_stack:
            ModuleProcessor.module_stack[tid] = []

        if self.module_stack[tid]:
            ModuleProcessor.module_node[full_name] = (
                self.module_stack[tid][-1] if not is_megatron() else [self.module_stack[tid][-1], get_micro_step()]
            )
        else:
            parent_name = ModuleProcessor.module_queue.find_last(full_name)
            ModuleProcessor.module_node[full_name] = (
                parent_name if not is_megatron() else [parent_name, get_micro_step()]
            )

        ModuleProcessor.module_queue.add_name(full_name)
        ModuleProcessor.module_stack[tid].append(full_name)
        ModuleProcessor.api_parent_node[tid] = full_name if not is_megatron() else [full_name, get_micro_step()]
        if self.scope:
            self.scope.begin_module(full_name)

    def set_construct_info_in_hook(self, full_name, is_forward=True):
        tid = threading.get_ident()
        if torch_version_above_or_equal_2 or is_forward:
            ModuleProcessor.module_queue.remove_name(full_name)
            ModuleProcessor.api_parent_node[tid] = None if not is_megatron() else [None, get_micro_step()]
            if self.module_stack.get(tid):
                ModuleProcessor.module_stack[tid].pop()
            if self.module_stack.get(tid):
                ModuleProcessor.api_parent_node[tid] = (
                    ModuleProcessor.module_stack[tid][-1]
                    if not is_megatron()
                    else [ModuleProcessor.module_stack[tid][-1], get_micro_step()]
                )
            if self.scope:
                self.scope.end_module(full_name)
        else:
            if self.scope:
                self.scope.begin_module(full_name)
            ModuleProcessor.api_parent_node[tid] = full_name if not is_megatron() else [full_name, get_micro_step()]
