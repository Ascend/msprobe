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
import threading
from contextlib import nullcontext

import torch

from msprobe.core.common.const import Const
from msprobe.pytorch.common.log import logger
from msprobe.core.common.runtime import Runtime
from msprobe.core.common.utils import replace_last_occurrence, ThreadSafe
from msprobe.core.dump.data_dump.data_processor.base import ModuleForwardInputsOutputs
from msprobe.core.dump.hook_manager import BaseHookManager, HookSet
from msprobe.pytorch.common.utils import (
    torch_version_above_or_equal_2,
    register_forward_hook,
    Const as PtConst,
    get_rank_if_initialized,
)
from msprobe.pytorch.dump.api_dump.hook_module import HOOKModule
from msprobe.pytorch.dump.module_dump.module_processor import ModuleProcessor

try:
    from torch.utils._python_dispatch import _disable_current_modes
except Exception:
    _disable_current_modes = None


class PytorchHookManager(BaseHookManager):
    def __init__(self, data_collector, config):
        super().__init__(data_collector, config)
        self.logger = logger

    @staticmethod
    def _no_grad_context():
        if _disable_current_modes is None:
            return nullcontext()
        return _disable_current_modes()

    @staticmethod
    def _add_count(name):
        HOOKModule.add_module_count(name)

    @staticmethod
    def _get_count(name):
        return HOOKModule.get_module_count(name)

    @staticmethod
    def _process_kwargs_and_output(module, tid, hook_type, kwargs_or_output, output_or_kwargs):
        if hook_type == Const.API:
            kwargs = kwargs_or_output
            output = output_or_kwargs
        else:
            kwargs = kwargs_or_output if torch_version_above_or_equal_2 else {}
            output = output_or_kwargs if torch_version_above_or_equal_2 else kwargs_or_output
        return kwargs, output

    @staticmethod
    def _get_current_rank():
        return get_rank_if_initialized()

    def build_hook(self, hook_type, name):
        if hook_type == Const.API:
            hook_set = HookSet(
                forward_pre_hook=self._build_forward_pre_hook(hook_type, name),
                distributed_forward_hook=self._build_distributed_forward_hook(),
            )
        else:
            full_backward_name = replace_last_occurrence(name, Const.FORWARD, Const.BACKWARD)
            hook_set = HookSet(
                forward_hook=self._build_forward_hook(hook_type, name),
                backward_hook=self._build_backward_hook(hook_type, full_backward_name),
            )
        return hook_set

    def _init_specific_components(self):
        pass

    def _on_forward_pre_hook(self):
        if self.config.task == Const.NAN_CHECK:
            device = torch.device(torch.npu.current_device())
            torch.ops.my_ns.npu_clear_over_flow(device)

    def _register_forward_hook(self, module, api_name):
        if not hasattr(module, 'msprobe_forward_hook'):
            register_forward_hook(module, self._build_forward_hook(Const.API, api_name))
            setattr(module, 'msprobe_forward_hook', True)

    def _register_backward_hook(self, module, full_backward_name, args):
        pass

    def _register_backward_pre_hook(self, module, full_backward_name, args, kwargs, output):
        if module.prefix_api_name in PtConst.DROPOUT_API_LIST:
            p = kwargs.get('p', 0.5)
            if len(args) >= 2:
                p = args[1]
            if p == 0:
                self.logger.debug(
                    f"The parameter 'p' is set to 0, so the {full_backward_name} data will not be collected."
                )
                return output

        var = output
        while not isinstance(var, torch.Tensor):
            if isinstance(var, dict):
                var = next((v for v in var.values() if isinstance(v, torch.Tensor)))
            elif isinstance(var, (list, tuple)):
                if var:
                    var = var[0]
                else:
                    return output
            else:
                return output

        if not (var.requires_grad and torch.is_grad_enabled()):
            return output

        grad_fn = var.grad_fn
        if grad_fn is not None:
            if hasattr(grad_fn, 'register_prehook') and self.config.task == Const.NAN_CHECK:

                def _nan_check_backward_prehook(_grad_outputs):
                    device = torch.device(torch.npu.current_device())
                    torch.ops.my_ns.npu_clear_over_flow(device)

                grad_fn.register_prehook(_nan_check_backward_prehook)

            backward_hook = self._build_backward_hook(Const.API, full_backward_name)
            wrapper = functools.partial(backward_hook, module)
            functools.update_wrapper(wrapper, backward_hook)
            grad_fn.register_hook(wrapper)

        return output

    def _need_exchange(self, module):
        return True

    def _get_params_dict(self, module):
        params_dict = {}
        if self.config.task != Const.STRUCTURE:
            params_dict = {key.split(Const.SEP)[-1]: value for key, value in module.named_parameters(recurse=False)}
        return params_dict

    def _build_distributed_forward_hook(self):
        def distributed_forward_hook(module, full_name, args, kwargs, output):
            if not full_name or not Runtime.is_running:
                return

            tid = threading.get_ident()
            with ThreadSafe():
                BaseHookManager.inner_switch[tid] = True
                self.data_collector.update_api_or_module_name(full_name)
                module_input_output = ModuleForwardInputsOutputs(args=args, kwargs=kwargs, output=output)
                with self._no_grad_context():
                    self.data_collector.forward_output_data_collect(full_name, module, self._pid, module_input_output)
                BaseHookManager.inner_switch[tid] = False

        return distributed_forward_hook

    def _register_param_hook(self, name, module, params_dict):
        ori_name = name.rsplit(Const.SEP, 2)[0]
        # data_mode为forward时，不注册参数hook
        if not (Const.FORWARD in self.config.data_mode and Const.BACKWARD not in self.config.data_mode):
            for param_name, param in params_dict.items():
                if param.requires_grad:
                    name = ori_name + Const.SEP + param_name
                    old_handle = BaseHookManager.hook_handle_dict.get(name)
                    if old_handle and hasattr(old_handle, "remove"):
                        old_handle.remove()
                    param_tmp = param.expand_as(param)
                    if param_tmp.grad_fn:
                        grad_acc = param_tmp.grad_fn.next_functions[0][0]
                        handle = grad_acc.register_hook(self._build_grad_hook(ori_name, param_name, param))
                        BaseHookManager.hook_handle_dict[name] = handle

    def _build_grad_hook(self, ori_name, param_name, param):
        def hook_fn(*args):
            tid = threading.get_ident()
            if not self._should_execute_hook(Const.MODULE, tid):
                return
            with ThreadSafe():
                original_state = self.ensure_gc_enabled()
                BaseHookManager.inner_switch[tid] = True
                index = self._get_grad_hook_call_index(ori_name, param_name)
                if ModuleProcessor.is_megatron_module and hasattr(param, "main_grad"):
                    grad = param.main_grad
                else:
                    grad = param.grad
                self.data_collector.params_data_collect(ori_name, param_name, self._pid, grad, str(index))
                BaseHookManager.inner_switch[tid] = False
                self.restore_gc_state(original_state)
            return

        return hook_fn
