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

import inspect
import threading
from collections import OrderedDict

from mindspore import Tensor
from mindspore.common.hook_handle import HookHandle
from mindspore.ops.operations import _inner_ops as inner

from msprobe.core.common.const import Const
from msprobe.core.common.exceptions import MsprobeException
from msprobe.core.common.runtime import Runtime
from msprobe.core.common.utils import ModuleQueue, ThreadSafe
from msprobe.core.dump.data_dump.scope import ModuleRangeScope, MixRangeScope, BaseScope
from msprobe.core.common.megatron_utils import wrap_megatron_step, get_micro_step, is_megatron
from msprobe.mindspore.common.const import Const as MsConst
from msprobe.mindspore.common.log import logger
from msprobe.mindspore.common.utils import (
    is_mindtorch,
    get_cells_and_names_with_index,
    has_kwargs_in_forward_hook,
    is_graph_mode_cell_dump_allowed,
    is_backward_hook_output_a_view,
)
from msprobe.mindspore.dump.debugger.debugger_config import DebuggerConfig
from msprobe.mindspore.dump.dump_processor.graph_mode_cell_dump import GraphModeCellDump


def get_cell_construct(construct):
    def _construct(self, *args, **kwargs):
        if hasattr(self, 'msprobe_hook'):
            setattr(self, 'msprobe_input_kwargs', kwargs)
        return construct(self, *args, **kwargs)

    return _construct


def patch_schedules_step():
    try:
        from mindspeed.mindspore.core.pipeline_parallel import schedules

        schedules.forward_step = wrap_megatron_step(schedules.forward_step)
        schedules.backward_step = wrap_megatron_step(schedules.backward_step, is_forward=False)
        logger.info_on_rank_0("Patch mindspeed.mindspore method success.")
    except ImportError:
        logger.info_on_rank_0("No mindspeed.mindspore find.")
    except Exception as e:
        logger.info_on_rank_0(f"Patch mindspeed.mindspore method failed, detail:{str(e)}")

    try:
        from megatron.core.pipeline_parallel import schedules

        schedules.forward_step = wrap_megatron_step(schedules.forward_step)
        schedules.backward_step = wrap_megatron_step(schedules.backward_step, is_forward=False)
        logger.info_on_rank_0("Patch megatron method success.")
    except ImportError:
        logger.info_on_rank_0("No megatron find.")
    except Exception as e:
        logger.info_on_rank_0(f"Patch megatron method failed, detail:{str(e)}")


class CellProcessor:
    cell_queue = ModuleQueue()
    cell_count = {}
    cell_stack = {}
    api_parent_node = {}
    module_node = {}
    cell_bw_hook_kernels = {}
    cell_backward_pre_hook = []
    cell_backward_hook = []

    def __init__(self, scope):
        self.scope = scope if isinstance(scope, (ModuleRangeScope, MixRangeScope)) else None

    @staticmethod
    def set_and_get_calls_number(cell_name):
        if cell_name not in CellProcessor.cell_count:
            CellProcessor.cell_count[cell_name] = 0
        else:
            CellProcessor.cell_count[cell_name] += 1
        return CellProcessor.cell_count[cell_name]

    @classmethod
    def reset_cell_stats(cls):
        cls.cell_queue = ModuleQueue()
        cls.cell_count = {}
        cls.cell_stack = {}
        cls.api_parent_node = {}
        cls.module_node = {}
        cls.cell_bw_hook_kernels = {}
        cls.cell_backward_pre_hook = []
        cls.cell_backward_hook = []

    def register_cell_hook(self, models, build_hook, config: DebuggerConfig):
        if not models:
            raise MsprobeException(
                MsprobeException.INVALID_PARAM_ERROR, 'The model cannot be None, when level is "L0" or "mix"'
            )

        patch_schedules_step()

        is_registered = False
        model_type = Const.MODULE if is_mindtorch() else Const.CELL
        cells_with_index_in_pynative_mode, cells_with_index_in_graph_mode = get_cells_and_names_with_index(models)
        construct_name = '_call_impl' if is_mindtorch() else '_run_construct'

        for index, cells_and_names in cells_with_index_in_pynative_mode.items():
            model = models if index == "-1" else models[int(index)]
            for name, cell in cells_and_names:
                if cell == model:
                    continue

                if not has_kwargs_in_forward_hook():
                    if not hasattr(cell.__class__, 'msprobe_construct'):
                        setattr(cell.__class__, 'msprobe_construct', True)
                        if hasattr(cell.__class__, construct_name):
                            setattr(
                                cell.__class__,
                                construct_name,
                                get_cell_construct(getattr(cell.__class__, construct_name)),
                            )
                setattr(cell, 'msprobe_hook', True)

                cell_index = (index + Const.SEP) if index != "-1" else ""
                prefix = f'{model_type}{Const.SEP}{cell_index}{name}{Const.SEP}{cell.__class__.__name__}{Const.SEP}'

                pre_hook_with_kwargs = 'with_kwargs' in inspect.signature(cell.register_forward_pre_hook).parameters
                forward_pre_hook = self.build_cell_hook(prefix, build_hook, pre_hook_with_kwargs)
                if pre_hook_with_kwargs:
                    cell.register_forward_pre_hook(forward_pre_hook, with_kwargs=True)
                else:
                    cell.register_forward_pre_hook(forward_pre_hook)

                if not is_registered:
                    logger.info("The cell hook function is successfully mounted to the model.")
                is_registered = True

        if is_graph_mode_cell_dump_allowed(config):
            cells_and_names_in_graph_mode = []
            for index, cells_and_names in cells_with_index_in_graph_mode.items():
                model = models if index == "-1" else models[int(index)]
                for name, cell, parent_cell in cells_and_names:
                    if cell == model:
                        continue
                    cell_index = (index + Const.SEP) if index != "-1" else ""
                    cells_and_names_in_graph_mode.append((f'{cell_index}{name}', cell, parent_cell))

            if cells_and_names_in_graph_mode:
                Runtime.run_mode = MsConst.PYNATIVE_GRAPH_MODE
                GraphModeCellDump(config, cells_and_names_in_graph_mode, strict=False).handle()

    def build_cell_hook(self, cell_name, build_data_hook, with_kwargs=False):
        @ThreadSafe.synchronized
        def forward_pre_hook(cell, args, kwargs=None):
            if kwargs is None:
                kwargs = {}

            if not Runtime.is_running:
                return (args, kwargs) if with_kwargs else args

            index = CellProcessor.set_and_get_calls_number(cell_name)
            full_forward_name = f'{cell_name}{Const.FORWARD}{Const.SEP}{index}'
            full_backward_name = f'{cell_name}{Const.BACKWARD}{Const.SEP}{index}'

            self.set_construct_info_in_pre_hook(full_forward_name)

            if not hasattr(cell, 'msprobe_forward_hook'):
                if is_mindtorch():
                    cell.register_forward_hook(forward_hook, prepend=True, with_kwargs=True)
                else:
                    forward_hook_dict = getattr(cell, '_forward_hook', OrderedDict())
                    if has_kwargs_in_forward_hook():
                        forward_hook_with_kwargs_dict = getattr(cell, '_forward_hook_with_kwargs', OrderedDict())
                        handle = HookHandle(forward_hook_dict, extra_dict=forward_hook_with_kwargs_dict)
                        forward_hook_with_kwargs_dict[handle.handle_id] = True
                    else:
                        handle = HookHandle(forward_hook_dict)
                    forward_hook_dict[handle.handle_id] = forward_hook
                    forward_hook_dict.move_to_end(handle.handle_id, last=False)

                setattr(cell, 'msprobe_forward_hook', True)

            hook_set = build_data_hook(BaseScope.Module_Type_Module, full_forward_name)

            if hook_set.module_forward_pre_hook:
                hook_set.module_forward_pre_hook(cell, args, kwargs)

            def get_backward_hook(backward_data_hook, full_backward_name):
                @ThreadSafe.synchronized
                def backward_hook_fn(cell, grad_input, grad_output):
                    new_output = backward_data_hook(cell, grad_input, grad_output)
                    self.set_construct_info_in_hook(full_backward_name)
                    cell.has_pre_hook_called = False
                    return new_output

                return backward_hook_fn

            enable_hooked = sum(
                isinstance(ele, Tensor) and ele.dtype not in MsConst.NonDifferentiableType for ele in args
            )
            if enable_hooked:
                backward_hook = OrderedDict()
                backward_hook[full_backward_name] = get_backward_hook(hook_set.backward_hook, full_backward_name)
                CellProcessor.cell_backward_hook.append(backward_hook)
                bw_hook = inner.CellBackwardHook(full_backward_name, cell, self.cell_backward_hook[-1])
                bw_hook.register_backward_hook()
                CellProcessor.cell_bw_hook_kernels[full_forward_name] = bw_hook

                args = bw_hook(args) if is_backward_hook_output_a_view() else bw_hook(*args)

            return (args, kwargs) if with_kwargs else args

        @ThreadSafe.synchronized
        def forward_hook(cell, args, kwargs_or_output, output_or_kwargs=None):
            index = CellProcessor.cell_count.get(cell_name, 0)
            full_forward_name = f'{cell_name}{Const.FORWARD}{Const.SEP}{index}'
            full_backward_name = f'{cell_name}{Const.BACKWARD}{Const.SEP}{index}'

            self.set_construct_info_in_hook(full_forward_name)

            hook_set = build_data_hook(BaseScope.Module_Type_Module, full_forward_name)
            hook_result = hook_set.forward_hook(cell, args, kwargs_or_output, output_or_kwargs)
            if hook_result is not None:
                outputs = hook_result
            else:
                outputs = output_or_kwargs if has_kwargs_in_forward_hook() else kwargs_or_output

            bw_hook = CellProcessor.cell_bw_hook_kernels.get(full_forward_name)
            if bw_hook:
                if not isinstance(outputs, (Tensor, tuple)):
                    logger.warning(
                        "For backward hooks to be called,"
                        " cell output should be a Tensor or a tuple of Tensors"
                        f" but received {type(outputs)}"
                    )
                if is_backward_hook_output_a_view():
                    new_outputs = bw_hook(outputs)
                else:
                    if isinstance(outputs, tuple):
                        new_outputs = bw_hook(*outputs)
                    else:
                        new_outputs = bw_hook(outputs)
                    if isinstance(outputs, tuple) and len(outputs) == 1:
                        new_outputs = (new_outputs,)
                outputs = new_outputs

            def get_backward_pre_hook(full_backward_name, backward_data_hook):
                @ThreadSafe.synchronized
                def backward_pre_hook_fn(cell, grad_output):
                    cell.has_pre_hook_called = True
                    self.set_construct_info_in_pre_hook(full_backward_name)
                    if backward_data_hook:
                        backward_data_hook(cell, (), grad_output)
                        self.set_construct_info_in_hook(full_backward_name)
                        cell.has_pre_hook_called = False

                return backward_pre_hook_fn

            backward_pre_hook = OrderedDict()
            backward_data_hook = None if bw_hook else hook_set.backward_hook
            backward_pre_hook[full_backward_name] = get_backward_pre_hook(full_backward_name, backward_data_hook)
            CellProcessor.cell_backward_pre_hook.append(backward_pre_hook)
            bw_pre_hook = inner.CellBackwardHook(full_backward_name, cell, self.cell_backward_pre_hook[-1])
            bw_pre_hook.register_backward_pre_hook()

            if is_backward_hook_output_a_view():
                result = bw_pre_hook(outputs)
            else:
                if isinstance(outputs, tuple):
                    result = bw_pre_hook(*outputs)
                else:
                    result = bw_pre_hook(outputs)
                if isinstance(outputs, tuple):
                    if len(outputs) == 1:
                        result = (result,)
                    if len(result) != len(outputs):
                        raise TypeError(
                            f"The backward pre hook return value size is {len(result)} "
                            f"not equal to output size {len(outputs)}"
                        )
            return result

        return forward_pre_hook

    def set_construct_info_in_pre_hook(self, full_name):
        tid = threading.get_ident()
        if tid not in self.cell_stack:
            CellProcessor.cell_stack[tid] = []

        if self.cell_stack[tid]:
            CellProcessor.module_node[full_name] = (
                self.cell_stack[tid][-1] if not is_megatron() else [self.cell_stack[tid][-1], get_micro_step()]
            )
        else:
            parent_name = CellProcessor.cell_queue.find_last(full_name)
            CellProcessor.module_node[full_name] = parent_name if not is_megatron() else [parent_name, get_micro_step()]

        CellProcessor.cell_queue.add_name(full_name)
        CellProcessor.cell_stack[tid].append(full_name)
        CellProcessor.api_parent_node[tid] = full_name if not is_megatron() else [full_name, get_micro_step()]
        if self.scope:
            self.scope.begin_module(full_name)

    def set_construct_info_in_hook(self, full_name):
        tid = threading.get_ident()
        CellProcessor.cell_queue.remove_name(full_name)
        CellProcessor.api_parent_node[tid] = None if not is_megatron() else [None, get_micro_step()]
        if self.cell_stack.get(tid):
            CellProcessor.cell_stack[tid].pop()
        if self.cell_stack.get(tid):
            CellProcessor.api_parent_node[tid] = (
                CellProcessor.cell_stack[tid][-1]
                if not is_megatron()
                else [CellProcessor.cell_stack[tid][-1], get_micro_step()]
            )
        if self.scope:
            self.scope.end_module(full_name)
