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
import os
import random
import sys
import types

from packaging.version import parse as version_parse

import mindspore as ms
from mindspore import ops
from mindspore.common.jit_config import JitConfig
from mindspore.mint import nn

try:
    from mindspore._c_expression import _set_init_iter
except ImportError:
    enable_dynamic_kbyk_dump = False
else:
    enable_dynamic_kbyk_dump = True

from msprobe.core.common.const import Const
from msprobe.core.common.decorator import recursion_depth_decorator
from msprobe.core.common.exceptions import DistributedNotInitializedError
from msprobe.core.common.file_utils import path_len_exceeds_limit, check_path_exists, save_npy
from msprobe.core.common.log import logger
from msprobe.core.common.utils import CompareException, check_seed_all, is_save_variable_valid
from msprobe.mindspore.common.const import Const as MsConst

mindtorch_check_result = None
register_backward_hook_functions = {}
kwargs_exist_in_forward_hook = None
is_output_of_backward_hook_a_view = None


class MsprobeStep(ms.train.Callback):
    def __init__(self, debugger):
        super().__init__()
        self.debugger = debugger

    def on_train_begin(self, run_context):
        self.debugger.start()
        if enable_dynamic_kbyk_dump:
            _set_init_iter(0)

    def on_train_step_begin(self, run_context):
        self.debugger.start()

    def on_train_step_end(self, run_context):
        self.debugger.stop()
        self.debugger.step()


class MsprobeInitStep(ms.train.Callback):
    def on_train_begin(self, run_context):
        try:
            from ms._c_expression import _set_init_iter
        except ImportError:
            logger.warning('MsprobeInitStep does not work on this version of MindSpore.')
            return
        cb_params = run_context.original_args()
        _set_init_iter(cb_params.cur_step_num)


def get_rank_if_initialized():
    if ms.communication.GlobalComm.INITED:
        return ms.communication.get_rank()
    else:
        raise DistributedNotInitializedError("mindspore distributed environment is not initialized")


def convert_bf16_to_fp32(tensor):
    if tensor.dtype == ms.bfloat16:
        tensor = tensor.to(ms.float32)
    return tensor


def save_tensor_as_npy(tensor, file_path):
    if not path_len_exceeds_limit(file_path):
        tensor = convert_bf16_to_fp32(tensor)
        saved_tensor = tensor.asnumpy()
        save_npy(saved_tensor, file_path)
    else:
        logger.warning(f'The file path {file_path} length exceeds limit.')


def convert_to_int(value):
    if isinstance(value, bool):
        logger.error('The value in rank_id or step should be int, please check!')
        raise CompareException(CompareException.INVALID_OBJECT_TYPE_ERROR)
    try:
        return int(value)
    except Exception:
        return -1


def clean_input_kwargs(cell):
    if hasattr(cell, 'msprobe_input_kwargs'):
        del cell.msprobe_input_kwargs


def list_lowest_level_directories(root_dir):
    check_path_exists(root_dir)
    lowest_level_dirs = []

    def recurse_dirs(current_dir, depth=0):
        if depth > Const.MAX_DEPTH:
            logger.error(f'The directory {current_dir} has more than {Const.MAX_DEPTH} levels.')
            raise CompareException(CompareException.RECURSION_LIMIT_ERROR)
        for entry in os.listdir(current_dir):
            full_path = os.path.join(current_dir, entry)
            if os.path.isdir(full_path):
                if any(os.path.isdir(os.path.join(full_path, subentry)) for subentry in os.listdir(full_path)):
                    recurse_dirs(full_path, depth=depth + 1)
                else:
                    lowest_level_dirs.append(full_path)

    recurse_dirs(root_dir)
    return lowest_level_dirs


def seed_all(seed=1234, mode=False, rm_dropout=False):
    check_seed_all(seed, mode, rm_dropout)
    os.environ['PYTHONHASHSEED'] = str(seed)
    ms.set_seed(seed)
    random.seed(seed)
    ms.set_context(deterministic="ON" if mode else "OFF")
    os.environ['HCCL_DETERMINISTIC'] = str(mode)
    if rm_dropout:
        remove_dropout()


class Dropout(ops.Dropout):
    def __init__(self, keep_prob=0.5, seed0=0, seed1=1):
        super().__init__(1.0, seed0, seed1)


class Dropout2D(ops.Dropout2D):
    def __init__(self, keep_prob=0.5):
        super().__init__(1.0)


class Dropout3D(ops.Dropout3D):
    def __init__(self, keep_prob=0.5):
        super().__init__(1.0)


class DropoutExt(nn.Dropout):
    def __init__(self, p=0.5):
        super().__init__(0)


def dropout_ext(input_tensor, p=0.5, training=True):
    return input_tensor


def remove_dropout():
    ops.Dropout = Dropout
    ops.operations.Dropout = Dropout
    ops.Dropout2D = Dropout2D
    ops.operations.Dropout2D = Dropout2D
    ops.Dropout3D = Dropout3D
    ops.operations.Dropout3D = Dropout3D
    nn.Dropout = DropoutExt
    nn.functional.dropout = dropout_ext


def is_mindtorch():
    global mindtorch_check_result
    if mindtorch_check_result is None:
        mindtorch_check_result = False
        if 'torch' not in sys.modules:
            return mindtorch_check_result
        try:
            import torch
        except ImportError:
            return mindtorch_check_result
        tensor = torch.tensor(0.0)
        if isinstance(tensor, ms.Tensor):
            mindtorch_check_result = True
    return mindtorch_check_result


def set_register_backward_hook_functions():
    if register_backward_hook_functions:
        return

    if is_mindtorch():
        import torch
        from msprobe.mindspore.dump.mindtorch import (
            _call_impl,
            register_full_backward_pre_hook,
            register_full_backward_hook,
        )

        if not hasattr(torch.nn.Module, "register_full_backward_hook"):
            setattr(torch.nn.Module, "_call_impl", _call_impl)
            setattr(torch.nn.Module, "register_full_backward_pre_hook", register_full_backward_pre_hook)
            setattr(torch.nn.Module, "register_full_backward_hook", register_full_backward_hook)
        register_backward_hook_functions["pre"] = torch.nn.Module.register_full_backward_pre_hook
        register_backward_hook_functions["full"] = torch.nn.Module.register_full_backward_hook
    else:
        register_backward_hook_functions["pre"] = ms.nn.Cell.register_backward_pre_hook
        register_backward_hook_functions["full"] = ms.nn.Cell.register_backward_hook


def check_save_param(variable, name, save_backward):
    # try catch this api to skip invalid call
    valid_data_types = (ms.Tensor, int, float, str)
    if not is_save_variable_valid(variable, valid_data_types):
        valid_data_types_with_nested_types = valid_data_types + (dict, tuple, list)
        logger.warning(
            "PrecisionDebugger.save variable type not valid, "
            f"should be one of {valid_data_types_with_nested_types}"
            "Skip current save process."
        )
        raise ValueError
    if not isinstance(name, str):
        logger.warning("PrecisionDebugger.save name not valid, should be string. skip current save process.")
        raise ValueError
    if not isinstance(save_backward, bool):
        logger.warning("PrecisionDebugger.save_backward name not valid, should be bool. Skip current save process.")
        raise ValueError


def is_graph_mode_cell_dump_allowed(config):
    if config.task not in [Const.TENSOR, Const.STATISTICS] or is_mindtorch() or not hasattr(ops, 'DumpGradient'):
        return False
    valid_mix_level = [MsConst.CELL_AND_API, Const.LEVEL_MIX]
    if config.level in valid_mix_level and config.execution_mode == MsConst.PYNATIVE_MODE:
        return True
    return config.level in (MsConst.CELL, Const.LEVEL_L0)


@recursion_depth_decorator('msprobe.mindspore.common.utils.is_decorated_by_jit')
def is_decorated_by_jit(func):
    closure = getattr(func, '__closure__', [])
    if closure:
        for obj in closure:
            if isinstance(obj.cell_contents, JitConfig):
                return True
            elif isinstance(obj.cell_contents, types.FunctionType) and hasattr(obj.cell_contents, '__closure__'):
                if is_decorated_by_jit(obj.cell_contents):
                    return True
    return False


@recursion_depth_decorator('msprobe.mindspore.common.utils.get_cells_and_names')
def get_cells_and_names(model, cells_set=None, name_prefix='', parent_cell=None):
    cells_set = cells_set if cells_set else set()
    if model in cells_set:
        return

    cells_set.add(model)
    jit_decorated = is_decorated_by_jit(model.construct)
    yield name_prefix, model, jit_decorated, parent_cell
    if jit_decorated:
        return

    children_cells = getattr(model, '_cells')
    for name, cell in children_cells.items():
        if cell:
            cells_name_prefix = f'{name_prefix}{Const.SEP}{name}' if name_prefix else name
            jit_decorated = is_decorated_by_jit(model.construct)
            if jit_decorated:
                yield cells_name_prefix, cell, jit_decorated, model
            else:
                yield from get_cells_and_names(cell, cells_set, cells_name_prefix, model)


def get_cells_and_names_with_index(models):
    cells_with_index_in_pynative_mode = {}
    cells_with_index_in_graph_mode = {}

    def distinguish_cells(cells):
        cells_in_pynative_mode = []
        cells_in_graph_mode = []
        for name, cell, jit_decorated, parent_cell in cells:
            if jit_decorated:
                cells_in_graph_mode.append((name, cell, parent_cell))
            else:
                cells_in_pynative_mode.append((name, cell))
        return cells_in_pynative_mode, cells_in_graph_mode

    if is_mindtorch():
        if isinstance(models, (list, tuple)):
            for index, model in enumerate(models):
                cells_with_index_in_pynative_mode[str(index)] = model.named_modules()
        else:
            cells_with_index_in_pynative_mode["-1"] = models.named_modules()
    else:
        if isinstance(models, (list, tuple)):
            for index, model in enumerate(models):
                cells = get_cells_and_names(model)
                cells_in_pynative_mode, cells_in_graph_mode = distinguish_cells(cells)
                cells_with_index_in_pynative_mode[str(index)] = cells_in_pynative_mode
                cells_with_index_in_graph_mode[str(index)] = cells_in_graph_mode
        else:
            cells = get_cells_and_names(models)
            cells_in_pynative_mode, cells_in_graph_mode = distinguish_cells(cells)
            cells_with_index_in_pynative_mode["-1"] = cells_in_pynative_mode
            cells_with_index_in_graph_mode["-1"] = cells_in_graph_mode

    return cells_with_index_in_pynative_mode, cells_with_index_in_graph_mode


def has_kwargs_in_forward_hook():
    global kwargs_exist_in_forward_hook

    if kwargs_exist_in_forward_hook is None:
        if is_mindtorch():
            kwargs_exist_in_forward_hook = True
            return kwargs_exist_in_forward_hook

        try:
            func_params = inspect.signature(nn.Cell.register_forward_hook).parameters
            kwargs_exist_in_forward_hook = 'with_kwargs' in func_params
        except Exception:
            kwargs_exist_in_forward_hook = False
        return kwargs_exist_in_forward_hook

    return kwargs_exist_in_forward_hook


def is_backward_hook_output_a_view():
    global is_output_of_backward_hook_a_view

    if is_output_of_backward_hook_a_view is None:
        is_output_of_backward_hook_a_view = False
        if version_parse(getattr(ms, '__version__', '2.4.0')) < version_parse('2.7.0'):
            return is_output_of_backward_hook_a_view
        try:
            from mindspore.ops.operations import _inner_ops as inner

            call_func = getattr(inner.CellBackwardHook, '__call__')
            func_params = inspect.signature(call_func).parameters
        except Exception:
            return is_output_of_backward_hook_a_view
        if 'args' in func_params and func_params['args'].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
            is_output_of_backward_hook_a_view = True

    return is_output_of_backward_hook_a_view


def wrap_backward_hook_call_func(call_func):
    if not is_backward_hook_output_a_view():
        return call_func

    from mindspore.common.api import _pynative_executor as executor
    from mindspore._c_expression import CreationType

    def new_call(self, args):
        outputs = call_func(self, args)
        if isinstance(outputs, ms.Tensor):
            executor.set_creation_type(outputs, CreationType.DEFAULT)
        elif isinstance(outputs, tuple):
            for item in outputs:
                if isinstance(item, ms.Tensor):
                    executor.set_creation_type(item, CreationType.DEFAULT)
        return outputs

    new_call.__name__ = '__call__'

    return new_call


def cast_to_float_if_fp8(tensor):
    dtype = str(tensor.dtype)
    if dtype in MsConst.FP8_TYPE_LIST:
        logger.debug(
            f"The {dtype} tensor analyzing/saving is unsupported in dump function.Casting to a float for processing"
        )
        tensor = tensor.to(ms.float32)
    return tensor
