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
import stat
from functools import wraps

import numpy as np
import torch
import torch.distributed as dist

from msprobe.core.common.exceptions import DistributedNotInitializedError
from msprobe.core.common.file_utils import (
    FileCheckConst,
    change_mode,
    check_file_or_directory_path,
    check_path_before_create
)
from msprobe.core.common.log import logger
from msprobe.core.common.utils import check_seed_all, is_save_variable_valid
from msprobe.pytorch.reproducibility.random_reproducibility import set_reproducibility

try:
    import torch_npu
except ImportError:
    is_gpu = True
else:
    is_gpu = False

torch_without_guard_version = torch.__version__ >= '2.1'
torch_version_above_or_equal_2 = torch.__version__.split('+')[0] >= '2.0'

if not is_gpu and not torch_without_guard_version:
    from torch_npu.utils.device_guard import torch_device_guard as torch_npu_device_guard

npu_distributed_api = ['isend', 'irecv']

# =========================
# 解决 PyTorch 2.6+ 安全反序列化问题
# =========================

def enable_torch_npu_pickle_compat():
    try:
        torch.serialization.add_safe_globals([
            torch_npu.utils.storage._rebuild_npu_tensor
        ])
    except Exception as e:
        logger.warning(f"Enable torch_npu pickle compatibility failed: {e}")

#  在 torch.load 之前执行
if not is_gpu:
    enable_torch_npu_pickle_compat()


def parameter_adapter(func):
    def handle_masked_select(input_tensor, indices):
        masked_select_func = getattr(torch._C._VariableFunctionsClass, "masked_select")
        if input_tensor.dtype == torch.bfloat16:
            # masked_select在NPU上输入数据dtype类型为bfloat16会报错，提示不支持此类型
            return masked_select_func(input_tensor.to(torch.float32), indices).to(torch.bfloat16)
        else:
            return masked_select_func(input_tensor, indices)

    @wraps(func)
    def inner(self, *args, **kwargs):
        if self.api_name == "__getitem__" and len(args) > 1 and isinstance(args[1], torch.Tensor):
            input_tensor = args[0]
            indices = args[1]
            if indices.dtype == torch.uint8:
                indices = indices.bool()
            if indices.dtype == torch.bool:
                if indices.shape == input_tensor.shape:
                    return handle_masked_select(input_tensor, indices)
                else:
                    indices = getattr(torch._C._VariableFunctionsClass, "nonzero")(indices, as_tuple=True)
                    return getattr(torch._C._TensorBase, "__getitem__")(input_tensor, indices)
            elif indices.dtype != torch.bool:
                if not indices.shape or len(indices.shape) == 1:
                    return func(self, input_tensor, indices.tolist())
                elif len(indices.shape) == 2:
                    result = [func(self, input_tensor, index) for index in indices.tolist()]
                    return getattr(torch._C._VariableFunctionsClass, "stack")(result, 0)
                else:
                    res = [input_tensor[tensor_index] for tensor_index in indices]
                    return getattr(torch._C._VariableFunctionsClass, "stack")(res, 0)
        if self.api_name == "__eq__" and len(args) > 1 and args[1] is None:
            return False
        return func(self, *args, **kwargs)

    return inner


def torch_device_guard(func):
    if is_gpu or torch_without_guard_version:
        return func

    # Parse args/kwargs matched torch.device objects
    @torch_npu_device_guard
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def get_rank_if_initialized():
    """
        return rank id if it is initialized or raise Exception: DistributedNotInitializedError
    """
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        raise DistributedNotInitializedError("torch distributed environment is not initialized")


def remove_dropout():
    if torch.__version__ > "1.8":
        logger.info_on_rank_0("For precision comparison, the probability p in the dropout method is set to 0.")
        import torch.nn.functional as F
        from torch import _VF
        from torch.overrides import has_torch_function_unary, handle_torch_function

        def function_dropout(input_tensor: torch.Tensor, p: float = 0.5, training: bool = True,
                             inplace: bool = False) -> torch.Tensor:
            if has_torch_function_unary(input_tensor):
                return handle_torch_function(
                    function_dropout, (input_tensor,), input_tensor, p=0., training=training, inplace=inplace)
            if p < 0.0 or p > 1.0:
                raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
            return _VF.dropout_(input_tensor, 0., training) if inplace else _VF.dropout(input_tensor, 0., training)

        def function_dropout2d(input_tensor: torch.Tensor, p: float = 0.5, training: bool = True,
                               inplace: bool = False) -> torch.Tensor:
            if has_torch_function_unary(input_tensor):
                return handle_torch_function(
                    function_dropout2d, (input_tensor,), input_tensor, p=0., training=training, inplace=inplace)
            if p < 0.0 or p > 1.0:
                raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
            return _VF.feature_dropout_(input_tensor, 0., training) if inplace else _VF.feature_dropout(input_tensor,
                                                                                                        0., training)

        def function_dropout3d(input_tensor: torch.Tensor, p: float = 0.5, training: bool = True,
                               inplace: bool = False) -> torch.Tensor:
            if has_torch_function_unary(input_tensor):
                return handle_torch_function(
                    function_dropout3d, (input_tensor,), input_tensor, p=0., training=training, inplace=inplace)
            if p < 0.0 or p > 1.0:
                raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
            return _VF.feature_dropout_(input_tensor, 0., training) if inplace else _VF.feature_dropout(input_tensor,
                                                                                                        0., training)

        F.dropout = function_dropout
        F.dropout2d = function_dropout2d
        F.dropout3d = function_dropout3d


def seed_all(seed=1234, mode=False, rm_dropout=False, is_enhanced=False):
    check_seed_all(seed, mode, rm_dropout)
    set_reproducibility(seed=seed, is_deterministic=mode, is_enhanced=is_enhanced)
    if rm_dropout:
        remove_dropout()


class Const:
    """
    Class for const
    """
    SEP = "."
    MODEL_TYPE = ['.onnx', '.pb', '.om']
    DIM_PATTERN = r"^(-?[0-9]+)(,-?[0-9]+)*"
    SEMICOLON = ";"
    COLON = ":"
    EQUAL = "="
    COMMA = ","
    DOT = "."
    DUMP_RATIO_MAX = 100
    SUMMERY_DATA_NUMS = 256
    FLOAT_EPSILON = np.finfo(float).eps
    SUPPORT_DUMP_MODE = ['api', 'acl']
    ON = 'ON'
    OFF = 'OFF'
    KWARGS = 'kwargs'
    INPUT = 'input'
    OUTPUT = 'output'
    BACKWARD = 'backward'
    FORWARD = 'forward'
    PRE_FORWARD = "pre_forward"
    INPUT_ARGS = 'input_args'
    INPUT_KWARGS = 'input_kwargs'
    GRAD_INPUT = 'grad_input'
    GRAD_OUTPUT = 'grad_output'
    START = "start"
    STOP = "stop"
    MAX = 'Max'
    MIN = 'Min'

    # dump mode
    ALL = "all"
    LIST = "list"
    RANGE = "range"
    STACK = "stack"
    ACL = "acl"
    API_LIST = "api_list"
    API_STACK = "api_stack"
    DUMP_MODE = [ALL, LIST, RANGE, STACK, ACL, API_LIST, API_STACK]
    AUTO = "auto"
    ONLINE_DUMP_MODE = [ALL, LIST, AUTO, OFF]
    SUMMARY = "summary"
    MD5 = "md5"
    SUMMARY_MODE = [ALL, SUMMARY, MD5]

    WRITE_FLAGS = os.O_WRONLY | os.O_CREAT
    OVERWRITE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    WRITE_MODES = stat.S_IWUSR | stat.S_IRUSR

    PKL_SUFFIX = ".pkl"
    NUMPY_SUFFIX = ".npy"
    ONE_GB = 1 * 1024 * 1024 * 1024
    TEN_GB = 10 * 1024 * 1024 * 1024
    FILE_PATTERN = r'^[a-zA-Z0-9_./-]+$'
    FILE_NAME_LENGTH = 255
    DIRECTORY_LENGTH = 4096
    DISTRIBUTED_PREFIX_LENGTH = 60
    SUMMARY_COLUMN_NUM = 6
    STACK_COLUMN_NUM = 2
    # env dump path
    ASCEND_WORK_PATH = "ASCEND_WORK_PATH"
    DUMP_DIR = "dump_data"
    DATA = "data"

    ENV_ENABLE = "1"
    ENV_DISABLE = "0"

    MAX_SEED_VALUE = 2 ** 32 - 1

    TASK_LIST = ["tensor", "statistics", "overflow_check", "free_benchmark"]
    LEVEL_LIST = ["L0", "L1", "L2", "mix"]
    STATISTICS = "statistics"
    TENSOR = "tensor"
    OVERFLOW_CHECK = "overflow_check"
    FREE_BENCHMARK = "free_benchmark"

    ATTR_NAME_PREFIX = "wrap_"

    FLOAT_TYPE = [np.half, np.single, float, np.double, np.float64, np.longdouble, np.float32, np.float16]
    BOOL_TYPE = [bool, np.uint8]
    INT_TYPE = [np.int32, np.int64]
    NPU = 'NPU'
    DISTRIBUTED = 'Distributed'

    HIFLOAT8_TYPE = "torch_npu.HiFloat8Tensor"
    FLOAT8_E5M2_TYPE = "torch.float8_e5m2"
    FLOAT8_E4M3FN_TYPE = "torch.float8_e4m3fn"
    FLOAT8_E8M0FNU_TYPE = "torch.float8_e8m0fnu"

    RAISE_PRECISION = {
        torch.float16: torch.float32,
        torch.bfloat16: torch.float32,
        torch.float32: torch.float64
    }
    CONVERT = {
        "int32_to_int64": ["torch.int32", "torch.int64"],
    }

    CONVERT_API = {
        "int32_to_int64": ["cross_entropy"]
    }

    DROPOUT_API_LIST = [
        "Functional.dropout.",
        "Functional.dropout1d.",
        "Functional.dropout2d.",
        "Functional.dropout3d.",
        "Functional.alpha_dropout.",
        "Functional.feature_alpha_dropout.",
        "Torch.dropout.",
        "Torch.dropout_.",
        "Torch.alpha_dropout.",
        "Torch.alpha_dropout_.",
        "Torch.feature_alpha_dropout.",
        "Torch.feature_alpha_dropout_.",
        "Torch.feature_dropout.",
        "Torch.feature_dropout_.",
        "NPU._npu_dropout."
    ]


def get_tensor_rank(in_feat, out_feat):
    if dist.is_initialized():
        return dist.get_rank()

    def get_tensor_rank_single(x):
        if isinstance(x, (list, tuple)):
            if len(x) > 0:
                return get_tensor_rank_single(x[0])
        elif isinstance(x, torch.Tensor):
            device = x.device
            if device.type != 'cpu':
                return device.index
        return None

    in_rank = get_tensor_rank_single(in_feat)
    out_rank = get_tensor_rank_single(out_feat)
    tensor_rank = in_rank if in_rank else out_rank
    return tensor_rank


def get_rank_id():
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def print_rank_0(message):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            logger.info(message)
    else:
        logger.info(message)


def load_pt(pt_path, to_cpu=False, weights_only=True):
    pt_path = os.path.realpath(pt_path)
    check_file_or_directory_path(pt_path)
    try:
        if to_cpu:
            pt = torch.load(pt_path, map_location=torch.device("cpu"), weights_only=weights_only)
        else:
            pt = torch.load(pt_path, weights_only=weights_only)
    except Exception as e:
        raise RuntimeError(f"load pt file {pt_path} failed") from e
    return pt


def save_pt(tensor, filepath):
    check_path_before_create(filepath)
    filepath = os.path.realpath(filepath)
    try:
        torch.save(tensor, filepath)
    except Exception as e:
        logger.error("Save pt file failed, please check according possible error causes: "
                     "1. out of disk space or disk error, "
                     "2. no permission to write files, etc.")
        raise RuntimeError(f"save pt file {filepath} failed") from e
    change_mode(filepath, FileCheckConst.DATA_FILE_AUTHORITY)


def is_recomputation(call_stack=None):
    """Check if the current operation is in the re-computation phase.

    This function inspects the current call stack to indicate whether the current operation is in the
    re-computation phase. We use a blacklist mechanism, now supported megatron and mindspeed framework.
    megatron: The 'backward' function is called by the 'torch/autograd/function.py' file.
    mindspeed: The 'checkpoint_function_backward' function is called by the 'torch/autograd/function.py'
    file or the custom module(use CheckpointWithoutOutput) with the 'recompute_fn' function is executed within the
    'torch/utils/checkpoint.py' file.

    Returns:
        bool: True if in the re-computation phase, False otherwise.
    """
    if not call_stack:
        try:
            call_stack = inspect.stack()
        except Exception as e:
            logger.warning(
                f"Failed to capture stack trace, recomputation validation may be incorrect, error info: {e}.")
            return False
    backward_function_indices = []

    # Identify the function 'backward' is being executed within the 'torch/_tensor.py' file.
    for frame_info in call_stack:
        if frame_info.function == "recompute_fn" and frame_info.filename.endswith('torch/utils/checkpoint.py'):
            del call_stack
            return True

    # Identify indices in the call stack where the specific function is being executed
    for idx, frame_info in enumerate(call_stack):
        if (frame_info.function == Const.BACKWARD or frame_info.function == 'checkpoint_function_backward' and
                "megatron" in frame_info.filename):
            backward_function_indices.append(idx)

    # Check if the execution is within 'torch/autograd/function.py' file
    for idx in backward_function_indices:
        # The Megatron and MindSpeed L0&L1 scenes
        if idx + 1 < len(call_stack) and call_stack[idx + 1].filename.endswith('torch/autograd/function.py'):
            del call_stack
            return True
        # The latest MindSpeed L2 and ModelLink scenes
        if idx + 2 < len(call_stack) and call_stack[idx + 2].filename.endswith('torch/autograd/function.py'):
            del call_stack
            return True

    del call_stack
    return False


def check_save_param(variable, name, save_backward):
    # try catch this api to skip invalid call
    valid_data_types = (torch.Tensor, int, float, str)
    if not is_save_variable_valid(variable, valid_data_types):
        valid_data_types_with_nested_types = valid_data_types + (dict, tuple, list)
        logger.warning("PrecisionDebugger.save variable type not valid, "
                       f"should be one of {valid_data_types_with_nested_types}"
                       "Skip current save process.")
        raise ValueError
    if not isinstance(name, str):
        logger.warning("PrecisionDebugger.save name not valid, "
                       "should be string. "
                       "skip current save process.")
        raise ValueError
    if not isinstance(save_backward, bool):
        logger.warning("PrecisionDebugger.save_backward name not valid, "
                       "should be bool. "
                       "Skip current save process.")
        raise ValueError


def is_torch_nn_module(variable):
    return isinstance(variable, torch.nn.Module) and not isinstance(variable, torch.jit.ScriptModule)


def register_forward_pre_hook(module, forward_pre_hook):
    if torch_version_above_or_equal_2:
        module.register_forward_pre_hook(forward_pre_hook, with_kwargs=True)
    else:
        module.register_forward_pre_hook(forward_pre_hook)


def register_forward_hook(module, forward_hook):
    if torch_version_above_or_equal_2:
        module.register_forward_hook(forward_hook, with_kwargs=True)
    else:
        module.register_forward_hook(forward_hook)


def is_hifloat8_tensor(tensor):
    if not is_gpu and hasattr(torch_npu, "HiFloat8Tensor") and isinstance(tensor, torch_npu.HiFloat8Tensor):
        return True
    return False


def is_float8_tensor(tensor):
    return (str(tensor.dtype) in [Const.FLOAT8_E5M2_TYPE, Const.FLOAT8_E4M3FN_TYPE, Const.FLOAT8_E8M0FNU_TYPE] 
            or is_hifloat8_tensor(tensor))
