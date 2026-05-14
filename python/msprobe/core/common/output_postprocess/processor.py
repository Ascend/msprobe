#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

import importlib
import importlib.util
import os
from functools import lru_cache
from typing import Any, Callable, Dict

import numpy as np
import torch

from msprobe.core.common.const import Const
from msprobe.core.common.file_utils import load_yaml, FileChecker, FileCheckConst
from msprobe.pytorch.common.log import logger


_DEFAULT_YAML_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "api_output_postprocess.yaml")
_DEFAULT_TRUSTED_HANDLER_DIRS = os.path.dirname(os.path.realpath(__file__))
_BACKENDS = ("golden", "target")


def should_postprocess_output(api_name: str, backend: str):
    """判断当前 API 是否命中指定后端侧(golden/target)的后处理 handler 配置。"""
    backend_enabled_api_names = _get_backend_enabled_api_names(backend, "enabled_postprocess_api_names")
    return api_name in backend_enabled_api_names


def should_postprocess_output_for_compare(op_name: str, backend: str):
    """判断当前 API 是否命中指定后端侧(golden/target)的后处理 handler 配置。"""
    backend_enabled_api_names = _get_backend_enabled_api_names(backend, "enabled_compare_api_names")

    for name in backend_enabled_api_names:
        if name in op_name:
            return True, name
        
    return False, None


def postprocess_output(api_name: str, output, args, kwargs: Dict[str, Any], backend: str):
    """
    对单个 API 的指定后端侧(golden/target)输出执行 handler 后处理。
    """
    backend_handlers = _get_backend_handlers(backend, "acc_check_handlers")
    handler_spec = backend_handlers.get(api_name)
    if handler_spec:
        return _run_acc_check_handler(handler_spec, api_name, output, args, kwargs)

    return output


def extract_valid_len(api_name: str, args, kwargs: Dict[str, Any], backend: str):
    """
    从参数中提取有效长度，供外部调用。

    参数:
        api_name (str): 当前 API 名称。
        args: 当前侧位置参数。
        kwargs (dict): 当前侧关键字参数。
        backend (str): 后端类型，如 'golden' 或 'target'。

    返回值:
        int 或 None: 提取到的有效长度；未找到或条件不满足时返回 None。
    """
    backend_handlers = _get_backend_handlers(backend, "compare_handlers")
    handler_spec = backend_handlers.get(api_name)
    if handler_spec:
        return _run_compare_handler(handler_spec, api_name, args, kwargs)

    return None


def _extract_valid_len_by_group_key(api_name: str, group_key: str, kwargs: Dict[str, Any]):
    return _get_valid_len_from_group_key(api_name, group_key, kwargs)


def clean_single_tensor(tensor, valid_len: int):
    """
    对外暴露的单 tensor 清理函数。

    参数:
        tensor: 待清理数据，支持 torch.Tensor 和 numpy.ndarray。
        valid_len (int): 有效长度。

    返回值:
        与输入类型保持一致的清理结果；输入不符合条件时返回原值。
    """
    if isinstance(tensor, np.ndarray):
        clean_tensor = _clean_single_tensor(torch.from_numpy(tensor), valid_len)
        return clean_tensor.numpy()

    return _clean_single_tensor(tensor, valid_len)


def _get_backend_enabled_api_names(backend: str, enabled_key: str):
    rules = _get_rules()
    enabled_api_names = rules.get(enabled_key, {})
    return enabled_api_names.get(backend, set())


def _get_backend_handlers(backend: str, handlers_key: str):
    rules = _get_rules()
    handlers = rules.get(handlers_key, {})
    return handlers.get(backend, {})


def _clean_by_group_key(api_name: str, group_key: str, output, kwargs: Dict[str, Any]):
    """
    通过 group_key 对应张量计算有效长度，并清理输出中的 dirty 区域。

    参数:
        api_name (str): 当前 API 名称。
        group_key (str): 计算有效长度使用的键名，如 group_index/group_list。
        output: 当前侧输出。
        kwargs (dict): 当前侧关键字参数。

    返回值:
        清理后的输出；若条件不满足则返回原输出。
    """
    valid_len = _get_valid_len_from_group_key(api_name, group_key, kwargs)
    if valid_len is None:
        return output

    return _clean_outputs(output, valid_len)


def _get_valid_len_from_group_key(api_name: str, group_key: str, kwargs: Dict[str, Any]):
    group_tensor = kwargs.get(group_key) if isinstance(kwargs, dict) else None
    if group_tensor is None or not isinstance(group_tensor, torch.Tensor):
        return None

    if group_tensor.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool):
        logger.warning(f"[{api_name}] '{group_key}' tensor dtype {group_tensor.dtype} "
                       f"is not supported for valid_len extraction.")
        return None

    valid_len = int(group_tensor.sum().item())
    if valid_len < 0:
        logger.warning(f"[{api_name}] Extracted valid_len {valid_len} from '{group_key}' is invalid.")
        return None

    logger.debug(f"[{api_name}] Cleaning dirty outputs by '{group_key}', valid_len={valid_len}")
    return valid_len


def _clean_outputs(outputs, valid_len: int):
    """
    清理输出中的脏数据，保持容器类型不变。

    参数:
        outputs: 输出对象，支持 Tensor/list/tuple/None。
        valid_len (int): 有效数据长度（按第 0 维截取）。

    返回值:
        清理后的输出对象；输入为 None 时返回 None。
    """
    if outputs is None:
        return None

    if isinstance(outputs, torch.Tensor):
        return _clean_single_tensor(outputs, valid_len)

    if isinstance(outputs, tuple):
        return tuple(_clean_single_tensor(t, valid_len) for t in outputs)

    if isinstance(outputs, list):
        return [_clean_single_tensor(t, valid_len) for t in outputs]

    return outputs


def _clean_single_tensor(tensor, valid_len: int):
    """
    清理单个张量：保留 [0, valid_len) 范围，其他位置置零。

    参数:
        tensor (torch.Tensor): 待清理张量。
        valid_len (int): 有效长度。

    返回值:
        清理后的张量；输入不符合条件时返回原值。
    """
    if tensor is None or not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
        return tensor

    clean_tensor = torch.zeros_like(tensor)
    if tensor.dim() == 0:
        return tensor

    safe_len = max(0, min(valid_len, tensor.shape[0]))
    if tensor.dim() == 1:
        clean_tensor[:safe_len] = tensor[:safe_len]
    else:
        clean_tensor[:safe_len, ...] = tensor[:safe_len, ...]
    return clean_tensor


def _run_acc_check_handler(handler_spec: str, api_name: str, output, args, kwargs: Dict[str, Any]):
    """
    安全执行预检场景的处理函数，失败时回退到原输出。

    参数:
        handler_spec (str): 处理函数定位串，格式为 Python 文件路径:函数名。
        api_name (str): 当前 API 名称。
        output: 当前侧输出。
        args: 当前侧位置参数。
        kwargs (dict): 当前侧关键字参数。

    返回值:
        处理后的输出；加载或执行失败时返回原输出。
    """
    ok, new_out = _try_run_handler(
        handler_spec,
        api_name,
        lambda handler: handler(api_name, output, args, kwargs),
        scene_name="postprocess",
    )
    if not ok:
        return output

    if new_out is None:
        logger.warning(f"[{api_name}] Postprocess handler returned None, keep original output.")
        return output
    return new_out


def _run_compare_handler(handler_spec: str, api_name: str, args, kwargs: Dict[str, Any]):
    """
    安全执行比对场景的 valid_len 提取函数。

    参数:
        handler_spec (str): 处理函数定位串，格式为 Python 文件路径:函数名。
        api_name (str): 当前 API 名称。
        args: 当前侧位置参数。
        kwargs (dict): 当前侧关键字参数。

    返回值:
        int 或 None: 提取得到的有效长度；加载或执行失败时返回 None。
    """
    ok, valid_len = _try_run_handler(
        handler_spec,
        api_name,
        lambda handler: handler(api_name, args, kwargs),
        scene_name="valid_len",
    )
    if not ok:
        return None

    if valid_len is None:
        logger.warning(f"[{api_name}] Valid_len handler returned None.")
        return None

    try:
        return int(valid_len)
    except (TypeError, ValueError):
        logger.warning(f"[{api_name}] Valid_len handler returned invalid value: {valid_len}")
        return None


def _try_run_handler(handler_spec: str, api_name: str, run_handler: Callable, scene_name: str):
    """加载并执行配置的处理函数，返回 (是否成功, 执行结果)。"""
    try:
        handler = _load_callable(handler_spec)
    except Exception as error:
        logger.warning(f"[{api_name}] Load {scene_name} handler failed: {error}")
        return False, None

    try:
        result = run_handler(handler)
    except Exception as error:
        logger.warning(f"[{api_name}] Execute {scene_name} handler failed: {error}")
        return False, None

    return True, result


def _load_callable(spec: str) -> Callable:
    """
    从配置串加载可调用对象。

    参数:
        spec (str): 函数定位串，支持 文件路径:函数名。

    返回值:
        Callable: 可调用的处理函数对象。

    异常:
        ValueError: 配置串格式不合法。
        TypeError: 目标对象不可调用。
    """
    if not isinstance(spec, str) or ":" not in spec:
        raise ValueError("handler spec must be '<py_path>:<function_name>'")

    py_path, function_name = spec.split(":", 1)
    if py_path.endswith(Const.PY_SUFFIX) or os.path.sep in py_path:
        module = _load_module_from_py_path(py_path)
    else:
        raise ValueError("handler spec must be '<py_path>:<function_name>'")

    handler = getattr(module, function_name, None)
    if handler is None or not callable(handler):
        logger.error(f"{function_name} in {py_path} is not callable")
        raise TypeError
    return handler


def _load_module_from_py_path(py_path: str):
    """
    从 Python 文件路径动态加载模块对象。

    参数:
        py_path (str): Python 文件路径。

    返回值:
        module: 动态加载后的模块对象。

    异常:
        FileNotFoundError: 文件不存在。
        ImportError: 模块规格构建失败。
    """
    py_path = _resolve_handler_py_path(py_path)
    path_checker = FileChecker(py_path, FileCheckConst.FILE, FileCheckConst.READ_ABLE, FileCheckConst.PY_SUFFIX)
    py_path = path_checker.common_check()
    _ensure_path_in_trusted_dirs(py_path)

    if not os.path.isfile(py_path):
        logger.error(f"handler file not found: {py_path}")
        raise FileNotFoundError

    module_name = f"msprobe_output_postprocess_{abs(hash(py_path))}"
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    if spec is None or spec.loader is None:
        logger.error(f"create module spec failed for {py_path}")
        raise ImportError

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_handler_py_path(py_path: str):
    """将相对 handler 路径解析到 output_postprocess 目录下。"""
    expanded_path = os.path.expanduser(py_path)
    if os.path.isabs(expanded_path):
        return expanded_path

    return os.path.join(_DEFAULT_TRUSTED_HANDLER_DIRS, expanded_path)


def _ensure_path_in_trusted_dirs(py_path: str):
    """仅允许从可信目录中加载处理脚本。"""
    trusted_dir = os.path.realpath(_DEFAULT_TRUSTED_HANDLER_DIRS)
    if os.path.dirname(py_path) == trusted_dir:
        return

    logger.error(f"handler path is outside trusted dirs: {py_path}")
    raise PermissionError


def _parse_handlers(handlers: Dict[str, Any]):
    parsed_handlers = {backend: {} for backend in _BACKENDS}
    if not isinstance(handlers, dict):
        logger.warning("postprocess handlers must be a dict in postprocess yaml, ignore handler config.")
        return parsed_handlers

    for backend in _BACKENDS:
        backend_handlers = handlers.get(backend, {})
        if isinstance(backend_handlers, dict):
            parsed_handlers[backend] = backend_handlers
    return parsed_handlers


@lru_cache(maxsize=1)
def _get_rules():
    """
    从内置 YAML 读取规则并缓存。

    返回值:
        dict: 规则字典，包含:
            - acc_check_handlers: 预检场景 API 到处理函数定位串的映射。
            - compare_handlers: 比对场景 API 到 valid_len 提取函数定位串的映射。
    """
    config_data = load_yaml(_DEFAULT_YAML_PATH) or {}
    if not isinstance(config_data, dict):
        logger.warning("Default postprocess yaml root must be dict, fallback to empty config.")
        config_data = {}

    acc_check_handlers = _parse_handlers(config_data.get("acc_check_handlers", {}))
    compare_handlers = _parse_handlers(config_data.get("compare_handlers", {}))

    return {
        "acc_check_handlers": acc_check_handlers,
        "compare_handlers": compare_handlers,
        "enabled_postprocess_api_names": {
            backend: set(acc_check_handlers.get(backend, {}).keys())
            for backend in _BACKENDS
        },
        "enabled_compare_api_names": {
            backend: set(compare_handlers.get(backend, {}).keys())
            for backend in _BACKENDS
        },
    }
