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

"""
融合算子标杆函数自动注册模块

功能：
  从 fusion_operator_config.yaml 读取融合算子配置，自动发现并注册前向/反向标杆函数。
  用户添加新融合算子时，只需：
    1. 在 bench_functions/ 下创建 Python 文件，实现标杆函数
    2. 在 bench_functions/fusion_operator_config.yaml 中注册算子信息
  无需修改本文件或任何其他源代码文件。
"""

import os

from msprobe.core.common.file_utils import load_yaml
from msprobe.pytorch.common.utils import logger
from msprobe.pytorch import bench_functions


_FUSION_CONFIG_CACHE = None
_FUSION_REGISTRIES_CACHE = None


def get_fusion_config():
    """加载融合算子注册配置（带缓存，仅首次从 YAML 加载）"""
    global _FUSION_CONFIG_CACHE
    if _FUSION_CONFIG_CACHE is not None:
        return _FUSION_CONFIG_CACHE
    bench_dir = os.path.dirname(bench_functions.__file__)
    config_path = os.path.join(bench_dir, "fusion_operator_config.yaml")
    if not os.path.exists(config_path):
        logger.warning(
            f"Fusion operator config file not found at {config_path}. No custom fusion operators will be loaded."
        )
        _FUSION_CONFIG_CACHE = {}
        return _FUSION_CONFIG_CACHE
    config = load_yaml(config_path)
    _FUSION_CONFIG_CACHE = config.get("operators", {})
    return _FUSION_CONFIG_CACHE


def _resolve_func(func_name, bench_functions_module):
    """从 bench_functions 模块中解析函数"""
    if func_name is None:
        return None
    func = getattr(bench_functions_module, func_name, None)
    if func is None:
        logger.warning(
            f"Fusion operator bench function '{func_name}' not found in "
            f"bench_functions module. Available functions starting with 'npu_': "
            f"{[n for n in dir(bench_functions_module) if n.startswith('npu_') and callable(getattr(bench_functions_module, n, None))]}"
        )
    return func


class Register(dict):
    """可调用对象注册表，按函数名注册可调用对象"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dict = {}

    def __call__(self, target_func_list):
        for target in target_func_list:
            self.register(target)

    def __setitem__(self, key, value):
        self._dict[key] = value

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def __str__(self):
        return str(self._dict)

    def __len__(self):
        return len(self._dict)

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def items(self):
        return self._dict.items()

    def register(self, target):
        def add_register_item(key, value):
            if key in self._dict:
                logger.warning(f"{value.__name__} has been registered before, so we will override it.")
            self[key] = value
            return value

        if callable(target):
            return add_register_item(target.__name__, target)
        else:
            raise TypeError(f"The func {target} is not callable.")


def _build_registries():
    """
    从 fusion_operator_config.yaml 构建前向/反向函数注册表。

    Returns:
        tuple: (npu_custom_functions, npu_custom_grad_functions)
            - npu_custom_functions: 前向标杆函数注册表
            - npu_custom_grad_functions: 反向标杆函数注册表
    """
    forward_registry = Register()
    backward_registry = Register()

    operators = get_fusion_config()
    if not operators:
        logger.warning("No fusion operators configured in fusion_operator_config.yaml")
        return forward_registry, backward_registry

    for op_name, op_config in operators.items():
        # 注册前向函数
        forward_func_name = op_config.get("forward")
        if forward_func_name:
            func = _resolve_func(forward_func_name, bench_functions)
            if func is not None:
                # 用算子名注册（用于 npu 侧查找），同时也用函数名注册
                forward_registry[op_name] = func

        # 注册反向函数
        backward_func_name = op_config.get("backward")
        if backward_func_name:
            func = _resolve_func(backward_func_name, bench_functions)
            if func is not None:
                backward_registry[backward_func_name] = func

    logger.info(
        f"Loaded {len(forward_registry)} forward and {len(backward_registry)} backward "
        f"fusion operator bench functions from config."
    )
    logger.debug(f"Registered forward operators: {list(forward_registry.keys())}")
    return forward_registry, backward_registry


def get_fusion_registries():
    """Build and cache fusion operator registries on first use."""
    global _FUSION_REGISTRIES_CACHE
    if _FUSION_REGISTRIES_CACHE is None:
        _FUSION_REGISTRIES_CACHE = _build_registries()
    return _FUSION_REGISTRIES_CACHE


def get_npu_custom_functions():
    return get_fusion_registries()[0]


# The backward operator is not used but is registered to support future feature expansion.
def get_npu_custom_grad_functions():
    return get_fusion_registries()[1]
