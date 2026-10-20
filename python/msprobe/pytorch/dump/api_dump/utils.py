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

import importlib
import inspect
import os

from msprobe.core.common.file_utils import load_yaml
from msprobe.core.common.log import logger


def get_ops():
    cur_path = os.path.dirname(os.path.realpath(__file__))
    yaml_path = os.path.join(cur_path, "support_wrap_ops.yaml")
    ops = load_yaml(yaml_path)
    if not isinstance(ops, dict):
        logger.warning("[msprobe] support_wrap_ops.yaml is empty or malformed, no api will be wrapped")
        return set()
    wrap_ops = set()
    for category in ["functional", "tensor", "torch", "torch_npu"]:
        api_list = ops.get(category)
        if api_list is None:
            logger.warning(f"[msprobe] support_wrap_ops.yaml has no '{category}' category, treated as empty")
            continue
        if not isinstance(api_list, list):
            logger.warning(f"[msprobe] '{category}' in support_wrap_ops.yaml is not a list, ignored")
            continue
        wrap_ops.update(api_list)
    return wrap_ops


def dynamic_import_op(package, white_list):
    package_name = package.__name__
    ops = {}
    ops_dir, _ = os.path.split(package.__file__)
    for file_name in os.listdir(ops_dir):
        if file_name in white_list:
            sub_module_name = file_name[:-3]
            module_name = f"{package_name}.{sub_module_name}"
            try:
                module = importlib.import_module(module_name)
            except Exception:  # noqa: BLE001 - 保持兜底，任意导入失败都跳过该模块
                logger.warning(f"import {module_name} failed!")
                continue

            func_members = inspect.getmembers(module, inspect.isfunction)
            for func_member in func_members:
                func_name, func = func_member[0], func_member[1]
                ops[f"{sub_module_name}.{func_name}"] = func
    return ops
