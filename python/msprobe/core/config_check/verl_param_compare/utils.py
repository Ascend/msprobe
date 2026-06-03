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
import os
from collections import OrderedDict


def flatten_dict(obj, prefix=''):
    """
    Recursively flatten a JSON object, returning a dictionary that maps each path to its value.
    --If the value is a non‑empty dictionary, recursively process its child keys and do not record the intermediate path.
    --If the value is an empty dictionary, treat it as a leaf node and record it as {}.
    --For all other types (list, string, number, boolean, null), record the value directly.
    """
    items = OrderedDict()
    if isinstance(obj, dict):
        if not obj:  # 空字典作为叶子
            items[prefix] = obj
        else:
            for key, value in obj.items():
                new_prefix = f"{prefix}/{key}" if prefix else key
                items.update(flatten_dict(value, new_prefix))
    elif isinstance(obj, list):
        items[prefix] = obj
    else:  # 标量: str, int, float, bool, None
        items[prefix] = obj
    return items


def compare_values(v1, v2, missing_marker=None):
    if v1 is missing_marker or v2 is missing_marker:
        return '否'
    return '是' if v1 == v2 else '否'


def value_to_str(v, missing_marker=None):
    if v is missing_marker:
        return 'NA'
    return repr(v)


def check_yaml_extension(file_path):
    _, ext = os.path.splitext(file_path)
    return ext.lower() in ('.yaml')
