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

from msprobe.core.common.framework_adapter import FmkAdp
from msprobe.core.common.log import logger
from msprobe.core.common.const import Const


def merge_keys(dir_0, dir_1):
    output_list = list(dir_0.keys())
    output_list.extend(list(dir_1.keys()))
    return set(output_list)


def compare_dict(bench_dict, cmp_dict):
    result = []
    for key in set(bench_dict.keys()) | set(cmp_dict.keys()):
        if key in bench_dict and key in cmp_dict:
            if bench_dict[key] != cmp_dict[key]:
                result.append(f"{key}: {bench_dict[key]} -> {cmp_dict[key]}")
        elif key in bench_dict:
            result.append(f"{key}: [deleted] -> {bench_dict[key]}")
        else:
            result.append(f"{key}: [added] -> {cmp_dict[key]}")
    return result


def config_checking_print(msg):
    logger.info(f"[config checking log] {msg}")


def get_tensor_features(tensor):
    features = {
        "max": FmkAdp.tensor_max(tensor),
        "min": FmkAdp.tensor_min(tensor),
        "mean": FmkAdp.tensor_mean(tensor),
        "norm": FmkAdp.tensor_norm(tensor),
    }

    return features


def compare_dicts(dict1, dict2, path=''):
    deleted = []
    added = []
    changed = []
    result = {}

    for key in dict1:
        if key not in dict2:
            deleted.append(f"[Deleted]: {path + key}")
            result[key] = "[deleted]"
        else:
            if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                sub_deleted, sub_added, sub_changed, sub_result = compare_dicts(
                    dict1[key], dict2[key], path + key + '/')
                deleted.extend(sub_deleted)
                added.extend(sub_added)
                changed.extend(sub_changed)
                if sub_result:
                    result[key] = sub_result
            elif dict1[key] != dict2[key]:
                changed.append(f"[Changed]: {path + key} : {dict1[key]} -> {dict2[key]}")
                result[key] = f"[changed]: {dict1[key]} -> {dict2[key]}"
    for key in dict2:
        if key not in dict1:
            added.append(f"[Added]: {path + key}")
            result[key] = "[added]"
    return deleted, added, changed, result


def update_dict(ori_dict, new_dict):
    for key, value in new_dict.items():
        if key in ori_dict and ori_dict[key] != value:
            if "values" in ori_dict.keys():
                ori_dict[key]["values"].append(new_dict[key])
            else:
                ori_dict[key] = {"description": "duplicate_value", "values": [ori_dict[key], new_dict[key]]}
        else:
            ori_dict[key] = value


def process_pass_check(data):
    if Const.CONFIG_CHECK_ERROR in data:
        return Const.CONFIG_CHECK_ERROR
    elif Const.CONFIG_CHECK_WARNING in data:
        return Const.CONFIG_CHECK_WARNING
    else:
        return Const.CONFIG_CHECK_PASS
