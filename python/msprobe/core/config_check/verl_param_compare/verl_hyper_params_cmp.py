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
import pandas as pd

from msprobe.core.common.file_utils import load_json, check_file_or_directory_path, write_df_to_csv, create_directory
from msprobe.core.common.log import logger
from msprobe.core.common.const import CompareConst


EXCLUDE_KEYWORDS = ['profiler/', 'ray_']


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


def verl_compare_hyper_params(npu_config, bench_config, output_dirpath):
    """Compare two config JSONs, ignoring non-critical data,
        to verify consistency of hyperparameters and export the results to a file.

    Args:
        npu_config (str): Path to npu config json file
        bench_config (str): Path to bench config json file
        output_dirpath (str): Path to save compared result csv
    """
    check_file_or_directory_path(npu_config)
    check_file_or_directory_path(bench_config)

    npu_cfg = load_json(npu_config)
    bench_cfg = load_json(bench_config)

    npu_flat = flatten_dict(npu_cfg)
    bench_flat = flatten_dict(bench_cfg)

    all_keys = sorted(set(npu_flat.keys()) | set(bench_flat.keys()))

    # Represent missing values with a unique sentinel to distinguish from None
    missing_marker = object()

    rows = []
    for key in all_keys:
        skip = False
        for kw in EXCLUDE_KEYWORDS:
            if kw in key:
                skip = True
                break
        if skip:
            continue
        v1 = npu_flat.get(key, missing_marker)
        v2 = bench_flat.get(key, missing_marker)
        consistent = compare_values(v1, v2, missing_marker)
        rows.append([key, value_to_str(v1, missing_marker), value_to_str(v2, missing_marker), consistent])

    real_output_dirpath = os.path.realpath(output_dirpath)
    if not os.path.isdir(real_output_dirpath):
        create_directory(real_output_dirpath)

    output_file = os.path.join(real_output_dirpath, "hyper_params_compare.csv")
    df = pd.DataFrame(rows, columns=CompareConst.VERL_HYPER_PARAM_COMPARE_COLUM)
    write_df_to_csv(df, output_file)

    logger.info(f"Saving verl compare file to disk: {output_file}")
