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
import pandas as pd

from msprobe.core.common.file_utils import load_json, check_file_or_directory_path, write_df_to_csv, create_directory
from msprobe.core.common.log import logger
from msprobe.core.common.const import CompareConst
from msprobe.core.config_check.verl_param_compare.utils import flatten_dict, compare_values, value_to_str


EXCLUDE_KEYWORDS = ['profiler/', 'ray_']


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
