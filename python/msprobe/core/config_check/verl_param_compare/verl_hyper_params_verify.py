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

from msprobe.core.common.file_utils import load_json, load_yaml, check_file_or_directory_path, write_df_to_csv
from msprobe.core.common.log import logger
from msprobe.core.common.const import CompareConst
from msprobe.core.config_check.verl_param_compare.utils import flatten_dict, compare_values, value_to_str


def load_config(file_path):
    if file_path.endswith('.json'):
        return load_json(file_path)
    elif file_path.endswith('.yaml'):
        return load_yaml(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {file_path}，仅支持 .json, .yaml格式")


def verl_verify_hyper_params(bench_config, tgt_config, output_dirpath):
    """Given a benchmark configuration,
       verify whether the key configurations in the target file are consistent with the benchmark

    Args:
        bench_config (str): Path to bench config yaml file
        tgt_config (str): Path to target log file
        output_dirpath (str): Path to save verified result excel
    """
    check_file_or_directory_path(bench_config)
    check_file_or_directory_path(tgt_config)

    tgt_cfg = load_config(tgt_config)
    bench_cfg = load_config(bench_config)
    tgt_flat = flatten_dict(tgt_cfg)
    bench_flat = flatten_dict(bench_cfg)

    all_keys = sorted(set(bench_flat.keys()))

    # Represent missing values with a unique sentinel to distinguish from None
    missing_marker = object()

    rows = []
    for key in all_keys:
        bench_value = bench_flat.get(key, missing_marker)
        tgt_value = tgt_flat.get(key, missing_marker)

        consistent = compare_values(bench_value, tgt_value, missing_marker)
        rows.append(
            [key, value_to_str(bench_value, missing_marker), value_to_str(tgt_value, missing_marker), consistent]
        )
    output_file = os.path.join(output_dirpath, "hyper_params_verify.csv")
    df = pd.DataFrame(rows, columns=CompareConst.VERL_HYPER_PARAM_VERIFY_COLUM)
    write_df_to_csv(df, output_file, encoding='utf-8-sig')

    logger.info(f"Saving verl verify file to disk: {output_file}")
