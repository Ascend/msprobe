# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You may use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
# http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON‑INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

import re
import os
from msprobe.core.common.file_utils import FileOpen, save_json, check_file_or_directory_path, create_directory
from msprobe.core.common.log import logger

START_MARKER = "------------------------ arguments ------------------------"
END_MARKER = "-------------------- end of arguments ---------------------"


def get_args_raw_text(full_text: str) -> str:
    """Extract text content between start marker and end marker
    Args:
        full_text: full‑content of log file
    Returns:
        str: raw argument block string
    Raises:
        ValueError: when marker cannot be located
    """
    pos_start = full_text.find(START_MARKER)
    if pos_start == -1:
        raise ValueError("Start arguments marker not found")
    pos_start += len(START_MARKER)
    pos_end = full_text.find(END_MARKER, pos_start)
    if pos_end == -1:
        raise ValueError("End arguments marker not found")
    content = full_text[pos_start:pos_end]
    return content


def parse_argument_lines(raw_block: str) -> dict:
    """
    Parse argument line formatted as '  name ......... value'
    """
    param_pattern = re.compile(r"^\s+(.*?)\s+\.+\s+(.*)$")
    result = {}
    for line in raw_block.splitlines():
        line = line.rstrip()
        match = param_pattern.fullmatch(line)
        if not match:
            continue
        param_name, param_val_str = match.groups()
        param_name = param_name.strip()
        raw_val = param_val_str.strip()

        # automatic type conversion
        if raw_val == "True":
            result[param_name] = True
        elif raw_val == "False":
            result[param_name] = False
        elif re.fullmatch(r"-?\d+", raw_val):
            result[param_name] = int(raw_val)
        elif re.fullmatch(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$", raw_val):
            result[param_name] = float(raw_val)
        else:
            result[param_name] = raw_val
    return result


def slime_get_config_file_path(file_folder):
    real_file_folder = os.path.realpath(file_folder)
    if not os.path.isdir(real_file_folder):
        create_directory(real_file_folder)
    npu_config_file = os.path.join(real_file_folder, "NPU_config.json")
    bench_config_file = os.path.join(real_file_folder, "bench_config.json")
    return npu_config_file, bench_config_file


def slime_filter_config_info(slime_train_log_path: str, out_file_path: str):
    """Filter out the configuration‑related parts from slime train log and save to generate configuration file.

    Args:
        slime_train_log_path (str): Path to slime train log
        out_file_path (str): Path to save configuration file
    """
    check_file_or_directory_path(slime_train_log_path)
    with FileOpen(slime_train_log_path, "r") as f:
        raw = f.read()

    block_text = get_args_raw_text(raw)
    if not block_text.strip():
        err_msg = "Extracted argument string is empty"
        logger.error(f"Failed extract config from {slime_train_log_path}: {err_msg}")
        raise ValueError(err_msg)

    args_dict = parse_argument_lines(block_text)
    save_json(out_file_path, args_dict, indent=2)
    logger.info(f"Saving json file to disk: {out_file_path}")
    return args_dict
