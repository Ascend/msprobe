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

import re
import ast
import os
import json

from msprobe.core.common.file_utils import FileOpen, save_json, check_file_or_directory_path, create_directory
from msprobe.core.common.log import logger


def check_is_end_of_list(s, stack, pre_stack) -> bool:
    pattern = r"^('[^']*'|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\]+}*,$"
    if not bool(re.fullmatch(pattern, s)):
        return False
    if (len(pre_stack) - len(stack)) == s.count(']'):
        return True
    return False


def detect_prefix(log_lines, target_key: str = "{'actor_rollout_ref':") -> re.Pattern:
    """
    Detect and verify the prefix from the list of log lines.
    """
    first_line = log_lines[0]
    idx = first_line.find(target_key)
    if idx == -1:
        logger.error("Target key not found in the first line.")
        raise ValueError("Target key not found in the first line.")
    candidate = first_line[:idx]
    pattern = re.compile(re.escape(candidate))
    for line in log_lines[1:]:
        if not line.startswith(candidate):
            logger.error(f"The prefix of line {line} does not match {candidate}.")
            raise ValueError(f"The prefix of line {line} does not match {candidate}.")
    return pattern


def split_line_by_prefix(log_line, prefix_pattern: re.Pattern) -> str:
    matches = list(prefix_pattern.finditer(log_line))
    if not matches:
        return [log_line.strip()] if log_line.strip() else []
    parts = []
    for i, m in enumerate(matches):
        start = m.end()  # end of predix
        end = matches[i + 1].start() if i + 1 < len(matches) else len(log_line)
        content = log_line[start:end].strip()
        if content:
            parts.append(content)
    return parts


def is_generic_format(s: str) -> bool:
    return bool(re.fullmatch(r"'[^']*',| -?\d+(?:\.\d+)?,", s))


def get_in_string_value(ch, in_string, quote_char):
    if not in_string:
        in_string = True
        quote_char = ch
    elif quote_char == ch:
        in_string = False
        quote_char = None
    return in_string, quote_char


def get_config_start_idx(lines, target_key):
    config_start_ids = []
    for i, line in enumerate(lines):
        if target_key in line:
            config_start_ids.append(i)
        elif '{"actor_rollout_ref":' in line:
            config_start_ids.append(i)
            target_key = '{"actor_rollout_ref":'

    if len(config_start_ids) == 0:
        logger.error("Not found the start with 'actor_rollout_ref' dict in log.")
        raise ValueError("Not found the start with 'actor_rollout_ref' dict in log.")
    elif len(config_start_ids) > 1:
        logger.warning(
            f"This log contains {len(config_start_ids)} pieces of verl's config information, \
                       only the last config configuration will be collected!"
        )

    return config_start_ids[-1], target_key


def update_depth_square_bracket(in_string, ch, depth, square_bracket_stack):
    if in_string:
        return depth, square_bracket_stack

    if ch == '{':
        depth += 1
    elif ch == '}':
        depth -= 1
    elif ch == '[':
        square_bracket_stack.append(ch)
    elif ch == ']':
        if square_bracket_stack[-1] == '[':
            square_bracket_stack.pop()

    return depth, square_bracket_stack


def extract_dict(text: str) -> list:
    lines = text.splitlines()
    target_key = "{'actor_rollout_ref':"
    check_count = 3
    start_idx, target_key = get_config_start_idx(lines, target_key)

    depth = 0
    pre_depth = 0
    config_lines = []
    in_string = False
    quote_char = None
    square_bracket_stack = []
    pre_square_bracket_stack = []
    prefix_pattern = detect_prefix(lines[start_idx : start_idx + check_count], target_key)
    for line in lines[start_idx:]:
        # Handle multi-line content printed on one line
        cleaned_lines = split_line_by_prefix(line, prefix_pattern)
        for cleaned in cleaned_lines:
            pre_square_bracket_stack = square_bracket_stack.copy()
            pre_depth = depth
            for ch in cleaned:
                if ch in ('"', "'"):
                    in_string, quote_char = get_in_string_value(ch, in_string, quote_char)
                    continue

                # Only count brace depth when not inside a string
                depth, square_bracket_stack = update_depth_square_bracket(in_string, ch, depth, square_bracket_stack)

            if ('\':' in cleaned) or ('[' in square_bracket_stack and is_generic_format(cleaned)):
                config_lines.append(cleaned)
            elif square_bracket_stack != pre_square_bracket_stack:
                if check_is_end_of_list(cleaned, square_bracket_stack, pre_square_bracket_stack):
                    config_lines.append(cleaned)
                else:
                    square_bracket_stack = pre_square_bracket_stack.copy()
                    depth = pre_depth

        if depth == 0:
            return config_lines
    if depth != 0:
        raise ValueError(f"Unclosed dictionary, mismatched brackets of {depth}")

    return []


def verl_get_config_file_path(file_folder):
    real_file_folder = os.path.realpath(file_folder)
    if not os.path.isdir(real_file_folder):
        create_directory(real_file_folder)
    npu_config_file = os.path.join(real_file_folder, "NPU_config.json")
    bench_config_file = os.path.join(real_file_folder, "bench_config.json")
    return npu_config_file, bench_config_file


def check_log_extension(file_path):
    _, ext = os.path.splitext(file_path)
    return ext.lower() in ('.log', '.txt')


def verl_filter_config_info(verl_train_log_path, out_file_path):
    """Filter out the configuration-related parts from verl train log and save to generate configuration file.

    Args:
        verl_train_log_path (str): Path to verl train log
        out_file_path (str): Path to save configuration file
    """
    check_file_or_directory_path(verl_train_log_path)

    with FileOpen(verl_train_log_path, "r") as f:
        raw = f.read()

    try:
        lines = extract_dict(raw)
    except ValueError as e:
        logger.error(f"Failed extract config from {verl_train_log_path}:{e}.")
        raise

    dict_str = '\n'.join(lines)

    try:
        config = json.loads(dict_str)
    except json.JSONDecodeError:
        try:
            config = ast.literal_eval(dict_str)
        except (SyntaxError, ValueError) as e:
            logger.error(f"Failed analysis the config of {verl_train_log_path}, err_msg:{e}.")
            raise

    save_json(out_file_path, config, indent=2)
    logger.info(f"Saving json file to disk: {out_file_path}")
