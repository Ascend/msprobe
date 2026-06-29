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


PRE_LINE_REDUCE_VALUE_PATTERN = re.compile(r"^(?:'[^']*'|-?\d+(?:\.\d+)?|None|True|False|\[\]|\{\})(?:[}\]\]]+,?|,)$")


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


def split_line_by_prefix(log_line, prefix_pattern: re.Pattern) -> list:
    matches = list(prefix_pattern.finditer(log_line))
    if not matches:
        return []
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
            f"This log contains {len(config_start_ids)} pieces of verl's config information, "
            "only the last config configuration will be collected!"
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


def handle_closing_bracket(line: str, pos: int, ch: str, stack: list, blocks: list):
    if not stack:
        return

    left_char, left_start = stack.pop()
    if (left_char == '{' and ch == '}') or (left_char == '[' and ch == ']'):
        content = line[left_start + 1 : pos]
        if left_char == '{':
            test_str = '{' + content + '}'
        else:
            test_str = '[' + content + ']'
        try:
            ast.parse(test_str, mode='eval')
            is_valid = True
        except SyntaxError:
            is_valid = False
        blocks.append((left_start, pos, left_char, is_valid))
    else:
        stack.append((left_char, left_start))


def update_quote_state(ch, in_single, in_double):
    if ch == "'" and not in_double:
        in_single = not in_single
    elif ch == '"' and not in_single:
        in_double = not in_double
    return in_single, in_double


def get_list_dict_blocks(line):
    """
    Scan and record all matching brace/bracket blocks and determine their validity
    """
    in_single = False
    in_double = False
    stack = []
    blocks = []

    i = 0
    while i < len(line):
        ch = line[i]
        in_single, in_double = update_quote_state(ch, in_single, in_double)
        if in_single or in_double:
            i += 1
            continue

        # If not inside quotes, process brackets, braces
        if ch in '{[':
            stack.append((ch, i))
        elif ch in '}]':
            handle_closing_bracket(line, i, ch, stack, blocks)
        i += 1
    return blocks


def find_cleaned_line_end_comma(line: str) -> int:
    if line.count(',') == 1 and line.endswith(','):
        return -1  # default all line is valid

    blocks = get_list_dict_blocks(line)
    # Valid block ranges (inside only, excluding brackets) indicate whether a comma is ignored.
    valid_ranges = []
    for start, end, _, valid in blocks:
        if valid:
            valid_ranges.append((start + 1, end))

    # Scan from left to right, to find the first comma not in effective as the end comma
    in_single = False
    in_double = False

    for idx, ch in enumerate(line):
        in_single, in_double = update_quote_state(ch, in_single, in_double)
        if in_single or in_double:
            continue

        if ch == ',':
            # check the comma is in any effective blocks
            inside_valid = False
            for start, end in valid_ranges:
                if start <= idx < end:
                    inside_valid = True
                    break
            if not inside_valid:
                return idx

    return -1


def delete_invalid_info(line):
    end_indx = find_cleaned_line_end_comma(line)
    if end_indx != -1 and end_indx + 1 <= len(line):
        return line[: end_indx + 1]

    return line


def is_pre_line_reduce_value(s, depth):
    if depth != 0 and not s.endswith(","):
        return False
    return bool(PRE_LINE_REDUCE_VALUE_PATTERN.fullmatch(s))


def check_is_valid_cleaned_line(cleaned, square_bracket_stack, config_lines, depth):
    if '\':' in cleaned:
        return True
    if '[' in square_bracket_stack and is_generic_format(cleaned):
        return True
    # subsequent judgment is based on the case where a key-value pair is separated,
    # and requires that config_lines has at least one line of data.
    if not config_lines:
        return False
    # handle cases where keys and values are printed separately in logs
    if config_lines[-1].endswith(": ") and is_pre_line_reduce_value(cleaned, depth):
        return True
    if config_lines[-1].endswith(":") and is_pre_line_reduce_value(cleaned, depth):
        config_lines[-1] = config_lines[-1] + ' '
        return True
    # handle comma is printed separately in logs
    if not config_lines[-1].endswith(",") and cleaned.endswith(","):
        return True
    # handle cases where keys and values are printed separately in logs, where value with ":"
    if bool(re.fullmatch(r'^(["\'])([^"\']+)\1$', cleaned)):
        return True
    if config_lines[-1].endswith("'") and cleaned.startswith(":"):
        # after 2 chars ": "
        if len(cleaned) > 2 and is_pre_line_reduce_value(cleaned[2:], depth):
            return True
        config_lines.pop()

    return False


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
            cleaned = delete_invalid_info(cleaned)
            for i, ch in enumerate(cleaned):
                if ch in ('"', "'"):
                    in_string, quote_char = get_in_string_value(ch, in_string, quote_char)
                    continue

                # Only count brace depth when not inside a string
                depth, square_bracket_stack = update_depth_square_bracket(in_string, ch, depth, square_bracket_stack)
                if depth == 0:
                    cleaned = cleaned[: i + 1]
                    break
            if check_is_valid_cleaned_line(cleaned, square_bracket_stack, config_lines, depth):
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
