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

from collections import defaultdict
import os

# Bandit B403 is suppressed here because pickle is only used to catch
# UnpicklingError from the guarded torch.load fallback path.
import pickle  # nosec B403
import re

from msprobe.infer.utils.constants import TENSOR_MAX_SIZE, EXT_SIZE_MAPPING, PATH_WHITE_LIST_REGEX
from msprobe.core.common.log import logger
from msprobe.infer.utils.file_open_check import is_legal_path_length

# 记录工具函数递归的深度
recursion_depth = defaultdict(int)


def confirmation_interaction(prompt):
    confirm_pattern = re.compile(r'y(?:es)?', re.IGNORECASE)

    try:
        user_action = input(prompt)
    except Exception:
        return False

    return bool(confirm_pattern.match(user_action))


def check_file_ext(path, ext: str):
    if not isinstance(path, str):
        raise TypeError(f"Expected first positional argument type 'str', got {type(path)} instead")

    if not isinstance(ext, str):
        raise TypeError(f"Expected second positional argument type 'str', got {type(ext)} instead")

    path_ext = os.path.splitext(path)[1]

    if path_ext != ext:
        return False

    return True


def check_file_size_based_on_ext(path, ext=None):
    """Check the file size based on extension. This function uses `os.stat` to get file size may lead to OSError"""

    if not isinstance(path, str):
        raise TypeError(f"Expected path to be 'str', got {type(path)} instead")

    ext = ext or os.path.splitext(path)[1]
    size = os.path.getsize(path)  # may lead to errors

    if ext in EXT_SIZE_MAPPING:
        if size > EXT_SIZE_MAPPING[ext]:
            return False
    else:
        if size > TENSOR_MAX_SIZE:
            confirmation_prompt = (
                "The file %r is larger than expected. "
                "Attempting to read such a file could potentially impact system performance.\n"
                "Please confirm your awareness of the risks associated with this action (y/n): " % path
            )
            return confirmation_interaction(confirmation_prompt)

    return True


def safe_torch_load(path, **kwargs):
    import torch  # Do not move it !!! it may caused Import Error

    kwargs['weights_only'] = True
    tensor = None

    while True:
        try:
            # Bandit B614 is suppressed here because the default path forces
            # weights_only=True, and the unsafe fallback requires user confirmation.
            tensor = torch.load(path, **kwargs)  # nosec B614
        except pickle.UnpicklingError:
            confirmation_prompt = (
                "Weights only load failed. Re-running `torch.load` with `weights_only` "
                "set to `False` will likely succeed, but it can result in arbitrary code "
                "execution. Do it only if you get the file from a trusted source.\n"
                "Please confirm your awareness of the risks associated with this action ([y]/n): "
            )
            if not confirmation_interaction(confirmation_prompt):
                raise
            kwargs['weights_only'] = False
        else:
            break

    return tensor


def load_file_to_read_common_check(path: str, exts=None):
    if not isinstance(path, str):
        raise TypeError("'path' should be 'str'")

    if isinstance(exts, (tuple, list)):
        if not any(check_file_ext(path, ext) for ext in exts):
            logger.error(f"Expected extension to be one of {exts}")
            raise ValueError

    elif exts is not None:
        logger.error(f"Expected 'exts' to be 'List[str]', got {type(exts)} instead")
        raise TypeError

    if re.search(PATH_WHITE_LIST_REGEX, path):
        logger.error(f"Invalid character: {path}")
        raise ValueError

    if not is_legal_path_length(path):
        logger.error("Invalid path length.")
        raise ValueError

    path = os.path.realpath(path)

    try:
        file_status = os.stat(path)
    except OSError as e:
        logger.error(f"{e.strerror}: {path}")
        raise

    if not os.st.S_ISREG(file_status.st_mode):
        logger.error(f"Not a regular file: {path}")
        raise ValueError

    if not check_file_size_based_on_ext(path):
        logger.error(f"File too large: {path}")
        raise ValueError

    if (os.st.S_IWOTH & file_status.st_mode) == os.st.S_IWOTH:
        logger.error(f"Vulnerable path: {path} should not be other writeable")
        raise PermissionError

    return path


def is_valid_command(arg_str, index):
    first_whitelist_pattern = re.compile(r"^[a-zA-Z0-9_\-./=:,\[\] ]+$")
    whitelist_pattern = re.compile(r"^[a-zA-Z0-9_\-./=:,\[\] ;]+$")

    if index == 0:
        return re.fullmatch(first_whitelist_pattern, arg_str), first_whitelist_pattern
    else:
        return re.fullmatch(whitelist_pattern, arg_str), whitelist_pattern


def filter_cmd(paras):
    filtered = []
    for index, arg in enumerate(paras):
        arg_str = str(arg)
        valid, pattern = is_valid_command(arg_str, index)
        if not valid:
            raise ValueError(f"The command contains invalid characters. Only the '{pattern}' pattern is allowed.")
        filtered.append(arg_str)
    return filtered
