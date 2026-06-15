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

import os
import sys
import stat
import re
import logging

from msprobe.core.common.log import logger
from msprobe.infer.utils.constants import PATH_WHITE_LIST_REGEX
from msprobe.infer.utils.constants import CONFIG_FILE_MAX_SIZE

MAX_SIZE_UNLIMITE = -1  # 不限制，必须显式表示不限制，读取必须传入
MAX_SIZE_LIMITE_CONFIG_FILE = 10 * 1024 * 1024  # 10M 普通配置文件，可以根据实际要求变更
MAX_SIZE_LIMITE_NORMAL_FILE = 4 * 1024 * 1024 * 1024  # 4G 普通模型文件，可以根据实际要求变更
MAX_SIZE_LIMITE_MODEL_FILE = 100 * 1024 * 1024 * 1024  # 100G 超大模型文件，需要确定能处理大文件，可以根据实际要求变更

PATH_WHITE_LIST_REGEX_WIN = re.compile(r"[^_:\\A-Za-z0-9/.-]")

SOLUTION_LEVEL = 35
SOLUTION_LEVEL_WIN = 45
logging.addLevelName(SOLUTION_LEVEL, "\033[1;32m" + "SOLUTION" + "\033[0m")  # green [SOLUTION]
logging.addLevelName(SOLUTION_LEVEL_WIN, "SOLUTION_WIN")


def is_legal_path_length(path):
    if len(path) > 4096 and not sys.platform.startswith("win"):  # linux total path length limit
        logger.error(f"file total path {path} length out of range (4096), please check the file(or directory) path")
        return False

    dirnames = path.split("/")
    for dirname in dirnames:
        if len(dirname) > 255:  # linux single file path length limit
            logger.error(f"file name {dirname} length out of range (255), please check the file(or directory) path")
            return False
    return True


def is_match_path_white_list(path):
    if PATH_WHITE_LIST_REGEX.search(path) and not sys.platform.startswith("win"):
        logger.error(f"path: {path} contains illegal char, legal chars include A-Z a-z 0-9 _ - / .")
        return False
    if PATH_WHITE_LIST_REGEX_WIN.search(path) and sys.platform.startswith("win"):
        logger.error(f"path: {path} contains illegal char, legal chars include A-Z a-z 0-9 _ - / . : \\")
        return False
    return True


def is_legal_args_path_string(path):
    # only check path string
    if not path:
        return True
    if not is_legal_path_length(path):
        return False
    if not is_match_path_white_list(path):
        return False
    return True


class OpenException(Exception):
    pass


class FileStat:
    def __init__(self, file) -> None:
        if not is_legal_path_length(file) or not is_match_path_white_list(file):
            raise OpenException("Path name is too long or contains invalid characters.")
        self.file = file
        self.is_file_exist = os.path.exists(file)
        if self.is_file_exist:
            self.file_stat = os.stat(file)
            self.realpath = os.path.realpath(file)
        else:
            self.file_stat = None

    @property
    def is_exists(self):
        return self.is_file_exist

    @property
    def is_file(self):
        return stat.S_ISREG(self.file_stat.st_mode) if self.file_stat else False

    @property
    def is_dir(self):
        return stat.S_ISDIR(self.file_stat.st_mode) if self.file_stat else False

    @property
    def file_size(self):
        return self.file_stat.st_size if self.file_stat else 0

    @property
    def permission(self):
        return stat.S_IMODE(self.file_stat.st_mode) if self.file_stat else 0o777

    def is_basically_legal(self, perm='none', strict_permission=True):
        if sys.platform.startswith("win"):
            return self.check_windows_permission(perm)
        else:
            return self.check_linux_permission(perm, strict_permission=strict_permission)

    def check_basic_permission(self, perm='none'):
        if not self.is_exists and perm != 'write':
            logger.error(f"path: {self.file} not exist, please check if file or dir is exist")
            return False
        return True

    def check_linux_permission(self, perm='none', strict_permission=True):
        if not self.check_basic_permission(perm=perm):
            return False
        if perm == 'read':
            if not os.access(self.realpath, os.R_OK) or self.permission & stat.S_IRUSR == 0:
                logger.error(
                    f"Current user doesn't have read permission to the file {self.file}, "
                    "as import file(or directory) permission should be at least 0o400(r--------)"
                )
                return False
        elif perm == 'write' and self.is_exists:
            if not os.access(self.realpath, os.W_OK):
                logger.error(
                    f"Current user doesn't have write permission to the file {self.file}, "
                    "as export file(or directory) permission should be at least 0o200(-w-------)"
                )
                return False
        return True

    def check_windows_permission(self, perm='none'):
        if not self.check_basic_permission(perm=perm):
            return False
        return True

    def is_legal_file_size(self, max_size):
        if not self.is_file:
            logger.error(f"path: {self.file} is not a file")
            return False
        if self.file_size > max_size:
            logger.error(f"file_size: {self.file_size} byte out of max limit {max_size} byte")
            return False
        else:
            return True

    def is_legal_file_type(self, file_types: list):
        if not self.is_file and self.is_exists:
            logger.error(f"path: {self.file} is not a file")
            return False
        for file_type in file_types:
            if os.path.splitext(self.file)[1] == f".{file_type}":
                return True
        logger.error(f"path: {self.file}, file type not in {file_types}")
        return False


def ms_open(file, mode="r", max_size=CONFIG_FILE_MAX_SIZE, **kwargs):
    file_stat = FileStat(file)

    if file_stat.is_exists and file_stat.is_dir:
        raise OpenException(f"Expecting a file, but it's a folder. {file}")

    if "r" in mode:
        if not file_stat.is_exists:
            raise OpenException(f"No such file or directory {file}")
        if max_size is None:
            raise OpenException(f"Reading files must have a size limit control. {file}")
        if max_size != MAX_SIZE_UNLIMITE and max_size < file_stat.file_size:
            raise OpenException(f"The file size has exceeded the specifications and cannot be read. {file}")

    if "w" in mode and file_stat.is_exists:
        os.remove(file)

    if "+" in mode:
        flags = os.O_RDONLY | os.O_RDWR
    elif "w" in mode or "a" in mode or "x" in mode:
        flags = os.O_RDONLY | os.O_WRONLY
    else:
        flags = os.O_RDONLY

    if "w" in mode or "x" in mode:
        flags = flags | os.O_TRUNC | os.O_CREAT
    if "a" in mode:
        flags = flags | os.O_APPEND | os.O_CREAT
    return os.fdopen(os.open(file, flags), mode, **kwargs)
