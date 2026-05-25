# coding=utf-8
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

"""
Function:
This file mainly involves the path check function.
"""
import os
import re
import stat
from enum import Enum

from cmp_utils import log
from cmp_utils import utils
from cmp_utils.reg_manager import RegManager
from cmp_utils.constant.compare_error import CompareError
from cmp_utils.constant.const_manager import ConstManager


class PathType(Enum):
    """
    The enum for path type
    """
    All = 0
    File = 1
    Directory = 2


def _check_path_file_or_directory(path: str, path_type: PathType) -> int:
    ret = CompareError.MSACCUCMP_NONE_ERROR
    if path_type == PathType.File:
        if os.path.exists(path) and not os.path.isfile(path):
            log.print_error_log('The path "%r" is not a file. Please check the path.' % path)
            ret = CompareError.MSACCUCMP_INVALID_PATH_ERROR
    elif path_type == PathType.Directory:
        if not os.path.isdir(path):
            log.print_error_log('The path "%r" is not a directory. Please check the path.' % path)
            ret = CompareError.MSACCUCMP_INVALID_PATH_ERROR
    return ret


def get_path_list_for_str(path_str: str) -> list:
    """
    Get path list for string
    :param path_str: the user input string
    :return: the path list
    """
    if ',' not in path_str:
        new_path = os.path.realpath(path_str)
        ret = check_path_valid(new_path, True, False)
        if ret != CompareError.MSACCUCMP_NONE_ERROR:
            raise CompareError(ret)
        return [new_path]
    input_path_list = []
    for input_path in path_str.split(','):
        new_path = os.path.realpath(input_path.strip())
        ret = check_path_valid(new_path, True, False)
        if ret != CompareError.MSACCUCMP_NONE_ERROR:
            continue
        input_path_list.append(new_path)
    if not input_path_list:
        log.print_error_log(
            'There is no valid file in "%r". Please check the path.' % path_str)
        raise CompareError(CompareError.MSACCUCMP_INVALID_PATH_ERROR)
    return input_path_list


def check_link(path: str):
    path = os.path.abspath(path)
    current = ""
    for part in path.split(os.sep):
        if not part:
            continue
        current = os.path.join(current, part)
        if os.path.islink(current):
            log.print_error_log(f'The path component {current} is a soft link.')
            raise CompareError(CompareError.MSACCUCMP_INVALID_PATH_ERROR)


def check_output_path_valid(path: str, exist: bool, path_type: PathType = PathType.Directory) -> int:
    """
    Check output path valid
    :param path: the path to check
    :param exist: the path exist
    :param path_type: the path type
    :return: VectorComparisonErrorCode
    """
    check_link(path)
    output_path = os.path.realpath(path)
    if path_type == PathType.File:
        output_path = os.path.dirname(output_path)
    if not os.path.exists(output_path):
        try:
            os.makedirs(output_path, mode=0o700)
        except OSError as ex:
            log.print_error_log('Failed to create "%r". %s' % (output_path, str(ex)))
            return CompareError.MSACCUCMP_INVALID_PATH_ERROR
        finally:
            pass
    return check_path_valid(path, exist, True, path_type)


def check_exec_file_valid(exist_path: str) -> int:
    """
    Check exec path valid
    :param path: the path to check
    :return: VectorComparisonErrorCode
    """
    file_stat = os.stat(exist_path)
    if file_stat.st_uid != 0 and file_stat.st_uid != os.getuid():
        log.print_error_log('You are not the owner of the path "%r".' % exist_path)
        return CompareError.MSACCUCMP_INVALID_PATH_ERROR

    if file_stat.st_gid != 0 and file_stat.st_gid not in os.getgroups():
        log.print_error_log('You are not in the group of the path "%r".' % exist_path)
        return CompareError.MSACCUCMP_INVALID_PATH_ERROR

    ret = check_others_permission(exist_path)
    if ret != CompareError.MSACCUCMP_NONE_ERROR:
        raise CompareError(ret)
    
    return check_path_valid(exist_path, True, False, PathType.File)


def check_path_all_file_exec_valid(custom_path):
    # all file and dir in custom path check safe
    file_count = 0
    for up_dir, dirs, files in os.walk(custom_path):
        if len(up_dir.split(os.path.sep)) > ConstManager.MAX_WALK_DIR_DEEP_NUM:
            raise CompareError(f"custom path is deep then {ConstManager.MAX_WALK_DIR_DEEP_NUM}")
        
        for name in files:
            sub_file_path = os.path.join(up_dir, name)
            ret = check_exec_file_valid(sub_file_path)
            if ret != CompareError.MSACCUCMP_NONE_ERROR:
                raise CompareError(ret)
            utils.check_file_size(sub_file_path, ConstManager.ONE_MB, is_raise=True)
            file_count = file_count + 1
            if file_count > ConstManager.MAX_WALK_FILE_NUM:
                raise CompareError(f"file count in custom path is more then {ConstManager.MAX_WALK_FILE_NUM}")

        for name in dirs:
            sub_dir_path = os.path.join(up_dir, name)
            ret = check_path_valid(sub_dir_path, True, False, PathType.Directory)
            if ret != CompareError.MSACCUCMP_NONE_ERROR:
                raise CompareError(ret)


def check_name_valid(name: str) -> int:
    """
    Check name valid
    :param name: the name to check
    :return: VectorComparisonErrorCode
    """
    if name == "":
        log.print_error_log("The parameter is null.")
        return CompareError.MSACCUCMP_INVALID_PARAM_ERROR
    name_pattern = re.compile(RegManager.SUPPORT_PATH_PATTERN)
    match = name_pattern.match(name)
    if match is None:
        log.print_only_support_error('name', name, '"A-Za-z0-9_\\./:()=-"')
        return CompareError.MSACCUCMP_INVALID_PARAM_ERROR
    return CompareError.MSACCUCMP_NONE_ERROR


def is_same_owner(path) -> bool:
    file_stat = os.stat(path)
    if os.getuid() != 0 and file_stat.st_uid != os.getuid():
        return False
    return True


def is_group_and_others_writable(path) -> bool:
    file_stat = os.stat(path)
    file_mode = file_stat.st_mode
    if bool(file_mode & stat.S_IWGRP) or bool(file_mode & stat.S_IWOTH):
        return True
    return False


def is_parent_dir_has_right_permission(path) -> bool:
    if not is_same_owner(path) or is_group_and_others_writable(path):
        return False
    return True


def check_path_valid(path: str, exist: bool, have_write_permission: bool = False,
                     path_type: PathType = PathType.All) -> int:
    """
    Check path valid
    :param path: the path to check
    :param exist: the path exist
    :param have_write_permission: have write permission
    :param path_type: the path type
    :return: VectorComparisonErrorCode
    """
    if path == "":
        log.print_error_log("The path is null.")
        return CompareError.MSACCUCMP_INVALID_PARAM_ERROR

    ret = check_name_valid(path)
    if ret != CompareError.MSACCUCMP_NONE_ERROR:
        return ret
    if os.path.islink(os.path.abspath(path)):
        log.print_error_log('The path "%r" is a softlink, not permitted.' % path)
        return CompareError.MSACCUCMP_INVALID_PATH_ERROR

    exist_path = os.path.realpath(path)
    if len(exist_path) >= ConstManager.LINUX_PATH_MAX_LEN:
        log.print_error_log(
            "The path length exceeds the maximum limit, "
            f"the maximum limit is {ConstManager.LINUX_PATH_MAX_LEN}."
        )
        return CompareError.MSACCUCMP_INVALID_PATH_ERROR

    for path_part in exist_path.split(os.sep):
        if path_part and len(path_part) >= ConstManager.LINUX_FILE_NAME_MAX_LEN:
            log.print_error_log(
                "The file name length exceeds the maximum limit, "
                f"the maximum limit is {ConstManager.LINUX_FILE_NAME_MAX_LEN}."
            )
            return CompareError.MSACCUCMP_INVALID_PATH_ERROR

    if not exist:
        exist_path = os.path.dirname(exist_path)

    if not os.path.exists(exist_path):
        log.print_error_log('The path "%r" does not exist.' % exist_path)
        return CompareError.MSACCUCMP_INVALID_PATH_ERROR

    if not os.access(exist_path, os.R_OK):
        log.print_error_log('You do not have permission to read the path "%r".' % exist_path)
        return CompareError.MSACCUCMP_INVALID_PATH_ERROR

    if have_write_permission and not os.access(exist_path, os.W_OK):
        log.print_error_log('You do not have permission to write the path "%r".' % exist_path)
        return CompareError.MSACCUCMP_INVALID_PATH_ERROR
    
    file_stat = os.stat(exist_path)
    if os.getuid() != 0 and file_stat.st_uid != os.getuid() and file_stat.st_gid not in os.getgroups():
        log.print_error_log('You are neither the owner nor in the group of the path "%r".' % exist_path)
        return CompareError.MSACCUCMP_INVALID_PATH_ERROR

    if bool(file_stat.st_mode & stat.S_IWGRP):
        log.print_warn_log(f"The file path is writable by group: {exist_path}.")

    if bool(file_stat.st_mode & stat.S_IWOTH):
        log.print_error_log(f"The file must not allow write access to others. File path: {exist_path}.")
        return CompareError.MSACCUCMP_INVALID_PATH_ERROR


    parent_directory = os.path.dirname(os.path.abspath(path))
    if not have_write_permission and not is_parent_dir_has_right_permission(parent_directory):
        log.print_warn_log('The permissions of the parent directory of the current file are incorrect.')

    return _check_path_file_or_directory(path, path_type)


def check_write_path_secure(path: str):
    if os.path.islink(path):
        os.unlink(path)
    if os.path.exists(path):
        os.remove(path)


def check_others_permission(exist_path: str) -> int:
    """
    Check others permission
    :param path: the path to check
    :return: VectorComparisonErrorCode
    """
    file_stat = os.stat(exist_path)
    file_mode = file_stat.st_mode
    # 判断others或group权限是否有写权限
    if file_stat.st_gid != 0 and bool(file_mode & stat.S_IWGRP):
        log.print_error_log(f"File {exist_path} is not safe. Groups have writing permission to this file.")
        return CompareError.MSACCUCMP_INVALID_PATH_ERROR

    if bool(file_mode & stat.S_IWOTH):
        log.print_error_log("File %r is dangerous. Others have writing "
                            "permission to this file. Please use chmod to dismiss the writing permission." % exist_path)
        return CompareError.MSACCUCMP_INVALID_PATH_ERROR

    return CompareError.MSACCUCMP_NONE_ERROR

