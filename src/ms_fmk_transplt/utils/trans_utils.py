#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.

import json
import os
import platform
import re
import shutil

import pandas as pd
from prettytable import PrettyTable

from . import transplant_logger as translog

try:
    import jedi
except ImportError:
    IS_JEDI_INSTALLED = False
else:
    IS_JEDI_INSTALLED = True

MAX_PYTHON_FILE_COUNT = 5000
MAX_SIZE_OF_INPUT_PATH = 50 * 1024 ** 3
MAX_SIZE_OF_RULE_FILE = 10 * 1024 ** 2
WINDOWS_PATH_LENGTH_LIMIT = 200
LINUX_FILE_NAME_LENGTH_LIMIT = 200
MAX_PYTHON_FILE_SIZE = 10 * 1024 ** 2
MAX_JSON_FILE_SIZE = 10 * 1024 ** 2
MAX_CSV_FILE_SIZE = 10 * 1024 ** 2


class TransplantException(Exception):
    pass


class InputCheckException(Exception):
    pass


class SoftlinkCheckException(Exception):
    pass


class DeleteFileException(Exception):
    pass


class JediCacheClearException(Exception):
    pass


def write_csv(content_list, output_dir, csv_name, header):
    if os.path.isfile(output_dir):
        csv_file = os.path.join(os.path.dirname(output_dir), '%s.csv' % csv_name)
    else:
        csv_file = os.path.join(output_dir, '%s.csv' % csv_name)
    if not os.path.exists(csv_file):
        data_frame = pd.DataFrame(columns=header)
        data_frame.to_csv(csv_file, index=False)

    new_data = pd.DataFrame(list(content for content in content_list))
    new_data.to_csv(csv_file, mode='a+', header=False, index=False)
    change_mode(csv_file)


def get_unsupported_op_dict(version):
    if version == '1.8.1':
        op_list_path = os.path.join(os.path.dirname(__file__), '../resource/op_list_1_8_1.json')
    else:
        op_list_path = os.path.join(os.path.dirname(__file__), '../resource/op_list_1_11_0.json')
    ops = get_file_content_bytes(op_list_path)
    return json.loads(ops).get('op_list')


def get_supported_op_dict(version):
    if version == '1.8.1':
        op_list_path = os.path.join(os.path.dirname(__file__), '../resource/supported_op_1_8_1.json')
    else:
        op_list_path = os.path.join(os.path.dirname(__file__), '../resource/supported_op_1_11_0.json')
    ops = get_file_content_bytes(op_list_path)
    return json.loads(ops).get("op_list")


def get_affinity_info_dict(version, need_type):
    need_type_list = ['class', 'function', 'torch']
    if need_type not in need_type_list:
        return {}
    if version == '1.8.1':
        op_list_path = os.path.join(os.path.dirname(__file__), '../resource/affinity_list_1_8_1.json')
    else:
        op_list_path = os.path.join(os.path.dirname(__file__), '../resource/affinity_list_1_11_0.json')
    ops = get_file_content_bytes(op_list_path)
    return json.loads(ops).get(need_type)


def get_precision_performance_advice_dict(version):
    if version == '1.8.1':
        op_list_path = os.path.join(os.path.dirname(__file__), '../resource/precision_performance_advice_1_8_1.json')
    else:
        op_list_path = os.path.join(os.path.dirname(__file__), '../resource/precision_performance_advice_1_11_0.json')
    ops = get_file_content_bytes(op_list_path)
    return json.loads(ops).get('api_precision_list'), json.loads(ops).get('api_performance_list')


def get_file_content_bytes(file):
    check_input_file_valid(file)
    with open(file, 'rb') as file_handle:
        return file_handle.read()


def get_file_content(file):
    check_input_file_valid(file)
    with open(file, 'r', encoding='utf8') as file_handle:
        return file_handle.read()


def write_file_content(file, code, permission=0o640):
    with os.fdopen(os.open(file, os.O_WRONLY | os.O_CREAT, permission),
                   'w', encoding='utf8', newline='') as file_handle:
        file_handle.truncate()
        file_handle.write(code)


def _compare_authority(origin_auth, advise_auth):
    new_auth = advise_auth[0]
    for i in range(1, 3):
        new_auth += str(int(origin_auth[i]) & int(advise_auth[i]))
    return int(new_auth, 8)


def _get_path_authority(path):
    authority = oct(os.stat(path).st_mode)[-3:]
    if os.path.isdir(path):
        new_auth = _compare_authority(authority, '750')
    elif path.endswith('.sh'):
        new_auth = _compare_authority(authority, '750')
    else:
        new_auth = _compare_authority(authority, '640')
    return new_auth


def change_mode(path):
    if not os.path.exists(path) or islink(path):
        return
    os.chmod(path, _get_path_authority(path))
    if os.path.isfile(path):
        return
    for root, dirs, files in os.walk(path):
        for dir_name in dirs:
            new_dir_path = os.path.join(root, dir_name)
            if not islink(new_dir_path):
                os.chmod(new_dir_path, _get_path_authority(new_dir_path))
        for file_name in files:
            file_path = os.path.join(root, file_name)
            if islink(file_path):
                continue
            os.chmod(file_path, _get_path_authority(file_path))


def generate_distributed_shell_file(path):
    code = '''export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29688
export HCCL_WHITELIST_DISABLE=1

NPUS=($(seq 0 7))
export RANK_SIZE=${#NPUS[@]}
rank=0
for i in ${NPUS[@]}
do
    export DEVICE_ID=${i}
    export RANK_ID=${rank}
    echo run process ${rank}
    please input your shell script here > output_npu_${i}.log 2>&1 &
    let rank++
done'''
    write_file_content(os.path.join(path, 'run_distributed_npu.sh'), code, permission=0o750)


def walk_input_path(path, output_free_size):
    py_file_counts = 0
    total_size = 0
    already_check_file_count_flag = False
    already_check_max_size_flag = False
    for root, _, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            if islink(file_path) or (not os.path.exists(file_path)):
                continue
            if check_file_need_analysis(file_path, path):
                py_file_counts += 1
            if not already_check_file_count_flag and py_file_counts >= MAX_PYTHON_FILE_COUNT:
                user_interactive_confirm(
                    f'The input path contains more than {MAX_PYTHON_FILE_COUNT} python files. '
                    f'Do you want to continue?')
                already_check_file_count_flag = True
            total_size += os.path.getsize(file_path)
            if total_size >= output_free_size:
                raise InputCheckException(
                    'The size of input path is too large, and the remaining disk space is not enough.')
            if not already_check_max_size_flag and total_size >= MAX_SIZE_OF_INPUT_PATH:
                user_interactive_confirm(
                    f'The size of the input path exceeds {int(MAX_SIZE_OF_INPUT_PATH / 1024 ** 3)}G. '
                    f'Do you want to continue?')
                already_check_max_size_flag = True
    return py_file_counts


def user_interactive_confirm(message):
    while True:
        check_message = input(message + " Enter 'continue' or 'c' to continue or enter 'exit' to exit: ")
        if check_message == "continue" or check_message == "c":
            break
        elif check_message == "exit":
            raise TransplantException("User canceled.")
        else:
            print("Input is error, please enter 'exit' or 'c' or 'continue'.")


def remove_path(path):
    if not os.path.exists(path):
        return
    try:
        if islink(path) or os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)
    except PermissionError as exp:
        raise DeleteFileException(f'Failed to delete {path}: {exp}') from exp


def check_path_owner_consistent(path):
    if platform.system().lower() == 'windows':
        return True
    # st_uid:user ID of owner, os.getuid: Return the current process's user id.
    return os.stat(path).st_uid == os.getuid()


def check_path_length_valid(path):
    path = os.path.realpath(path)
    if platform.system().lower() == 'windows':
        return len(path) <= WINDOWS_PATH_LENGTH_LIMIT
    else:
        return len(os.path.basename(path)) <= LINUX_FILE_NAME_LENGTH_LIMIT


def check_api_file_valid(path):
    filed_names = pd.read_csv(path).columns
    if '3rd-party API' not in filed_names:
        raise ValueError('The unsupported api file %s should contain 3rd-party API field!' % path)


def check_path_pattern_valid(path):
    if platform.system().lower() == 'windows':
        pattern = re.compile(r'(\.|\\|/|:|_|-|\s|[~0-9a-zA-Z])+')
        if not pattern.fullmatch(path):
            raise ValueError('Only the following characters are allowed in the path: A-Z a-z 0-9 - _ . / \\ :')
    else:
        pattern = re.compile(r'(\.|/|:|_|-|\s|[~0-9a-zA-Z])+')
        if not pattern.fullmatch(path):
            raise ValueError('Only the following characters are allowed in the path: A-Z a-z 0-9 - _ . / :')


def check_file_need_analysis(file, commonprefix, record=False):
    if not os.path.exists(file):
        return False
    if not file.endswith('.py'):
        return False
    file_relative_path = os.path.relpath(file, commonprefix)
    if islink(file):
        if record:
            translog.warning(f'{file_relative_path} is a soft link, skip.')
        return False
    if os.path.getsize(file) > MAX_PYTHON_FILE_SIZE:
        if record:
            translog.warning(
                f'The size of {file_relative_path} exceeds {int(MAX_PYTHON_FILE_SIZE / 1024 ** 2)}M, skip.')
        return False
    if not check_path_length_valid(file):
        if record:
            translog.warning(f'The real path or file name of {file_relative_path} is too long, skip.')
        return False
    return True


def get_main_file(main_file_path, input_path):
    if os.path.isfile(input_path):
        return os.path.basename(main_file_path)
    return os.path.relpath(os.path.realpath(main_file_path), os.path.realpath(input_path))


def name_to_jedi_position(file, line, name):
    if not os.path.isfile(file):
        return {}
    check_input_file_valid(file)
    with open(file, 'r', encoding='utf-8') as file_handler:
        file_lines = file_handler.readlines()
        if line > len(file_lines):
            return {}
        content = file_lines[line - 1]
    if not name or not content:
        return {}
    column = content.find(name)
    if column == -1:
        return {}
    return {'line': line, 'column': column}


def check_model_name_valid(name):
    if not re.match("^([a-zA-Z_]\\w*\\.)*([a-zA-Z_]\\w*)$", name):
        raise ValueError('Target model variable name is not valid!')


def clear_parso_cache():
    from jedi.settings import cache_directory
    if not os.path.exists(cache_directory):
        return
    try:
        remove_path(cache_directory)
    except DeleteFileException as exp:
        translog.warning(exp)


def refresh_parso_cache():
    from jedi.settings import cache_directory
    clear_parso_cache()
    if os.path.exists(cache_directory):
        raise JediCacheClearException('Failed to delete jedi cache. Please delete it manually.')
    os.makedirs(cache_directory, mode=0o700, exist_ok=True)


def check_is_subdirectory(path_may_be_parent, path_may_be_child):
    path_may_be_parent = os.path.realpath(path_may_be_parent)
    path_may_be_child = os.path.realpath(path_may_be_child)
    if path_may_be_parent[0] != path_may_be_child[0]:
        return False
    commonpath = os.path.commonpath([path_may_be_parent, path_may_be_child])
    return commonpath == path_may_be_parent


def islink(path):
    path = os.path.abspath(path)
    return os.path.islink(path)


def check_input_file_valid(input_path, max_file_size=MAX_JSON_FILE_SIZE):
    if islink(input_path):
        raise SoftlinkCheckException("Input path doesn't support soft link.")

    input_path = os.path.realpath(input_path)
    if not os.path.exists(input_path):
        raise ValueError('Input file %s does not exist!' % input_path)

    if not os.access(input_path, os.R_OK):
        raise PermissionError('Input file %s is not readable!' % input_path)

    if not check_path_length_valid(input_path):
        raise ValueError('The real path or file name of input is too long.')

    if os.path.getsize(input_path) > max_file_size:
        raise ValueError(f'The file is too large, exceeds {max_file_size // 1024 ** 2}MB')


def read_unsupported_op_csv(input_path):
    check_input_file_valid(input_path)
    apis_list = pd.read_csv(input_path)['3rd-party API'].values.tolist()
    apis_dict = {}
    for api in apis_list:
        for api_name in api.split():
            apis_dict[api_name] = ""
            if api_name.endswith('.forward'):
                apis_dict[api_name[:-1 * len('.forward')]] = ""
    return apis_dict


def get_analysis_result_statistics(result_dict: dict, output_path):
    if result_dict:
        tb = PrettyTable()
        tb.add_column('files', list(result_dict.keys()))
        tb.add_column('statistics', list(result_dict.values()))
        info = '   The detailed transplant result files are in the output path you defined, the relative path is ' \
               + output_path + '.' + '\n' + str(tb)
        translog.info_without_format(info)
