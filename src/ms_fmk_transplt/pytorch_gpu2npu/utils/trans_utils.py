#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.

import json
import os
import platform
import re
import shutil

import pandas as pd

import pytorch_gpu2npu.common_rules.common_rule as rule_module
from pytorch_gpu2npu.distributed_rules import distributed_rule
from pytorch_gpu2npu.pytorch_v1_5_0 import InitApexRule, Amp2Apex
from pytorch_gpu2npu.utils import transplant_logger as translog

try:
    import jedi
except ImportError:
    IS_JEDI_INSTALLED = False
else:
    IS_JEDI_INSTALLED = True

MAX_PYTHON_FILE_COUNT = 5000
MAX_SIZE_OF_INPUT_PATH = 50 * 1024 ** 3
MAX_SIZE_OF_RULE_FILE = 10 * 1024 ** 2
WINDOWS_PATH_LENGTH_LIMIT = 256
LINUX_FILE_NAME_LENGTH_LIMIT = 255
MAX_PYTHON_FILE_SIZE = 10 * 1024 ** 2


class TransplantException(Exception):
    pass


class InputCheckException(Exception):
    pass


class SoftlinkCheckException(Exception):
    pass


class DeleteFileException(Exception):
    pass


def write_csv(content_list, script_file, script_dir, csv_type):
    header_dict = {
        "change_list": ('File', 'Start Line', 'End Line', 'Operation Type', 'Message'),
        "unsupported_op": ('File', 'Start Line', 'End Line', 'OP', 'Tips')
    }
    if os.path.isfile(script_dir):
        csv_file = os.path.join(os.path.dirname(script_dir), '%s.csv' % csv_type)
    else:
        csv_file = os.path.join(script_dir, '%s.csv' % csv_type)
    header = header_dict.get(csv_type)
    if os.path.exists(csv_file):
        remove_path(csv_file)
    if not os.path.exists(csv_file):
        data_frame = pd.DataFrame(columns=header)
        data_frame.to_csv(csv_file, index=False)

    if os.path.isdir(script_dir):
        rel_script_file_name = os.path.relpath(script_file, script_dir)
    else:
        rel_script_file_name = os.path.basename(script_file)
    new_data = pd.DataFrame(list(([rel_script_file_name] + content) for content in content_list))
    new_data.to_csv(csv_file, mode='a+', header=False, index=False)
    change_mode(csv_file)


def get_op_list(version):
    if version == '1.8.1':
        op_list_path = os.path.join(os.path.dirname(__file__), '../pytorch_v1_8_1/op_list_1_8_1.json')
    else:
        op_list_path = os.path.join(os.path.dirname(__file__), '../pytorch_v1_5_0/op_list_1_5_0.json')
    ops = get_file_content_bytes(op_list_path)
    op_list = json.loads(ops).get('op_list')
    return op_list


def get_file_content_bytes(file):
    with open(file, 'rb') as file_handle:
        return file_handle.read()


def get_file_content(file):
    with open(file, 'r', encoding='utf8') as file_handle:
        return file_handle.read()


def write_file_content(file, code, permission=0o640):
    with os.fdopen(os.open(file, os.O_WRONLY | os.O_CREAT, permission),
                   'w', encoding='utf8', newline='') as file_handle:
        file_handle.truncate()
        file_handle.write(code)


def get_custom_rule(file, rule_list):
    key_set = ('ArgsModifyRule', 'FuncNameModifyRule', 'ModuleNameModifyRule')
    rules = get_file_content_bytes(file)
    rule_dict = json.loads(rules).get('rules', {})
    custom_rule_list = []

    for key in rule_dict:
        if key not in key_set:
            raise TransplantException('%s is not supported customization!' % key)
        init_rule_to_list(key, rule_dict, custom_rule_list, ['normal'])
    custom_rule_list.extend(rule_list)
    return custom_rule_list


def get_builtin_rule(feature_switch, args):
    rule_list = get_special_rule(args)
    # rules for different version
    if args.version == '1.8.1':
        rules_json_file_1_8_0 = os.path.join(os.path.dirname(__file__), '../pytorch_v1_8_1/builtin_rules_1_8_1.json')
        get_rule_from_json_file(feature_switch, rule_list, rules_json_file_1_8_0)
    # common rules
    common_rules_json_file = os.path.join(os.path.dirname(__file__), '../common_rules/builtin_rules.json')
    get_rule_from_json_file(feature_switch, rule_list, common_rules_json_file)

    return rule_list


def get_rule_from_json_file(feature_switch, rule_list, json_file):
    if not os.path.exists(json_file):
        return
    json_file_content = get_file_content(json_file)
    rule_dict = json.loads(json_file_content).get('rules', {})
    for key in rule_dict:
        init_rule_to_list(key, rule_dict, rule_list, feature_switch)


def init_rule_to_list(key, rule_dict, rule_list, feature_switch):
    tmp = []
    if not hasattr(rule_module, key):
        return
    for kwargs in rule_dict.get(key, []):
        if not set(kwargs.get('feature_switch', ['normal'])).intersection(set(feature_switch)):
            continue
        if kwargs.get('feature_switch', []):
            del kwargs['feature_switch']
        rule = getattr(rule_module, key)
        tmp.append(rule(**kwargs))

    rule_list.extend(tmp)


def get_special_rule(args):
    special_rule_list = [rule_module.PythonVersionConvertRule()]
    if args.amp_model:
        if hasattr(args, 'main'):
            special_rule_list.extend([InitApexRule(), Amp2Apex(args.amp_model, args.main)])
        else:
            special_rule_list.extend([InitApexRule(), Amp2Apex(args.amp_model, '')])
    if hasattr(args, 'main'):
        special_rule_list.extend([distributed_rule.DataLoaderRule(),
                                  distributed_rule.DistributedDataParallelRule(args.target_model, args.amp_model)])
    return special_rule_list


def change_mode(dir_path):
    if os.path.isfile(dir_path):
        if dir_path.endswith('.sh'):
            os.chmod(dir_path, 0o750)
        else:
            os.chmod(dir_path, 0o640)
        return
    os.chmod(dir_path, 0o750)
    for root, dirs, files in os.walk(dir_path):
        for dir_itr in dirs:
            dir_iter_path = os.path.join(root, dir_itr)
            if not os.path.islink(dir_iter_path):
                os.chmod(dir_iter_path, 0o750)
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.islink(file_path):
                continue
            if file.endswith('.sh'):
                os.chmod(os.path.join(root, file), 0o750)
                continue
            os.chmod(file_path, 0o640)


def generate_distributed_shell_file(path):
    code = '''export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29688
export HCCL_WHITELIST_DISABLE=1

NPUS=($(seq 0 7))
export NPU_WORLD_SIZE=${#NPUS[@]}
rank=0
for i in ${NPUS[@]}
do
    export NPU_CALCULATE_DEVICE=${i}
    export RANK=${rank}
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
            if os.path.islink(file_path) or (not os.path.exists(file_path)):
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
        if os.path.islink(path) or os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)
    except PermissionError as exp:
        raise DeleteFileException(f'Failed to delete {path}: {exp}')


def check_path_owner_consistent(path):
    if platform.system().lower() == 'windows':
        return True
    try:
        import pwd
    except ImportError:
        user_interactive_confirm(f'Failed to check owner consistency for path {path}. Do you want to continue?')
        return True

    file_owner = pwd.getpwuid(os.stat(path).st_uid).pw_name
    return file_owner == pwd.getpwuid(os.getuid()).pw_name


def check_path_length_valid(path):
    path = os.path.realpath(path)
    if platform.system().lower() == 'windows':
        return len(path) <= WINDOWS_PATH_LENGTH_LIMIT
    else:
        return len(os.path.basename(path)) <= LINUX_FILE_NAME_LENGTH_LIMIT


def check_file_need_analysis(file, commonprefix, record=False):
    if not os.path.exists(file):
        return False
    if not file.endswith('.py'):
        return False
    file_relative_path = os.path.relpath(file, commonprefix)
    if os.path.islink(file):
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
    return {'line':line, 'column': column}


def check_model_name_valid(name):
    if not re.match("^([a-zA-Z_]\\w*\\.)*([a-zA-Z_]\\w*)$", name):
        raise ValueError('Target model variable name is not valid!')
