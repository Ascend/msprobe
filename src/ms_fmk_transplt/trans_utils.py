#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.

import json
import os
import pandas as pd
import rule as rule_module


class TransplantException(Exception):
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
    if not os.path.exists(csv_file):
        data_frame = pd.DataFrame(columns=header)
        data_frame.to_csv(csv_file, index=False)
    rel_script_file_name = os.path.relpath(script_file, script_dir) if os.path.isdir(script_dir) else \
        os.path.basename(script_file)
    new_data = pd.DataFrame([[rel_script_file_name] + content for content in content_list])
    new_data.to_csv(csv_file, mode='a+', header=False, index=False)
    change_mode(csv_file)


def get_op_list():
    ops = get_file_content_bytes(os.path.join(os.path.dirname(__file__), 'op_list.json'))
    op_list = json.loads(ops).get('op_list')
    return op_list


def get_file_content_bytes(file):
    with open(file, 'rb') as file_handle:
        return file_handle.read()


def get_file_content(file):
    with open(file, 'r', encoding='utf8') as file_handle:
        return file_handle.read()


def write_file_content(file, code):
    with open(file, 'w', encoding='utf8', newline='') as file_handle:
        return file_handle.write(code)


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
    rules = get_file_content(os.path.join(os.path.dirname(__file__), 'builtin_rules.json'))
    rule_dict = json.loads(rules).get('rules', {})
    for key in rule_dict:
        init_rule_to_list(key, rule_dict, rule_list, feature_switch)

    return rule_list


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
            special_rule_list.extend([rule_module.InitApexRule(), rule_module.Amp2Apex(args.amp_model, args.main)])
        else:
            special_rule_list.extend([rule_module.InitApexRule(), rule_module.Amp2Apex(args.amp_model, '')])
    if hasattr(args, 'main'):
        special_rule_list.extend([rule_module.InitProcessGroupRule(),
                                  rule_module.DataLoaderRule(),
                                  rule_module.DistributedDataParallelRule(args.target_model, args.amp_model)])
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
            os.chmod(os.path.join(root, dir_itr), 0o750)
        for file in files:
            if file.endswith('.sh'):
                os.chmod(os.path.join(root, file), 0o750)
                continue
            os.chmod(os.path.join(root, file), 0o640)


def generate_distributed_shell_file(path):
    code = '''export MASTER_ADDR=localhost
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
    with open(os.path.join(path, 'run_distributed_npu.sh'), 'w', encoding='utf-8') as shell_file:
        shell_file.write(code)
    change_mode(os.path.join(path, 'run_distributed_npu.sh'))
