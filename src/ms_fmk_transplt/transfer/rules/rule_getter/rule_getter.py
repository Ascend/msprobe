#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
import json
import os

from utils.trans_utils import get_file_content_bytes, TransplantException, get_file_content
from .. import common_rules as rule_module
from ..common_rules.common_rule import RemoveImportRule
from ..distributed_rules import distributed_rule
from ..modelarts_rules import get_modelarts_rule
from ..pytorch_npu_patch_rules import InsertAheadRule


def get_builtin_rule(feature_switch, args):
    rule_list = get_special_rule(args)
    # rules for different version
    if args.modelarts:
        rule_list.extend(get_modelarts_rule())
    # use torch_npu.npu to replace torch.npu since 1.8.1
    rule_list.append(InsertAheadRule())
    rule_list.append(RemoveImportRule())
    rules_json_file_pytorch_npu = os.path.join(
        os.path.dirname(__file__), '../pytorch_npu_patch_rules/builtin_rules_pytorch_npu.json')
    get_rule_from_json_file(feature_switch, rule_list, rules_json_file_pytorch_npu)
    # common rules
    common_rules_json_file = os.path.join(os.path.dirname(__file__),
                                          '../common_rules/builtin_rules.json')
    get_rule_from_json_file(feature_switch, rule_list, common_rules_json_file)

    return rule_list


def get_rule_from_json_file(feature_switch, rule_list, json_file):
    if not os.path.exists(json_file):
        return
    json_file_content = get_file_content(json_file)
    try:
        json_file = json.loads(json_file_content)
    except ValueError:
        return
    if not isinstance(json_file, dict):
        return
    rule_dict = json_file.get('rules', {})
    if not isinstance(rule_dict, dict):
        return
    for key in rule_dict:
        init_rule_to_list(key, rule_dict, rule_list, feature_switch)


def init_rule_to_list(key, rule_dict, rule_list, feature_switch):
    tmp = []
    feature_key = 'feature_switch'
    if not hasattr(rule_module, key):
        return
    for kwargs in rule_dict.get(key, []):
        if not set(kwargs.get(feature_key, ['normal'])).intersection(set(feature_switch)):
            continue
        if kwargs.get(feature_key, []):
            del kwargs[feature_key]
        rule = getattr(rule_module, key)
        tmp.append(rule(**kwargs))

    rule_list.extend(tmp)


def get_special_rule(args):
    special_rule_list = [rule_module.PythonVersionConvertRule()]
    if hasattr(args, 'main'):
        special_rule_list.extend([distributed_rule.DataLoaderRule(),
                                  distributed_rule.DistributedDataParallelRule(args.target_model)])
    return special_rule_list
