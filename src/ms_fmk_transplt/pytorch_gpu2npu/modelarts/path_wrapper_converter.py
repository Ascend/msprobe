#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

import libcst

from pytorch_gpu2npu.common_rules.code_visitor import OperatorType, RuleVisitor
from pytorch_gpu2npu.common_rules.common_rule import BaseInsertGlobalRule


class ModelArtsPathWrapperRule(RuleVisitor):
    def __init__(self, data_api_dict, add_import_rule: BaseInsertGlobalRule):
        super(ModelArtsPathWrapperRule, self).__init__()
        self.data_api_dict = data_api_dict
        self.add_import_rule = add_import_rule

    def leave_Call(
            self, original_node: "libcst.Call", updated_node: "libcst.Call"
    ) -> "libcst.BaseExpression":
        full_name = self.get_full_name_for_node(original_node)
        api_info = self.data_api_dict.get(full_name)
        if not api_info:
            return updated_node
        self.add_import_rule.insert_flag = True
        arg_no = api_info.get('arg_no')
        args = updated_node.args
        new_args = list(args)
        if arg_no < len(args):
            self._record_position(original_node, OperatorType.MODIFY,
                                  f'[ModelArts] Wrap argument {arg_no} of func {full_name} for path mapping.')
            new_arg = libcst.parse_expression('ModelArtsPathManager().get_path()') \
                .with_changes(args=[args[arg_no].with_changes(comma=libcst.MaybeSentinel.DEFAULT)])
            new_args[arg_no] = libcst.Arg(new_arg)
            updated_node = updated_node.with_changes(args=tuple(new_args))

        return updated_node

    def clean(self):
        super().clean()
        self.add_import_rule.clean()
