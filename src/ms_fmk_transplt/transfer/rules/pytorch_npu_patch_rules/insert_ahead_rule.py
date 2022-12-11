#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

from typing import Optional, Union
import libcst
from libcst import FlattenSentinel, RemovalSentinel
from ..common_rules.base_rule import BaseRule, OperatorType


class InsertAheadRule(BaseRule):
    def __init__(self):
        super(InsertAheadRule, self).__init__()
        self.insert_flag = False
        self.already_insert = False

    def visit_ImportAlias(self, node: "libcst.ImportAlias") -> Optional[bool]:
        if not self.already_insert and not self.insert_flag and 'torch' in node.evaluated_name:
            self.insert_flag = True
        return True

    def visit_ImportFrom(self, node: "libcst.ImportFrom") -> Optional[bool]:
        not_insert = not self.already_insert and not self.insert_flag
        if not_insert and node.module and 'torch' in self.get_full_name_for_node(
                node.module):
            self.insert_flag = True
        return True

    def leave_SimpleStatementLine(
            self, original_node: "libcst.SimpleStatementLine", updated_node: "libcst.SimpleStatementLine"
    ) -> Union["libcst.BaseStatement", FlattenSentinel["libcst.BaseStatement"], RemovalSentinel]:
        if not self.insert_flag:
            return updated_node
        self.insert_flag = False
        self.already_insert = True
        code_need_to_add = 'import torch_npu'
        position = self.get_metadata(libcst.metadata.PositionProvider, original_node)
        self.changes_info.append(
            [position.start.line, position.start.line, OperatorType.INSERT.name, code_need_to_add])
        return FlattenSentinel([libcst.parse_statement(code_need_to_add), updated_node])

    def clean(self):
        super().clean()
        self.insert_flag = False
        self.already_insert = False
