#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.

import os
from typing import Optional
from typing import Union

import libcst
from libcst._flatten_sentinel import FlattenSentinel
from libcst._removal_sentinel import RemovalSentinel

import ms_fmk_transplt.utils.trans_utils as utils
import ms_fmk_transplt.utils.transplant_logger as translog
from ms_fmk_transplt.common_rules.code_visitor import ApiVisitor
from ms_fmk_transplt.common_rules import InsertMainFileRule
from ms_fmk_transplt.utils.trans_utils import TransplantException


class Transplant(object):
    def __init__(self, script_dir, rule_list, args):
        self.script_dir = script_dir
        self.rule_list = rule_list
        self.main_file = utils.get_main_file(args)
        self.args = args
        self.py_file_counts = 0

    @staticmethod
    def __need_analysis(file, commonprefix):
        return utils.check_file_need_analysis(file, commonprefix, record=True)

    def __analysis_code(self, file):
        code = utils.get_file_content_bytes(file)
        wrapper = libcst.metadata.MetadataWrapper(libcst.parse_module(code))

        api_visitor = ApiVisitor(utils.get_op_list(self.args.version))
        module = wrapper.visit(api_visitor)
        op_list = api_visitor.print_unsupported_ops()
        utils.write_csv(op_list, file, self.script_dir, "unsupported_op")

        new_module = self.__visit_rule(file, module)
        utils.write_file_content(file, new_module.code)

    def run(self):
        translog.info('Analysis start...')

        if not os.access(self.script_dir, os.R_OK):
            raise TransplantException('%s is not readable.' % self.script_dir)

        if os.path.isfile(self.script_dir) and self.__need_analysis(self.script_dir, os.path.dirname(self.script_dir)):
            self.__analysis_file(self.script_dir, os.path.dirname(self.script_dir))

        if os.path.isdir(self.script_dir):
            self.__analysis_dir()

    def set_py_file_counts(self, py_file_counts):
        self.py_file_counts = py_file_counts

    def __analysis_dir(self):
        count = 0
        translog.set_progress_info(f'[Progress:{count / self.py_file_counts * 100:6.2f}%]')
        for root, dirs, files in os.walk(self.script_dir):
            for f in files:
                file = os.path.join(root, f)
                if not self.__need_analysis(file, self.script_dir):
                    continue
                self.__analysis_file(file, self.script_dir)
                count += 1
                translog.set_progress_info(f'[Progress:{count / self.py_file_counts * 100:6.2f}%]')

    def __analysis_file(self, file, commonprefix):
        file_relative_path = os.path.relpath(file, commonprefix)
        translog.info(f'Start analysis {file_relative_path}.')
        self.__analysis_code(file)
        translog.info(f'Analysis {file_relative_path} complete.')

    def __visit_rule(self, file, module):
        current_file_name = os.path.basename(file) if os.path.isfile(self.script_dir) else \
            os.path.relpath(file, self.script_dir)
        code_transformer = CodeTransformer(self.rule_list,
                                           current_file_name == self.main_file if self.main_file else False)
        wrapper = libcst.metadata.MetadataWrapper(module)
        new_module = wrapper.visit(code_transformer)
        change_info_list = code_transformer.print_change_info()
        utils.write_csv(change_info_list, file, self.script_dir, "change_list")
        for rule in self.rule_list:
            rule.clean()
        return new_module


class CodeTransformer(libcst.CSTTransformer):
    METADATA_DEPENDENCIES = (libcst.metadata.PositionProvider, libcst.metadata.ScopeProvider,
                             libcst.metadata.QualifiedNameProvider, libcst.metadata.ParentNodeProvider)

    def __init__(self, rule_list, is_main_file):
        super().__init__()
        self.rule_list = rule_list
        for rule in self.rule_list:
            if isinstance(rule, InsertMainFileRule):
                rule.visit_main_file(is_main_file)
            rule.set_warp_visitor(self)

    def visit_Module(self, node: "libcst.Module") -> Optional[bool]:
        for rule in self.rule_list:
            rule.visit_Module(node)
        return True

    def visit_Assign(self, node: "libcst.Assign") -> Optional[bool]:
        for rule in self.rule_list:
            rule.visit_Assign(node)
        return True

    def visit_ImportAlias(self, node: "libcst.ImportAlias") -> Optional[bool]:
        for rule in self.rule_list:
            rule.visit_ImportAlias(node)
        return True

    def visit_ImportFrom(self, node: "libcst.ImportFrom") -> Optional[bool]:
        for rule in self.rule_list:
            rule.visit_ImportFrom(node)
        return True

    def visit_If(self, node: "libcst.If") -> Optional[bool]:
        for rule in self.rule_list:
            rule.visit_If(node)
        return True

    def leave_For(
            self, original_node: "libcst.For", updated_node: "libcst.For"
    ) -> Union["libcst.BaseStatement", libcst.RemovalSentinel]:
        for rule in self.rule_list:
            updated_node = rule.leave_For(original_node, updated_node)
        return updated_node

    def leave_Module(self, original_node: "libcst.Module", updated_node: "libcst.Module") -> "libcst.Module":
        for rule in self.rule_list:
            updated_node = rule.leave_Module(original_node, updated_node)
        return updated_node

    def leave_Name(
            self, original_node: "libcst.Name", updated_node: "libcst.Name"
    ) -> "libcst.Name":
        for rule in self.rule_list:
            updated_node = rule.leave_Name(original_node, updated_node)
        return updated_node

    def leave_SimpleString(
            self, original_node: "libcst.SimpleString", updated_node: "libcst.SimpleString"
    ) -> "libcst.BaseExpression":
        for rule in self.rule_list:
            updated_node = rule.leave_SimpleString(original_node, updated_node)
        return updated_node

    def leave_FormattedStringText(
            self, original_node: "libcst.FormattedStringText", updated_node: "libcst.FormattedStringText"
    ) -> "libcst.BaseExpression":
        for rule in self.rule_list:
            updated_node = rule.leave_FormattedStringText(original_node, updated_node)
        return updated_node

    def visit_Call(self, node: "libcst.Call") -> Optional[bool]:
        for rule in self.rule_list:
            rule.visit_Call(node)
        return True

    def leave_Call(
            self, original_node: "libcst.Call", updated_node: "libcst.Call"
    ) -> "libcst.BaseExpression":
        for rule in self.rule_list:
            updated_node = rule.leave_Call(original_node, updated_node)
        return updated_node

    def print_change_info(self):
        change_info_list = []
        for rule in self.rule_list:
            change_info_list.extend(rule.print_change_info())
        return change_info_list

    def leave_Attribute(self, original_node: "libcst.Attribute", updated_node: "libcst.Attribute") \
            -> "libcst.Attribute":
        for rule in self.rule_list:
            updated_node = rule.leave_Attribute(original_node, updated_node)
        return updated_node

    def leave_Assign(
            self, original_node: "libcst.Assign", updated_node: "libcst.Assign"
    ) -> Union[
        "libcst.BaseSmallStatement", FlattenSentinel["libcst.BaseSmallStatement"], RemovalSentinel
    ]:
        for rule in self.rule_list:
            updated_node = rule.leave_Assign(original_node, updated_node)
        return updated_node

    def leave_SimpleStatementLine(
            self, original_node: "libcst.SimpleStatementLine", updated_node: "libcst.SimpleStatementLine"
    ) -> Union["libcst.BaseStatement", FlattenSentinel["libcst.BaseStatement"], RemovalSentinel]:
        for rule in self.rule_list:
            updated_node = rule.leave_SimpleStatementLine(original_node, updated_node)
        return updated_node

    def leave_IfExp(
            self, original_node: "libcst.IfExp", updated_node: "libcst.IfExp"
    ) -> "libcst.BaseExpression":
        for rule in self.rule_list:
            updated_node = rule.leave_IfExp(original_node, updated_node)
        return updated_node

    def leave_With(self, original_node: "libcst.With", updated_node: "libcst.With") \
            -> Union["libcst.BaseStatement", FlattenSentinel["libcst.BaseStatement"], RemovalSentinel]:
        for rule in self.rule_list:
            updated_node = rule.leave_With(original_node, updated_node)
        return updated_node

    def leave_If(
        self, original_node: "libcst.If", updated_node: "libcst.If"
    ) -> Union["libcst.BaseStatement", FlattenSentinel["libcst.BaseStatement"], RemovalSentinel]:
        for rule in self.rule_list:
            updated_node = rule.leave_If(original_node, updated_node)
        return updated_node
