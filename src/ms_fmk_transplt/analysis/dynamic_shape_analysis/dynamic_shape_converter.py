#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

import builtins
from typing import Optional, Union

import libcst as cst
import libcst.matchers as m
from libcst.metadata import PositionProvider, ParentNodeProvider, QualifiedNameProvider
from utils import transplant_logger as translog


class DynamicShapeTransformer(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (PositionProvider, QualifiedNameProvider, ParentNodeProvider)

    def __init__(self):
        super().__init__()
        self.line_number_func_dict = {}

    @staticmethod
    def verify_import_position(body_item):
        if m.matches(body_item, m.DoesNotMatch(m.SimpleStatementLine())):
            return True
        if m.matches(body_item.body[0], m.OneOf(m.Import(), m.ImportFrom(), m.ImportStar())):
            return False
        if m.matches(body_item.body[0], m.Expr()) and m.matches(body_item.body[0].value, m.SimpleString()):
            return False
        return True

    def leave_Call(
            self, original_node: "Call", updated_node: "Call"
    ) -> "BaseExpression":
        if not self._check_if_need_hook(original_node):
            return updated_node
        func_arg = cst.Arg(value=updated_node.func)
        args = []
        index = 0
        for arg in updated_node.args:
            if arg.keyword:
                index += 1
            if arg.keyword and arg.keyword.value == 'self':
                position = self.get_metadata(cst.metadata.PositionProvider, original_node)
                msg = 'In Python, `self` is usually used to represent an instance object of a class, ' \
                      'please do not use it as a keyword argument'
                translog.warning(
                    "%-21s %-35s %s" % ("line: %s ~ %s" % (position.start.line, position.end.line), ":", msg))
                if index == 1:
                    arg = arg.with_changes(keyword=None, equal=cst.MaybeSentinel.DEFAULT)
                else:
                    continue
            if isinstance(arg.value, cst.GeneratorExp):
                args.append(arg.deep_clone().with_changes(
                    value=cst.ListComp(elt=arg.value.elt, for_in=arg.value.for_in)))
            else:
                args.append(arg)
        string_quote = "'"
        node = original_node
        while not isinstance(node, cst.Module):
            node = self.get_metadata(ParentNodeProvider, node)
            if isinstance(node, cst.FormattedString):
                # Avoid conflicts between func_name and the quotation marks of the formatted string
                string_quote = "'" if node.end == '"' else '"'
                break
        func_name = self._get_func_name(original_node)
        position = self.get_metadata(PositionProvider, original_node)
        line_key = (position.start.line, position.end.line, func_name)
        if line_key not in self.line_number_func_dict:
            self.line_number_func_dict[line_key] = -1
        self.line_number_func_dict[line_key] += 1
        return updated_node.with_changes(func=cst.Attribute(value=cst.Name("DETECTOR"), attr=cst.Name("hook_func")),
                                         args=[func_arg,
                                               cst.Arg(value=cst.SimpleString(
                                                   value=string_quote + func_name + string_quote)),
                                               cst.Arg(value=cst.Integer(f"{self.line_number_func_dict[line_key]}")),
                                               *args])

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        self.line_number_func_dict = {}
        body = list(updated_node.body)
        for i, body_item in enumerate(body):
            if self.verify_import_position(body_item):
                new_body = body[:i] + [cst.parse_statement("from msft_dynamic_analysis.hook import DETECTOR")] + body[
                                                                                                                 i:]
                return updated_node.with_changes(body=tuple(new_body))
        return updated_node

    def on_visit(self, node: "CSTNode") -> bool:
        getattr(self, f'visit_{type(node).__name__}')(node)
        return True

    def get_full_name_for_node(self, node: Union[str, cst.CSTNode]) -> Optional[str]:
        name_list = list(self.get_metadata(QualifiedNameProvider, node))
        if name_list:
            qualified_name = list(self.get_metadata(QualifiedNameProvider, node))[0].name
        else:
            qualified_name = cst.helpers.get_full_name_for_node(node)
        return qualified_name

    def _check_if_need_hook(self, original_node):
        if isinstance(original_node.func, cst.Name) and original_node.func.value in dir(builtins):
            return False
        if m.matches(original_node, m.Call(func=m.Attribute(attr=m.Name('__init__')))):
            return False
        parent_node = self._get_parent_node(original_node, m.FunctionDef())
        if not isinstance(parent_node, cst.Module):
            if parent_node.decorators:
                return self.get_full_name_for_node(parent_node.decorators[0].decorator) not in \
                       ('torch.jit.script', 'torch.jit.script_method')
            if self.get_full_name_for_node(parent_node.name) == '__init__':
                return False
        parent_node = self._get_parent_node(parent_node, m.ClassDef())
        if not isinstance(parent_node, cst.Module) and parent_node.decorators:
            return self.get_full_name_for_node(parent_node.decorators[0].decorator) not in \
                   ('torch.jit.script', 'torch.jit.script_method')
        return True

    def _get_parent_node(self, node, condition):
        parent_node = node
        while not m.matches(parent_node, condition | m.Module()):
            parent_node = self.get_metadata(ParentNodeProvider, parent_node)
        return parent_node

    def _get_func_name(self, original_node):
        func_name = self.get_full_name_for_node(original_node)
        if func_name is None:
            return 'None'
        if func_name == 'builtins.None' and m.matches(original_node, m.Call(func=m.Attribute(attr=m.Name()))):
            return original_node.func.attr.value
        return func_name
