#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.

from enum import Enum
from enum import auto
from typing import Optional, Union, Type, TypeVar

import libcst
import libcst.helpers as helper
import libcst.matchers as m

import utils.transplant_logger as translog

_T = TypeVar("_T")
_UNDEFINED_DEFAULT = object()


class OperatorType(Enum):
    INSERT = auto()
    MODIFY = auto()
    DELETE = auto()
    UNSUPPORTED = auto()


class ApiVisitor(libcst.CSTVisitor):
    METADATA_DEPENDENCIES = (libcst.metadata.PositionProvider, libcst.metadata.QualifiedNameProvider)

    def __init__(self, op_list):
        super(ApiVisitor, self).__init__()
        self.op_list = op_list
        self.unsupported_op_list = []

    def visit_Call(self, node: "libcst.Call") -> Optional[bool]:
        full_name = self.get_full_name_for_node(node)
        if full_name in self.op_list:
            position = self.get_metadata(libcst.metadata.PositionProvider, node)
            self.unsupported_op_list.append([position.start.line, position.end.line, full_name,
                                             self.op_list.get(full_name)])
        return True

    def print_unsupported_ops(self):
        for unsupported_op in self.unsupported_op_list:
            if unsupported_op[3]:
                unsupported_op_info = "Message: %s" % unsupported_op[3]
            else:
                unsupported_op_info = "Message: %s is not supported now!" % unsupported_op[2]
            msg = "%-21s %-35s %s" % ("line: %s ~ %s" % (unsupported_op[0], unsupported_op[1]),
                                      "Operation Type: UNSUPPORTED", unsupported_op_info)
            translog.warning(msg)
        return self.unsupported_op_list

    def get_full_name_for_node(self, node: Union[str, libcst.CSTNode]) -> Optional[str]:
        name_list = list(self.get_metadata(libcst.metadata.QualifiedNameProvider, node))
        if name_list:
            qualified_name = list(self.get_metadata(libcst.metadata.QualifiedNameProvider, node))[0].name
        else:
            qualified_name = helper.get_full_name_for_node(node)
        return qualified_name


class RuleVisitor(libcst.CSTTransformer):
    METADATA_DEPENDENCIES = (libcst.metadata.PositionProvider, libcst.metadata.ScopeProvider,
                             libcst.metadata.QualifiedNameProvider, libcst.metadata.ParentNodeProvider)

    def __init__(self):
        super(RuleVisitor, self).__init__()
        self.warp_visitor = None
        self.changes_info = []
        self.variable_dict = {}

    @staticmethod
    def get_code_for_node(node: libcst.CSTNode) -> str:
        return libcst.Module('').code_for_node(node)

    def set_warp_visitor(self, warp_visitor):
        self.warp_visitor = warp_visitor

    def visit_Assign(self, node: "libcst.Assign") -> Optional[bool]:
        for target in node.targets:
            self.manage_variable_definition(target, node)
        return True

    def manage_variable_definition(self, target, node):
        if isinstance(target.target, libcst.Tuple) and isinstance(node.value, libcst.Tuple):
            for tar, val in zip(target.target.elements, node.value.elements):
                # escape "output = model(input)[-1]","output = model[-1]"
                if m.findall(node, m.Call() | m.Subscript()):
                    continue
                assign_key = self.get_full_name_for_node(tar.value)
                assign_value = self.get_full_name_for_node(val.value)
                # escape "xxx=None","xxx=False"
                if assign_key and assign_value and not assign_value.startswith('builtins.'):
                    self.variable_dict[assign_key] = assign_value
        else:
            if m.findall(node, m.Call() | m.Subscript()):
                return
            assign_key = self.get_full_name_for_node(target.target)
            assign_value = self.get_full_name_for_node(node.value)
            if assign_key and assign_value and not assign_value.startswith('builtins.'):
                self.variable_dict[assign_key] = assign_value

    def print_change_info(self):
        for change_info in self.changes_info:
            msg = "%-21s %-35s %s" % ("line: %s ~ %s" % (change_info[0], change_info[1]),
                                      "Operation Type: %s" % change_info[2], "Message: %s" % change_info[3])
            translog.info(msg)
        return self.changes_info

    def get_full_name_for_node(self, node: Union[str, libcst.CSTNode], with_variable_replace=True) -> Optional[str]:
        name_list = list(self.get_metadata(libcst.metadata.QualifiedNameProvider, node))
        if name_list:
            qualified_name = list(self.get_metadata(libcst.metadata.QualifiedNameProvider, node))[0].name
        else:
            qualified_name = helper.get_full_name_for_node(node)

        if qualified_name:
            split_name = qualified_name.split('>.')[-1].split('.')
            if with_variable_replace:
                split_name[0] = self.variable_dict.get(split_name[0], split_name[0])
            qualified_name = '.'.join(split_name)

        return qualified_name

    def clean(self):
        self.changes_info = []
        self.variable_dict = {}

    def get_metadata(
        self,
        key: Type["libcst.BaseMetadataProvider[_T]"],
        node: "libcst.CSTNode",
        default: _T = _UNDEFINED_DEFAULT,
    ) -> _T:
        if self.warp_visitor:
            return self.warp_visitor.get_metadata(key, node, default)
        return super(RuleVisitor, self).get_metadata(key, node, default)

    def _record_position(self, original_node: "libcst.CSTNode", opt: "OperatorType", desc: "str"):
        original_position = self.get_metadata(libcst.metadata.PositionProvider, original_node)
        self.changes_info.append([original_position.start.line, original_position.end.line, opt.name, desc])
