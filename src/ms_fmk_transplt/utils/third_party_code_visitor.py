#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.

from typing import Optional, Union
from collections import namedtuple

import libcst
import libcst.helpers as helper
import libcst.matchers as m

from libcst.metadata import PositionProvider, QualifiedNameProvider
from analysis.third_party.function_node import ApiInstance

NodeInfo = namedtuple('NodeInfo', ['has_unsupported_api', 'unsupported_list', 'has_unknown_api', 'unknown_api_list',
                                   'file_path'])

try:
    import jedi
except ImportError:
    jedi = None


class ApiVisitor(libcst.CSTVisitor):
    METADATA_DEPENDENCIES = (PositionProvider, QualifiedNameProvider)

    def __init__(self, op_list, unsupported_op_list, global_reference_visitor, function_graph):
        super(ApiVisitor, self).__init__()
        self.op_list = op_list
        self.unsupported_op_list = unsupported_op_list
        self.global_reference_visitor = global_reference_visitor
        self.function_graph = function_graph

    def visit_FunctionDef(self, node: "FunctionDef") -> Optional[bool]:
        function_line, function_column = self._get_func_def_position(node)
        full_name, file_path = self.global_reference_visitor.get_full_name_for_function(function_line, function_column)
        self.function_graph.addnode(full_name)
        function_body_list = self._visit_function_body(node)
        defined_call_list, has_unsupported_api, unsupported_list, has_unknown_api, unknown_api_list = \
            self._visit_call_body(function_body_list, file_path)
        self._update_node(self.function_graph.getnode(full_name),
                          NodeInfo(has_unsupported_api, unsupported_list, has_unknown_api, unknown_api_list, file_path))
        for call_function in defined_call_list:
            # Avoid recursion
            if call_function != full_name:
                self.function_graph.addedge(call_function, full_name)
                self.function_graph.getnode(full_name).in_degree += 1

    @staticmethod
    def _update_node(node, nodeinfo):
        node.has_unsupported_api = nodeinfo.has_unsupported_api
        node.unsupported_list = nodeinfo.unsupported_list
        node.has_unknown_api = nodeinfo.has_unknown_api
        node.unknown_api_list = nodeinfo.unknown_api_list
        node.file_path = nodeinfo.file_path

    @staticmethod
    def _visit_function_body(node):
        function_body_list = []
        body = node.body.body
        for idx, element in enumerate(body):
            call_function = m.findall(element, m.Call())
            function_body_list.extend(call_function)
        return function_body_list

    def _visit_call_body(self, node_list, file_path):
        defined_call_list = set()
        unsupported_list = []
        unknown_api_list = []
        has_unsupported_api = False
        has_unknown_api = False
        for call_node in node_list:
            position = self._get_call_position(call_node)
            is_defined, full_name = self.global_reference_visitor.is_belong_with_self_project(
                position.start.line, position.start.column)
            libcst_full_name = self.get_full_name_for_node(call_node)
            if m.matches(call_node.func, m.Attribute(attr=m.Name(), value=m.Call())) and \
                    libcst_full_name.startswith('torch.'):
                libcst_full_name = '.'.join([libcst_full_name.split('.')[0], 'Tensor', libcst_full_name.split('.')[-1]])
            if is_defined:
                defined_call_list.add(full_name)
            elif libcst_full_name and libcst_full_name.startswith('torch.') and \
                    libcst_full_name in self.unsupported_op_list:
                has_unsupported_api = True
                unsupported_list.append(ApiInstance(libcst_full_name, position, file_path))
            elif libcst_full_name and libcst_full_name.startswith('torch.') and libcst_full_name not in self.op_list:
                has_unknown_api = True
                unknown_api_list.append(ApiInstance(libcst_full_name, position, file_path))
        return defined_call_list, has_unsupported_api, unsupported_list, has_unknown_api, unknown_api_list

    def _get_func_def_position(self, node):
        node_start_line = self.get_metadata(PositionProvider, node).start.line
        if node.asynchronous is not None:
            node_start_column = self.get_metadata(PositionProvider, node).start.column + 10
        else:
            node_start_column = self.get_metadata(PositionProvider, node).start.column + 4
        return node_start_line, node_start_column

    def _get_call_position(self, node):
        if m.matches(node.func, m.Attribute()):
            node = node.func.attr
        position = self.get_metadata(PositionProvider, node)
        return position

    def get_full_name_for_node(self, node: Union[str, libcst.CSTNode]) -> Optional[str]:
        name_list = list(self.get_metadata(libcst.metadata.QualifiedNameProvider, node))
        if name_list:
            qualified_name = list(self.get_metadata(libcst.metadata.QualifiedNameProvider, node))[0].name
        else:
            qualified_name = helper.get_full_name_for_node(node)
        return qualified_name
