#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.

from typing import Optional, Union
from collections import namedtuple

import libcst
import libcst.matchers as m

from libcst.metadata import PositionProvider, QualifiedNameProvider
from global_analysis import GlobalReferenceVisitor
from analysis.unsupported_api_analysis.unsupported_api_visitor import UnsupportedApiVisitor

NodeInfo = namedtuple('NodeInfo', ['has_unsupported_api', 'unsupported_list', 'has_unknown_api', 'unknown_api_list',
                                   'file_path'])


class ThirdPartyApiVisitor(UnsupportedApiVisitor):
    METADATA_DEPENDENCIES = (PositionProvider, QualifiedNameProvider)

    def __init__(self, op_info, global_reference_visitor: GlobalReferenceVisitor, function_graph):
        super(ThirdPartyApiVisitor, self).__init__(op_info, global_reference_visitor)
        self.function_graph = function_graph

    @staticmethod
    def _update_node(node, node_info):
        node.has_unsupported_api = node_info.has_unsupported_api
        node.unsupported_list = node_info.unsupported_list
        node.has_unknown_api = node_info.has_unknown_api
        node.unknown_api_list = node_info.unknown_api_list
        node.file_path = node_info.file_path

    @staticmethod
    def _visit_function_body(node):
        function_body_list = []
        body = node.body.body
        for element in body:
            call_function = m.findall(element, m.Call())
            function_body_list.extend(call_function)
        return function_body_list

    def visit_Call(self, node: "libcst.Call") -> Optional[bool]:
        return True

    def visit_FunctionDef(self, node: libcst.FunctionDef) -> Optional[bool]:
        function_line, function_column = self._get_func_def_position(node)
        full_name, file_path = self.global_reference_visitor.get_full_name_for_function(function_line, function_column)
        self.function_graph.addnode(full_name)
        function_body_list = self._visit_function_body(node)
        defined_call_list, unsupported_list, unknown_api_list = self._visit_call_body(function_body_list, file_path)
        self._update_node(self.function_graph.getnode(full_name),
                          NodeInfo(bool(unsupported_list), unsupported_list, bool(unknown_api_list), unknown_api_list,
                                   file_path))
        for call_function in defined_call_list:
            # Avoid recursion
            if call_function != full_name:
                self.function_graph.addedge(call_function, full_name)
                self.function_graph.getnode(full_name).in_degree += 1

    def _visit_call_body(self, node_list, file_path):
        defined_call_set = set()
        unsupported_list = []
        unknown_api_list = []
        for call_node in node_list:
            position = self._get_call_position(call_node)
            infer_func_list, infer_func_not_in_project_list = \
                self.global_reference_visitor.get_infer_func_list_in_project(position.start.line, position.start.column)
            if infer_func_list:  # if find infer function in project, use infer function
                defined_call_set.update(infer_func_list)
                continue
            if infer_func_not_in_project_list:
                continue
            libcst_full_name = self.get_full_name_for_node(call_node)
            if not libcst_full_name:
                continue
            unsupported_apis, unknown_apis = self.get_api_instances(call_node, libcst_full_name, position, file_path)
            unsupported_list.extend(unsupported_apis)
            unknown_api_list.extend(unknown_apis)
        return defined_call_set, unsupported_list, unknown_api_list

    def _get_func_def_position(self, node):
        node_start_line = self.get_metadata(PositionProvider, node).start.line
        if node.asynchronous is not None:
            node_start_column = self.get_metadata(PositionProvider, node).start.column + 10
        else:
            node_start_column = self.get_metadata(PositionProvider, node).start.column + 4
        return node_start_line, node_start_column
