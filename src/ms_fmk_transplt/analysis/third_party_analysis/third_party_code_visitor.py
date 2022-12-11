#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.

from typing import Optional, Union
from collections import namedtuple
import re

import libcst
import libcst.helpers as helper
import libcst.matchers as m

from libcst.metadata import PositionProvider, QualifiedNameProvider
from .function_node import ApiInstance
from ...global_analysis import GlobalReferenceVisitor

NodeInfo = namedtuple('NodeInfo', ['has_unsupported_api', 'unsupported_list', 'has_unknown_api', 'unknown_api_list',
                                   'file_path'])


class ThirdPartyApiVisitor(libcst.CSTVisitor):
    METADATA_DEPENDENCIES = (PositionProvider, QualifiedNameProvider)

    def __init__(self, op_list, unsupported_op_list, cuda_op_list,
                 global_reference_visitor: GlobalReferenceVisitor, function_graph):
        super(ThirdPartyApiVisitor, self).__init__()
        self.op_list = op_list
        self.unsupported_op_list = unsupported_op_list
        self.cuda_op_list = cuda_op_list
        self.unsupported_instance_op_dict = {}
        for unsupported_op in unsupported_op_list:
            class_name = unsupported_op.split(".")[-2]
            if class_name.lower() == class_name:
                continue
            op_name = unsupported_op.split(".")[-1]
            if op_name not in self.unsupported_instance_op_dict:
                self.unsupported_instance_op_dict[op_name] = []
            self.unsupported_instance_op_dict[op_name].append(unsupported_op)

        self.global_reference_visitor = global_reference_visitor
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

    @staticmethod
    def _get_call_obj_name(full_name):
        if full_name.startswith(("torch.nn.functional", "torch.nn.init")):
            return "torch.Tensor"
        elif full_name.startswith("torch.") and full_name.lower() == full_name:
            return "torch.Tensor"
        elif full_name.startswith("torch."):
            return full_name
        elif full_name.startswith("numpy"):
            return "numpy"
        return ""

    @staticmethod
    def _get_module_column_and_name(define_node, start_index, end_index=None):
        assign_value_stmt = define_node.description[start_index:end_index]
        assign_by_self_obj_column_range = re.search(f"[^\\w]{define_node.name}[^\\w]", assign_value_stmt)
        if assign_by_self_obj_column_range:
            column_range = assign_by_self_obj_column_range.span()
            column_range = (column_range[0] + 1, column_range[1] - 1)
        else:
            column_range = re.search("[\\w\\.]+", assign_value_stmt)
            if column_range is None:
                return -1, ""
            column_range = column_range.span()
        name_index_range = re.search(f"^{define_node.name}[^\\w]", define_node.description)
        if not name_index_range:
            name_index_range = re.search(f"[^\\w]{define_node.name}[^\\w]", define_node.description)
            name_index_start = name_index_range.span()[0] + 1
        else:
            name_index_start = name_index_range.span()[0]
        module_column = start_index + column_range[0]
        object_column = start_index + column_range[1]
        assign_module_column = define_node.column - name_index_start + module_column
        return assign_module_column, define_node.description[module_column:object_column]

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
            infer_func_list = self.global_reference_visitor.get_infer_func_list_in_project(
                position.start.line, position.start.column)
            if infer_func_list:  # if find infer function in project, use infer function
                defined_call_set.update(infer_func_list)
                continue
            libcst_full_name = self._get_full_name_for_node(call_node)
            if not libcst_full_name:
                continue
            if self._match_cuda_op(call_node, libcst_full_name):
                unsupported_list.append(ApiInstance(libcst_full_name, position, file_path))
            elif libcst_full_name.startswith('torch.') and not m.findall(call_node.func, m.Call()):
                if libcst_full_name in self.unsupported_op_list:
                    unsupported_list.append(ApiInstance(libcst_full_name, position, file_path))
                elif libcst_full_name not in self.op_list:
                    unknown_api_list.append(ApiInstance(libcst_full_name, position, file_path))
            else:  # handle instance api
                _unsupported_list, _unknown_list = \
                    self._handle_torch_instance_func(libcst_full_name, call_node, file_path)
                unsupported_list.extend(_unsupported_list)
                unknown_api_list.extend(_unknown_list)
        return defined_call_set, unsupported_list, unknown_api_list

    def _match_cuda_op(self, call_node, full_name):
        for cuda_op in self.cuda_op_list:
            if '.' in cuda_op.func_name:
                if not (full_name == cuda_op.func_name or full_name.endswith('.' + cuda_op.func_name)):
                    continue
            else:
                if not full_name.endswith('.' + cuda_op.func_name):
                    continue
            if cuda_op.max_args_num == -1:
                return True
            if cuda_op.min_args_num <= len(call_node.args) <= cuda_op.max_args_num:
                return True
        return False

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

    def _handle_torch_instance_func(self, full_name, call_node, file_path):
        if "." not in full_name or full_name.startswith("builtins.") or \
                not isinstance(call_node.func, libcst.Attribute):
            return [], []
        func_name = full_name.split(".")[-1]
        if func_name not in self.unsupported_instance_op_dict:
            return [], []
        call_obj_name_set = None
        position = self.get_metadata(PositionProvider, call_node)
        module_defined_list = self.global_reference_visitor.goto(position.start.line, position.start.column)
        for defined_node in module_defined_list:
            if defined_node.type == 'module':
                full_call_obj_name = (defined_node.full_name if defined_node.full_name else defined_node.name) + \
                                     full_name[full_name.index("."):full_name.rfind(".")]
                call_obj_name = self._get_call_obj_name(full_call_obj_name)
                call_obj_name_set = {call_obj_name} if call_obj_name else {}
                break
        call_position = self._get_call_position(call_node)
        if call_obj_name_set:
            return self._get_instance_func_list(call_position, file_path, full_name, call_obj_name_set)
        call_obj_position = self.get_metadata(PositionProvider, call_node.func.value)
        call_obj_define_list = self.global_reference_visitor.goto(call_obj_position.end.line,
                                                                  call_obj_position.end.column - 1)
        if call_obj_define_list and \
                all(call_obj_define.full_name and call_obj_define.full_name.startswith("builtins.")
                    for call_obj_define in call_obj_define_list):
            return [], []
        call_obj_name_set = self._get_call_obj_name_set_by_define_nodes(call_obj_define_list)
        return self._get_instance_func_list(call_position, file_path, full_name, call_obj_name_set)

    def _get_instance_func_list(self, call_position, file_path, full_name, call_obj_name_set):
        unsupported_list = []
        unknown_list = []
        func_name = full_name.split(".")[-1]
        if call_obj_name_set:
            unsupported_instance_func_list = self._get_unsupported_instance_func_list(func_name, call_obj_name_set)
            unsupported_list.extend(ApiInstance(instance_func_name, call_position, file_path)
                                    for instance_func_name in unsupported_instance_func_list)
        elif func_name not in ("get", "set", "add"):
            possible_func_names = ', '.join(instance_func_name
                                            for instance_func_name in self.unsupported_instance_op_dict.get(func_name))
            print_func_name = f"{full_name} ({possible_func_names})"
            unknown_list.append(ApiInstance(print_func_name, call_position, file_path))
        return unsupported_list, unknown_list

    def _get_unsupported_instance_func_list(self, func_name, call_obj_name_set):
        unsupported_set = set()
        for call_obj_name in call_obj_name_set:
            self._add_adapt_func_to_set(func_name, call_obj_name, unsupported_set)
        return unsupported_set

    def _add_adapt_func_to_set(self, func_name, call_obj_name, unsupported_set):
        has_adapt_func = False
        while not has_adapt_func:
            for instance_func_name in self.unsupported_instance_op_dict.get(func_name):
                if instance_func_name.startswith(call_obj_name) and instance_func_name.endswith(func_name):
                    has_adapt_func = True
                    unsupported_set.add(instance_func_name)
            last_seg_index = call_obj_name.rfind(".")
            if last_seg_index == -1:
                break
            call_obj_name = call_obj_name[:last_seg_index]

    def _get_call_obj_name_set_by_define_nodes(self, define_nodes):
        queue = []
        queue.extend(define_nodes)
        call_obj_name_set = set()
        while queue:
            define_node = queue.pop(0)
            if "\\" in define_node.description or len(define_node.description) > 1000:
                continue
            if define_node.type == 'statement':
                self._handle_define_type_statement(define_node, queue, call_obj_name_set)
            elif define_node.type == 'param':
                self._handle_define_type_param(define_node, call_obj_name_set)
            elif define_node.type == 'class':
                self._handle_define_type_class(define_node, call_obj_name_set)
            elif define_node.type == 'property':
                self._handle_define_type_property(define_node, call_obj_name_set)
        return call_obj_name_set

    def _handle_define_type_param(self, define_node, call_obj_name_set):
        if ":" not in define_node.description:
            func_context = self.global_reference_visitor.get_context(define_node.line)
            if not func_context or not func_context.full_name:
                return
            function_full_name = func_context.full_name
            if function_full_name.split(".")[-1] != "forward":
                return
            class_name_end_index = function_full_name.rfind(".")
            if class_name_end_index == -1:
                return
            class_full_name = function_full_name[:class_name_end_index]
            class_nodes = self.global_reference_visitor.search_in_project(class_full_name)
            for class_node in class_nodes:
                if "torch.nn.Module" in self.global_reference_visitor.get_super_class(
                        class_node.name, str(class_node.module_path)):
                    call_obj_name_set.add("torch.Tensor")
            return
        start_index = define_node.description.index(":")
        self._analyse_type_declaration(define_node, call_obj_name_set, start_index)

    def _handle_define_type_statement(self, define_node, queue, call_obj_name_set):
        if define_node.description.startswith(("for ", "with ")) or "=" not in define_node.description:
            return
        module_column, name = self._get_module_column_and_name(define_node, define_node.description.index("="))
        if module_column == -1:
            return
        try:
            next_define_nodes = self.global_reference_visitor.goto(define_node.line, module_column)
        except ValueError:
            return
        for node in next_define_nodes:
            if node.type == 'module':
                full_name = (node.full_name if node.full_name else node.name) + \
                            (name[name.index("."):] if "." in name else "")
                call_obj_name = self._get_call_obj_name(full_name)
                if call_obj_name:
                    call_obj_name_set.add(call_obj_name)
            else:
                queue.append(node)

    def _handle_define_type_class(self, define_node, call_obj_name_set):
        super_class_list = self.global_reference_visitor.get_super_class(define_node.name, str(define_node.module_path))
        for super_class in super_class_list:
            if super_class.startswith("torch"):
                call_obj_name_set.add(super_class)

    def _handle_define_type_property(self, define_node, call_obj_name_set):
        if "->" not in define_node.description[:define_node.description.index(":")]:
            return
        start_index = define_node.description.index("->")
        end_index = start_index + define_node.description[start_index:].index(":")
        self._analyse_type_declaration(define_node, call_obj_name_set, start_index, end_index)

    def _get_multi_module_column_and_name(self, define_node, start_index, end_index=None):
        module_column = 0
        define_type_column_dict = dict()
        while module_column != -1:
            module_column, name = self._get_module_column_and_name(define_node, start_index, end_index)
            if module_column != -1:
                define_type_column_dict[name] = module_column
                start_index += define_node.description[start_index:].index(name) + len(name)
        return define_type_column_dict

    def _analyse_type_declaration(self, define_node, call_obj_name_set, start_index, end_index=None):
        define_type_column_dict = self._get_multi_module_column_and_name(define_node, start_index, end_index)
        for name, column in define_type_column_dict.items():
            try:
                next_define_nodes = self.global_reference_visitor.goto(define_node.line, column)
            except ValueError:
                continue
            for node in next_define_nodes:
                if node.type != 'module':
                    continue
                full_name = (node.full_name if node.full_name else node.name) + \
                            (name[name.index("."):] if "." in name else "")
                call_obj_name = self._get_call_obj_name(full_name)
                if call_obj_name:
                    call_obj_name_set.add(call_obj_name)

    def _get_full_name_for_node(self, node: Union[str, libcst.CSTNode]) -> Optional[str]:
        name_list = list(self.get_metadata(libcst.metadata.QualifiedNameProvider, node))
        if name_list:
            qualified_name = list(self.get_metadata(libcst.metadata.QualifiedNameProvider, node))[0].name
        else:
            qualified_name = helper.get_full_name_for_node(node)
        return qualified_name
