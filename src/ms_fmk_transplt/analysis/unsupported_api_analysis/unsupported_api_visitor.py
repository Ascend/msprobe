#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
import re
from collections import namedtuple
from typing import Optional, Union

import libcst
import libcst.helpers as helper
from libcst import matchers as m

from utils import transplant_logger as translog

OpInfo = namedtuple("OpInfo", ["supported_op_dict", "unsupported_op_dict", "cuda_op_list"])


class UnsupportedApiVisitor(libcst.CSTVisitor):
    METADATA_DEPENDENCIES = (libcst.metadata.PositionProvider, libcst.metadata.QualifiedNameProvider)

    def __init__(self, op_info, global_reference_visitor=None):
        super(UnsupportedApiVisitor, self).__init__()
        self.supported_op_dict = op_info.supported_op_dict
        self.unsupported_op_dict = op_info.unsupported_op_dict
        self.cuda_op_list = op_info.cuda_op_list
        self.unsupported_instance_op_dict = {}
        unsupported_op_module_set = set()
        for unsupported_op in op_info.unsupported_op_dict:
            if "." not in unsupported_op:
                continue
            unsupported_op_name_list = unsupported_op.split(".")
            class_name = unsupported_op_name_list[-2]
            if class_name.lower() == class_name:
                continue
            module_name = unsupported_op_name_list[0]
            unsupported_op_module_set.add(f"{module_name}.")
            op_name = unsupported_op_name_list[-1]
            if op_name not in self.unsupported_instance_op_dict:
                self.unsupported_instance_op_dict[op_name] = []
            self.unsupported_instance_op_dict[op_name].append(unsupported_op)
        self.unsupported_op_module_tuple = tuple(unsupported_op_module_set)
        all_module_name_set = set()
        for func_name in [*self.unsupported_op_dict.keys(), *self.supported_op_dict.keys(), *self.cuda_op_list]:
            if "." not in func_name:
                continue
            all_module_name_set.add(f'{func_name.split(".")[0]}.')
        self.all_module_names = tuple(all_module_name_set)
        self.global_reference_visitor = global_reference_visitor
        self.unsupported_op_result = []
        self.unknown_api_result = []

    @staticmethod
    def _get_call_obj_name(full_name):
        if full_name.startswith(("torch.nn.functional", "torch.nn.init")):
            return "torch.Tensor"
        elif full_name.startswith("torch.") and full_name.lower() == full_name:
            return "torch.Tensor"
        elif full_name.startswith("numpy"):
            return "numpy"
        else:
            return full_name

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

    def visit_Call(self, node: "libcst.Call") -> Optional[bool]:
        full_name = self.get_full_name_for_node(node)
        position = self.get_metadata(libcst.metadata.PositionProvider, node)
        if full_name:
            unsupported_apis, unknown_apis = self.get_api_instances(node, full_name, position, None)
            self.unsupported_op_result.extend(unsupported_apis)
            self.unknown_api_result.extend(unknown_apis)
        return True

    def visit_ClassDef(self, node) -> Optional[bool]:
        for base in node.bases:
            full_name = self.get_full_name_for_node(base.value)
            position = self.get_metadata(libcst.metadata.PositionProvider, node)
            if full_name:
                unsupported_apis, unknown_apis = self.get_class_api_instances(full_name, position, None)
                self.unsupported_op_result.extend(unsupported_apis)
                self.unknown_api_result.extend(unknown_apis)

    def visit_Decorator(self, node: "libcst.Decorator") -> Optional[bool]:
        full_name = self.get_full_name_for_node(node)
        position = self.get_metadata(libcst.metadata.PositionProvider, node)
        if full_name:
            unsupported_apis, unknown_apis = self.get_class_api_instances(full_name, position, None)
            self.unsupported_op_result.extend(unsupported_apis)
            self.unknown_api_result.extend(unknown_apis)
        return True


    def print_unsupported_ops(self):
        for unsupported_op in self.unsupported_op_result:
            if unsupported_op.info:
                unsupported_op_info = "Message: %s" % unsupported_op.info
            else:
                unsupported_op_info = "Message: %s is not supported now!" % unsupported_op.name
            msg = "%-21s %-35s %s" % ("line: %s ~ %s" % (unsupported_op.start_line, unsupported_op.end_line),
                                      "Operation Type: UNSUPPORTED", unsupported_op_info)
            translog.warning(msg)

    def get_full_name_for_node(self, node: Union[str, libcst.CSTNode]) -> Optional[str]:
        name_list = list(self.get_metadata(libcst.metadata.QualifiedNameProvider, node))
        if name_list:
            qualified_name = list(self.get_metadata(libcst.metadata.QualifiedNameProvider, node))[0].name
        else:
            qualified_name = helper.get_full_name_for_node(node)
        return qualified_name

    def get_api_instances(self, call_node, full_name, position, file_path):
        if full_name.endswith(".__init__"):
            full_name = full_name[:-1 * len(".__init__")]
        if self._match_cuda_op(call_node, full_name):
            return [ApiInstance(full_name, position, file_path, "User-defined CUDA Operator.")], []
        elif not m.findall(call_node.func, m.Call()) and self._is_class_api(call_node, full_name):
            return self.get_class_api_instances(full_name, position, file_path)
        else:  # handle instance api
            if not self.global_reference_visitor:
                return [], []
            return self._handle_instance_func(full_name, call_node, file_path)

    def get_class_api_instances(self, full_name, position, file_path):
        if full_name in self.unsupported_op_dict:
            return [ApiInstance(full_name, position, file_path, self.unsupported_op_dict.get(full_name))], []
        elif full_name.startswith('torch.') and full_name not in self.supported_op_dict:
            return [], [ApiInstance(full_name, position, file_path)]
        else:
            return [], []

    def _match_cuda_op(self, call_node, full_name):
        if "<locals>" in full_name:
            return False
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

    def _is_class_api(self, call_node, full_name):
        if self.global_reference_visitor:
            position = self.get_metadata(libcst.metadata.PositionProvider, call_node)
            infer_list = self.global_reference_visitor.infer(position.start.line, position.start.column)
            if not infer_list:
                return full_name.startswith(self.all_module_names)
            infer_name_type = self.global_reference_visitor.get_type(infer_list[0])
            if not infer_name_type:
                return full_name.startswith(self.all_module_names)
            return infer_name_type == 'module'
        return full_name.startswith(self.all_module_names)

    def _handle_instance_func(self, full_name, call_node, file_path):
        if "." not in full_name or full_name.startswith("builtins.") or \
                not isinstance(call_node.func, libcst.Attribute):
            return [], []
        func_name = full_name.split(".")[-1]
        if func_name in ("get", "set", "add") or func_name not in self.unsupported_instance_op_dict:
            return [], []
        call_obj_name_set = None
        position = self.get_metadata(libcst.metadata.PositionProvider, call_node)
        module_defined_list = self.global_reference_visitor.goto(position.start.line, position.start.column)
        for defined_node in module_defined_list:
            defined_node_type = self.global_reference_visitor.get_type(defined_node)
            if not defined_node_type:
                return [], []
            if defined_node_type == 'module':
                defined_node_full_name = defined_node.full_name if defined_node.full_name else defined_node.name
                full_call_obj_name = defined_node_full_name + full_name[
                                                              len(defined_node_full_name):full_name.rfind(".")]
                call_obj_name = self._get_call_obj_name(full_call_obj_name)
                call_obj_name_set = {call_obj_name} if call_obj_name else {}
                break
        call_position = self._get_call_position(call_node)
        if call_obj_name_set:
            return self._get_instance_func_list(call_position, file_path, full_name, call_obj_name_set)
        call_obj_position = self.get_metadata(libcst.metadata.PositionProvider, call_node.func.value)
        call_obj_define_list = self.global_reference_visitor.goto(call_obj_position.end.line,
                                                                  call_obj_position.end.column)
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
        elif func_name not in ("forward", "wait"):
            possible_func_names = ', '.join(self.unsupported_instance_op_dict.get(func_name))
            print_func_name = f"{full_name} ({possible_func_names})"
            unknown_list.append(ApiInstance(print_func_name, call_position, file_path))
        return unsupported_list, []  # instance API results that fail to be inferred are not returned

    def _get_unsupported_instance_func_list(self, func_name, call_obj_name_set):
        unsupported_set = set()
        for call_obj_name in call_obj_name_set:
            for instance_func_name in self.unsupported_instance_op_dict.get(func_name):
                if instance_func_name.startswith(call_obj_name) and instance_func_name.endswith(func_name):
                    unsupported_set.add(instance_func_name)
        return unsupported_set

    def _get_call_obj_name_set_by_define_nodes(self, define_nodes):
        queue = []
        queue.extend(define_nodes)
        call_obj_name_set = set()
        while queue:
            define_node = queue.pop(0)
            if "\\" in define_node.description or len(define_node.description) > 1000:
                continue
            define_node_type = self.global_reference_visitor.get_type(define_node)
            if define_node_type == 'statement':
                self._handle_define_type_statement(define_node, call_obj_name_set, queue)
            elif define_node_type == 'param':
                self._handle_define_type_param(define_node, call_obj_name_set)
            elif define_node_type == 'class':
                self._handle_define_type_class(define_node, call_obj_name_set)
            elif define_node_type == 'property':
                self._handle_define_type_property(define_node, call_obj_name_set)
            elif define_node_type == 'instance':
                self._handle_define_type_instance(define_node, call_obj_name_set, queue)
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

    def _handle_define_type_statement(self, define_node, call_obj_name_set, queue):
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
            if self.global_reference_visitor.get_type(node) == 'module':
                full_name = (node.full_name if node.full_name else node.name) + \
                            (name[name.index("."):] if "." in name else "")
                call_obj_name = self._get_call_obj_name(full_name)
                if call_obj_name:
                    call_obj_name_set.add(call_obj_name)
            elif node != define_node:
                queue.append(node)

    def _handle_define_type_class(self, define_node, call_obj_name_set):
        super_class_list = self.global_reference_visitor.get_super_class(define_node.name, str(define_node.module_path))
        for super_class in super_class_list:
            if super_class and super_class.startswith(self.unsupported_op_module_tuple):
                call_obj_name_set.add(super_class)
        if define_node.full_name:
            call_obj_name_set.add(define_node.full_name)

    def _handle_define_type_property(self, define_node, call_obj_name_set):
        if "->" not in define_node.description[:define_node.description.index(":")]:
            return
        start_index = define_node.description.index("->")
        end_index = start_index + define_node.description[start_index:].index(":")
        self._analyse_type_declaration(define_node, call_obj_name_set, start_index, end_index)

    def _handle_define_type_instance(self, define_node, call_obj_name_set, queue):
        if not str(define_node.module_path).startswith(str(self.global_reference_visitor.project.path)):
            call_obj_name_set.add(define_node.full_name)
        else:
            queue.extend(self.global_reference_visitor.get_jedi_script(define_node.module_path).infer(
                define_node.line, define_node.column))

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
                if self.global_reference_visitor.get_type(node) != 'module':
                    continue
                full_name = (node.full_name if node.full_name else node.name) + \
                            (name[name.index("."):] if "." in name else "")
                call_obj_name = self._get_call_obj_name(full_name)
                if call_obj_name:
                    call_obj_name_set.add(call_obj_name)

    def _get_call_position(self, node):
        if m.matches(node.func, m.Attribute()):
            node = node.func.attr
        position = self.get_metadata(libcst.metadata.PositionProvider, node)
        return position


def analyse_unsupported_api(wrapper, op_info, global_reference_visitor=None):
    api_visitor = UnsupportedApiVisitor(op_info, global_reference_visitor)
    module = wrapper.visit(api_visitor)
    api_visitor.print_unsupported_ops()
    return (api_visitor.unsupported_op_result, api_visitor.unknown_api_result), module, wrapper


class ApiInstance:
    def __init__(self, name, position=None, file_path="", info=""):
        null_val = "NA"
        self.name = name
        self.start_line = position.start.line if position else null_val
        self.end_line = position.end.line if position else null_val
        self.file_path = file_path if file_path else null_val
        self.info = info if info else null_val
