#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from collections import namedtuple
from typing import Optional, Union
import libcst
import libcst.matchers as m
import libcst.helpers as helper
from libcst.metadata import PositionProvider, QualifiedNameProvider
from utils import trans_utils
from utils import transplant_logger as translog

AffinityInfo = namedtuple("AffinityInfo", ["affinity_api_def_list", "affinity_api_call_list", "affinity_special_list"])

INIT = '__init__'
FORWARD = 'forward'


class AffinityApiVisitor(libcst.CSTVisitor):
    METADATA_DEPENDENCIES = (PositionProvider, QualifiedNameProvider)

    def __init__(self, pytorch_version, global_reference_visitor=None):
        super().__init__()
        self.global_reference_visitor = global_reference_visitor
        self.affinity_function_dict = trans_utils.get_affinity_info_dict(pytorch_version, 'function')
        self.affinity_class_dict = trans_utils.get_affinity_info_dict(pytorch_version, 'class')
        self.affinity_torch_dict = trans_utils.get_affinity_info_dict(pytorch_version, 'torch')
        self.affinity_list = []
        self.affinity_call_list = []
        self.affinity_special_list = []

    @staticmethod
    def _visit_class_body(node):
        class_body_list = []
        body = node.body.body
        for element in body:
            call_function = m.findall(element, m.FunctionDef())
            class_body_list.extend(call_function)
        return class_body_list

    @staticmethod
    def _get_affinity_msg(api):
        affinity_api_info = "Message: %s has an affinity api that can be replaced!" % api.name
        return "%-21s %-35s %s" % ("line: %s ~ %s" % (api.start_line, api.end_line),
                                   "Operation Type: HAVE AFFINITY API", affinity_api_info)

    def visit_Assign(self, node: "libcst.Assign") -> Optional[bool]:
        pattern = m.Assign(
            targets=[
                m.AssignTarget(
                    target=m.Subscript(
                        value=m.Name(),
                        slice=[m.SubscriptElement(slice=m.Index(value=m.Name()))]
                    )
                )
            ],
            value=m.Name()
        )
        if m.matches(node, pattern):
            name = 'input[condition] = value'
            info = 'torch_npu.contrib.function.npu_fast_condition_index_put'
            position = self.get_metadata(libcst.metadata.PositionProvider, node)
            self.affinity_special_list.append(ApiInstance(name, name, position, api_type='special', info=info))

    def visit_Call(self, node: "libcst.Call") -> Optional[bool]:
        full_name = self.get_full_name_for_node(node)
        position = self._get_call_position(node)
        if full_name and full_name.startswith('torch.'):
            name = full_name.split('.')[-1]
            if name in self.affinity_torch_dict.keys():
                affinity_full_name = self.affinity_torch_dict.get(name).get('full_name')
                if full_name == affinity_full_name:
                    info = self.affinity_torch_dict.get(name).get('affinity_api')
                    self.affinity_call_list.append(
                        ApiInstance(full_name, full_name, position, api_type='torch', info=info))
        else:
            infer_func_list, infer_func_not_in_project_list = \
                self.global_reference_visitor.get_infer_func_list_in_project(position.start.line, position.start.column)
            if infer_func_list:
                full_name = infer_func_list[0]
                full_name_split = full_name.split(".")
                if full_name.endswith(INIT) and len(full_name_split) >= 2:
                    name = full_name_split[-2]
                    full_name = full_name.replace('.__init__', '')
                    api_type = 'class'
                else:
                    name = full_name_split[-1]
                    api_type = 'function'
                self.affinity_call_list.append(ApiInstance(full_name, name, position, api_type=api_type))

    def visit_ClassDef(self, node: libcst.ClassDef) -> Optional[bool]:
        position, class_line, class_column = self._get_class_def_position(node)
        full_name, file_path = self.global_reference_visitor.get_full_name_for_function(class_line, class_column)
        class_name = node.name.value
        if class_name in self.affinity_class_dict.keys():
            info_dict = self.affinity_class_dict.get(class_name)
            init_params_list = info_dict.get(INIT, [])
            forward_params_list = info_dict.get(FORWARD, [])
            function_def_list = self._visit_class_body(node)
            init_and_forward_dict = {}
            for function_def in function_def_list:
                func_name = function_def.name.value
                param_list = []
                for param in function_def.params.params:
                    param_list.append(param.name.value)
                init_and_forward_dict[func_name] = param_list
            have_affinity_api = True
            if init_and_forward_dict.get(INIT, []) != init_params_list:
                have_affinity_api = False
            if init_and_forward_dict.get(FORWARD, []) != forward_params_list:
                have_affinity_api = False
            if have_affinity_api:
                info = self.affinity_class_dict.get(class_name).get('affinity_class')
                self.affinity_list.append(ApiInstance(full_name, class_name, position, file_path, 'class', info))

    def visit_FunctionDef(self, node: libcst.FunctionDef) -> Optional[bool]:
        function_line, function_column = self._get_func_def_position(node)
        full_name, file_path = self.global_reference_visitor.get_full_name_for_function(function_line, function_column)
        function_name = node.name.value
        parameter_list = []
        if function_name in self.affinity_function_dict.keys():
            params_list = node.params.params
            for param in params_list:
                parameter_list.append(param.name.value)
            if parameter_list:
                parameter_template_list = self.affinity_function_dict.get(function_name, {}).get('parameter', [])
                if parameter_list == parameter_template_list:
                    position = self.get_metadata(libcst.metadata.PositionProvider, node)
                    name = full_name.split(".")[-1]
                    info = self.affinity_function_dict.get(function_name, {}).get('affinity_function', [])
                    self.affinity_list.append(ApiInstance(full_name, name, position, file_path, 'function', info))

    def print_affinity_ops(self):
        for api in self.affinity_list:
            msg = self._get_affinity_msg(api)
            translog.warning(msg)
        for api in self.affinity_call_list:
            if api.full_name.startswith('torch'):
                msg = self._get_affinity_msg(api)
                translog.warning(msg)
        for api in self.affinity_special_list:
            msg = self._get_affinity_msg(api)
            translog.warning(msg)

    def get_full_name_for_node(self, node: Union[str, libcst.CSTNode]) -> Optional[str]:
        name_list = list(self.get_metadata(libcst.metadata.QualifiedNameProvider, node))
        return name_list[0].name if name_list else helper.get_full_name_for_node(node)

    def _get_func_def_position(self, node):
        position = self.get_metadata(PositionProvider, node)
        node_start_line = position.start.line
        if node.asynchronous:
            # The starting position of the function name inside the class is 10 spaces
            node_start_column = position.start.column + 10
        else:
            # The starting position of the function name outside the class is 4 spaces
            node_start_column = position.start.column + 4
        return node_start_line, node_start_column

    def _get_class_def_position(self, node):
        position = self.get_metadata(PositionProvider, node)
        # The starting position of the class name is 6 spaces
        return position, position.start.line, position.start.column + 6

    def _get_call_position(self, node):
        if m.matches(node.func, m.Attribute()):
            node = node.func.attr
        position = self.get_metadata(libcst.metadata.PositionProvider, node)
        return position


def analyse_affinity_api(wrapper, pytorch_version, global_reference_visitor=None):
    api_visitor = AffinityApiVisitor(pytorch_version, global_reference_visitor)
    wrapper.visit(api_visitor)
    api_visitor.print_affinity_ops()
    return AffinityInfo(api_visitor.affinity_list, api_visitor.affinity_call_list, api_visitor.affinity_special_list)


class ApiInstance:
    def __init__(self, full_name, name, position, file_path="", api_type="", info=""):
        self.full_name = full_name
        self.name = name
        self.start_line = position.start.line
        self.end_line = position.end.line
        self.file_path = file_path
        self.api_type = api_type
        self.info = info
