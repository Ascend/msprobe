#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
from typing import Optional
from collections import namedtuple

import libcst
import libcst.matchers as m
from libcst.metadata import PositionProvider, QualifiedNameProvider
from analysis.unsupported_api_analysis.unsupported_api_visitor import UnsupportedApiVisitor, ApiInstance, OpInfo
from utils import transplant_logger as translog

AdviceInfo = namedtuple("AdviceInfo",
                        ["precision_advice_dict", "performance_advice_dict", "api_parameters_performance_dict"])


class PrecisionPerformanceAdviceVisitor(UnsupportedApiVisitor):
    METADATA_DEPENDENCIES = (PositionProvider, QualifiedNameProvider)

    def __init__(self, op_info, advice_info, global_reference_visitor=None):
        super().__init__(op_info, global_reference_visitor)
        self.precision_advice_dict = advice_info.precision_advice_dict
        self.performance_advice_dict = advice_info.performance_advice_dict
        self.precision_advice_result = []
        self.performance_advice_result = []
        all_module_name_set = set()
        for func_name in [*self.precision_advice_dict.keys(), *self.performance_advice_dict.keys()]:
            if "." not in func_name:
                continue
            all_module_name_set.add(f'{func_name.split(".")[0]}.')
        self.all_module_names = tuple(all_module_name_set)
        self.api_parameters_performance_dict = advice_info.api_parameters_performance_dict

    def visit_Call(self, node: "libcst.Call") -> Optional[bool]:
        full_name = self.get_full_name_for_node(node)
        position = self.get_metadata(libcst.metadata.PositionProvider, node)
        if full_name:
            precision_advice_apis, _ = self.get_advice_api_instances(node, full_name, position, None,
                                                                     self.precision_advice_dict)
            performance_advice_apis, _ = self.get_advice_api_instances(node, full_name, position, None,
                                                                       self.performance_advice_dict)
            api_parameters_performance_advice = \
                self.get_api_parameters_performance_advice_instances(node, full_name, position, None)

            self.precision_advice_result.extend(precision_advice_apis)
            self.performance_advice_result.extend(performance_advice_apis)
            self.performance_advice_result.extend(api_parameters_performance_advice)
        return True

    def visit_ClassDef(self, node) -> Optional[bool]:
        for base in node.bases:
            full_name = self.get_full_name_for_node(base.value)
            position = self.get_metadata(libcst.metadata.PositionProvider, node)
            if full_name:
                precision_advice_apis, _ = self.get_advice_class_api_instances(full_name, position, None,
                                                                               self.precision_advice_dict)
                performance_advice_apis, _ = self.get_advice_class_api_instances(full_name, position, None,
                                                                                 self.performance_advice_dict)
                self.precision_advice_result.extend(precision_advice_apis)
                self.performance_advice_result.extend(performance_advice_apis)

    def print_unsupported_ops(self):
        for precision_op in self.precision_advice_result:
            precision_advice_info = "Message: %s has a suggestion about precision" % precision_op.name
            msg = "%-21s %-35s %s" % ("line: %s ~ %s" % (precision_op.start_line, precision_op.end_line),
                                      "Operation Type: SUGGESTION", precision_advice_info)
            translog.info(msg)
        for performance_op in self.performance_advice_result:
            performance_advice_info = "Message: %s has a suggestion about performance" % performance_op.name
            msg = "%-21s %-35s %s" % ("line: %s ~ %s" % (performance_op.start_line, performance_op.end_line),
                                      "Operation Type: SUGGESTION", performance_advice_info)
            translog.info(msg)

    def get_advice_api_instances(self, call_node, full_name, position, file_path, need_analyze_dict):
        if not m.findall(call_node.func, m.Call()) and self._is_class_api(call_node, full_name):
            return self.get_advice_class_api_instances(full_name, position, file_path, need_analyze_dict)
        else:  # handle instance api
            if not self.global_reference_visitor:
                return [], []
            return self._handle_instance_func(full_name, call_node, file_path)

    def get_advice_class_api_instances(self, full_name, position, file_path, need_analyze_dict):
        if full_name in need_analyze_dict:
            return [ApiInstance(full_name, position, file_path, need_analyze_dict.get(full_name))], []
        else:
            return [], []

    def get_api_parameters_performance_advice_instances(self, node, full_name, position, file_path):
        info_dict = self.api_parameters_performance_dict.get(full_name)
        if info_dict:
            args = node.args
            for arg in args:
                keyword = None if arg.keyword is None else libcst.parse_module("").code_for_node(arg.keyword)
                value = None if arg.value is None else libcst.parse_module("").code_for_node(arg.value)
                if keyword == info_dict.get('parameter') and str(value) != info_dict.get('expected_value'):
                    return [ApiInstance(full_name, position, file_path,
                                        self.api_parameters_performance_dict.get(full_name).get('msg'))]
                if keyword == info_dict.get('parameter') and str(value) == info_dict.get('expected_value'):
                    return []
            if info_dict.get('default_value') != info_dict.get('expected_value'):
                return [ApiInstance(full_name, position, file_path,
                                    self.api_parameters_performance_dict.get(full_name).get('msg'))]
        return []


def analyse_precision_performance_advice_api(wrapper, advice_info, global_reference_visitor=None):
    op_info = OpInfo({}, {}, [])
    api_visitor = PrecisionPerformanceAdviceVisitor(op_info, advice_info, global_reference_visitor)
    module = wrapper.visit(api_visitor)
    api_visitor.print_unsupported_ops()
    return (api_visitor.precision_advice_result, api_visitor.performance_advice_result), module, wrapper
