#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
from typing import Optional, Dict, List, Tuple

import libcst
import libcst.matchers as m
from libcst.metadata import PositionProvider, QualifiedNameProvider
from analysis.unsupported_api_analysis.unsupported_api_visitor import UnsupportedApiVisitor, ApiInstance, OpInfo
from utils import transplant_logger as translog
from .prec_perf_utils import PerfApiSuggest


class PrecisionPerformanceAdviceVisitor(UnsupportedApiVisitor):
    METADATA_DEPENDENCIES = (PositionProvider, QualifiedNameProvider)

    def __init__(self, op_info, advice_info, global_reference_visitor=None):
        super().__init__(op_info, global_reference_visitor)
        self.precision_advice_dict = advice_info.api_prec_dict
        self.performance_advice_dict = advice_info.api_perf_dict
        self.precision_advice_result = []
        self.performance_advice_result = []
        all_module_name_set = set()
        for func_name in [*self.precision_advice_dict.keys(), *self.performance_advice_dict.keys()]:
            if "." not in func_name:
                continue
            all_module_name_set.add(f'{func_name.split(".")[0]}.')
        self.all_module_names = tuple(all_module_name_set)
        self.api_params_perf_dict = advice_info.api_params_perf_dict
        self.perf_api_suggest = advice_info.perf_api_suggest

    def visit_Call(self, node: "libcst.Call") -> Optional[bool]:
        full_name = self.get_full_name_for_node(node)
        position = self.get_metadata(libcst.metadata.PositionProvider, node)
        if full_name:
            self._update_perf_api_suggest(full_name)
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
        advice_list = []
        func_name = full_name.split(".")[-1]
        info_dict_full = self.api_params_perf_dict.get(full_name)
        info_dict_func = self.api_params_perf_dict.get(func_name)
        info_dict = info_dict_full if info_dict_full else info_dict_func
        match_name = full_name if info_dict_full else func_name  # The name that matches the name in the json file
        if not info_dict:
            return advice_list

        args, kwargs = self._parse_args(node)
        # generate advice
        params = info_dict.get("parameter", {})
        for param_name, comp_info in params.items():
            param_idx_list = sorted(comp_info.get("parameter_idx", []))
            expect_val = comp_info.get("expected_value")
            unexpect_val = comp_info.get("unexpected_value")
            default_val = comp_info.get("default_value")
            msg = comp_info.get("msg")
            advice_inst = ApiInstance(match_name, position, file_path, msg)
            # set parameter by key-value pairs
            if param_name in kwargs:
                if not self._is_parameter_match(expect_val, unexpect_val, kwargs[param_name]):
                    advice_list.append(advice_inst)
                continue
            # set parameter by location
            sat_idx_param = False
            valid_param_idx = False
            for idx in param_idx_list:
                if len(args) < int(idx):
                    break
                valid_param_idx = True
                if self._is_parameter_match(expect_val, unexpect_val, args[int(idx) - 1]):
                    sat_idx_param = True
                    break
            if valid_param_idx:
                if not sat_idx_param:
                    advice_list.append(advice_inst)
                continue
            if not self._is_parameter_match(expect_val, unexpect_val, default_val): # using default value
                advice_list.append(advice_inst)

        return advice_list

    def _update_perf_api_suggest(self, full_name: str):
        func_name = full_name.split(".")[-1]
        if full_name in self.perf_api_suggest.dependency:
            self.perf_api_suggest.dependency[full_name] = True
        elif func_name in self.perf_api_suggest.dependency:
            self.perf_api_suggest.dependency[func_name] = True
        if full_name in self.perf_api_suggest.suggest_apis:
            self.perf_api_suggest.suggest_apis[full_name] = True
        elif func_name in self.perf_api_suggest.suggest_apis:
            self.perf_api_suggest.suggest_apis[func_name] = True

    def _parse_args(self, node: "libcst.Call") -> Tuple[List[str], Dict[str, str]]:
        args = []    # parameters set by location
        kwargs = {}  # parameters set by key-value pairs
        node_args = node.args

        # parse node parameters
        for arg in node_args:
            if not arg.keyword:
                args.append(libcst.parse_module("").code_for_node(arg.value))
            else:
                keyword = libcst.parse_module("").code_for_node(arg.keyword)
                value = libcst.parse_module("").code_for_node(arg.value)
                kwargs[keyword] = str(value)

        return args, kwargs

    def _is_parameter_match(self, expect_val: Optional[str], unexpect_val: Optional[str], real_val: str):
        # maybe real_val is ""
        if expect_val is not None:
            return expect_val == real_val
        elif unexpect_val is not None:
            return unexpect_val != real_val
        return False


def analyse_precision_performance_advice_api(wrapper, advice_info, global_reference_visitor=None):
    op_info = OpInfo({}, {}, [])
    api_visitor = PrecisionPerformanceAdviceVisitor(op_info, advice_info, global_reference_visitor)
    module = wrapper.visit(api_visitor)
    api_visitor.print_unsupported_ops()
    return (api_visitor.precision_advice_result, api_visitor.performance_advice_result), module, wrapper


def generate_perf_suggest(perf_api_suggest: PerfApiSuggest) -> List[ApiInstance]:
    """Generate performance suggestions about unused API."""
    suggest_list = []
    dependency = perf_api_suggest.dependency
    suggest_apis = perf_api_suggest.suggest_apis
    suggest_apis_info = perf_api_suggest.suggest_apis_info
    try:
        for api_name, infos in suggest_apis_info.items():
            dep_apis = infos.get("dependency", [])
            dep_meet = [dependency.get(dep, False) for dep in dep_apis]
            if sum(dep_meet) != len(dep_meet):
                continue
            # Add suggesttion api instance to list if all dependencies are satisfied.
            if api_name in suggest_apis and not suggest_apis.get(api_name):
                suggest_list.append(ApiInstance(api_name, info=infos.get("msg")))
    except AttributeError as err:
        raise RuntimeError("Inner precision and performance config file is incorrect!") from err

    return suggest_list
