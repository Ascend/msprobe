#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.

import os
from typing import Optional
from typing import Union

import libcst
from libcst._flatten_sentinel import FlattenSentinel
from libcst._removal_sentinel import RemovalSentinel

from utils import trans_utils as utils
from utils import transplant_logger as translog
from analysis import analyse_unsupported_api, analyse_cuda_ops, OpInfo
from analysis.affinity_api_analysis.affinity_api_visitor import analyse_affinity_api
from analysis.affinity_api_analysis.affinity_api_analyzer import AffinityApiAnalyzer
from analysis.precision_performance_advice_analysis.precision_performance_advice_visitor import (
    analyse_precision_performance_advice_api, generate_perf_suggest
)
from analysis.precision_performance_advice_analysis.prec_perf_utils import AdviceInfo
from analysis.precision_performance_advice_analysis.prec_perf_utils import PerfApiSuggest

from analysis.unsupported_api_analysis.unsupported_api_analyzer import export_performance_configuration
from .rules.distributed_rules import DataLoaderRule
from .rules.common_rules import InsertMainFileRule


class Transplant(object):
    def __init__(self, script_dir, rule_list, args, analysis_result_dir):
        self.script_dir = script_dir
        self.rule_list = rule_list
        self.main_file = utils.get_main_file(args.main, args.input) if hasattr(args, 'main') else ''
        self.args = args
        self.py_file_counts = 0
        self.current_file_rel_path = ''

        self.global_reference_visitor = None

        # init dict for precision and performance advice analyse
        prec_perf_advice_dict = utils.parse_precision_performance_advice_file()
        if not isinstance(prec_perf_advice_dict, dict):
            raise TypeError("Inner precision and performance config file is incorrect!")
        api_prec_dict = prec_perf_advice_dict.get("api_precision_dict", {})
        api_perf_dict = prec_perf_advice_dict.get("api_performance_dict", {})
        api_params_perf_dict = prec_perf_advice_dict.get("api_parameters_performance_dict", {})
        perf_api_suggest_dict = prec_perf_advice_dict.get("performance_api_suggest_use", {})
        perf_api_suggest = PerfApiSuggest(perf_api_suggest_dict)
        self.perf_config_dict = prec_perf_advice_dict.get("performance_configuration_dict", {})
        self.advice_info = AdviceInfo(api_prec_dict, api_perf_dict, api_params_perf_dict, perf_api_suggest)

        self.op_info = OpInfo(utils.get_supported_op_dict(args.version), utils.get_unsupported_op_dict(args.version),
                              analyse_cuda_ops(script_dir, analysis_result_dir))
        self.transplant_result_statistics = {}
        self.analysis_result_dir = analysis_result_dir
        self.affinity_api_analyzer = AffinityApiAnalyzer(script_dir, analysis_result_dir, args.version)

    @staticmethod
    def __need_analysis(file, commonprefix):
        return utils.check_file_need_analysis(file, commonprefix, record=True)

    def init_global_visitor(self, global_reference_visitor):
        self.global_reference_visitor = global_reference_visitor

    def run(self):
        export_performance_configuration(self.perf_config_dict, self.transplant_result_statistics,
                                         self.analysis_result_dir)
        translog.info('Analysis start...')

        if not os.access(self.script_dir, os.R_OK):
            raise utils.TransplantException('%s is not readable.' % self.script_dir)
        self.__analysis_dir()
        self.affinity_api_analyzer.write_csv()
        self.transplant_result_statistics.update(self.affinity_api_analyzer.result_dict)

    def set_py_file_counts(self, py_file_counts):
        self.py_file_counts = py_file_counts

    def __analysis_code(self, file):
        code = utils.get_file_content_bytes(file)
        try:
            wrapper = libcst.metadata.MetadataWrapper(libcst.parse_module(code))
        except BaseException:
            translog.warning(f'{file} has unsupported python syntax, skip.')
            return
        (unsupported_list, unknown_list), module, wrapper = analyse_unsupported_api(wrapper, self.op_info,
                                                                                    self.global_reference_visitor)
        (precision_advice_list, performance_advice_list), _, _ = \
            analyse_precision_performance_advice_api(wrapper, self.advice_info,
                                                     self.global_reference_visitor)
        self.affinity_api_analyzer.affinity_info = analyse_affinity_api(wrapper, self.args.version,
                                                                        self.global_reference_visitor)
        result_dicts = {
            'cuda_op_list.csv': self.op_info.cuda_op_list,
            'unsupported_api.csv': unsupported_list,
            'unknown_api.csv': unknown_list,
            'api_precision_advice.csv': precision_advice_list,
            'api_performance_advice.csv': performance_advice_list
        }
        for result_dict in result_dicts.items():
            self.transplant_result_statistics.update({result_dict[0]: self.transplant_result_statistics.get(
                result_dict[0], 0) + len(result_dict[1])})

        csv_title = ('File', 'Start Line', 'End Line', 'OP', 'Tips')
        utils.write_csv(list((self.current_file_rel_path, api.start_line, api.end_line, api.name, api.info)
                             for api in unsupported_list), self.analysis_result_dir, "unsupported_api",
                        csv_title)
        utils.write_csv(list((self.current_file_rel_path, api.start_line, api.end_line, api.name)
                             for api in unknown_list), self.analysis_result_dir, "unknown_api",
                        csv_title)
        utils.write_csv(
            self.__get_content_list(precision_advice_list),
            self.analysis_result_dir,
            "api_precision_advice",
            csv_title
        )
        utils.write_csv(
            self.__get_content_list(performance_advice_list),
            self.analysis_result_dir,
            "api_performance_advice",
            csv_title
        )
        self.affinity_api_analyzer.collect_affinity_analysis_results()

        new_module = self.__visit_rule(file, module)
        utils.write_file_content(file, new_module.code)

    def __analysis_dir(self):
        count = 0
        translog.set_progress_info(f'[Progress:{count / self.py_file_counts * 100:6.2f}%]')
        for root, _, files in os.walk(self.script_dir):
            for current_file in files:
                file = os.path.join(root, current_file)
                if not self.__need_analysis(file, self.script_dir):
                    continue
                self.__analysis_file(file, self.script_dir)
                count += 1
                translog.set_progress_info(f'[Progress:{count / self.py_file_counts * 100:6.2f}%]')
        # Give performance suggestion about the api not used.
        suggest_list = generate_perf_suggest(self.advice_info.perf_api_suggest)
        self.transplant_result_statistics.update({'api_performance_advice.csv': self.transplant_result_statistics.get(
            'api_performance_advice.csv', 0) + len(suggest_list)})
        utils.write_csv(
            self.__get_content_list(suggest_list, with_file=False),
            self.analysis_result_dir,
            "api_performance_advice",
            ('File', 'Start Line', 'End Line', 'OP', 'Tips')
        )

    def __analysis_file(self, file, commonprefix):
        self.current_file_rel_path = os.path.relpath(file, commonprefix)
        if self.global_reference_visitor:
            self.global_reference_visitor.visit_file(self.current_file_rel_path)
        translog.info(f'Start the analysis of {self.current_file_rel_path}.')
        self.__analysis_code(file)
        translog.info(f'Analysis of {self.current_file_rel_path} completed.')

    def __visit_rule(self, file, module):
        code_transformer = CodeTransformer(self.rule_list,
                                           self.current_file_rel_path == self.main_file if self.main_file else False,
                                           global_reference_visitor=self.global_reference_visitor)
        wrapper = libcst.metadata.MetadataWrapper(module)
        new_module = wrapper.visit(code_transformer)
        change_info_list = code_transformer.print_change_info()
        self.transplant_result_statistics.update({'change_list.csv': self.transplant_result_statistics.get(
            'change_list.csv', 0) + len(change_info_list)})
        utils.write_csv(list([self.current_file_rel_path] + change_info for change_info in change_info_list),
                        self.analysis_result_dir, "change_list",
                        ('File', 'Start Line', 'End Line', 'Operation Type', 'Message'))
        for rule in self.rule_list:
            rule.clean()
        return new_module

    def __get_content_list(self, result_list, with_file=True):
        if with_file:
            result = [(self.current_file_rel_path, api.start_line, api.end_line,
                        api.name, api.info) for api in result_list]
        else:
            result = [("NA", api.start_line, api.end_line,
                        api.name, api.info) for api in result_list]
        return result


class CodeTransformer(libcst.CSTTransformer):
    METADATA_DEPENDENCIES = (
        libcst.metadata.PositionProvider, libcst.metadata.ScopeProvider,
        libcst.metadata.QualifiedNameProvider, libcst.metadata.ParentNodeProvider
    )

    def __init__(self, rule_list, is_main_file, global_reference_visitor=None):
        super().__init__()
        self.rule_list = rule_list
        for rule in self.rule_list:
            if isinstance(rule, InsertMainFileRule):
                rule.visit_main_file(is_main_file)
            if isinstance(rule, DataLoaderRule):
                rule.set_global_reference_visitor(global_reference_visitor)
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

    def leave_Module(self, original_node: libcst.Module, updated_node: libcst.Module) -> libcst.Module:
        for rule in self.rule_list:
            updated_node = rule.leave_Module(original_node, updated_node)
        return updated_node

    def leave_Name(
            self, original_node: libcst.Name, updated_node: libcst.Name
    ) -> libcst.Name:
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

    def leave_Import(
        self, original_node: "libcst.Import", updated_node: "libcst.Import"
    ) -> "libcst.BaseExpression":
        for rule in self.rule_list:
            updated_node = rule.leave_Import(original_node, updated_node)
        return updated_node

    def print_change_info(self):
        change_info_list = []
        for rule in self.rule_list:
            change_info_list.extend(rule.print_change_info())
        return change_info_list

    def leave_Attribute(self, original_node: libcst.Attribute, updated_node: libcst.Attribute) \
            -> libcst.Attribute:
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
