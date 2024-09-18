#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

import os
from pathlib import Path
import libcst

from utils import trans_utils as utils
from utils import transplant_logger as translog
from analysis.base_analyzer import BaseAnalyzer
from .third_party_code_visitor import ThirdPartyApiVisitor
from .function_graph import Graph
from ..unsupported_api_analysis.unsupported_api_visitor import OpInfo
from ..unsupported_api_analysis.cuda_cpp_visitor import analyse_cuda_ops


class ThirdPartyAnalyzer(BaseAnalyzer):
    def __init__(self, script_dir, output_dir, pytorch_version, unsupported_third_party_file_list=None):
        super().__init__(script_dir, output_dir, pytorch_version, unsupported_third_party_file_list)
        self.function_graph = Graph()
        self.cuda_ops = analyse_cuda_ops(script_dir, output_dir, write_csv=False)
        self.simple_names_dict = dict()

    @staticmethod
    def _get_write_csv_list(api_name, affected_set):
        affected_list = list(affected_set)
        split_affected_list = [affected_list[i:i + 100] for i in range(0, len(affected_list), 100)]
        return [[api_name, '\n'.join(split_lst)] for split_lst in split_affected_list]

    @staticmethod
    def _check_cuda_op_valid(unknown_torch_api):
        api_name = unknown_torch_api.name
        if api_name.startswith('torch.cuda') or api_name.split('.')[-1] == 'cuda' \
               or not api_name.startswith('torch.'):
            return True
        if any([device in api_name for device in ['mlu', 'mps']]):
            return True
        if unknown_torch_api.info == 'User-defined CUDA Operator.':
            return True
        else:
            return False

    def run(self):
        super().run()

        self.traverse_function_graph()
        self.write_info()

    def traverse_function_graph(self):
        function_queue = self.function_graph.get_leaf_api()
        for function in function_queue:
            self.bfs(function)

    def bfs(self, node):
        queue = [node]
        node.vis = True
        while queue:
            top = queue.pop(0)
            for adj_node in top.connected_function:
                adj_node.in_degree -= 1
                adj_node.has_unsupported_api = adj_node.has_unsupported_api or top.has_unsupported_api
                adj_node.has_unknown_api = adj_node.has_unknown_api or top.has_unknown_api
                adj_node.unsupported_list.extend(top.unsupported_list)
                adj_node.unknown_api_list.extend(top.unknown_api_list)
                if not adj_node.vis and adj_node.in_degree == 0:
                    queue.append(adj_node)
                    adj_node.vis = True

    def write_info(self):
        unsupported_api_list, unknown_api_list = self.function_graph.get_apis()

        cuda_op = self._get_operator_adaptation_needed_list(unsupported_api_list)
        framework_unsupported_op = self._get_framework_adaptation_needed_list(unsupported_api_list)
        full_unsupported_results = self._get_full_unsupported_results(unsupported_api_list)
        migration_needed_op = self._get_migration_needed_list(unsupported_api_list)
        unknown_op = self._get_manual_confirmation_needed_list(unknown_api_list)

        self.result_dict.update({'cuda_op.csv': self.result_dict.get(
            'cuda_op.csv', 0) + len(cuda_op)})
        self.result_dict.update({'framework_unsupported_op.csv': self.result_dict.get(
            'framework_unsupported_op.csv', 0) + len(framework_unsupported_op)})
        self.result_dict.update({'full_unsupported_results.csv': self.result_dict.get(
            'full_unsupported_results.csv', 0) + len(full_unsupported_results)})
        self.result_dict.update({'migration_needed_op.csv': self.result_dict.get(
            'migration_needed_op.csv', 0) + len(migration_needed_op)})
        self.result_dict.update({'unknown_op.csv': self.result_dict.get(
            'unknown_op.csv', 0) + len(unknown_op)})

        csv_title = ('Torch API', 'Affected 3rd-party API')
        utils.write_csv(full_unsupported_results, self.output_path, 'full_unsupported_results',
                        ('File', '3rd-party API', 'Message'))
        utils.write_csv(unknown_op, self.output_path, 'unknown_op',
                        csv_title)
        utils.write_csv(framework_unsupported_op, self.output_path, 'framework_unsupported_op',
                        csv_title)
        utils.write_csv(cuda_op, self.output_path, 'cuda_op',
                        ('OP Name', 'Affected 3rd-party API'))
        utils.write_csv(migration_needed_op, self.output_path, 'migration_needed_op',
                        csv_title)

    def _analysis_code(self, file):
        code = utils.get_file_content_bytes(file)
        try:
            wrapper = libcst.metadata.MetadataWrapper(libcst.parse_module(code))
        except BaseException:
            translog.warning(f'{file} has unsupported python syntax, skip.')
            return
        file_path = Path(file)
        format_file_path = str(file_path)
        format_parent_path = str(file_path.parent)
        if file_path.name == "__init__.py":
            last_sep_index = format_file_path.rfind(os.path.sep)
            while last_sep_index != -1:
                if format_file_path in self.package_env_path_set:
                    self._analysis_init_file(format_parent_path[len(format_file_path) + 1:].replace(os.path.sep, "."))
                    break
                format_file_path = format_file_path[:last_sep_index]
                last_sep_index = format_file_path.rfind(os.path.sep)
        api_visitor = ThirdPartyApiVisitor(OpInfo(self.supported_op_dict, self.unsupported_op_dict, self.cuda_ops),
                                           self.global_reference_visitor, self.function_graph)
        wrapper.visit(api_visitor)

    def _analysis_file(self, file, commonprefix):
        if self.global_reference_visitor:
            self.global_reference_visitor.visit_file(os.path.relpath(file, self.script_dir))
        file_relative_path = os.path.relpath(file, commonprefix)
        translog.info(f'Start the analysis of {file_relative_path}.')
        self._analysis_code(file)
        translog.info(f'Analysis of {file_relative_path} completed.')

    def _analysis_init_file(self, package_path):
        defined_names = self.global_reference_visitor.complete()
        for define_name in defined_names:
            define_name_type = self.global_reference_visitor.get_type(define_name)
            if not define_name_type:
                continue
            if define_name.module_name == "builtins" or define_name.complete.startswith("__") \
                    or define_name_type in ("module", "instance"):
                continue
            try:
                infer_func_list = self.global_reference_visitor.get_jedi_script(
                    str(define_name.module_path)).infer(line=define_name.line, column=define_name.column)
            except BaseException:
                infer_func_list = []
            for func_name in infer_func_list:
                if not func_name.full_name:
                    continue
                if func_name.full_name not in self.simple_names_dict:
                    self.simple_names_dict[func_name.full_name] = []
                self.simple_names_dict.get(func_name.full_name).append(f"{package_path}.{define_name.name}")

    def _get_full_unsupported_results(self, unsupported_api_list):
        full_unsupported_results = []
        for api in unsupported_api_list:
            api_info_list = []
            for unsupported_torch_api in api.unsupported_list:
                api_info = f"file_path:{unsupported_torch_api.file_path}, start_line:" \
                           f"{unsupported_torch_api.start_line}, api_name:{unsupported_torch_api.name} \n"
                api_info_list.append(api_info)
            full_unsupported_results.append(
                [api.file_path, '\n'.join(self._get_simple_names(api.key)), ''.join(api_info_list)])
        return full_unsupported_results

    def _get_framework_adaptation_needed_list(self, unsupported_api_list):
        api_dict = {}
        for api in unsupported_api_list:
            for unsupported_api in api.unsupported_list:
                api_name = unsupported_api.name
                if self._check_cuda_op_valid(unsupported_api):
                    continue
                if api_name in api_dict.keys():
                    api_dict.get(api_name).extend(self._get_simple_names(api.key))
                else:
                    api_dict[api_name] = self._get_simple_names(api.key)

        framework_adaptation_needed_list = []
        for api_name in api_dict.keys():
            affected_set = set(api_dict.get(api_name))
            if len(affected_set) <= 100:
                framework_adaptation_needed_list.append([api_name, '\n'.join(affected_set)])
                continue
            framework_adaptation_needed_list.extend(self._get_write_csv_list(api_name, affected_set))
        return framework_adaptation_needed_list

    def _get_operator_adaptation_needed_list(self, unsupported_api_list):
        op_dict = {}
        for cuda_op in self.cuda_ops:
            op_dict[cuda_op.func_name] = []

        for api in unsupported_api_list:
            for unsupported_op in api.unsupported_list:
                op_name = unsupported_op.name.split('.')[-1]
                if op_name in op_dict.keys():
                    op_dict.get(op_name).extend(self._get_simple_names(api.key))

        operator_adaptation_needed_list = []
        for op_name in op_dict.keys():
            affected_list = op_dict.get(op_name)
            if affected_list:
                operator_adaptation_needed_list.append([op_name, '\n'.join(set(affected_list))])
        return operator_adaptation_needed_list

    def _get_migration_needed_list(self, unsupported_api_list):
        api_dict = {}
        for api in unsupported_api_list:
            for unsupported_api in api.unsupported_list:
                api_name = unsupported_api.name
                is_belongs_to_third_party_device = api_name.startswith('torch.') and \
                                                   any([device in api_name for device in ['mlu', 'mps']])
                if api_name.startswith('torch.cuda') or api_name.split('.')[-1] == 'cuda' or \
                        is_belongs_to_third_party_device:
                    api_dict.setdefault(api_name, []).extend(self._get_simple_names(api.key))

        migration_needed_list = []
        for api_name in api_dict.keys():
            affected_set = set(api_dict.get(api_name))
            if len(affected_set) <= 100:
                migration_needed_list.append([api_name, '\n'.join(affected_set)])
                continue
            migration_needed_list.extend(self._get_write_csv_list(api_name, affected_set))
        return migration_needed_list

    def _get_manual_confirmation_needed_list(self, unknown_api_list):
        api_dict = {}
        for api in unknown_api_list:
            for unknown_torch_api in api.unknown_api_list:
                api_name = unknown_torch_api.name
                if api_name.startswith(("torch.npu", "torch_npu")):
                    continue
                if api_name in api_dict.keys():
                    api_dict.get(api_name).extend(self._get_simple_names(api.key))
                else:
                    api_dict[api_name] = self._get_simple_names(api.key)

        manual_confirmation_needed_list = []
        for api_name in api_dict.keys():
            affected_set = set(api_dict.get(api_name))
            if len(affected_set) <= 100:
                manual_confirmation_needed_list.append([api_name, '\n'.join(affected_set)])
                continue
            manual_confirmation_needed_list.extend(self._get_write_csv_list(api_name, affected_set))
        return manual_confirmation_needed_list

    def _get_simple_names(self, full_name):
        name_set = set()
        for infer_func_name, simple_name_list in self.simple_names_dict.items():
            if not full_name.startswith(infer_func_name):
                continue
            name_set.update(f"{simple_name}{full_name[len(infer_func_name):]}" for simple_name in simple_name_list)
        name_set.add(full_name)
        return sorted(list(name_set), key=len)
