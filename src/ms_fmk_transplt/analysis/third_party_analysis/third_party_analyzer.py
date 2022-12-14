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
from ..unsupported_api_analysis.api_visitor import OpInfo
from ..unsupported_api_analysis.cuda_cpp_visitor import analyse_cuda_ops


class ThirdPartyAnalyzer(BaseAnalyzer):
    def __init__(self, script_dir, output_dir, pytorch_version, unsupported_third_party_file_list=None):
        super().__init__(script_dir, output_dir, pytorch_version, unsupported_third_party_file_list)
        self.function_graph = Graph()
        self.cuda_ops = analyse_cuda_ops(script_dir, output_dir)
        self.simple_names_dict = dict()

    def run(self):
        super().run()

        self.traverse_function_graph()
        self.write_info()

    def _analysis_code(self, file):
        code = utils.get_file_content_bytes(file)
        wrapper = libcst.metadata.MetadataWrapper(libcst.parse_module(code))
        if os.path.basename(file) == "__init__.py":
            for env_path in self.package_env_path_set:
                if file.startswith(env_path):
                    self._analysis_init_file(os.path.dirname(file)[len(env_path) + 1:].replace(os.path.sep, "."))
        api_visitor = ThirdPartyApiVisitor(OpInfo(self.supported_op_list, self.unsupported_op_list, self.cuda_ops),
                                           self.global_reference_visitor, self.function_graph)
        wrapper.visit(api_visitor)

    def _analysis_file(self, file, commonprefix):
        if self.global_reference_visitor:
            self.global_reference_visitor.visit_file(os.path.relpath(file, self.script_dir))
        file_relative_path = os.path.relpath(file, commonprefix)
        translog.info(f'Start analysis {file_relative_path}.')
        self._analysis_code(file)
        translog.info(f'Analysis {file_relative_path} complete.')

    def _analysis_init_file(self, package_path):
        defined_names = self.global_reference_visitor.complete()
        for define_name in defined_names:
            if define_name.module_name == "builtins" or define_name.complete.startswith("__") \
                    or define_name.type in ("module", "instance"):
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
        unsupported_info_list = []
        unknown_info_list = []
        for api in unsupported_api_list:
            api_info_list = []
            for unsupported_torch_api in api.unsupported_list:
                api_info = f"file_path:{unsupported_torch_api.file_path}, start_line:" \
                           f"{unsupported_torch_api.start_line}, api_name:{unsupported_torch_api.name} \n"
                api_info_list.append(api_info)
            unsupported_info_list.append([api.file_path, '\n'.join(self._get_simple_names(api.key)),
                                          ''.join(api_info_list)])

        for api in unknown_api_list:
            api_info_list = []
            for unknown_torch_api in api.unknown_api_list:
                api_info = f"file_path:{unknown_torch_api.file_path}, start_line:" \
                           f"{unknown_torch_api.start_line}, api_name:{unknown_torch_api.name} \n"
                api_info_list.append(api_info)
            unknown_info_list.append([api.file_path, '\n'.join(self._get_simple_names(api.key)),
                                      ''.join(api_info_list)])

        utils.write_csv(unsupported_info_list, self.output_path, 'unsupported_api', ('File', 'Api', 'Message'))
        utils.write_csv(unknown_info_list, self.output_path, 'unknown_api', ('File', 'Api', 'Message'))

    def _get_simple_names(self, full_name):
        name_set = set()
        for infer_func_name, simple_name_list in self.simple_names_dict.items():
            if not full_name.startswith(infer_func_name):
                continue
            name_set.update(f"{simple_name}{full_name[len(infer_func_name):]}" for simple_name in simple_name_list)
        name_set.add(full_name)
        return sorted(list(name_set), key=len)
