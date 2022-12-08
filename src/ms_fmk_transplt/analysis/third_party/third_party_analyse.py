#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

import os
from pathlib import Path
import libcst

import utils.trans_utils as utils
import utils.transplant_logger as translog
from utils.third_party_code_visitor import ThirdPartyApiVisitor
from analysis.analyse import PyTorchAnalyze
from analysis.third_party.function_graph import Graph
from analysis.third_party.cuda_cpp_visitor import CudaOpVisitor


class ThirdPartyAnalyse(PyTorchAnalyze):
    def __init__(self, script_dir, output_dir, pytorch_version):
        super().__init__(script_dir, output_dir, pytorch_version)
        self.script_dir = script_dir
        self.output_dir = output_dir
        self.py_file_counts = 0
        self.pytorch_version = pytorch_version
        self.global_reference_visitor = None
        self.function_graph = Graph()
        self.cuda_ops = self._get_cuda_ops()
        self.package_env_path_set = self._search_package_env_path()
        self.simple_names_dict = dict()

    def _search_package_env_path(self):
        package_env_path_set = set()
        search_file_list = [self.script_dir]
        while search_file_list:
            file_path = search_file_list.pop()
            if not os.path.isdir(file_path):
                continue
            if os.path.exists(os.path.join(file_path, "__init__.py")):
                package_env_path_set.add(str(Path(file_path).parent))
                continue
            for sub_file in os.listdir(file_path):
                full_path = os.path.join(file_path, sub_file)
                if os.path.isdir(full_path):
                    search_file_list.append(full_path)
        return package_env_path_set

    def init_global_visitor(self, global_reference_visitor):
        self.global_reference_visitor = global_reference_visitor

    def run(self):
        super().run()

        self.traverse_function_graph()
        self.write_info()

    def set_py_file_counts(self, py_file_counts):
        self.py_file_counts = py_file_counts

    def _analysis_code(self, file):
        code = utils.get_file_content_bytes(file)
        wrapper = libcst.metadata.MetadataWrapper(libcst.parse_module(code))
        if os.path.basename(file) == "__init__.py":
            for env_path in self.package_env_path_set:
                if file.startswith(env_path):
                    self._analysis_init_file(os.path.dirname(file)[len(env_path) + 1:].replace(os.path.sep, "."))
        api_visitor = ThirdPartyApiVisitor(utils.get_supported_op_list(), utils.get_op_list(self.pytorch_version),
                                           self.cuda_ops, self.global_reference_visitor, self.function_graph)
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

    def _get_cuda_ops(self):
        cuda_op_visitor = CudaOpVisitor(self.script_dir)
        cuda_op_visitor.visit_cuda_files()
        cuda_op_list = cuda_op_visitor.cuda_ops
        cuda_op_info_list = [[cuda_op.file_path, cuda_op.func_name] for cuda_op in cuda_op_list]
        utils.write_csv(cuda_op_info_list, '', self.output_dir, 'cuda_op_list')
        return cuda_op_list

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

        utils.write_csv(unsupported_info_list, '', self.output_dir, 'unsupported_api')
        utils.write_csv(unknown_info_list, '', self.output_dir, 'unknown_api')

    def _get_simple_names(self, full_name):
        name_set = set()
        for infer_func_name, simple_name_list in self.simple_names_dict.items():
            if not full_name.startswith(infer_func_name):
                continue
            name_set.update(f"{simple_name}{full_name[len(infer_func_name):]}" for simple_name in simple_name_list)
        name_set.add(full_name)
        return sorted(list(name_set), key=len)










