#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.

import os
import libcst

import utils.trans_utils as utils
import utils.transplant_logger as translog
from utils.third_party_code_visitor import ApiVisitor
from analysis.analyse import PyTorchAnalyze
from analysis.third_party.function_graph import Graph


class ThirdPartyAnalyse(PyTorchAnalyze):
    def __init__(self, script_dir, output_dir, pytorch_version):
        super().__init__(script_dir, output_dir, pytorch_version)
        self.script_dir = script_dir
        self.output_dir = output_dir
        self.py_file_counts = 0
        self.pytorch_version = pytorch_version
        self.global_reference_visitor = None
        self.function_graph = Graph()

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

        api_visitor = ApiVisitor(utils.get_supported_op_list(), utils.get_op_list(self.pytorch_version),
                                 self.global_reference_visitor, self.function_graph)
        wrapper.visit(api_visitor)

    def _analysis_file(self, file, commonprefix):
        if self.global_reference_visitor:
            self.global_reference_visitor.visit_file(os.path.relpath(file, self.script_dir))
        file_relative_path = os.path.relpath(file, commonprefix)
        translog.info(f'Start analysis {file_relative_path}.')
        self._analysis_code(file)
        translog.info(f'Analysis {file_relative_path} complete.')

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
            unsupported_info_list.append([api.file_path, api.key, ''.join(api_info_list)])

        for api in unknown_api_list:
            api_info_list = []
            for unknown_torch_api in api.unknown_api_list:
                api_info = f"file_path:{unknown_torch_api.file_path}, start_line:" \
                           f"{unknown_torch_api.start_line}, api_name:{unknown_torch_api.name} \n"
                api_info_list.append(api_info)
            unknown_info_list.append([api.file_path, api.key, ''.join(api_info_list)])

        utils.write_csv(unsupported_info_list, '', self.output_dir, 'unsupported_api')
        utils.write_csv(unknown_info_list, '', self.output_dir, 'unknown_api')








