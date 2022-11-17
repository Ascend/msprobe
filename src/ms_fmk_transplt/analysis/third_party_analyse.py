#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.

import os
import libcst

import utils.trans_utils as utils
import utils.transplant_logger as translog
from analysis.code_visitor import ApiVisitor
from utils.trans_utils import TransplantException
from analysis.function_graph import Graph


class ThirdPartyAnalyse(object):
    def __init__(self, script_dir, output_dir, pytorch_version):
        self.script_dir = script_dir
        self.output_dir = output_dir
        self.py_file_counts = 0
        self.pytorch_version = pytorch_version
        self.global_reference_visitor = None
        self.function_graph = Graph()

    @staticmethod
    def __need_analysis(file, commonprefix):
        return utils.check_file_need_analysis(file, commonprefix, record=True)

    def init_global_visitor(self, global_reference_visitor):
        self.global_reference_visitor = global_reference_visitor

    def run(self):
        translog.info('Analysis start...')

        if not os.access(self.script_dir, os.R_OK):
            raise TransplantException('%s is not readable.' % self.script_dir)

        if os.path.isfile(self.script_dir) and self.__need_analysis(self.script_dir, os.path.dirname(self.script_dir)):
            self.__delete_csv_file_for_file_transplant()
            self.__analysis_file(self.script_dir, os.path.dirname(self.script_dir))

        if os.path.isdir(self.script_dir):
            self.__analysis_dir()

        self.traverse_function_graph()
        self.write_info()

    def set_py_file_counts(self, py_file_counts):
        self.py_file_counts = py_file_counts

    def __delete_csv_file_for_file_transplant(self):
        for csv_type in ['change_list', 'unsupported_op']:
            csv_file = os.path.join(os.path.dirname(self.script_dir), '%s.csv' % csv_type)
            if os.path.exists(csv_file):
                utils.remove_path(csv_file)

    def __analysis_code(self, file):
        code = utils.get_file_content_bytes(file)
        wrapper = libcst.metadata.MetadataWrapper(libcst.parse_module(code))

        api_visitor = ApiVisitor(utils.get_supported_op_list(), self.global_reference_visitor, self.function_graph)
        module = wrapper.visit(api_visitor)

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

    def __analysis_file(self, file, commonprefix):
        if self.global_reference_visitor:
            self.global_reference_visitor.visit_file(os.path.relpath(file, self.script_dir))
        file_relative_path = os.path.relpath(file, commonprefix)
        translog.info(f'Start analysis {file_relative_path}.')
        self.__analysis_code(file)
        translog.info(f'Analysis {file_relative_path} complete.')

    def traverse_function_graph(self):
        function_queue = self.function_graph.get_leaf_api()
        for function in function_queue:
            self.bfs(function)

    def bfs(self, node):
        queue = [node]
        node.vis = True
        while len(queue):
            top = queue.pop(0)
            for adj_node in top.connected_function:
                adj_node.in_degree -= 1
                adj_node.has_unsupported_api = adj_node.has_unsupported_api or top.has_unsupported_api
                adj_node.unsupported_list.extend(top.unsupported_list)
                if not adj_node.vis and adj_node.in_degree == 0:
                    queue.append(adj_node)
                    adj_node.vis = True

    def write_info(self):
        unsupported_api_list = self.function_graph.get_unsupported_apis()
        content_list = []
        for api in unsupported_api_list:
            api_info_list = []
            for unsupported_torch_api in api.unsupported_list:
                api_info = f"file_path:{unsupported_torch_api.file_path}, start_line:" \
                           f"{unsupported_torch_api.start_line}, api_name:{unsupported_torch_api.name} \n"
                api_info_list.append(api_info)
            content_list.append(''.join(api_info_list))
        info_list = []
        for inx in range(len(unsupported_api_list)):
            api = unsupported_api_list[inx]
            content = content_list[inx]
            info_list.append((api.file_path, api.key, content))
        utils.write_csv_third_party(info_list, self.output_dir)

    def print_graph_node_num(self):
        file_dict = {}
        for key, node in self.function_graph.nodelist.items():
            if not node.file_path:
                print(key)
            if node.file_path not in file_dict:
                file_dict[node.file_path] = 1
            else:
                file_dict[node.file_path] += 1
        with open("../scripts/file.txt", 'a+') as f:
            for file_path, num in file_dict.items():
                f.write(file_path + ' ' + str(num) + '\n')
        print(file_dict)








