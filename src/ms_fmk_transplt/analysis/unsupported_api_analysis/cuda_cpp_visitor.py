#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

import os

from utils import trans_utils as utils
from .cuda_cpp_parser import PybindModuleParser, TorchLibraryParser


class CudaOpVisitor:
    def __init__(self, project_path):
        self.project_path = project_path
        self._cuda_ops = []
        self._file_lines = []
        self.rel_file_path = ''

    @property
    def cuda_ops(self):
        return self._cuda_ops

    def visit_cuda_files(self):
        for root, _, files in os.walk(self.project_path):
            for file in files:
                if not file.endswith('.cpp') and not file.endswith('.cu'):
                    continue
                file_path = os.path.join(root, file)
                if utils.islink(file_path):
                    continue
                utils.check_input_file_valid(file_path, utils.InputInfo())
                self.rel_file_path = os.path.relpath(file_path, self.project_path)
                self.visit_file(file_path)

    def visit_file(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file_reader:
                self._file_lines = file_reader.readlines()
        except Exception as e:
            raise RuntimeError("Can't open file: " + file_path) from e
        # parse cuda ops via "PYBIND11_MODULE"
        self._visit_py_bind_module()
        # parse cuda ops via "TORCH_LIBRARY"
        self._visit_torch_library_module()

    def _visit_py_bind_module(self):
        in_pybind_body = False
        in_func_or_class_declare = False
        declare_line = ''
        declare_lines = []
        for line in self._file_lines:
            # escape annotation
            if line.strip().startswith(('/*', '//', '*/')):
                continue
            line = line.split('//')[0].strip()
            if line.startswith('PYBIND11_MODULE('):
                in_pybind_body = True
            if not in_pybind_body:
                continue
            if line.startswith(('m.def(', 'py::class_')):
                if declare_line:
                    declare_lines.append(declare_line)
                    declare_line = ''
                in_func_or_class_declare = True
            if in_func_or_class_declare:
                declare_line += line
                if line.endswith('}'):
                    declare_lines.append(declare_line)
                    declare_line = ''
                    in_pybind_body = False
                    in_func_or_class_declare = False

        pybind_module_parser = PybindModuleParser(self._cuda_ops, self._file_lines, self.rel_file_path)
        for declare_line in declare_lines:
            if declare_line.startswith('m.def('):
                pybind_module_parser.parse_m_def(declare_line)
            elif declare_line.startswith('py::class_'):
                pybind_module_parser.parse_class_declare(declare_line)

    def _visit_torch_library_module(self):
        in_torch_library_body = False
        in_func_or_class_declare = False
        declare_line = ''
        declare_lines = []
        for line in self._file_lines:
            # escape annotation
            if line.strip().startswith(('/*', '//', '*/')):
                continue
            line = line.split('//')[0].strip()
            if line.startswith(('TORCH_LIBRARY(', 'TORCH_LIBRARY_FRAGMENT(', 'TORCH_LIBRARY_IMPL(')):
                in_torch_library_body = True
            if not in_torch_library_body:
                continue
            if line.startswith(('m.def(', 'm.impl(', 'm.class')):
                if declare_line:
                    declare_lines.append(declare_line)
                    declare_line = ''
                in_func_or_class_declare = True
            if in_func_or_class_declare:
                declare_line += line
                if line.endswith('}'):
                    declare_lines.append(declare_line)
                    declare_line = ''
                    in_torch_library_body = False
                    in_func_or_class_declare = False

        torch_library_parser = TorchLibraryParser(self._cuda_ops, self._file_lines, self.rel_file_path)
        for declare_line in declare_lines:
            if declare_line.startswith('m.def('):
                # m.def(xxx);
                torch_library_parser.parse_m_def(declare_line)
            elif declare_line.startswith('m.impl('):
                # m.impl(xxx);
                torch_library_parser.parse_m_impl(declare_line)
            elif declare_line.startswith('m.class'):
                # m.class xxx
                torch_library_parser.parse_class_declare(declare_line)


def analyse_cuda_ops(script_dir, output_path, write_csv=True):
    cuda_op_visitor = CudaOpVisitor(script_dir)
    cuda_op_visitor.visit_cuda_files()
    cuda_op_list = cuda_op_visitor.cuda_ops
    if write_csv:
        cuda_op_info_list = [[cuda_op.file_path, cuda_op.func_name] for cuda_op in cuda_op_list]
        utils.write_csv(cuda_op_info_list, output_path, "cuda_op_list", ('File', 'Api'))
    return cuda_op_list
