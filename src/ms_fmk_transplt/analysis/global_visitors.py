#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
import os
import sys

import jedi

from utils import trans_utils as utils


class GlobalReferenceVisitor:
    def __init__(self, project_path):
        self.project_path = project_path
        self.project = jedi.Project(path=self.project_path, sys_path=sys.path)
        self.file_path = ''

    def visit_file(self, file_relative_path):
        self.file_path = os.path.join(self.project_path, file_relative_path)

    def get_func_def_line(self, func_name):
        if not os.path.exists(self.file_path):
            return -1
        with open(self.file_path, 'r', encoding='utf-8') as file_handle:
            lines = file_handle.readlines()
        for index, line in enumerate(lines):
            if line.strip().startswith(f'def {func_name}('):
                return index + 1
        return -1

    def get_jedi_script(self, file_path):
        code = utils.get_file_content(file_path)
        return jedi.Script(code, path=file_path, project=self.project)

    def get_function_define(self, line, column):
        self.get_jedi_script(self.file_path).infer(line, column)

    def get_full_name_for_function(self, line, column):
        func_list = self.get_jedi_script(self.file_path).infer(line, column)
        try:
            full_name = func_list[0].full_name
            if full_name is None:
                # solve the function within the function problem
                full_name = '_'.join((os.path.basename(self.file_path), str(line), str(column),
                                      func_list[0].description.split()[-1]))
            return full_name, os.path.basename(self.file_path)
        except IndexError:
            print("func_list is None")

    def is_belong_with_self_project(self, line, column):
        func_list = self.get_jedi_script(self.file_path).infer(line, column)
        if func_list:
            is_defined = str(func_list[0].module_path).startswith(str(self.project.path))
            full_name = func_list[0].full_name
            if full_name is None:
                full_name = '_'.join((str(func_list[0].line), str(func_list[0].column),
                                      func_list[0].description.split()[-1]))
            if func_list[0].description.startswith('class') and is_defined:
                full_name = full_name + '.__init__'
            elif func_list[0].description.startswith('instance') and is_defined:
                full_name = full_name + '.forward'
            elif func_list[0].description.startswith('module'):
                full_name = ''
                is_defined = False
            return is_defined, full_name
        return False, ''
