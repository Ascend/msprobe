#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
import os
import sys

import jedi

from pytorch_gpu2npu.utils import trans_utils as utils


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
            line = line.strip()
            if line.startswith(f'def {func_name}('):
                return index + 1

    def get_jedi_script(self, file_path):
        code = utils.get_file_content(file_path)
        return jedi.Script(code, path=file_path, project=self.project)

    def find_usages(self, line, name):
        if not self.file_path:
            return []
        position = utils.name_to_jedi_position(self.file_path, line, name)
        usages = self.get_jedi_script(self.file_path).get_references(**position)
        if usages:
            # escape definition, focus on usage
            return list(usage for usage in usages if not usage.is_definition())
        else:
            return []
