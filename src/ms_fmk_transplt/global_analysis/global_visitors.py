#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
import os
from functools import lru_cache

try:
    import jedi
except ImportError:
    jedi = None

from utils import trans_utils as utils


class GlobalReferenceVisitor:
    complete_variable_cache = {}

    def __init__(self, project_path, sys_path=None):
        self.project_path = project_path
        self.project = jedi.Project(path=self.project_path, sys_path=sys_path)
        self.file_path = ''

    @staticmethod
    @lru_cache()
    def _readlines(file_path):
        utils.check_input_file_valid(file_path)
        with open(file_path, 'r', encoding='utf-8') as file_handler:
            return file_handler.readlines()

    @classmethod
    def complete_undefined_name(cls, jedi_script, name, line, column):
        complete_full_name = ''
        try:
            completions = jedi_script.complete(line, column)
        except BaseException:
            return complete_full_name
        for completion in completions:
            if completion.name == name:
                complete_full_name = completion.full_name
                break
        cls.complete_variable_cache[name] = complete_full_name
        return complete_full_name

    def visit_file(self, file_relative_path):
        self.file_path = os.path.join(self.project_path, file_relative_path)
        self.complete_variable_cache.clear()

    def get_func_def_line(self, func_name):
        if not os.path.exists(self.file_path):
            return -1
        utils.check_input_file_valid(self.file_path)
        with open(self.file_path, 'r', encoding='utf-8') as file_handle:
            lines = file_handle.readlines()
        for index, line in enumerate(lines):
            if line.strip().startswith(f'def {func_name}('):
                return index + 1
        return -1

    @lru_cache()
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

    def get_full_name_for_function(self, line, column):
        try:
            func_list = self.get_jedi_script(self.file_path).infer(line, column)
        except BaseException:
            func_list = []
        if func_list:
            full_name = func_list[0].full_name
            if full_name is None:
                # solve the function within the function problem
                full_name = '_'.join((os.path.basename(self.file_path), str(line), str(column),
                                      func_list[0].description.split()[-1]))
            return full_name, os.path.basename(self.file_path)
        return '', ''

    def get_infer_func_list_in_project(self, line, column):
        try:
            func_list = self.get_jedi_script(self.file_path).infer(line, column)
        except BaseException:
            func_list = []
        infer_func_list = []
        for func in func_list:
            if not str(func.module_path).startswith(str(self.project.path)):
                continue
            full_name = func.full_name
            if full_name is None:
                full_name = '_'.join((os.path.basename(self.file_path), str(func.line),
                                      str(func.column), func.description.split()[-1]))
            if func.type == 'class':
                full_name = full_name + '.__init__'
            elif func.type == 'instance':
                full_name = full_name + '.forward'
            elif func.type == 'module':
                continue
            infer_func_list.append(full_name)
        return infer_func_list

    def goto(self, line, column):
        return self.get_jedi_script(self.file_path).goto(line, column)

    def get_context(self, line=None, column=None):
        return self.get_jedi_script(self.file_path).get_context(line, column)

    def search_in_project(self, string):
        return self.project.search(string)

    def complete(self, line=None, column=None, *, fuzzy=False):
        try:
            return self.get_jedi_script(self.file_path).complete(line=line, column=column, fuzzy=fuzzy)
        except BaseException:
            return []

    def get_super_class(self, class_name, file_path=''):
        super_class_list = []
        if class_name.startswith('.'):
            class_name, class_file_path = self.parse_class_name(class_name)
            file_path = file_path or class_file_path

        file_path = file_path or self.file_path
        if not os.path.exists(file_path):
            return super_class_list
        content = self._readlines(file_path)
        for line_id, line in enumerate(content):
            if not line.strip().startswith(f'class {class_name}('):
                continue
            super_class_name = line[line.find('(') + 1:line.find(')')]
            if not super_class_name:
                return super_class_list
            super_class_names = super_class_name.split(',')
            script = self.get_jedi_script(file_path)
            last_found_index = line.find('(')
            for super_class_name in super_class_names:
                super_class_name = super_class_name.strip()

                delta_index = next(i for i, c in enumerate(line[last_found_index:]) if c not in (' ', ',', ')'))
                last_found_index += delta_index
                delta_index = next(i for i, c in enumerate(line[last_found_index:]) if c in (' ', ',', ')'))
                last_found_index += delta_index
                final_delim_index = last_found_index
                try:
                    definitions = script.infer(line_id + 1, final_delim_index)
                except BaseException:
                    definitions = None
                if definitions:
                    definition = definitions[0]
                    super_class_list.append(definition.full_name)
                    super_class_list.extend(self.get_super_class(definition.name, str(definition.module_path)))
                else:
                    super_class_list.append(self._complete_super_class_name(
                        line_id, final_delim_index - len(super_class_name) + 1, script, super_class_name))
            break
        return super_class_list

    def parse_class_name(self, class_name):
        prefix_dir = os.path.dirname(self.file_path)
        class_name = class_name[1:]
        for character in class_name:
            if character != '.':
                break
            prefix_dir = os.path.join(prefix_dir, '../')
        relative_path, _, class_name = class_name.lstrip('.').rpartition('.')
        class_file_path = os.path.join(prefix_dir, relative_path)
        class_file_path = os.path.realpath(class_file_path + ".py")
        return class_name, class_file_path

    def _complete_super_class_name(self, index, col, script, super_class_name):
        split_names = super_class_name.split('.')
        complete_full_name = self.complete_undefined_name(script, split_names[0], index + 1, col)
        if complete_full_name and not complete_full_name.startswith('builtins.'):
            split_names[0] = complete_full_name
        return '.'.join(split_names)

