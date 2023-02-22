#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
import os
from pathlib import Path

from utils import trans_utils as utils, transplant_logger as translog


class BaseAnalyzer:
    def __init__(self, script_dir, output_path, pytorch_version, unsupported_third_party_file_list=None):
        self.script_dir = script_dir
        self.output_path = output_path
        self.pytorch_version = pytorch_version
        self.py_file_counts = 0
        self.current_file_rel_path = ''
        self.unsupported_op_dict = utils.get_unsupported_op_dict(self.pytorch_version)
        if unsupported_third_party_file_list:
            for file_path in unsupported_third_party_file_list:
                self.unsupported_op_dict.update(utils.read_unsupported_op_csv(file_path))
        self.supported_op_dict = utils.get_supported_op_dict(self.pytorch_version)
        self.global_reference_visitor = None
        self.package_env_path_set = self._search_package_env_path()

    @staticmethod
    def __need_analysis(file, commonprefix):
        return utils.check_file_need_analysis(file, commonprefix, record=True)

    @staticmethod
    def _analysis_file(file, commonprefix):
        raise NotImplementedError()

    def init_global_visitor(self, global_reference_visitor):
        self.global_reference_visitor = global_reference_visitor

    def set_py_file_counts(self, py_file_counts):
        self.py_file_counts = py_file_counts

    def run(self):
        translog.info('Analysis start...')
        self._analysis_dir()

    def _analysis_dir(self):
        count = 0
        translog.set_progress_info(f'[Progress:{count / self.py_file_counts * 100:6.2f}%]')
        for root, _, files in os.walk(self.script_dir):
            for current_file in files:
                file = os.path.join(root, current_file)
                if not self.__need_analysis(file, self.script_dir):
                    continue
                self._analysis_file(file, self.script_dir)
                count += 1
                translog.set_progress_info(f'[Progress:{count / self.py_file_counts * 100:6.2f}%]')

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
