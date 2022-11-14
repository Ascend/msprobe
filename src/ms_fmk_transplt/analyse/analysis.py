#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

import os
import libcst
from utils.code_visitor import ApiVisitor
import utils.transplant_logger as translog
import utils.trans_utils as utils


class PytorchAnalyze:
    def __init__(self, script_dir, output_path, pytorch_version):
        self.script_dir = script_dir
        self.output_path = output_path
        self.pytorch_version = pytorch_version
        self.py_file_counts = 0
        self.current_file_rel_path = ''

    @staticmethod
    def __need_analysis(file, commonprefix):
        return utils.check_file_need_analysis(file, commonprefix, record=True)

    def set_py_file_counts(self, py_file_counts):
        self.py_file_counts = py_file_counts

    def run(self):
        translog.info('Analysis start...')
        if os.path.isfile(self.script_dir) and self.__need_analysis(self.script_dir, os.path.dirname(self.script_dir)):
            self.__analysis_file(self.script_dir, os.path.dirname(self.script_dir))
        if os.path.isdir(self.script_dir):
            self.__analysis_dir()

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
        self.current_file_rel_path = os.path.relpath(file, commonprefix)
        translog.info(f'Start analysis {self.current_file_rel_path}.')
        self.__analysis_code(file)
        translog.info(f'Analysis {self.current_file_rel_path} complete.')

    def __analysis_code(self, file):
        code = utils.get_file_content_bytes(file)
        wrapper = libcst.metadata.MetadataWrapper(libcst.parse_module(code))
        api_visitor = ApiVisitor(utils.get_op_list(self.pytorch_version))
        module = wrapper.visit(api_visitor)
        op_list = api_visitor.print_unsupported_ops()
        utils.write_csv(op_list, self.current_file_rel_path, self.output_path, "unsupported_op")
