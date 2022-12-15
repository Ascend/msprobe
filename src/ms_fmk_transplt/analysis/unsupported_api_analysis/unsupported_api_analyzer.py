#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
import os

from analysis.base_analyzer import BaseAnalyzer
from utils import trans_utils as utils, transplant_logger as translog
from .api_visitor import analyse_unsupported_api, OpInfo
from .cuda_cpp_visitor import analyse_cuda_ops


class UnsupportedApiAnalyzer(BaseAnalyzer):
    def __init__(self, script_dir, output_path, pytorch_version, unsupported_third_party_file_list=None):
        super(UnsupportedApiAnalyzer, self).__init__(script_dir, output_path, pytorch_version,
                                                     unsupported_third_party_file_list)
        self.cuda_op_list = analyse_cuda_ops(script_dir, output_path)

    def _analysis_file(self, file, commonprefix):
        if self.global_reference_visitor:
            self.global_reference_visitor.visit_file(os.path.relpath(file, self.script_dir))
        self.current_file_rel_path = os.path.relpath(file, commonprefix)
        translog.info(f'Start analysis {self.current_file_rel_path}.')
        self._analysis_code(file)
        translog.info(f'Analysis {self.current_file_rel_path} complete.')

    def _analysis_code(self, file):
        code = utils.get_file_content_bytes(file)
        unsupported_op_list, unknown_op_list, _, _ = analyse_unsupported_api(
            code, OpInfo(self.supported_op_dict, self.unsupported_op_dict, self.cuda_op_list),
            self.global_reference_visitor)
        utils.write_csv(list((self.current_file_rel_path, api.start_line, api.end_line, api.name, api.info)
                             for api in unsupported_op_list), self.output_path, "unsupported_api",
                        ('File', 'Start Line', 'End Line', 'OP', 'Tips'))
        utils.write_csv(list((self.current_file_rel_path, api.start_line, api.end_line, api.name)
                             for api in unknown_op_list), self.output_path, "unknown_api",
                        ('File', 'Start Line', 'End Line', 'OP', 'Tips'))
