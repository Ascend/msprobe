#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

import os
import libcst as cst

from analysis.base_analyzer import BaseAnalyzer
from utils import trans_utils as utils
from utils import transplant_logger as translog
from .dynamic_shape_converter import DynamicShapeTransformer


class DynamicShapeAnalyzer(BaseAnalyzer):
    def __init__(self, script_dir, output_path, pytorch_version, unsupported_third_party_file_list=None):
        super().__init__(script_dir, output_path, pytorch_version, unsupported_third_party_file_list)

    def _analysis_file(self, file, commonprefix):
        if self.global_reference_visitor:
            self.global_reference_visitor.visit_file(os.path.relpath(file, self.output_path))
        self.current_file_rel_path = os.path.relpath(file, commonprefix)
        translog.info(f'Start the analysis of {self.current_file_rel_path}.')
        self._analysis_code(file)
        translog.info(f'Analysis of {self.current_file_rel_path} completed.')

    def _analysis_code(self, file):
        code = utils.get_file_content_bytes(file)
        try:
            wrapper = cst.metadata.MetadataWrapper(cst.parse_module(code))
        except Exception:
            translog.warning(f'{file} has unsupported python syntax, skip.')
            return
        module = wrapper.visit(DynamicShapeTransformer())
        code = module.code
        utils.write_file_content(file, code)
