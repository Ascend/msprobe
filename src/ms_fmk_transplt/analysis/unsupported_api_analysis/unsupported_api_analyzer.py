#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
import os

from utils import trans_utils as utils, transplant_logger as translog
from .api_visitor import get_op_visit_result
from ..base_analyzer import BaseAnalyzer


class UnsupportedApiAnalyzer(BaseAnalyzer):
    def _analysis_file(self, file, commonprefix):
        self.current_file_rel_path = os.path.relpath(file, commonprefix)
        translog.info(f'Start analysis {self.current_file_rel_path}.')
        self._analysis_code(file)
        translog.info(f'Analysis {self.current_file_rel_path} complete.')

    def _analysis_code(self, file):
        code = utils.get_file_content_bytes(file)
        op_list, _, _ = get_op_visit_result(code, self.pytorch_version)
        utils.write_csv(op_list, self.current_file_rel_path, self.output_path, "unsupported_op")
