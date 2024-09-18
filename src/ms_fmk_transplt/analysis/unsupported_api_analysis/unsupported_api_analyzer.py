#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
import os
import libcst

from analysis.base_analyzer import BaseAnalyzer
from utils import trans_utils as utils, transplant_logger as translog
from .unsupported_api_visitor import analyse_unsupported_api, OpInfo
from ..precision_performance_advice_analysis.precision_performance_advice_visitor import \
    analyse_precision_performance_advice_api
from ..precision_performance_advice_analysis.prec_perf_utils import AdviceInfo
from .cuda_cpp_visitor import analyse_cuda_ops


class UnsupportedApiAnalyzer(BaseAnalyzer):
    def __init__(self, script_dir, output_path, pytorch_version, unsupported_third_party_file_list=None):
        super(UnsupportedApiAnalyzer, self).__init__(script_dir, output_path, pytorch_version,
                                                     unsupported_third_party_file_list)
        self.cuda_op_list = analyse_cuda_ops(script_dir, output_path)

    def run(self):
        export_performance_configuration(self.perf_config_dict, self.result_dict, self.output_path)
        super().run()
        self.gen_perf_suggest_use()

    def _analysis_file(self, file, commonprefix):
        if self.global_reference_visitor:
            self.global_reference_visitor.visit_file(os.path.relpath(file, self.script_dir))
        self.current_file_rel_path = os.path.relpath(file, commonprefix)
        translog.info(f'Start the analysis of {self.current_file_rel_path}.')
        self._analysis_code(file)
        translog.info(f'Analysis of {self.current_file_rel_path} completed.')

    def _analysis_code(self, file):
        code = utils.get_file_content_bytes(file)
        try:
            wrapper = libcst.metadata.MetadataWrapper(libcst.parse_module(code))
        except BaseException:
            translog.warning(f'{file} has unsupported python syntax, skip.')
            return
        (unsupported_op_list, unknown_op_list), _, _ = analyse_unsupported_api(wrapper, OpInfo(self.supported_op_dict,
                                                                                               self.unsupported_op_dict,
                                                                                               self.cuda_op_list),
                                                                               self.global_reference_visitor)
        (precision_advice_list, performance_advice_list), _, _ = \
            analyse_precision_performance_advice_api(wrapper, self.advice_info, self.global_reference_visitor)
        result_dicts = {
            'cuda_op_list.csv': self.cuda_op_list,
            'unsupported_api.csv': unsupported_op_list,
            'unknown_api.csv': unknown_op_list,
            'api_precision_advice.csv': precision_advice_list,
            'api_performance_advice.csv': performance_advice_list
        }
        for result_dict in result_dicts.items():
            self.result_dict.update({result_dict[0]: self.result_dict.get(
                result_dict[0], 0) + len(result_dict[1])})
        csv_title = ('File', 'Start Line', 'End Line', 'OP', 'Tips')
        utils.write_csv(self._get_content_list(unsupported_op_list), self.output_path, "unsupported_api",
                        csv_title)
        utils.write_csv(list((self.current_file_rel_path, api.start_line, api.end_line, api.name)
                             for api in unknown_op_list), self.output_path, "unknown_api",
                        csv_title)
        utils.write_csv(self._get_content_list(precision_advice_list), self.output_path, "api_precision_advice",
                        csv_title)
        utils.write_csv(self._get_content_list(performance_advice_list), self.output_path, "api_performance_advice",
                        csv_title)


def export_performance_configuration(configuration_dict, result_dict, output_path):
    result_list = []
    na = 'NA'
    for key, value in configuration_dict.items():
        result_list.append((na, na, na, key, value))
    if result_list:
        result_dict.update({'api_performance_advice.csv': result_dict.get(
            'api_performance_advice.csv', 0) + len(result_list)})
        utils.write_csv(result_list, output_path, "api_performance_advice",
                        ('File', 'Start Line', 'End Line', 'OP', 'Tips'))
