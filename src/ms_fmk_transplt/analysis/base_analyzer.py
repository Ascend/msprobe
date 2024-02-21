#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
import os
from pathlib import Path

from utils import trans_utils as utils, transplant_logger as translog
from analysis.precision_performance_advice_analysis.prec_perf_utils import PerfApiSuggest
from analysis.precision_performance_advice_analysis.prec_perf_utils import AdviceInfo
from analysis.precision_performance_advice_analysis.precision_performance_advice_visitor import generate_perf_suggest


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
        self.package_env_path_set = utils.search_package_env_path(self.script_dir)
        self.result_dict = {}
        # init dict for precision and performance advice analyse
        prec_perf_advice_dict = utils.parse_precision_performance_advice_file()
        api_prec_dict = prec_perf_advice_dict.get("api_precision_dict", {})
        api_perf_dict = prec_perf_advice_dict.get("api_performance_dict", {})
        api_params_perf_dict = prec_perf_advice_dict.get("api_parameters_performance_dict", {})
        perf_api_suggest_dict = prec_perf_advice_dict.get("performance_api_suggest_use", {})
        perf_api_suggest = PerfApiSuggest(perf_api_suggest_dict)
        self.perf_config_dict = prec_perf_advice_dict.get("performance_configuration_dict", {})
        self.advice_info = AdviceInfo(api_prec_dict, api_perf_dict, api_params_perf_dict, perf_api_suggest)

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

    def gen_perf_suggest_use(self):
        # Give performance suggestion about the api not used.
        suggest_list = generate_perf_suggest(self.advice_info.perf_api_suggest)
        self.result_dict.update({'api_performance_advice.csv': self.result_dict.get(
            'api_performance_advice.csv', 0) + len(suggest_list)})
        utils.write_csv(
            self._get_content_list(suggest_list, with_file=False),
            self.output_path,
            "api_performance_advice",
            ('File', 'Start Line', 'End Line', 'OP', 'Tips')
        )

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

    def _get_content_list(self, result_list, with_file=True):
        if with_file:
            result = [(self.current_file_rel_path, api.start_line, api.end_line,
                        api.name, api.info) for api in result_list]
        else:
            result = [("NA", api.start_line, api.end_line,
                        api.name, api.info) for api in result_list]
        return result
