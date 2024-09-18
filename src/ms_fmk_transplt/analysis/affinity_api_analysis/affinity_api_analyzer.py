#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

import os
import libcst

from analysis.base_analyzer import BaseAnalyzer
from utils import trans_utils as utils
from utils import transplant_logger as translog
from .affinity_api_visitor import analyse_affinity_api, AffinityInfo


class AffinityApiAnalyzer(BaseAnalyzer):
    def __init__(self, script_dir, output_path, pytorch_version, unsupported_third_party_file_list=None):
        super().__init__(script_dir, output_path, pytorch_version, unsupported_third_party_file_list)
        self.affinity_api_dict = {}
        self.affinity_api_call_list = []
        self.affinity_info = AffinityInfo([], [], [])
        self.current_file_rel_path = ''

    def run(self):
        super().run()
        self.write_csv()

    def write_csv(self):
        affinity_api_call_set = set()
        for call in self.affinity_api_call_list:
            if call.full_name.startswith('torch') or call.api_type == 'special':
                affinity_api_call_set.add(call)
                continue
            call_full_name_list = call.full_name.split('.')
            if len(call_full_name_list) < 2:
                continue
            call_full_name = call_full_name_list[-2] + '.' + call_full_name_list[-1]
            if call_full_name in self.affinity_api_dict.keys():
                call.info = self.affinity_api_dict.get(call_full_name)
                affinity_api_call_set.add(call)

        utils.write_csv(list((api.file_path, api.start_line, api.end_line, api.api_type, api.name, api.info)
                             for api in affinity_api_call_set), self.output_path, "affinity_api_call",
                        ('File', 'Start Line', 'End Line', 'Api Type', 'Api Call Name', 'Affinity Api Name'))
        self.result_dict.update({'affinity_api_call.csv': len(affinity_api_call_set)})

    def collect_affinity_analysis_results(self):
        if self.affinity_info.affinity_api_def_list:
            for api in self.affinity_info.affinity_api_def_list:
                full_name_list = api.full_name.split('.')
                if len(full_name_list) < 2:
                    continue
                full_name = full_name_list[-2] + '.' + full_name_list[-1]
                self.affinity_api_dict[full_name] = api.info
        if self.affinity_info.affinity_api_call_list:
            for api in self.affinity_info.affinity_api_call_list:
                api.file_path = self.current_file_rel_path
                self.affinity_api_call_list.append(api)
        if self.affinity_info.affinity_special_list:
            for api in self.affinity_info.affinity_special_list:
                api.file_path = self.current_file_rel_path
                self.affinity_api_call_list.append(api)

    def _analysis_file(self, file, commonprefix):
        if self.global_reference_visitor:
            self.global_reference_visitor.visit_file(os.path.relpath(file, self.script_dir))
        self.current_file_rel_path = os.path.relpath(file, commonprefix)
        info_msg = f'Start the analysis of {self.current_file_rel_path}.'
        translog.info(info_msg)
        self._analysis_code(file)
        info_msg = f'Analysis of {self.current_file_rel_path} completed.'
        translog.info(info_msg)

    def _analysis_code(self, file):
        code = utils.get_file_content_bytes(file)
        try:
            wrapper = libcst.metadata.MetadataWrapper(libcst.parse_module(code))
        except BaseException:
            translog.warning(f'{file} has unsupported python syntax, skip.')
            return
        self.affinity_info = analyse_affinity_api(wrapper, self.pytorch_version, self.global_reference_visitor)
        self.collect_affinity_analysis_results()
