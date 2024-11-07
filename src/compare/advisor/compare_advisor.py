
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
"""
Function:
Make advisor, perform comparative analysis, This class mainly involves the main function.
"""

import pandas as pd

from cmp_utils import log
from cmp_utils.constant.compare_error import CompareError
from advisor.advisor_const import AdvisorConst
from advisor.advisor_result import AdvisorResult
from advisor.input_advisor import InputAdvisor
from advisor.node_advisor import NodeAdvisor
from advisor.overflow_advisor import OverflowAdvisor


class CompareAdvisor:
    """
    Class for generate advisor
    """

    def __init__(self, input_file, input_nodes=None, out_path=""):
        self.input_file = input_file
        self.input_nodes = input_nodes
        self.out_path = out_path

    @staticmethod
    def _overflow_check(advisor_result, analyze_data):
        if not advisor_result.match_advisor:
            overflow_advisor = OverflowAdvisor(analyze_data, advisor_result)
            advisor_result = overflow_advisor.start_analyze()
            if advisor_result.match_advisor:
                log.print_info_log("The FP16 Overflow detection matches successfully.")
            log.print_info_log("End FP16 Overflow detection.")
        return advisor_result

    @staticmethod
    def _net_nodes_check(advisor_result, analyze_data):
        if not advisor_result.match_advisor:
            node_advisor = NodeAdvisor(analyze_data, advisor_result)
            advisor_result = node_advisor.start_analyze()
            if advisor_result.match_advisor:
                log.print_info_log("The Global Consistency detection matches successfully.")
            log.print_info_log("End Global Consistency detection.")
        return advisor_result

    def advisor(self):
        analyze_data = self._parse_input_file()
        log.print_info_log('Start analyzing the comparison results: "%r" .' % self.input_file)
        advisor_result = AdvisorResult()
        advisor_result = self._overflow_check(advisor_result, analyze_data)
        advisor_result = self._input_check(advisor_result, analyze_data)
        advisor_result = self._net_nodes_check(advisor_result, analyze_data)
        log.print_info_log("Comparison result analysis is over.")
        return advisor_result

    def _input_check(self, advisor_result, analyze_data):
        if not advisor_result.match_advisor and self.input_nodes:
            input_advisor = InputAdvisor(analyze_data, advisor_result, self.input_nodes)
            advisor_result = input_advisor.start_analyze()
            if advisor_result.match_advisor:
                log.print_info_log("The Input Inconsistent detection matches successfully.")
            log.print_info_log("End Input Inconsistent detection.")
        return advisor_result

    def _parse_input_file(self):
        if self.input_file.endswith(".csv"):
            try:
                df = pd.read_csv(self.input_file, on_bad_lines='skip')
            except (OSError, SystemError, ValueError, TypeError, RuntimeError, MemoryError) as io_err:
                log.print_error_log('Failed to parse the input file %r. %s'
                                    % (self.input_file, str(io_err)))
                raise CompareError(CompareError.MSACCUCMP_OPEN_FILE_ERROR) from io_err
            data_columns = df.columns.values
            if not {AdvisorConst.INDEX, AdvisorConst.NPU_DUMP}.issubset(data_columns):
                log.print_error_log('Input csv file does not contain %s, %s columns.'
                                    % (AdvisorConst.INDEX, AdvisorConst.NPU_DUMP))
                raise CompareError(CompareError.MSACCUCMP_INVALID_FILE_ERROR)
            return df
        else:
            log.print_error_log("Advisor only support csv file from msaccucmp result.")
            raise CompareError(CompareError.MSACCUCMP_INVALID_FILE_ERROR)
