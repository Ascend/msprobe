#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2012-2022. All rights reserved.
"""
Function:
Make advisor, perform comparative analysis, This class mainly involves the main function.
"""

import os
import sys
import argparse
import pandas as pd

from src.compare.cmp_utils import log
from src.compare.cmp_utils.constant.compare_error import CompareError
from src.compare.cmp_utils.constant.const_manager import ConstManager
from src.compare.advisor.advisor_const import AdvisorConst
from src.compare.advisor.advisor_result import AdvisorResult
from src.compare.advisor.input_advisor import InputAdvisor
from src.compare.advisor.node_advisor import NodeAdvisor
from src.compare.advisor.overflow_advisor import OverflowAdvisor


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
        log.print_info_log('Start analyzing the comparison results: "%s" .' % self.input_file)
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
            except OSError as os_err:
                log.print_error_log('Failed to parse the input file %s. %s'
                                    % (self.input_file, str(os_err)))
                raise CompareError(CompareError.MSACCUCMP_OPEN_FILE_ERROR) from os_err
            data_columns = df.columns.values
            if not {AdvisorConst.INDEX, AdvisorConst.NPU_DUMP}.issubset(data_columns):
                log.print_error_log('Input csv file does not contain %s, %s columns.'
                                    % (AdvisorConst.INDEX, AdvisorConst.NPU_DUMP))
                raise CompareError(CompareError.MSACCUCMP_INVALID_FILE_ERROR)
            return df
        else:
            log.print_error_log("Advisor only support csv file from msaccucmp result.")
            raise CompareError(CompareError.MSACCUCMP_INVALID_FILE_ERROR)


def parse_input_nodes(input_nodes):
    """
    Convert input_nodes string to nodes list
    :param input_nodes: string of input nodes
    """
    nodes_list = []
    if input_nodes:
        nodes = input_nodes.strip().split(";")
        for node in nodes:
            if node.strip():
                nodes_list.append(node.strip())
    return nodes_list


def _compare_advisor_parser(parser):
    parser.add_argument("-i", "--input_file", dest="input_file", default="",
                        help="<Required> The compare result file: generate from msaccucmp compare command, a csv file.",
                        required=True)
    parser.add_argument('-input_nodes', dest="input_nodes", default="",
                        help="<optional> Input nodes designated by user. Separate multiple nodes with semicolons(;)."
                             " E.g: \"node_name1;node_name2;node_name3\"", required=False)
    parser.add_argument("-o", "--out_path", dest="out_path", default="",
                        help="<optional> The compare advice out path.",
                        required=False)


def _do_advisor():
    parser = argparse.ArgumentParser()
    _compare_advisor_parser(parser)
    args = parser.parse_args(sys.argv[1:])
    input_file = os.path.realpath(args.input_file)
    check_file_size(input_file)
    input_nodes = parse_input_nodes(args.input_nodes)
    out_path = os.path.realpath(args.out_path) if args.out_path else ""
    compare_advisor = CompareAdvisor(input_file, input_nodes, out_path)
    advisor_result = compare_advisor.advisor()
    message_list = advisor_result.print_advisor_log()
    if out_path:
        advisor_result.gen_summary_file(out_path, message_list)


def check_file_size(input_file):
    try:
        file_size = os.path.getsize(input_file)
    except OSError as os_error:
        log.print_error_log('Failed to open "%s". %s' % (input_file, str(os_error)))
        raise CompareError(CompareError.MSACCUCMP_OPEN_FILE_ERROR) from os_error
    if file_size > ConstManager.ONE_HUNDRED_MB:
        log.print_error_log(
            'The size (%d) of %s exceeds 100MB, tools not support.'
            % (file_size, input_file))
        raise CompareError(CompareError.MSACCUCMP_INVALID_FILE_ERROR)


if __name__ == '__main__':
    try:
        _do_advisor()
    except CompareError as err:
        sys.exit(err.code)
    finally:
        pass
    log.print_info_log("Advisor completed.")
    sys.exit(CompareError.MSACCUCMP_NONE_ERROR)

