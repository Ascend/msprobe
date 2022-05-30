#!/usr/bin/env python
# coding=utf-8
"""
Function:
Make advisor, perform comparative analysis, This class mainly involves the main function.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2021-2022
"""

import os
import sys
import argparse
import pandas as pd

import log

from advisor.advisor_const import AdvisorConst
from advisor.advisor_result import AdvisorResult
from compare_error import CompareError
from advisor.input_advisor import InputAdvisor
from advisor.node_advisor import NodeAdvisor
from advisor.overflow_advisor import OverflowAdvisor


class CompareAdvisor:
    """
    Class for generate advisor
    """

    def __init__(self, input_file, input_nodes=[], out_path=""):
        self.input_file = input_file
        self.input_nodes = input_nodes
        self.out_path = out_path

    def advisor(self):
        analyze_data = self.parse_input_file()
        log.print_info_log('Start analyzing the comparison results: "%s" .' % self.input_file)
        advisor_result = AdvisorResult()
        advisor_result = self.overflow_check(advisor_result, analyze_data)
        advisor_result = self.input_check(advisor_result, analyze_data)
        advisor_result = self.net_nodes_check(advisor_result, analyze_data)
        log.print_info_log("Comparison result analysis is over.")
        return advisor_result

    @staticmethod
    def overflow_check(advisor_result, analyze_data):
        if not advisor_result.match_advisor:
            overflow_advisor = OverflowAdvisor(analyze_data, advisor_result)
            advisor_result = overflow_advisor.start_analyze()
            log.print_info_log("End analysis of operator overflow problem.")
        return advisor_result

    def input_check(self, advisor_result, analyze_data):
        if not advisor_result.match_advisor and self.input_nodes:
            input_advisor = InputAdvisor(analyze_data, advisor_result, self.input_nodes)
            advisor_result = input_advisor.start_analyze()
            log.print_info_log("End analysis of input nodes precision problem.")
        return advisor_result

    @staticmethod
    def net_nodes_check(advisor_result, analyze_data):
        if not advisor_result.match_advisor:
            node_advisor = NodeAdvisor(analyze_data, advisor_result)
            advisor_result = node_advisor.start_analyze()
            log.print_info_log("End analysis of net nodes precision problem.")
        return advisor_result

    def parse_input_file(self):
        if self.input_file.endswith(".csv"):
            try:
                df = pd.read_csv(self.input_file)
                data_columns = df.columns.values
                if not {AdvisorConst.INDEX, AdvisorConst.NPUDump}.issubset(data_columns):
                    log.print_error_log('Input csv file does not contain %s, %s columns.'
                                        % (AdvisorConst.INDEX, AdvisorConst.NPUDump))
                    raise CompareError(CompareError.MSACCUCMP_INVALID_FILE_ERROR)
                return df
            except OSError as os_err:
                log.print_error_log('Failed to parse the input file %s. %s'
                                    % (self.input_file, str(os_err)))
                raise CompareError(CompareError.MSACCUCMP_OPEN_FILE_ERROR)
        else:
            log.print_error_log("Advisor only support csv file from msaccucmp result.")
            raise CompareError(CompareError.MSACCUCMP_INVALID_FILE_ERROR)


def parse_input_nodes(input_nodes):
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
                             " E.g: node_name1;node_name2;node_name3", required=False)
    parser.add_argument("-o", "--out_path", dest="out_path", default="",
                        help="<optional> The compare advice out path.",
                        required=False)


def do_advisor():
    parser = argparse.ArgumentParser()
    _compare_advisor_parser(parser)
    args = parser.parse_args(sys.argv[1:])
    input_file = os.path.realpath(args.input_file)
    input_nodes = parse_input_nodes(args.input_nodes)
    out_path = ""
    if args.out_path:
        out_path = os.path.realpath(args.out_path)
    compare_advisor = CompareAdvisor(input_file, input_nodes, out_path)
    advisor_result = compare_advisor.advisor()
    message_list = advisor_result.print_advisor_log()
    if out_path:
        advisor_result.gen_summary_file(out_path, message_list)


if __name__ == '__main__':
    try:
        do_advisor()
    except CompareError as err:
        sys.exit(err.code)
    finally:
        pass
    log.print_info_log("Advisor completed.")
    sys.exit(CompareError.MSACCUCMP_NONE_ERROR)

