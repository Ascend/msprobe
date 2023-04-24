
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2012-2022. All rights reserved.
"""
Function:
Make advisor, perform comparative analysis, This class mainly involves the main function.
"""

import os
import sys
import argparse
from cmp_utils import log
from cmp_utils.constant.compare_error import CompareError
from cmp_utils.constant.const_manager import ConstManager
from advisor.compare_advisor import CompareAdvisor


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

