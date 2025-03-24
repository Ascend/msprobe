
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2012-2022. All rights reserved.
"""
Function:
Make advisor, perform comparative analysis, This class mainly involves the main function.
"""

import os
import sys
import argparse
import re
from cmp_utils import log, path_check, file_utils
from cmp_utils.utils import safe_path_string
from cmp_utils.constant.compare_error import CompareError
from cmp_utils.constant.const_manager import ConstManager
from advisor.compare_advisor import CompareAdvisor
MAX_STRING_LENGTH = 1024
NODE_WHITE_LIST_REGEX = re.compile(r"[^_A-Za-z0-9/.,;-]")


def parse_input_nodes(input_nodes):
    """
    Convert input_nodes string to nodes list
    :param input_nodes: string of input nodes
    """
    if not input_nodes:
        return []
    else:
        check_safe_string(input_nodes)
        check_string_length(input_nodes)
        return [node.strip() for node in input_nodes.strip().split(";") if node.strip()]


def _compare_advisor_parser(parser):
    parser.add_argument("-i", "--input_file", dest="input_file", default="", type=safe_path_string,
                        help="<Required> The compare result file: generate from msaccucmp compare command, a csv file.",
                        required=True)
    parser.add_argument('-input_nodes', dest="input_nodes", default="",
                        help="<optional> Input nodes designated by user. Separate multiple nodes with semicolons(;)."
                             " E.g: \"node_name1;node_name2;node_name3\"", required=False)
    parser.add_argument("-o", "--out_path", dest="out_path", default="", type=safe_path_string,
                        help="<optional> The compare advice out path.",
                        required=False)


def _do_advisor():
    parser = argparse.ArgumentParser()
    _compare_advisor_parser(parser)
    args = parser.parse_args(sys.argv[1:])
    input_file = os.path.realpath(args.input_file)
    check_file_size(input_file)
    _check_input_file(input_file, ConstManager.CSV_SUFFIX)
    input_nodes = parse_input_nodes(args.input_nodes)
    if args.out_path:
        if os.path.islink(os.path.abspath(args.out_path)):
            log.print_error_log('The path "%r" is a softlink, not permitted.' % args.out_path)
            raise CompareError(CompareError.MSACCUCMP_INVALID_PATH_ERROR)
        out_path = os.path.realpath(args.out_path) 
    else:
        out_path = ""
    compare_advisor = CompareAdvisor(input_file, input_nodes, out_path)
    advisor_result = compare_advisor.advisor()
    message_list = advisor_result.print_advisor_log()
    if out_path:
        path_check.check_output_path_valid(out_path, exist=True)
        advisor_result.gen_summary_file(out_path, message_list)


def check_file_size(input_file):
    try:
        file_size = os.path.getsize(input_file)
    except OSError as os_error:
        log.print_error_log('Failed to open "%r". %s' % (input_file, str(os_error)))
        raise CompareError(CompareError.MSACCUCMP_OPEN_FILE_ERROR) from os_error
    if file_size > ConstManager.ONE_HUNDRED_MB:
        log.print_error_log('The size (%d) of %r exceeds 100MB, tools not support.' % (file_size, input_file))
        raise CompareError(CompareError.MSACCUCMP_INVALID_FILE_ERROR)


def check_string_length(s):
    byte_length = len(s.encode('utf-8'))
    if byte_length > MAX_STRING_LENGTH:
        log.print_error_log('The length (%d) of %s exceeds 1024, tools not support.' % (byte_length, s))
        raise CompareError(CompareError.MSACCUCMP_INVALID_FILE_ERROR)


def check_safe_string(s):
    if re.search(NODE_WHITE_LIST_REGEX, s):
        raise ValueError("String parameter contains invalid characters.")


def _check_input_file(input_file:str, file_type:str) -> None:
    if not input_file.endswith(file_type):
        log.print_error_log("[file_compare] The file %r is invalid.Only support %r file." % (input_file, file_type))
        raise CompareError(CompareError.MSACCUCMP_INVALID_TYPE_ERROR)
    ret = path_check.check_exec_file_valid(input_file)
    if ret != CompareError.MSACCUCMP_NONE_ERROR:
        raise CompareError(ret)


if __name__ == '__main__':
    with file_utils.UmaskWrapper():
        try:
            _do_advisor()
        except CompareError as err:
            sys.exit(err.code)
        except Exception as base_err:
            log.print_error_log(f'Basic error running {sys.argv[0]}: {base_err}')
            sys.exit(1)

    log.print_info_log("Advisor completed.")
    sys.exit(CompareError.MSACCUCMP_NONE_ERROR)
