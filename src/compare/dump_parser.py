# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
"""
Function:
This class mainly involves the main function.
"""

import argparse
import sys
import time

from cmp_utils import log, file_utils
from cmp_utils.utils import safe_path_string
from cmp_utils.constant.compare_error import CompareError
from dump_parse.dump_data_parser import DumpDataParser


def _save_log_parser(save_log_parser: argparse.ArgumentParser) -> None:
    save_log_parser.add_argument(
        '-d', '--dump_file', dest='dump_path', default='', type=safe_path_string,
        help='<Required> the dump file path, supports one AICPU custom operator dump file.',
        required=True)
    save_log_parser.add_argument('-out', '--output', dest='output_path', default='', type=safe_path_string,
                                 help='<Optional> the output path')


def _do_cmd() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='commands')
    save_log_parser = subparsers.add_parser('save_log', help='Save AICPU custom operator log.')

    _save_log_parser(save_log_parser)

    args = parser.parse_args(sys.argv[1:])
    if len(sys.argv) < 2:
        parser.print_help()
        raise CompareError(CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    if sys.argv[1] == 'save_log':
        args.dump_version = 2
        args.output_file_type = None
        ret = _do_save_log(args)
        return ret
    else:
        return 0


def _do_save_log(args: argparse.Namespace) -> int:
    ret = DumpDataParser(args).parse_log_data()
    return ret


def main() -> None:
    """
    parse argument and run command
    :return:
    """
    start = time.time()
    with file_utils.UmaskWrapper():
        try:
            ret = _do_cmd()
        except CompareError as err:
            ret = err.code
        except Exception as base_err:
            log.print_error_log(f'Basic error running {sys.argv[0]}: {base_err}')
            sys.exit(1)
    end = time.time()
    if ret != 0:
        log.print_error_log("Failed to parse dump log.")
    log.print_info_log(
        'The command was completed and took %d seconds.' % (end - start))


if __name__ == '__main__':
    main()
