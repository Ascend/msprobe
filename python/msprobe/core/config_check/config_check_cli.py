# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
import os
from msprobe.core.config_check.config_checker import ConfigChecker
from msprobe.core.config_check.ckpt_compare.ckpt_comparator import compare_checkpoints
from msprobe.core.config_check.verl_param_compare.verl_log_filter import (
    verl_filter_config_info,
    verl_get_config_file_path,
    check_log_extension,
)
from msprobe.core.config_check.verl_param_compare.utils import check_yaml_extension
from msprobe.core.common.file_utils import create_directory
from msprobe.core.config_check.verl_param_compare.verl_hyper_params_cmp import verl_compare_hyper_params
from msprobe.core.config_check.verl_param_compare.verl_hyper_params_verify import verl_verify_hyper_params
from msprobe.core.common.log import logger


def pack(shell_path, output_path):
    ConfigChecker(shell_path=shell_path, output_zip_path=output_path)


def compare(bench_zip_path, cmp_zip_path, output_path):
    ConfigChecker.compare(bench_zip_path, cmp_zip_path, output_path)


def _get_verl_verify_error_message(verl_verify_args):
    """Return error message when verl-verify arguments are invalid, None when valid."""
    n_args = len(verl_verify_args)
    if n_args > 2:
        return f"verl-verify supports up to two files, but received {n_args} files"

    if n_args == 1:
        if check_log_extension(verl_verify_args[0]):
            return None
        if check_yaml_extension(verl_verify_args[0]):
            return (
                "verl-verify requires tgt_log (log or txt file) as the mandatory argument. "
                "When providing only one file, it must be a log or txt file, not yaml."
            )
        return (
            "verl-verify requires tgt_log in log or txt format. "
            "Optionally, provide bench_config (yaml) as the first argument "
            "and tgt_log (log or txt) as the second argument."
        )

    first, second = verl_verify_args[0], verl_verify_args[1]
    first_is_yaml = check_yaml_extension(first)
    first_is_log = check_log_extension(first)
    second_is_yaml = check_yaml_extension(second)
    second_is_log = check_log_extension(second)

    if first_is_yaml and second_is_log:
        return None

    if first_is_log and second_is_yaml:
        return (
            "verl-verify parameter order error: the first argument must be bench_config "
            "(yaml file), and the second argument must be tgt_log (log or txt file). "
            f"Received '{first}' as first and '{second}' as second."
        )

    if first_is_yaml and not second_is_log:
        return (
            "verl-verify requires tgt_log (log or txt file) as the second argument "
            f"when bench_config (yaml) is provided as the first argument. "
            f"Received '{second}' with unsupported file extension."
        )

    if second_is_log and not first_is_yaml:
        return (
            "verl-verify requires bench_config (yaml file) as the first argument "
            f"when two files are provided. Received '{first}' with unsupported file extension."
        )

    return (
        "verl-verify requires tgt_log (log or txt file) as mandatory argument. "
        "When providing two files, the first must be bench_config (yaml file) "
        "and the second must be tgt_log (log or txt file)."
    )


def _config_checking_parser(parser):
    group = parser.add_argument_group('Select one of the following operations, multiple selections are not permitted.')
    mutex = group.add_mutually_exclusive_group(required=False)
    mutex.add_argument('-d', '--dump', nargs='*', help='Collect the train config into a zip file')
    mutex.add_argument('-c', '--compare', nargs=2, help='Compare two zip files or checkpoints')
    mutex.add_argument(
        '-vc',
        '--verl-compare',
        nargs=2,
        help='Compare the parameter info in the configuration file filtered from the verl train logs for NPU and bench,'
        'the first argument is the log to be compared(eg:NPU_log), and the second argument is the bench log(eg:bench_log).',
    )
    mutex.add_argument(
        '-vv',
        '--verl-verify',
        nargs='+',
        help='Verify the parameter info in the configuration file for target and bench, the first argument is an optional '
        'benchmark configuration file(eg:bench_yaml), and the second argument is a mandatory target log that must be passed in(eg:target_log)',
    )
    parser.add_argument(
        '-o',
        '--output',
        help='output path, default is ./config_check_pack.zip for dump mode and'
        ' ./config_check_result for compare mode and'
        ' ./verl_param_compare_result a folder for verl compare mode.'
        ' ./verl_param_verify_result for verl verify mode',
    )


def _run_config_checking_command(args):
    if args.dump is not None:
        output_dirpath = args.output if args.output else "./config_check_pack.zip"
        pack(args.dump, output_dirpath)
    elif args.compare:
        if args.compare[0].endswith('zip'):
            logger.info('The input paths is zip files, comparing packed config.')
            output_dirpath = args.output if args.output else "./config_check_result"
            compare(args.compare[0], args.compare[1], output_dirpath)
        else:
            logger.info('Comparing model checkpoint.')
            output_dirpath = args.output if args.output else "./ckpt_similarity.json"
            compare_checkpoints(args.compare[0], args.compare[1], output_dirpath)
    elif args.verl_compare:
        if check_log_extension(args.verl_compare[0]) and check_log_extension(args.verl_compare[1]):
            output_dirpath = args.output if args.output else "./verl_param_compare_result"
            npu_config_file, bench_config_file = verl_get_config_file_path(output_dirpath)
            verl_filter_config_info(args.verl_compare[0], npu_config_file)
            verl_filter_config_info(args.verl_compare[1], bench_config_file)
            verl_compare_hyper_params(npu_config_file, bench_config_file, output_dirpath)
        else:
            ext_err_msg = (
                "The param of verl-compare require two log files, and the file format just support '.log' or '.txt'."
            )
            logger.error(ext_err_msg)
            raise Exception(ext_err_msg)  # pylint: disable=broad-exception-raised
    elif args.verl_verify:
        verl_verify_error = _get_verl_verify_error_message(args.verl_verify)
        if verl_verify_error is None:
            output_dirpath = args.output if args.output else "./verl_param_verify_result"
            real_file_folder = os.path.realpath(output_dirpath)
            if not os.path.isdir(real_file_folder):
                create_directory(real_file_folder)
            tgt_config = os.path.join(output_dirpath, "tgt_config.json")
            verl_filter_config_info(args.verl_verify[-1], tgt_config)
            core_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            default_yaml_path = os.path.join(
                core_dir, "config_check", "verl_param_compare", "verl_hyper_params_verify.yaml"
            )
            bench_config = default_yaml_path if len(args.verl_verify) == 1 else args.verl_verify[0]
            verl_verify_hyper_params(bench_config, tgt_config, output_dirpath)
        else:
            logger.error(verl_verify_error)
            raise Exception(verl_verify_error)  # pylint: disable=broad-exception-raised
    else:
        logger.error(
            "The param is not correct, you need to give '-d' for dump or '-c' for compare "
            "or '-vc' for verl compare or '-vv' for verl verify."
        )
        # pylint: disable=broad-exception-raised
        raise Exception(
            "The param is not correct, you need to give '-d' for dump or '-c' for compare "
            "or '-vc' for verl compare or '-vv' for verl verify."
        )
