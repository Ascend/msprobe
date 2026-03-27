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

from msprobe.core.config_check.config_checker import ConfigChecker
from msprobe.core.config_check.ckpt_compare.ckpt_comparator import compare_checkpoints
from msprobe.core.common.log import logger


def pack(shell_path, output_path):
    ConfigChecker(shell_path=shell_path, output_zip_path=output_path)


def compare(bench_zip_path, cmp_zip_path, output_path):
    ConfigChecker.compare(bench_zip_path, cmp_zip_path, output_path)


def _config_checking_parser(parser):
    parser.add_argument('-d', '--dump', nargs='*', help='Collect the train config into a zip file')
    parser.add_argument('-c', '--compare', nargs=2, help='Compare two zip files or checkpoints')
    parser.add_argument('-o', '--output',
                        help='output path, default is ./config_check_pack.zip for dump mode and'
                             ' ./config_check_result for compare mode')


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

    else:
        logger.error("The param is not correct, you need to give '-d' for dump or '-c' for compare.")
        raise Exception("The param is not correct, you need to give '-d' for dump or '-c' for compare.")
