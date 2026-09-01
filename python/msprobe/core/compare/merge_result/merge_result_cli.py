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

from msprobe.core.compare.merge_result.merge_result import merge_result


def _merge_result_parser(parser):
    parser.add_argument(
        "-i",
        "--input_dir",
        dest="input_dir",
        type=str,
        help="<Required> The compare result path, a directory.",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        dest="output_dir",
        type=str,
        help="<Required> The result merge output path, a directory.",
        required=True,
    )
    parser.add_argument(
        "-config",
        "--config-path",
        dest="config_path",
        type=str,
        help="<Required> YAML path containing distributed APIs and compare indexes for merging data "
        "from compare results.",
        required=True,
    )


def merge_result_cli(args):
    merge_result(args.input_dir, args.output_dir, args.config_path)
