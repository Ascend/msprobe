# -*- coding: utf-8 -*-
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
import subprocess  # nosec
import sys

from msprobe.core.common.log import logger
from msprobe.core.compare.utils import print_compare_ends_info
from msprobe.infer.offline.compare.msquickcmp.adapter_cli.args_adapter import CmpArgsAdapter
from msprobe.infer.offline.compare.msquickcmp.cmp_process import cmp_process
from msprobe.infer.offline.compare.msquickcmp.common.args_check import (
    check_model_path_legality,
    check_output_path_legality,
    check_dict_kind_string,
    check_rank_range_valid,
    check_number_list,
    check_dym_range_string,
    str2bool,
    check_input_data_path,
)
from msprobe.infer.offline.compare.msquickcmp.dump.args_adapter import DumpArgsAdapter
from msprobe.infer.offline.compare.msquickcmp.dump.dump_process import dump_process
from msprobe.core.common.file_utils import check_file_or_directory_path

CANN_PATH = os.environ.get('ASCEND_TOOLKIT_HOME', "/usr/local/Ascend/ascend-toolkit/latest")


def _offline_dump_parser(parser):
    parser.add_argument(
        '--model_path',
        required=True,
        dest="model_path",
        type=check_model_path_legality,
        help='The original model .onnx or .om file path',
    )
    parser.add_argument(
        '--input_data',
        default='',
        dest="input_data",
        type=check_input_data_path,
        help='The input data path of the model. Separate multiple inputs with commas(,). E.g: input_0.bin,input_1.bin',
    )
    parser.add_argument(
        '-o',
        '--output_path',
        dest="output_path",
        default='./output',
        type=check_output_path_legality,
        help='The output path',
    )
    parser.add_argument(
        '--input_shape',
        type=check_dict_kind_string,
        dest="input_shape",
        default='',
        help="Shape of input shape. Separate multiple nodes with semicolons(;)."
        " E.g: \"input_name1:1,224,224,3;input_name2:3,300\"",
    )
    parser.add_argument('--rank', type=check_rank_range_valid, dest="rank", default='0', help='Input rank ID [0, 255].')
    parser.add_argument(
        '--dym_shape_range',
        type=check_dym_range_string,
        dest="dym_shape_range",
        default='',
        help="Dynamic shape range using in dynamic model, "
        "using this means ignore input_shape"
        " E.g: \"input_name1:1,3,200~224,224-230;input_name2:1,300\"",
    )
    parser.add_argument(
        '--output_size',
        dest="output_size",
        default='',
        help='The size of output. Separate multiple sizes with commas(,). E.g: 10200,34000',
    )
    parser.add_argument(
        '--onnx_fusion_switch',
        dest="onnx_fusion_switch",
        default=True,
        type=str2bool,
        help='Onnxruntime fusion switch, set False for dump complete onnx data when '
        'necessary.Usage: --onnx_fusion_switch False',
    )


def offline_dump_cli(args):
    dump_args = DumpArgsAdapter(
        args.model_path,
        input_data=args.input_data,
        cann_path=CANN_PATH,
        output_path=args.output_path,
        input_shape=args.input_shape,
        rank=args.rank,
        dym_shape_range=args.dym_shape_range,
        onnx_fusion_switch=args.onnx_fusion_switch,
        output_size=args.output_size,
    )
    dump_process(dump_args)


def compare_offline_model_mode(args):
    args = set_args_default(args)
    check_compare_args(args)

    if not args.golden_path:
        logger.error("The following args are required: -gp/--golden_path")
        return
    cmp_args = CmpArgsAdapter(
        args.golden_path,
        args.target_path,
        args.input_data,
        CANN_PATH,
        args.output_path,
        args.input_shape,
        args.rank,
        args.output_size,
        args.dym_shape_range,
        args.onnx_fusion_switch,
    )
    cmp_process(cmp_args)
    print_compare_ends_info()


def set_args_default(args):
    if not hasattr(args, 'rank') or not args.rank:
        args.rank = '0'
    if not hasattr(args, 'input_data'):
        args.input_data = ''
    if not hasattr(args, 'input_shape'):
        args.input_shape = ''
    if not hasattr(args, 'output_size'):
        args.output_size = ''
    if not hasattr(args, 'dym_shape_range'):
        args.dym_shape_range = ''
    if not hasattr(args, 'onnx_fusion_switch'):
        args.onnx_fusion_switch = True
    return args


def check_compare_args(args):
    if args.target_path:
        check_file_or_directory_path(args.target_path, False, False, [".om"])
    if args.golden_path:
        check_file_or_directory_path(args.golden_path, False, False, [".onnx", ".om"])
    if args.output_path:
        args.output_path = check_output_path_legality(args.output_path)
    if args.input_data:
        args.input_data = check_input_data_path(args.input_data)
    if args.input_shape:
        args.input_shape = check_dict_kind_string(args.input_shape)
    if args.rank:
        args.rank = check_rank_range_valid(args.rank)
    if args.output_size:
        args.output_size = check_number_list(args.output_size)
    if args.dym_shape_range:
        args.dym_shape_range = check_dym_range_string(args.dym_shape_range)
    if args.onnx_fusion_switch:
        args.onnx_fusion_switch = str2bool(args.onnx_fusion_switch)


def _install_offline_deps_parser(parser):
    parser.add_argument(
        "--no_check",
        dest="no_check",
        action="store_true",
        help="<optional> Whether to skip checking the target website's certificate information "
        "when installing aclruntime and ais_bench poses a certain security risk. "
        "This poses a certain security risk, "
        "and users should use it with caution and bear the consequences themselves.",
        required=False,
    )


def install_offline_deps_cli(args):
    pip_install_onnx_cmd = [sys.executable, "-m", "pip", "install", "onnx>=1.14.0", "onnxruntime>=1.14.1,!=1.16.0"]
    subprocess.run(pip_install_onnx_cmd, shell=False, check=False)  # nosec

    offline_extra_install_cmd = [
        "/bin/bash",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "install_aclruntime_aisbench.sh")),
        str(args.no_check),
    ]
    subprocess.run(offline_extra_install_cmd, shell=False, check=False)  # nosec
