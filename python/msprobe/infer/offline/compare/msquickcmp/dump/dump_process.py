# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
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
# pylint: disable=duplicate-code

"""
Function:
This class mainly dump model ops inputs and outputs.
"""

import os
import time

from msprobe.core.common.file_utils import check_file_or_directory_path, create_directory
from msprobe.core.common.log import logger
from msprobe.core.common.const import Const
from msprobe.infer.offline.compare.msquickcmp.atc import atc_utils
from msprobe.infer.offline.compare.msquickcmp.common import utils
from msprobe.infer.offline.compare.msquickcmp.common.convert import convert_npy_to_bin
from msprobe.infer.offline.compare.msquickcmp.common.utils import AccuracyCompareException, get_shape_to_directory_name
from msprobe.infer.offline.compare.msquickcmp.dump.args_adapter import DumpArgsAdapter
from msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data import NpuDumpData
from msprobe.infer.offline.compare.msquickcmp.npu.om_parser import OmParser


def dump_process(args: DumpArgsAdapter):
    """
    Function Description:
        main process function
    Exception Description:
        exit the program when an AccuracyCompare Exception  occurs
    """
    args.golden_path = os.path.realpath(args.golden_path)
    args.cann_path = os.path.realpath(args.cann_path)
    args.input_data = convert_npy_to_bin(args.input_data)
    check_and_dump(args)
    logger.info(Const.TOOL_ENDS_SUCCESSFULLY)


def check_and_dump(args):
    check_file_or_directory_path(args.golden_path, False)
    utils.check_device_param_valid(args.rank)
    time_dir = time.strftime("%Y%m%d%H%M%S", time.localtime())
    original_out_path = os.path.realpath(os.path.join(args.output_path, time_dir))
    args.output_path = original_out_path
    create_directory(args.output_path)
    # deal with the dymShape_range param if exists
    input_shapes = []
    if args.dym_shape_range:
        input_shapes = utils.parse_dym_shape_range(args.dym_shape_range)
    if not input_shapes:
        input_shapes.append("")
    for input_shape in input_shapes:
        dump_data(args, input_shape, original_out_path)


def dump_data(args, input_shape, original_out_path):
    if input_shape:
        args.input_shape = input_shape
        args.output_path = os.path.join(original_out_path, get_shape_to_directory_name(args.input_shape))

    _, extension = utils.get_model_name_and_extension(args.golden_path)
    if extension == ".onnx":
        from msprobe.infer.offline.compare.msquickcmp.onnx_model.onnx_dump_data import OnnxDumpData

        dumper = OnnxDumpData(args, npu_dump_npy_path=None)
        dumper.generate_inputs_data(npu_dump_data_path=None, use_aipp=False)
        dumper.generate_dump_data()
    elif extension == ".om":
        output_json_path = atc_utils.convert_model_to_json(args.cann_path, args.golden_path, args.output_path)
        om_parser = OmParser(output_json_path)
        use_aipp = True if om_parser.get_aipp_config_content() else False  # pylint: disable=simplifiable-if-expression

        args.target_path = args.golden_path
        dumper = NpuDumpData(args, is_golden=False)
        dumper.generate_inputs_data(use_aipp=use_aipp)
        dumper.generate_dump_data()
    else:
        logger.error("Dump model files whose names end with .onnx or .om are supported, please check your model type.")
        raise AccuracyCompareException(utils.ACCURACY_COMPARISON_MODEL_TYPE_ERROR)
