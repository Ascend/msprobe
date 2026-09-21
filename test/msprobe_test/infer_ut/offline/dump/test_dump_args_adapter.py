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

from msprobe.infer.offline.compare.msquickcmp.dump.args_adapter import DumpArgsAdapter

# Test data
VALID_MODEL_PATH = "tests/model"
VALID_INPUT_DATA = "tests/input"
VALID_CANN_PATH = "/usr/local/Ascend/ascend-toolkit/latest"
VALID_OUTPUT_PATH = "tests/output"
VALID_INPUT_SHAPE = "input_shape"
VALID_RANK = "0"
VALID_DYM_SHAPE_RANGE = "dym_shape_range"
VALID_ONNX_FUSION_SWITCH = True
VALID_CUSTOM_OP = "custom_op"
VALID_DUMP = True
VALID_SINGLE_OP = "single_op"
VALID_OUTPUT_SIZE = "test_output_size"


# Test cases
def test_given_valid_arguments_when_init_then_success():
    args = DumpArgsAdapter(
        model_path=VALID_MODEL_PATH,
        input_data=VALID_INPUT_DATA,
        cann_path=VALID_CANN_PATH,
        output_path=VALID_OUTPUT_PATH,
        input_shape=VALID_INPUT_SHAPE,
        rank=VALID_RANK,
        dym_shape_range=VALID_DYM_SHAPE_RANGE,
        onnx_fusion_switch=VALID_ONNX_FUSION_SWITCH,
        custom_op=VALID_CUSTOM_OP,
        dump=VALID_DUMP,
        single_op=VALID_SINGLE_OP,
        output_size=VALID_OUTPUT_SIZE,
    )
    assert args.golden_path == VALID_MODEL_PATH
    assert args.input_data == VALID_INPUT_DATA
    assert args.cann_path == VALID_CANN_PATH
    assert args.output_path == VALID_OUTPUT_PATH
    assert args.input_shape == VALID_INPUT_SHAPE
    assert args.rank == VALID_RANK
    assert args.dym_shape_range == VALID_DYM_SHAPE_RANGE
    assert args.onnx_fusion_switch == VALID_ONNX_FUSION_SWITCH
    assert args.custom_op == VALID_CUSTOM_OP
    assert args.dump == VALID_DUMP
    assert args.single_op == VALID_SINGLE_OP
    assert args.output_size == VALID_OUTPUT_SIZE


def test_given_default_arguments_when_init_then_success():
    args = DumpArgsAdapter(model_path=VALID_MODEL_PATH)
    assert args.golden_path == VALID_MODEL_PATH
    assert args.input_data == ""
    assert args.output_path == "./output"
    assert args.input_shape == ""
    assert args.rank == "0"
    assert args.dym_shape_range == ""
    assert args.onnx_fusion_switch
    assert args.custom_op == ""
    assert args.dump
    assert args.single_op == ""
    assert args.output_size == ""
