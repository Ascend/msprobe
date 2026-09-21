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

import os


CANN_PATH = os.environ.get('ASCEND_TOOLKIT_HOME', "/usr/local/Ascend/ascend-toolkit/latest")


class CmpArgsAdapter:
    def __init__(
        self,
        golden_path,
        target_path,
        input_data="",
        cann_path=CANN_PATH,
        output_path="./output",
        input_shape="",
        rank="0",
        output_size="",
        dym_shape_range="",
        onnx_fusion_switch=True,
    ):
        self.golden_path = golden_path
        self.target_path = target_path
        self.input_data = input_data
        self.cann_path = cann_path
        self.output_path = output_path
        self.input_shape = input_shape
        self.rank = rank
        self.output_size = output_size
        self.dym_shape_range = dym_shape_range
        self.onnx_fusion_switch = onnx_fusion_switch
