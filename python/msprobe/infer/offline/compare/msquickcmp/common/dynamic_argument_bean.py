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

from enum import Enum


class Arg(object):  # pylint: disable=useless-object-inheritance
    def __init__(self, benchmark_arg, atc_arg, msquickcmp_arg):
        self.atc_arg = atc_arg
        self.benchmark_arg = benchmark_arg
        self.msquickcmp_arg = msquickcmp_arg


class DynamicArgumentEnum(Enum):
    # enum struct Arg(benchmark_arg, atc_arg, msquickcmp_arg)
    DYM_BATCH = Arg("--dymBatch", "--dynamic_batch_size", None)
    DYM_SHAPE = Arg("--dymShape", "--input_shape_range", "input_shape")
    DYM_DIMS = Arg("--dymDims", "--dynamic_dims", "input_shape")

    @staticmethod
    def get_all_args() -> list:
        """
        get all argument enum, return as a list
        """
        return list(map(lambda arg: arg, DynamicArgumentEnum))
