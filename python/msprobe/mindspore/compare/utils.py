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

import numpy as np

from msprobe.core.common.const import Const
from msprobe.core.common.file_utils import load_npy, FileChecker, FileCheckConst
from msprobe.core.common.utils import detect_framework_by_dump_json, CompareException, check_op_str_pattern_valid
from msprobe.core.common.log import logger


def read_npy_data(dir_path, file_name):
    if not file_name:
        return None

    data_path = os.path.join(dir_path, file_name)
    path_checker = FileChecker(data_path, FileCheckConst.FILE, FileCheckConst.READ_ABLE,
                               FileCheckConst.NUMPY_SUFFIX)
    data_path = path_checker.common_check()
    data_value = load_npy(data_path)
    data_value = np.squeeze(data_value)
    return data_value


def check_cross_framework(bench_json_path):
    framework = detect_framework_by_dump_json(bench_json_path)
    return framework == Const.PT_FRAMEWORK


def check_name_map_dict(name_map_dict):
    if not isinstance(name_map_dict, dict):
        logger.error("'map_dict' should be a dict, please check!")
        raise CompareException(CompareException.INVALID_OBJECT_TYPE_ERROR)
    check_op_str_pattern_valid(str(name_map_dict))
