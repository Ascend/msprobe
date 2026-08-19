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


import numpy as np
from msprobe.core.common.const import CompareConst
from msprobe.core.common.utils import format_value
from msprobe.core.compare.utils import flatten_numpy


def column_name() -> str:
    return CompareConst.MAX_ABS_ERR


def compare(n_value: np.ndarray, b_value: np.ndarray):
    n_value, b_value = flatten_numpy(n_value, b_value)

    max_value = np.max(np.abs(n_value - b_value))
    if np.isnan(max_value):
        return CompareConst.NAN, "Cannot compare by MaxAbsError, the data contains nan/inf/-inf in dump data."
    return format_value(max_value), ""
