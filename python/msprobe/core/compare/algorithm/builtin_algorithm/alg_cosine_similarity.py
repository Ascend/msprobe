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
from msprobe.core.common.const import Const, CompareConst
from msprobe.core.common.utils import format_value
from msprobe.core.compare.utils import validate_numpy, flatten_numpy


def column_name() -> str:
    return CompareConst.COSINE


def compare(n_value: np.ndarray, b_value: np.ndarray):
    result, err = validate_numpy(n_value, b_value)
    if err:
        return result, err
    n_value, b_value = flatten_numpy(n_value, b_value)
    with np.errstate(divide="ignore", invalid="ignore"):
        num = np.dot(n_value, b_value)
        a_norm = np.linalg.norm(n_value)
        b_norm = np.linalg.norm(b_value)

        if a_norm <= Const.FLOAT_EPSILON and b_norm <= Const.FLOAT_EPSILON:
            return 1.0, ""
        if a_norm <= Const.FLOAT_EPSILON:
            return CompareConst.NAN, "Cannot compare by Cosine Similarity, All the data is Zero in npu dump data."
        if b_norm <= Const.FLOAT_EPSILON:
            return CompareConst.NAN, "Cannot compare by Cosine Similarity, All the data is Zero in Bench dump data."

        cos = num / (a_norm * b_norm)
        if np.isnan(cos):
            return CompareConst.NAN, "Cannot compare by Cosine Similarity, the dump data has NaN."
        result = format_value(cos)
        if result > CompareConst.COSINE_THRESHOLD:
            result = round(result, 6)
    return result, ""
