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
from msprobe.core.compare.utils import validate_compare_numpy, get_relative_err, calc_err_ratio


def column_name() -> str:
    return CompareConst.ONE_THOUSANDTH_ERR_RATIO


def compare(n_value: np.ndarray, b_value: np.ndarray):
    result, err = validate_compare_numpy(n_value, b_value)
    if err:
        return result, err
    return calc_err_ratio(get_relative_err(n_value, b_value), CompareConst.THOUSAND_RATIO_THRESHOLD), ""
