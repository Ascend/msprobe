#!/usr/bin/env python3
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

from msprobe.pytorch.api_accuracy_checker.compare.algorithm import get_rel_err_ratio
from msprobe.core.common.const import CompareConst
from msprobe.pytorch.api_accuracy_checker.precision_standard.base_standard import BaseCompare


class ThousandthStdCompare(BaseCompare):
    """
    Thousandth standard comparison class for calculating accuracy metrics.

    A subclass of BaseCompare, specifically designed to compare the relative error
    between benchmark and device outputs, focusing on errors within a thousandth (0.001) threshold.

    Attributes:
        rel_err_orign (float or array-like): The original relative error values to be compared.
        compare_column (object): An object to store and update comparison metrics.

    Methods:
        _compute_metrics(): Computes the relative error metrics, specifically the thousandth error ratio.
    """

    def __init__(self, input_data):
        super().__init__(input_data)
        self.rel_err_orign = input_data.rel_err_orign
        self.compare_column = input_data.compare_column

    def _pre_compare(self):
        pass

    def _compute_metrics(self):
        """
        Computes the relative error metrics for the comparison, specifically focusing on errors within a thousandth
        (0.001) threshold.

        This method calculates the proportion of relative errors that are within the thousandth threshold.
        It uses the `get_rel_err_ratio` function to determine the ratio of relative errors that are less than or
        equal to the
        specified threshold defined in `CompareConst.THOUSAND_RATIO_THRESHOLD`.

        Returns:
        dict: A dictionary containing the computed relative error metric.
            The dictionary has the following key:
            - 'rel_err_thousandth': The proportion of relative errors within the thousandth threshold.
        """
        rel_err_thousandth, _ = get_rel_err_ratio(self.rel_err_orign, CompareConst.THOUSAND_RATIO_THRESHOLD)

        return {'rel_err_thousandth': rel_err_thousandth}
