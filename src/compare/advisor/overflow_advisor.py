# coding=utf-8
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

"""
Function:
This file mainly involves the overflow advisor function.
"""

from cmp_utils import log
from advisor.advisor_const import AdvisorConst
from advisor.advisor_result import AdvisorResult


class OverflowAdvisor:
    """
    Class for generate overflow advisor
    """

    def __init__(self, input_file, result):
        self.analyze_data = input_file
        self.result = result

    def start_analyze(self):
        """
        Analyze result by overflow detection
        """
        log.print_info_log('Start FP16 Overflow detection.')
        data_columns = self.analyze_data.columns.values
        if AdvisorConst.OVERFLOW not in data_columns:
            log.print_warn_log('Input csv file does not contain %s columns, Skip FP16 Overflow detection.'
                               % AdvisorConst.OVERFLOW)
        else:
            overflow_df = self.analyze_data[self.analyze_data[AdvisorConst.OVERFLOW] == "YES"]
            # check overflow dataframe lines
            if overflow_df.shape[0] == 0:
                log.print_info_log('After analysis, input csv file does not have FP16 Overflow problem.')
                return self.result
            overflow_df.reset_index(drop=True, inplace=True)
            index = overflow_df.at[0, AdvisorConst.INDEX]
            self.result = AdvisorResult(True, AdvisorConst.OVERFLOW_DETECTION, str(index),
                                        AdvisorConst.OVERFLOW_SUGGEST)
        return self.result


