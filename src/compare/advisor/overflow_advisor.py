#!/usr/bin/env python
# coding=utf-8
"""
Function:
This file mainly involves the overflow advisor function.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2021-2022
"""

import log

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
        log.print_info_log('Start analysis operator overflow problem.')
        data_columns = self.analyze_data.columns.values
        if AdvisorConst.OVERFLOW not in data_columns:
            log.print_warn_log('Input csv file does not contain %s columns, Skip overflow detection analysis.'
                               % AdvisorConst.OVERFLOW)
            return self.result
        else:
            overflow_df = self.analyze_data[self.analyze_data[AdvisorConst.OVERFLOW] == "YES"]
            # check overflow dataframe lines
            if overflow_df.shape[0] == 0:
                log.print_info_log('After analysis, input csv file does not have operator overflow problem.')
                return self.result
            overflow_df.reset_index(drop=True, inplace=True)
            index = overflow_df.at[0, AdvisorConst.INDEX]
            self.result = AdvisorResult(True, AdvisorConst.OVERFLOW_DETECTION, str(index),
                                        AdvisorConst.OVERFLOW_SUGGEST)
            return self.result


