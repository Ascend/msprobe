
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2012-2022. All rights reserved.
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


