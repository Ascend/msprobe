
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2012-2022. All rights reserved.
"""
Function:
This file mainly involves the node advisor function.
"""

from cmp_utils import log
from advisor.advisor_const import AdvisorConst
from advisor.advisor_result import AdvisorResult


class NodeAdvisor:
    """
    Class for generate node advisor
    """

    def __init__(self, input_file, result):
        self.analyze_data = input_file
        self.result = result

    def start_analyze(self):
        """
        Analyze result by node detection
        """
        log.print_info_log('Start Global Consistency detection.')
        data_columns = self.analyze_data.columns.values
        if AdvisorConst.COSINE_SIMILARITY not in data_columns:
            log.print_warn_log('Input csv file does not contain %s columns, Skip Global Consistency detection.'
                               % AdvisorConst.COSINE_SIMILARITY)
            return self.result
        else:
            have_cos_df = self.analyze_data.dropna(subset=[AdvisorConst.COSINE_SIMILARITY])
            # check cosine dataframe lines
            if have_cos_df.shape[0] == 0:
                log.print_warn_log('After analysis, input csv file %s column, does not have valid value. '
                                   'May all values be NAN, please check.'
                                   % AdvisorConst.COSINE_SIMILARITY)
                return self.result
            cos_df = have_cos_df.reset_index(drop=True, inplace=False)
            err_cos_df = have_cos_df[have_cos_df[AdvisorConst.COSINE_SIMILARITY] < AdvisorConst.ACCURACY_THRESHOLD]
            if err_cos_df.shape[0] == 0:
                return AdvisorResult(True, AdvisorConst.CONSISTENCY_DETECTION,
                                     AdvisorConst.NO_ERROR_OP, AdvisorConst.CONSISTENCY_SUGGEST)
            err_cos_df.reset_index(drop=True, inplace=True)
            index = err_cos_df.at[0, AdvisorConst.INDEX]
            cos_df_len = cos_df.shape[0]
            last_line_cos = cos_df.at[cos_df_len - 1, AdvisorConst.COSINE_SIMILARITY]
            if last_line_cos < AdvisorConst.ACCURACY_THRESHOLD:
                return AdvisorResult(True, AdvisorConst.CONSISTENCY_DETECTION,
                                     str(index), AdvisorConst.PROBLEM_SUGGEST)
            return AdvisorResult(True, AdvisorConst.CONSISTENCY_DETECTION,
                                 str(index), AdvisorConst.DEVIATION_SUGGEST)
