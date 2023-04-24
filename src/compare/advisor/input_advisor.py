
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2012-2022. All rights reserved.
"""
Function:
This file mainly involves the input advisor function.
"""

from cmp_utils import log
from advisor.advisor_const import AdvisorConst
from advisor.advisor_result import AdvisorResult


class InputAdvisor:
    """
    Class for generate input advisor
    """

    def __init__(self, input_file, result, input_nodes):
        self.analyze_data = input_file
        self.result = result
        self.input_nodes = input_nodes

    def start_analyze(self):
        """
        Analyze result by input detection
        """
        log.print_info_log('Start Input Inconsistent detection.')
        data_columns = self.analyze_data.columns.values
        if AdvisorConst.COSINE_SIMILARITY not in data_columns:
            log.print_warn_log('Input csv file does not contain %s columns, Skip Input Inconsistent detection.'
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
            err_cos_df = have_cos_df[have_cos_df['CosineSimilarity'] < AdvisorConst.ACCURACY_THRESHOLD]
            for input_node in self.input_nodes:
                err_input_df = err_cos_df[err_cos_df[AdvisorConst.NPU_DUMP] == input_node]
                err_input_df.reset_index(drop=True, inplace=True)
                if err_input_df.shape[0] > 0:
                    index = err_input_df.at[0, AdvisorConst.INDEX]
                    return AdvisorResult(True, AdvisorConst.INPUT_DETECTION, str(index),
                                         AdvisorConst.INPUT_SUGGEST)
            return self.result


