
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2012-2022. All rights reserved.
"""
Function:
This class mainly involves the advisor result function.
"""
import os

from advisor.advisor_const import AdvisorConst
from cmp_utils.constant.const_manager import ConstManager
from cmp_utils import path_check
from cmp_utils import log


class AdvisorResult:
    """
    Class for generate advisor result
    """

    def __init__(self, match_advisor=False, advisor_type="NA", operator_index="NA", advisor_message="NA"):
        self.match_advisor = match_advisor
        self.advisor_type = advisor_type
        self.operator_index = operator_index
        self.advisor_message = advisor_message

    @staticmethod
    def gen_summary_file(out_path, message_list):
        """
        Generate advisor summary file
        :param out_path: advisor summary file out path
        :param  message_list: summary message
        """
        result_file = os.path.join(out_path, "advisor_summary.txt")
        try:
            path_check.check_write_path_secure(result_file)
            with os.fdopen(os.open(result_file, ConstManager.WRITE_FLAGS, ConstManager.WRITE_MODES),
                           'w+') as output_file:
                output_file.truncate(0)
                message_list = [message + AdvisorConst.NEW_LINE for message in message_list]
                output_file.writelines(message_list)
        except IOError as io_error:
            log.print_error_log("Failed to save the advisor summary, the reason is %s." % io_error)
        else:
            log.print_info_log('The advisor summary (.txt) is saved in: "%r" .' % result_file)

    def print_advisor_log(self):
        """
        Log and print advisor summary
        """
        log.print_info_log("The summary of the expert advice is as follows: ")
        message_list = [
            AdvisorConst.DETECTION_TYPE + AdvisorConst.COLON + self.advisor_type,
            AdvisorConst.OPERATOR_INDEX + AdvisorConst.COLON + self.operator_index,
            AdvisorConst.ADVISOR_SUGGEST + AdvisorConst.COLON + self.advisor_message
        ]
        for message in message_list:
            log.print_info_log(message)
        return message_list
