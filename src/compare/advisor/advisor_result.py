#!/usr/bin/env python
# coding=utf-8
"""
Function:
This class mainly involves the advisor result function.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2021-2022
"""
import os

import log

from advisor.advisor_const import AdvisorConst


class AdvisorResult:
    """
    Class for generate advisor result
    """

    def __init__(self, match_advisor=False, advisor_type="NA", operator_index="NA", advisor_message="NA"):
        self.match_advisor = match_advisor
        self.advisor_type = advisor_type
        self.operator_index = operator_index
        self.advisor_message = advisor_message

    def print_advisor_log(self):
        log.print_info_log("A summary of the expert advice is as follows: ")
        message_list = [AdvisorConst.DETECTION_TYPE + AdvisorConst.COLON +
                        self.advisor_type,
                        AdvisorConst.OPERATOR_INDEX + AdvisorConst.COLON +
                        self.operator_index,
                        AdvisorConst.ADVISOR_SUGGEST + AdvisorConst.COLON +
                        self.advisor_message]
        for message in message_list:
            log.print_info_log(message)
        return message_list

    @staticmethod
    def gen_summary_file(out_path, message_list):
        result_file = os.path.join(out_path, "advisor_summary.txt")
        try:
            with open(result_file, 'w') as f:
                message_list = [message + AdvisorConst.NEW_LINE for message in message_list]
                f.writelines(message_list)
            log.print_info_log('The advisor summary (.txt) is saved in: "%s" .' % result_file)
        except IOError as io_error:
            log.print_error_log("Failed to save the advisor summary, the reason is %s." % io_error)
        finally:
            pass


