#!/usr/bin/env python
# coding=utf-8
"""
Function:
This file mainly involves the const value.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2021-2022
"""


class AdvisorConst:
    """
    The class for advisor const
    """
    # column const
    COSINE_SIMILARITY = "CosineSimilarity"
    INDEX = "Index"
    NPU_DUMP = "NPUDump"
    OVERFLOW = "OverFlow"

    # advisor summary key
    DETECTION_TYPE = "Detection Type"
    OPERATOR_INDEX = "Operator Index"
    ADVISOR_SUGGEST = "Advisor Suggest"

    # detection type
    OVERFLOW_DETECTION = "Overflow Detection"
    INPUT_DETECTION = "Input Detection"
    CONSISTENCY_DETECTION = "Consistency Detection"
    PROBLEM_DETECTION = "Problem Node Detection"
    DEVIATION_DETECTION = "Deviation Detection"

    # operator index
    NO_ERROR_OP = "NA"

    # advisor suggest
    OVERFLOW_SUGGEST = "Have overflow, please check!"
    INPUT_SUGGEST = "Have input error, please check!"
    CONSISTENCY_SUGGEST = "Consistency check good!"
    PROBLEM_SUGGEST = "Have problem node, please check!"
    DEVIATION_SUGGEST = "Have deviation node, please ignore!"

    # text symbol
    NEW_LINE = "\n"
    COLON = ": "

    ACCURACY_THRESHOLD = 0.99

