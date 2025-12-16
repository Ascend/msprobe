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
This file mainly involves the const value.
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
    ADVISOR_SUGGEST = "Expert Advice"

    # detection type
    OVERFLOW_DETECTION = "FP16 Overflow"
    INPUT_DETECTION = "Input Inconsistent"
    CONSISTENCY_DETECTION = "Global Consistency"

    # operator index
    NO_ERROR_OP = "NA"

    # advisor suggest
    OVERFLOW_SUGGEST = "Float16 data overflow occurs. Rectify the fault and perform comparison again."
    INPUT_SUGGEST = "The input data of NPUDump is inconsistent with that of GroundTruth. Use the same data " \
                    "or check the data preprocessing process."
    CONSISTENCY_SUGGEST = "All data in the comparison result meets the accuracy requirements. " \
                          "If data accuracy of the model is still not up to standard in practical application, " \
                          "please check the post-processing process of model outputs."
    PROBLEM_SUGGEST = "The accuracy of some tensors is low, resulting in an unqualified final accuracy. " \
                      "This may be caused by quantization. Calibrate the data or contact Huawei for further diagnosis. "
    DEVIATION_SUGGEST = "The accuracy of some tensors is low, while the final accuracy is qualified. " \
                        "This may be caused by Ascend internal optimization. " \
                        "Ignore or contact Huawei for further diagnosis. "

    # text symbol
    NEW_LINE = "\n"
    COLON = ": "

    ACCURACY_THRESHOLD = 0.99

