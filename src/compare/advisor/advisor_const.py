
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2012-2022. All rights reserved.
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

