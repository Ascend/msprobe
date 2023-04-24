
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
"""
Function:
ShapeConversion class. This class mainly involves the convert_shape function.
"""
import sys
import time
from src.compare.cmp_utils import log
from src.compare.conversion.shape_format_conversion import ShapeConversionMain


if __name__ == "__main__":
    log.print_deprecated_warning(sys.argv[0])
    START = time.time()
    SHAPE_CONVERSION = ShapeConversionMain()
    RET = SHAPE_CONVERSION.process()
    END = time.time()
    log.print_info_log("The format conversion was completed and took %.2f seconds." % (END - START))
    sys.exit(RET)
