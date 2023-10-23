
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
"""
Function:
ShapeConversion class. This class mainly involves the convert_shape function.
"""
import sys
import time
from cmp_utils import log, file_utils
from conversion.shape_format_conversion import ShapeConversionMain


if __name__ == "__main__":
    log.print_deprecated_warning(sys.argv[0])
    START = time.time()
    SHAPE_CONVERSION = ShapeConversionMain()
    RET = 0
    with file_utils.UmaskWrapper():
        try:
            RET = SHAPE_CONVERSION.process()
        except Exception as base_err:
            log.print_error_log(f'Basic error running {sys.argv[0]}: {base_err}')
            sys.exit(1)
    END = time.time()
    log.print_info_log("The format conversion was completed and took %.2f seconds." % (END - START))
    sys.exit(RET)
