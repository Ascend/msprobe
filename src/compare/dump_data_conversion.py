
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
"""
Function:
DumpDataConversion class. This class mainly involves the convert_data function.
"""

import sys
import time
from src.compare.cmp_utils import log
from src.compare.cmp_utils.constant.compare_error import CompareError
from src.compare.conversion.data_conversion import DumpDataConversion

if __name__ == "__main__":
    log.print_deprecated_warning(sys.argv[0])
    START = time.time()
    CONVERSION = DumpDataConversion()
    try:
        RET = CONVERSION.convert_data()
    except CompareError as err:
        RET = err.code
    END = time.time()
    log.print_info_log("The dump data conversion was completed and took %.2f seconds." % (END - START))
    sys.exit(RET)
