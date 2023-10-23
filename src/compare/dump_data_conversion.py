#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
"""
Function:
DumpDataConversion class. This class mainly involves the convert_data function.
"""
import sys
import time
from cmp_utils import log, file_utils
from cmp_utils.constant.compare_error import CompareError
from conversion.data_conversion import DumpDataConversion


if __name__ == "__main__":
    log.print_deprecated_warning(sys.argv[0])
    START = time.time()
    CONVERSION = DumpDataConversion()
    RET = 0
    with file_utils.UmaskWrapper():
        try:
            RET = CONVERSION.convert_data()
        except CompareError as err:
            RET = err.code
        except Exception as base_err:
            log.print_error_log(f'Basic error running {sys.argv[0]}: {base_err}')
            sys.exit(1)

    END = time.time()
    log.print_info_log("The dump data conversion was completed and took %.2f seconds." % (END - START))
    sys.exit(RET)
