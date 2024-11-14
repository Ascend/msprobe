
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
"""
Function:
VectorComparison class. This class mainly involves the compare function.
"""

import sys
import time
import signal
from cmp_utils import log, file_utils
from cmp_utils.constant.compare_error import CompareError
from vector_cmp.vector_comparison import VectorComparison


if __name__ == "__main__":
    log.print_deprecated_warning(sys.argv[0])
    START = time.time()
    for SIG in [signal.SIGINT, signal.SIGHUP, signal.SIGTERM]:
        signal.signal(SIG, lambda sig, frame : sys.exit(-1))
    VECTOR_COMPARISON = VectorComparison()
    RET = 0
    with file_utils.UmaskWrapper():
        try:
            RET = VECTOR_COMPARISON.compare()
        except CompareError as err:
            RET = err.code
        except Exception as base_err:
            log.print_error_log(f'Basic error running {sys.argv[0]}: {base_err}')
            sys.exit(1)
    END = time.time()
    log.print_info_log("The comparison was completed and took " + str(END - START) + " seconds.")
    sys.exit(RET)
