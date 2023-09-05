
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
"""
Function:
VectorComparison class. This class mainly involves the compare function.
"""

import sys
import time
import signal
from cmp_utils import log
from cmp_utils.constant.compare_error import CompareError
from vector_cmp.vector_comparison import VectorComparison


def _handle_stop(sig: any, frame: any) -> None:
    _ = sig
    _ = frame
    sys.exit(-1)


if __name__ == "__main__":
    log.print_deprecated_warning(sys.argv[0])
    START = time.time()
    for SIG in [signal.SIGINT, signal.SIGHUP, signal.SIGTERM]:
        signal.signal(SIG, _handle_stop)
    VECTOR_COMPARISON = VectorComparison()
    try:
        RET = VECTOR_COMPARISON.compare()
    except CompareError as err:
        RET = err.code
    finally:
        pass
    END = time.time()
    log.print_info_log("The comparison was completed and took " + str(END - START) + " seconds.")
    sys.exit(RET)
