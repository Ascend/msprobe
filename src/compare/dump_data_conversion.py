#!/usr/bin/env python
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
