# -*- coding: utf-8 -*-
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

import os
import sys
import numpy as np

from msprobe.core.common.log import logger
from msprobe.infer.offline.compare.msquickcmp.common.utils import execute_command
from msprobe.infer.utils.util import load_file_to_read_common_check, filter_cmd


MSACCUCMP_FILE_PATH = "toolkit/tools/operator_cmp/compare/msaccucmp.py"


def convert_bin_file_to_npy(bin_file_path, npy_dir_path, cann_path):
    """
    Function Description:
        convert a bin file to npy file.
    Parameter:
        bin_file_path: the path of the bin file needed to be converted to npy
        npy_dir_path: the dest dir to save the converted npy file
        cann_path: user or system cann_path for using msaccucmp.py
    """
    python_version = sys.executable.rsplit('/', maxsplit=1)[-1]
    msaccucmp_command_file_path = os.path.join(cann_path, MSACCUCMP_FILE_PATH)
    logger.info(f"convert dump data: {bin_file_path} to npy file")
    bin2npy_cmd = [python_version, msaccucmp_command_file_path, "convert", "-d", bin_file_path, "-out", npy_dir_path]
    bin2npy_cmd = filter_cmd(bin2npy_cmd)
    execute_command(bin2npy_cmd)


def convert_npy_to_bin(npy_input_path):
    """
    Function Description:
        convert a  file to bin file.
    Parameter:
        npy_file_path: the path of the npy file needed to be converted to bin
    """
    input_initial_path = npy_input_path.split(",")
    outputs = []
    for input_item in input_initial_path:
        input_item_path = os.path.realpath(input_item)
        if input_item_path.endswith('.npy'):
            bin_item = input_item[:-4] + '.bin'
            bin_path = input_item_path[:-4] + '.bin'
            input_item_path = load_file_to_read_common_check(input_item_path)
            npy_data = np.load(input_item_path)
            if os.path.islink(bin_path):
                os.unlink(bin_path)
            if os.path.exists(bin_path):
                os.remove(bin_path)
            npy_data.tofile(bin_path)
            os.chmod(bin_path, 0o640)
            outputs.append(bin_item)
        else:
            outputs.append(input_item)
    return ",".join(outputs)
