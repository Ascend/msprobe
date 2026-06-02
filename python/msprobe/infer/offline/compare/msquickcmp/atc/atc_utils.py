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

"""
Function:
This class mainly involves convert model to json function.
"""

import os

from msprobe.core.common.log import logger
from msprobe.core.common.file_utils import check_file_or_directory_path
from msprobe.infer.utils.util import filter_cmd
from msprobe.infer.offline.compare.msquickcmp.common.utils import (
    execute_command,
    get_model_name_and_extension,
    check_file_size_valid,
    ACCURACY_COMPARISON_MODEL_TYPE_ERROR,
    ACCURACY_COMPARISON_INVALID_PATH_ERROR,
    MAX_READ_FILE_SIZE_4G,
    AccuracyCompareException,
)

ATC_FILE_PATH = "compiler/bin/atc"
OLD_ATC_FILE_PATH = "atc/bin/atc"
NEW_ATC_FILE_PATH = "bin/atc"


def convert_model_to_json(cann_path, offline_model_path, out_path):
    """
    Function Description:
        convert om model to json
    Return Value:
        output json path
    Exception Description:
        when the model type is wrong throw exception
    """
    model_name, extension = get_model_name_and_extension(offline_model_path)
    if extension not in [".om", ".txt"]:
        logger.error(f"The offline model file not ends with .om or .txt, Please check {offline_model_path}.")
        raise AccuracyCompareException(ACCURACY_COMPARISON_MODEL_TYPE_ERROR)

    cann_path = os.path.realpath(cann_path)
    if not os.path.isdir(cann_path):
        logger.error(f'The cann path {cann_path} is not a directory.Please check.')
        raise AccuracyCompareException(ACCURACY_COMPARISON_INVALID_PATH_ERROR)

    atc_command_file_path = get_atc_path(cann_path)
    check_file_or_directory_path(atc_command_file_path)
    output_json_path = os.path.join(out_path, "model", model_name + ".json")
    if os.path.exists(output_json_path):
        logger.info(f"The {output_json_path} file is exists.")
    else:
        # do the atc command to convert om to json
        logger.info('Start to converting the model to json')
        if extension == ".om":
            mode_type = "1"
        else:
            mode_type = "5"
        atc_cmd = [
            atc_command_file_path,
            "--mode=" + mode_type,
            "--om=" + offline_model_path,
            "--json=" + output_json_path,
        ]
        logger.info(f"ATC command line {' '.join(atc_cmd)}")
        atc_cmd = filter_cmd(atc_cmd)
        execute_command(atc_cmd)
        logger.info(f"Complete model conversion to json {output_json_path}.")

    check_file_size_valid(output_json_path, MAX_READ_FILE_SIZE_4G)
    return output_json_path


def get_atc_path(cann_path):
    atc_command_file_path = os.path.join(cann_path, ATC_FILE_PATH)
    if not os.path.exists(atc_command_file_path):
        atc_command_file_path = os.path.join(cann_path, OLD_ATC_FILE_PATH)
    if not os.path.exists(atc_command_file_path):
        atc_command_file_path = os.path.join(cann_path, NEW_ATC_FILE_PATH)
    if not os.path.exists(atc_command_file_path):
        raise AccuracyCompareException(ACCURACY_COMPARISON_INVALID_PATH_ERROR)

    atc_command_file_path = os.path.realpath(atc_command_file_path)
    if not os.access(atc_command_file_path, os.X_OK):
        logger.error('ATC path is not permitted for executing.')
        raise AccuracyCompareException(ACCURACY_COMPARISON_INVALID_PATH_ERROR)
    return atc_command_file_path
