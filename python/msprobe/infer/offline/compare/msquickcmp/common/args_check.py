# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
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

import argparse
import re

from msprobe.core.common.log import logger
from msprobe.core.common.const import FileCheckConst
from msprobe.core.common.file_utils import check_file_or_directory_path, check_output_dir_path, check_file_type


def check_model_path_legality(path):
    check_file_or_directory_path(path, False, False, [".onnx", ".om"])
    return path


def check_input_data_path(path):
    if not isinstance(path, str):
        logger.error(f"input data path:{path} is illegal. Please check.")
        raise argparse.ArgumentTypeError
    if path == '':
        return path
    input_item_paths = path.split(',')
    for input_item_path in input_item_paths:
        file_type = check_file_type(input_item_path)
        if file_type != FileCheckConst.FILE:
            logger.error("The '--input_data' parameter only supports file paths, not folder paths. Please check.")
            raise argparse.ArgumentTypeError
        check_file_or_directory_path(input_item_path, False, False, [".npy", ".bin"])
    return path


def check_output_path_legality(path):
    if not path:
        return path
    check_output_dir_path(path)
    return path


def check_dict_kind_string(value):
    # just like "input_name1:1,224,224,3;input_name2:3,300"
    if not value:
        return value
    regex = re.compile(r"[^_A-Za-z0-9,;:/.-]")
    if regex.search(value):
        logger.error(f"string '{value}' is not a legal string")
        raise argparse.ArgumentTypeError
    return value


def check_rank_range_valid(rank):
    min_value = 0
    max_value = 255
    try:
        int_rank = int(rank)
    except Exception as e:
        logger.error("The rank is illegal. The value must be of type that can be cast to an int variable!")
        raise argparse.ArgumentTypeError from e
    if int_rank < min_value or int_rank > max_value:
        logger.error(f"rank:{rank} is invalid. valid value range is [{min_value}, {max_value}]")
        raise argparse.ArgumentTypeError
    return rank


def check_number_list(value):
    # just like "1241414,124141,124424"
    if not value:
        return value
    outsize_list = value.split(',')
    for outsize in outsize_list:
        regex = re.compile(r"[^0-9]")
        if regex.search(outsize):
            logger.error(f"output size \"{outsize}\" is not a legal string")
            raise argparse.ArgumentTypeError
    return value


def check_dym_range_string(value):
    if not value:
        return value
    dym_string = value
    regex = re.compile(r"[^_A-Za-z0-9,;:/.\-~]")
    if regex.search(dym_string):
        logger.error(f"dym range string \"{dym_string}\" is not a legal string")
        raise argparse.ArgumentTypeError
    return value


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        logger.error('Boolean value expected true, 1, false, 0 with case insensitive.')
        raise argparse.ArgumentTypeError
