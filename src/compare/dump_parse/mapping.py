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
This file mainly involves the mapping function.
"""
import os
import csv

from cmp_utils import log
from cmp_utils.constant.const_manager import ConstManager
from cmp_utils.utils import check_file_size


def _handle_csv_object(csv_object: any, mapping_file_path: str) -> dict:
    hash_to_file_name_map = {}
    for item in csv_object:
        if len(item) == 2:
            hash_to_file_name_map[item[0]] = item[1]
        else:
            log.print_error_log(
                'The content (%s) of the mapping file "%r" is invalid.' % (item, mapping_file_path))
    return hash_to_file_name_map


def read_mapping_file(mapping_file_path: str) -> dict:
    """
    Read mapping file
    :param mapping_file_path: mapping file path
    :return: hash_to_file_name_map
    """
    hash_to_file_name_map = {}
    if not os.path.isfile(mapping_file_path):
        return hash_to_file_name_map
    check_file_size(mapping_file_path, ConstManager.ONE_HUNDRED_MB)
    try:
        with open(mapping_file_path, "r") as mapping_file:
            csv_object = csv.reader(mapping_file)
            return _handle_csv_object(csv_object, mapping_file_path)
    except csv.Error:
        log.print_error_log('Failed to read csv object. The content of the mapping file "%r" is invalid.'
                            % mapping_file_path)
    except (OSError, SystemError, ValueError, TypeError, RuntimeError, MemoryError) as error:
        log.print_open_file_error(mapping_file_path, error)
    finally:
        pass
    return hash_to_file_name_map
