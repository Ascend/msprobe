# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
"""
Function:
This file mainly involves the common function.
"""
import os

from cmp_utils.constant.const_manager import ConstManager
from cmp_utils import log
from dump_parse.dump_utils import check_valid_timestamp


class SortMode:
    """
    The class of sort mode
    """
    hash_to_file_name_map = {}

    def __init__(self, parameter):
        self.parameter = parameter

    def __call__(self: any, wrap_function):
        """
        the wrapper of get info to sort
        @param wrap_function: file name
        @return: Basis of sorted
        """
        @wraps(wrap_function)
        def inner(*args, **kwargs):
            file_path = wrap_function(*args, **kwargs)
            file_name = os.path.basename(file_path)
            file_name = self.hash_to_file_name_map.get(file_name) if file_name.isdigit() else file_name
            file_split = file_name.split('.')
            if self.parameter == ConstManager.NORMAL_MODE or \
                    self.parameter == ConstManager.FFTS_TIMESTAMP:
                return self._parameter_timestamp(file_split, file_name)
            elif self.parameter == ConstManager.AUTOMATIC_MODE:
                return self._parameter_auto(file_split, file_name)
            elif self.parameter == ConstManager.MANUAL_MODE:
                return self._parameter_manual(file_split, file_name)
            else:
                log.print_warn_log('The sort mode parameter is invalid, failed to sort')
                return ConstManager.INVALID_SORT_MODE
        return inner

    @staticmethod
    def _parameter_manual(file_split, file_name):
        # Conv2D.partition0_rank2_new_sub_graph15_sgt_graph_0_fp32_vars_conv2d_39_Conv2D_lxslice0. \
        # 2.9.1670205071724946.4.487.0.0
        slice_x = file_split[1][-1]
        if not slice_x.isdigit():
            log.print_warn_log(
                'The file name \"{}\"\'s slice_x is invalid.'.format(file_name))
            return ConstManager.INVALID_SLICE_X
        return int(slice_x)

    @staticmethod
    def _parameter_auto(file_split, file_name):
        thread_id = file_split[-1]
        if not thread_id.isdigit():
            log.print_warn_log(
                'The file name \"{}\"\'s thread_id is invalid.'.format(file_name))
            return ConstManager.INVALID_THREAD_ID
        return int(thread_id)

    def _parameter_timestamp(self, file_split, file_name):
        if self.parameter == ConstManager.FFTS_TIMESTAMP:
            timestamp = file_split[4]
        elif file_name.endswith(
                (ConstManager.STANDARD_SUFFIX, ConstManager.NUMPY_SUFFIX, ConstManager.QUANT_SUFFIX)):
            timestamp = file_split[2]
        else:
            timestamp = file_split[-1]
        if not check_valid_timestamp(timestamp):
            log.print_warn_log(
                'The file name \"{}\"\'s timestamp is invalid.'.format(file_name))
            return ConstManager.INVALID_TIMESTAMP
        return int(timestamp)


@SortMode(ConstManager.AUTOMATIC_MODE)
def get_ffts_auto(file_name):
    """
    get thread id of ffts auto mode from file name
    @param file_name: file name
    @return: thread id
    """
    return file_name


@SortMode(ConstManager.MANUAL_MODE)
def get_ffts_manual(file_name):
    """
    get slice X of ffts manual mode from file name
    @param file_name: file name
    @return: slice X
    """
    return file_name


@SortMode(ConstManager.NORMAL_MODE)
def get_normal_timestamp(file_name):
    """
    get timestamp of normal mode
    @param file_name: file name
    @return: timestamp
    """
    return file_name


@SortMode(ConstManager.FFTS_TIMESTAMP)
def get_ffts_timestamp(file_name):
    """
    get timestamp of ffts mode
    @param file_name: file name
    @return: timestamp
    """
    return file_name


def sort_dump_file_list(dump_file_type: int, dump_file_list: list) -> list:
    """
    sort dump file list by different dump mode
    @param dump_file_type: dump data mode
    @param dump_file_list: dump file list
    @return: sorted dump file list
    """
    if dump_file_type == ConstManager.NORMAL_MODE:
        dump_file_list.sort(key=get_normal_timestamp)
    elif dump_file_type == ConstManager.AUTOMATIC_MODE or dump_file_type == ConstManager.MANUAL_MODE:
        dump_file_list.sort(key=get_ffts_timestamp)
        if dump_file_type == ConstManager.AUTOMATIC_MODE:
            dump_file_list.sort(key=get_ffts_auto)
        elif dump_file_type == ConstManager.MANUAL_MODE:
            dump_file_list.sort(key=get_ffts_manual)
    return dump_file_list

