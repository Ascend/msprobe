#!/usr/bin/env python
# coding=utf-8
"""
Function:
This file mainly involves the print function.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2019-2021
"""

import os
import sys
import time


def _print_log(level: str, msg: str) -> None:
    current_time = time.strftime("%Y-%m-%d %H:%M:%S",
                                 time.localtime(int(time.time())))
    pid = os.getpid()
    print("%s (%d) - [%s] %s" % (current_time, pid, level, msg))
    sys.stdout.flush()


def print_error_log(error_msg: str) -> None:
    """
    print error log
    :param error_msg: the error message
    """
    _print_log("ERROR", error_msg)


def print_warn_log(warn_msg: str) -> None:
    """
    print warn log
    :param warn_msg: the warn message
    """
    _print_log("WARNING", warn_msg)


def print_info_log(info_msg: str) -> None:
    """
    print info log
    :param info_msg: the info message
    """
    _print_log("INFO", info_msg)


def print_no_left_dump_file_error(op_name: str, op_type: str, is_error: bool = False) -> str:
    """
    Print warn or error log for no my output dump file error
    :param op_name: the op name
    :param op_type: the op type
    :param is_error: the log lever is error
    :return: message
    """
    msg = '[%s] There is no dump file for my output operator "%s". The type is %s.' \
          % (op_name, op_name, op_type)
    if is_error:
        print_error_log(msg)
    else:
        print_warn_log(msg)
    return msg


def print_no_right_dump_file_error(op_name: str, tensor_id: str, is_error: bool = False) -> str:
    """
    Print warn or error log for no right dump file error
    :param op_name: the op name
    :param tensor_id: the tensor id
    :param is_error: the log lever is error
    :return message
    """
    msg = '[%s] There is no the ground truth dump file for %s.' % (op_name, tensor_id)
    if is_error:
        print_error_log(msg)
    else:
        print_warn_log(msg)
    return msg


def print_start_to_compare_op(op_name: str) -> None:
    """
    Print info log for start to compare op
    :param op_name: the op name
    """
    print_info_log('[%s] Start to compare op "%s".' % (op_name, op_name))


def print_open_file_error(path: str, io_error: any) -> None:
    """
    Print error log for open file error
    :param path: the path
    :param io_error: error info
    """
    print_error_log('Failed to open "%s". %s' % (path, str(io_error)))


def print_write_result_info(prefix: str, path: str) -> None:
    """
    Print info log for write result to file
    :param prefix: the info
    :param path: the path
    """
    print_info_log('The %s have been written to "%s".' % (prefix, path))


def print_only_support_error(prefix: str, value: any, support_info: list) -> None:
    """
    Print error log for only supports error
    :param prefix: the info
    :param value: the value no support
    :param support_info: the support info
    """
    print_error_log(
        "The %s '%s' is invalid. It only supports '%s'." % (prefix, str(value), str(support_info)))


def print_not_match_error(op_name: str, prefix: str, left_value: str, right_value: str, tensor_id: str = None) -> str:
    """
    Print not match error
    :param op_name: the op name
    :param prefix: the info
    :param left_value: the left value
    :param right_value: the right value
    :param tensor_id: the tensor id
    :return message
    """
    line = '[%s] The %s does not match (%s vs %s)' % (op_name, prefix, left_value, right_value)
    if tensor_id:
        line = '%s for %s.' % (line, tensor_id)
        print_error_log(line)
    else:
        line = '%s.' % line
        print_warn_log(line)
    return line


def print_cannot_compare_warning(op_name: str, left_shape: str, right_shape: str) -> str:
    """
    Print cannot compare warning
    :param op_name: the op name
    :param left_shape: the left_shape
    :param right_shape: the right shape
    :return message
    """
    prefix = '[%s] ' % op_name if op_name else ''
    message = '%sDue to the different shapes on the left and right,the left dump data%s can not ' \
              'be compared to the right dump data%s. Please check the batch of the dump data or ' \
              'the shape may be changed due to optimization.' % (prefix, left_shape, right_shape)
    print_warn_log(message)
    return message


def print_npu_path_valid_message(npu_dump_dir: str, dump_file_path_format: str) -> str:
    """
    Print npu path valid message
    :param npu_dump_dir: the npu dump directory
    :param dump_file_path_format : correct dump file path format
    :return message
    """
    message = "The {0} does not match the path format," \
              "please save dump files in the {1} path format".format(npu_dump_dir, dump_file_path_format)
    print_error_log(message)
    return message


def print_out_of_range_error(op_name: str, index_type: str, index: int, range_str: str) -> None:
    """
    Print out of range error
    :param op_name: the op name
    :param index_type: the tensor type
    :param index: the index
    :param range_str: the count
    """
    prefix = ''
    if op_name:
        prefix = '[%s] ' % op_name
    print_error_log('%sThe %s index (%d) is out of range %s. Please check the index.'
                    % (prefix, index_type, index, range_str))


def print_skip_inner_op_msg(op_name: str, is_error: bool) -> None:
    """
    Print warn or error log for skip inner operator
    :param op_name: the op name
    :param is_error: the log lever is error
    :return message
    """
    msg = '[%s] The op "%s" is inner node for multi to multi relation. Skip the op "%s".' \
          % (op_name, op_name, op_name)
    if is_error:
        print_error_log(msg)
    else:
        print_warn_log(msg)


def print_deprecated_warning(file_name: str) -> None:
    """
    Print deprecated warning
    :param file_name: the file name
    """
    print_warn_log('Note that "%s" will be deprecated in a future release. It'
                   ' is recommended to use the next-generation "msaccucmp.py".' % file_name)


def print_skip_quant_info(op_name: str) -> None:
    """
    Print the op skipped info

    :param op_name: the op name
    """

    print_info_log('[%s] This op is in a quant/dequant op pair. Skip the op.' % op_name)
