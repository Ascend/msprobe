
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
"""
Function:
This file mainly involves the common function.
"""
import os
import re
import math
import collections
import csv
import numpy as np

from cmp_utils import common
from cmp_utils import log
from cmp_utils.utils_type import ShapeType
from cmp_utils.constant.const_manager import ConstManager
from cmp_utils.constant.compare_error import CompareError


PATH_BLACK_LIST_REGEX = re.compile(r"[^_A-Za-z0-9/.,-]")  # Includes `,`
MALICIOUS_CSV_PATTERN = re.compile(r'^[＝＋－\+\-=%@]|;[＝＋－\+\-=%@]')


def safe_path_string(value):
    if re.search(PATH_BLACK_LIST_REGEX, value):
        raise ValueError("String parameter contains invalid characters.")
    return value


def sanitize_csv_value(value: str, errors='strict'):
    if errors == 'ignore' or not isinstance(value, str):
        return value

    sanitized_value = value
    try:
        float(value) # in case value is a digit but in str format
    except ValueError as e: # not digit
        if not MALICIOUS_CSV_PATTERN.search(value):
            pass
        elif errors == 'replace':
            sanitized_value = ' ' + value
        else:
            msg = f'Malicious value is not allowed to be written to the csv: {value}'
            log.print_error_log("Please check the value written to the csv")
            raise ValueError(msg) from e

    return sanitized_value


def make_msnpy_file_name(file_path: str, op_name: str, tensor_type: str, index: int, tensor_format: int) -> str:
    """
    Make file name for msnpy
    :param file_path: the file path
    :param op_name: the op name
    :param tensor_type: the tensor type, input or output
    :param index: the tensor index
    :param tensor_format: the tensor format
    :return: the msnpy file name
    """
    name_split = os.path.basename(file_path).split('.')
    if len(name_split) == ConstManager.OFFLINE_FILE_NAME_COUNT and op_name:
        # the index 1 for op_name
        name_split[1] = op_name.split('/')[-1]
        origin_file_name = ".".join(name_split)
    else:
        origin_file_name = os.path.basename(file_path)
    return '%s.%s.%d.%s.npy' % (origin_file_name, tensor_type, index, common.get_format_string(tensor_format))


def check_shape_valid_in_nz(shape: list, tensor_shape: list, is_convert_mode: bool = True) -> None:
    """
    check fractal nz dump data shape is valid
    param:
        shape: target shape
        tensor_shape: current tensor shape
        is_convert_shape: the method is used for two mode, one is compare mode, the other is convert mode
    return: None
    """
    if len(shape) == 0:
        error_msg = 'The format before transfer is FRACTAL_NZ. Please enter a valid shape.'
        _raise_exception_by_convert_mode(is_convert_mode, error_msg)
    origin_shape = []
    for index in range(len(tensor_shape) - 4):
        origin_shape.append(tensor_shape[index])
    origin_shape.append(tensor_shape[-2] * tensor_shape[-3])
    origin_shape.append(tensor_shape[-1] * tensor_shape[-4])
    is_invalid_shape = shape[-1] > origin_shape[-1] or \
                     shape[-1] <= origin_shape[-1] - 16 or \
                     shape[-2] > origin_shape[-2] or \
                     shape[-2] <= origin_shape[-2] - 16
    if len(shape) != len(origin_shape) or is_invalid_shape:
        error_msg = 'The target shape %s is invalid. The recommended shape is %s.' \
            % (convert_shape_to_string(shape), convert_shape_to_string(origin_shape))
        _raise_exception_by_convert_mode(is_convert_mode, error_msg)
    for index in range(len(origin_shape) - 2):
        if shape[index] != origin_shape[index]:
            error_msg = 'The target shape %s is invalid, the recommended shape is %s.' \
                % (convert_shape_to_string(shape), convert_shape_to_string(origin_shape))
            _raise_exception_by_convert_mode(is_convert_mode, error_msg)


def get_string_from_list(string_list: list, splitter: str = ',') -> str:
    """
    Get string from list splitter by splitter
    :param string_list: the list to string
    :param splitter: the splitter, default is ','
    :return: the string
    """
    list_str = []
    for item in string_list:
        if isinstance(item, str):
            list_str.append(item)
        else:
            list_str.append(str(item))
    return splitter.join(list_str)


def convert_ndarray_to_bytes(array: np.ndarray) -> bytes:
    """
    convert ndarray to bytes
    @param array: ndarray
    @return:bytes
    """
    return array.tobytes()


def convert_shape_to_string(shape: list) -> str:
    """
    Convert shape to string
    :param shape: the shape
    :return: the shape string
    """
    return "(%s)" % get_string_from_list(shape, ', ')


def format_value(value: float) -> str:
    """
    Format value, 6 decimal places
    :param value: the value to format
    :return: value with 6 decimal places
    """
    return '{:.6f}'.format(value)


def space_to_comma(value: str) -> str:
    """
    Format convert(space to comma)
    :param value: the value to convert
    :return: the value after convert
    """
    new_value = value.replace(',', '|')
    new_value = new_value.replace(' ', ',')
    return new_value.replace('|', ' ')


def merge_dict(dict_dst: dict, dict_src: dict) -> None:
    """
    Merge dict2 into dict1
    :param dict_dst:
    :param dict_src:
    """
    for key in dict_src.keys():
        if key in dict_dst:
            dict_dst[key] = dict_dst[key] + dict_src[key]
        else:
            dict_dst[key] = dict_src[key]


def sort_result_file_by_index(result_file: str, csv_file: bool = True) -> None:
    """
    Sort compare result
    :param result_file: output file path
    :param csv_file: the result is csv or not
    """
    try:
        # read result file and sort result
        if result_file:
            _sort_result_file_exec(result_file, csv_file)
    except (OSError, SystemError, ValueError, TypeError, RuntimeError, MemoryError) as error:
        log.print_open_file_error(result_file, error)


def get_shape_type(shape_dim_array: list) -> ShapeType:
    """
    Get shape type
    :param shape_dim_array: the shape info
    :return: ShapeType
    """
    return ShapeType.Scalar if sum(shape_dim_array) == len(shape_dim_array) else ShapeType.Tensor


def get_data_type(dump_data_type: str) -> str:
    """
    Get data type
    :param dump_data_type: the shape info
    :return: data type
    """
    if dump_data_type not in ConstManager.DATA_TYPE_TO_STR_DTYPE_MAP:
        return ConstManager.NAN
    return ConstManager.DATA_TYPE_TO_STR_DTYPE_MAP.get(dump_data_type)


def get_address_from_tensor(tensor: any):
    """
    get address from tensor
    args:tensor
    return:address
    """
    if hasattr(tensor, "address") and tensor.address != 0:
        return tensor.address
    else:
        return ConstManager.NAN


def dump_path_contains_npy(dump_path: str) -> bool:
    """
    check dump_file is npy file in dump path
    args: dump_path
    returns: bool
    """
    if dump_path and os.path.isfile(dump_path):
        return dump_path.endswith(ConstManager.NUMPY_SUFFIX)
    if dump_path and os.path.isdir(dump_path):
        return has_npy_at_dir(dump_path)
    return False


def has_npy_at_dir(dump_path: str) -> bool:
    """
    check there is npy file at dump_path
    args:dump_path
    return:bool
    """
    file_list = os.listdir(dump_path)
    for file_path in file_list:
        if str(file_path).endswith(ConstManager.NUMPY_SUFFIX):
            return True
    return False


def _raise_exception_by_convert_mode(is_convert_mode: bool, error_msg: str):
    if is_convert_mode:
        log.print_invalid_nz_dump_data(error_msg, is_error=True)
        raise CompareError(CompareError.MSACCUCMP_INVALID_PARAM_ERROR, error_msg)
    else:
        raise CompareError(CompareError.MSACCUCMP_INVALID_FRACTAL_NZ_DUMP_DATA_ERROR, error_msg)


def _write_sorted_result(result_file: str, sorted_result_line: list, header_list: list, table_header_info: str,
                         csv_file: bool) -> None:
    with os.fdopen(os.open(result_file, ConstManager.WRITE_FLAGS, ConstManager.WRITE_MODES), 'w',
                   newline="") as fp_write:
        for item in sorted_result_line:
            if len(item) < 2:
                log.print_error_log('Failed to write sorted result')
                raise IndexError("Invalid data to save, please check it!")
        if csv_file:
            # write header to file
            writer = csv.writer(fp_write)
            writer.writerow((sanitize_csv_value(ii) for ii in header_list))
            # write sorted result to file
            for item in sorted_result_line:
                writer.writerow((sanitize_csv_value(ii) for ii in item[1]))
        else:
            # write header to file
            fp_write.write(table_header_info)
            # write value to file
            for item in sorted_result_line:
                fp_write.write(item[1])


def _get_header_and_data(csv_file: bool, fp_read: any) -> (str, list, list):
    table_header_info = next(fp_read)
    header_list = []
    origin_result_line = []
    if csv_file:
        header_list = table_header_info.strip().split(',')
        result_reader = csv.reader(fp_read)
        for line in result_reader:
            origin_result_line.append((int(line[0]), line))
    else:
        result_reader = fp_read.readlines()
        for line in result_reader:
            origin_result_line.append((int(line.split(" ")[0]), line))
    return table_header_info, header_list, origin_result_line


def _sort_result_file_exec(result_file: str, csv_file: bool = True) -> None:
    check_file_size(result_file, ConstManager.ONE_HUNDRED_MB)
    with open(result_file, 'r') as fp_read:
        table_header_info, header_list, origin_result_line = _get_header_and_data(csv_file, fp_read)
        sorted_result_line = sorted(origin_result_line, key=lambda s: s[0])
    _write_sorted_result(result_file, sorted_result_line, header_list, table_header_info, csv_file)


def check_file_size(file_path: str, size_limit: int, is_raise=False) -> None:
    try:
        file_size = os.path.getsize(file_path)
    except OSError as os_error:
        log.print_open_file_error(file_path, os_error)
        raise CompareError(CompareError.MSACCUCMP_OPEN_FILE_ERROR) from os_error
    if file_size > size_limit:
        log.print_warn_log(
            'The size (%d) of %r exceeds %dMB, it may task more time to run, please wait.'
            % (file_size, file_path, size_limit / 1024 / 1024))
        if is_raise:
            raise CompareError("%r file size (%d) exceeds %d" % (file_path, file_size, size_limit))


def least_common_multiple(left: int, right: int) -> int:
    """
    Least common multiple, in this file, n could not zero
    :param left: One of the calculation parameters
    :param right: One of the calculation parameters
    :return: left, right Least common multiple
    """
    if left == 0 or right == 0:
        return 0
    return (left * right) // math.gcd(left, right)


def ceiling_divide(left: int, right: int) -> int:
    """
    Ceiling divide, in this file, n could not zero
    :param left: One of the calculation parameters
    :param right: One of the calculation parameters
    :return: left, right Ceiling divide
    """
    if right == 0:
        raise ZeroDivisionError('Can not divide zero.')
    return (left + right - 1) // right


ResultInfo = collections.namedtuple(
    "ResultInfo",
    ["op_name", "dump_match", "result_list",
     "ret", "input_list", "input_result_list",
     "output_result_list", "is_ffts",
     "op_name_origin_output_index_map", "npu_vs_npu"])

