#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
"""
Function:
This file mainly involves the common function.
"""

import os
import re
import math
from functools import wraps
from enum import Enum
import csv
import numpy as np
from dump_data_pb2 import DumpData

from src.compare.cmp_utils import common
from src.compare.cmp_utils import log
from src.compare.cmp_utils.utils_type import ShapeType, PathType
from src.compare.cmp_utils.constant.const_manager import ConstManager
from src.compare.cmp_utils.reg_manager import RegManager
from src.compare.cmp_utils.constant.compare_error import CompareError
from src.compare.dump_parse.big_dump_data import DumpDataHandler
from src.compare.dump_parse.dump_data_object import DumpDataObj, DumpTensor


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
    def check_valid_timestamp(timestamp) -> bool:
        """
        Check if timestamp format is valid
        @param timestamp: timestamp from dump_file_path
        @return: True or False
        """
        return len(timestamp) == ConstManager.TIMESTAMP_LENGTH and timestamp.isdigit()

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


def deserialize_dump_data_to_array(tensor: any) -> any:
    """
    Deserialize dump data to array
    :param tensor: the dump data for input or output
    :return: the numpy array
    """
    if 0 in tensor.shape.dim:
        return np.array([]).reshape(tensor.shape.dim)
    result = np.frombuffer(tensor.data, dtype=common.get_dtype_by_data_type(tensor.data_type))
    return result if tensor.data_type not in ConstManager.SPECIAL_DTYPE else np.unpackbits(result)


def check_hdf5_file_valid(file_path: str) -> bool:
    """
    Check file is hdf5
    :param file_path: the file path
    :return bool
    """
    return os.path.isfile(os.path.realpath(file_path)) and file_path.endswith(".h5")


def get_path_list_for_str(path_str: str) -> list:
    """
    Get path list for string
    :param path_str: the user input string
    :return: the path list
    """
    if ',' not in path_str:
        new_path = os.path.realpath(path_str)
        ret = check_path_valid(new_path, True, False)
        if ret != CompareError.MSACCUCMP_NONE_ERROR:
            raise CompareError(ret)
        return [new_path]
    input_path_list = []
    for input_path in path_str.split(','):
        new_path = os.path.realpath(input_path.strip())
        ret = check_path_valid(new_path, True, False)
        if ret != CompareError.MSACCUCMP_NONE_ERROR:
            continue
        input_path_list.append(new_path)
    if not input_path_list:
        log.print_error_log(
            'There is no valid file in "%s". Please check the path.' % path_str)
        raise CompareError(CompareError.MSACCUCMP_INVALID_PATH_ERROR)
    return input_path_list


def check_name_valid(name: str) -> int:
    """
    Check name valid
    :param name: the name to check
    :return: VectorComparisonErrorCode
    """
    if name == "":
        log.print_error_log("The parameter is null.")
        return CompareError.MSACCUCMP_INVALID_PARAM_ERROR
    name_pattern = re.compile(RegManager.SUPPORT_PATH_PATTERN)
    match = name_pattern.match(name)
    if match is None:
        log.print_only_support_error('name', name, '"A-Za-z0-9_\\./:()=-"')
        return CompareError.MSACCUCMP_INVALID_PARAM_ERROR
    return CompareError.MSACCUCMP_NONE_ERROR


def _check_path_file_or_directory(path: str, path_type: PathType) -> int:
    ret = CompareError.MSACCUCMP_NONE_ERROR
    if path_type == PathType.File:
        if os.path.exists(path) and not os.path.isfile(path):
            log.print_error_log('The path "%s" is not a file. Please check the path.' % path)
            ret = CompareError.MSACCUCMP_INVALID_PATH_ERROR
    elif path_type == PathType.Directory:
        if not os.path.isdir(path):
            log.print_error_log('The path "%s" is not a directory. Please check the path.' % path)
            ret = CompareError.MSACCUCMP_INVALID_PATH_ERROR
    return ret


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


def check_output_path_valid(path: str, exist: bool, path_type: PathType = PathType.Directory) -> int:
    """
    Check output path valid
    :param path: the path to check
    :param exist: the path exist
    :param path_type: the path type
    :return: VectorComparisonErrorCode
    """
    output_path = os.path.realpath(path)
    if path_type == PathType.File:
        output_path = os.path.dirname(output_path)
    if not os.path.exists(output_path):
        try:
            os.makedirs(output_path, mode=0o700)
        except OSError as ex:
            log.print_error_log('Failed to create "%s". %s' % (output_path, str(ex)))
            return CompareError.MSACCUCMP_INVALID_PATH_ERROR
        finally:
            pass
    return check_path_valid(path, exist, True, path_type)


def check_path_valid(path: str, exist: bool, have_write_permission: bool = False,
                     path_type: PathType = PathType.All) -> int:
    """
    Check path valid
    :param path: the path to check
    :param exist: the path exist
    :param have_write_permission: have write permission
    :param path_type: the path type
    :return: VectorComparisonErrorCode
    """
    if path == "":
        log.print_error_log("The path is null.")
        return CompareError.MSACCUCMP_INVALID_PARAM_ERROR

    ret = check_name_valid(path)
    if ret != CompareError.MSACCUCMP_NONE_ERROR:
        return ret

    exist_path = os.path.realpath(path)
    if not exist:
        exist_path = os.path.dirname(exist_path)

    if not os.path.exists(exist_path):
        log.print_error_log('The path "%s" does not exist.' % exist_path)
        return CompareError.MSACCUCMP_INVALID_PATH_ERROR

    if not os.access(exist_path, os.R_OK):
        log.print_error_log('You do not have permission to read the path "%s".' % exist_path)
        return CompareError.MSACCUCMP_INVALID_PATH_ERROR

    if have_write_permission and not os.access(exist_path, os.W_OK):
        log.print_error_log('You do not have permission to write the path "%s".' % exist_path)
        return CompareError.MSACCUCMP_INVALID_PATH_ERROR

    return _check_path_file_or_directory(path, path_type)


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
    is_valid_shape = shape[-1] > origin_shape[-1] or \
                     shape[-1] <= origin_shape[-1] - 16 or \
                     shape[-2] > origin_shape[-2] or \
                     shape[-2] <= origin_shape[-2] - 16
    if len(shape) != len(origin_shape) or is_valid_shape:
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


def read_numpy_file(path: str) -> any:
    """
    Read numpy file
    :param path: the numpy file path
    :return: numpy data
    """
    return DumpDataHandler(path).read_numpy_file()


def convert_dump_data_object(wrap_function: any) -> any:
    """
    This is a wrapper
    @param wrap_function: function need to be wrapped
    @return: inner function
    """
    @wraps(wrap_function)
    def inner(*args, **kwargs):
        try:
            dump_data = wrap_function(*args, **kwargs)
        except CompareError as error:
            if error.code == CompareError.MSACCUCMP_UNMATCH_STANDARD_DUMP_SIZE:
                dump_data = DumpData()
            else:
                raise error
        dump_data_object = convert_dump_data(dump_data)
        return dump_data_object
    return inner


def build_dump_tensor(dump_data_object_data: list) -> None:
    """
    replace the input or output object of DD.DumpData to DumpyTensor
    @param dump_data_object_data: input or output object of DD.DumpData
    @return: None
    """
    for index, tensor in enumerate(dump_data_object_data):
        data_to_np = deserialize_dump_data_to_array(tensor)
        dump_tensor = DumpTensor(index, tensor.data_type, tensor.format, list(tensor.shape.dim),
                                 data_to_np, tensor.size, list(tensor.original_shape.dim),
                                 tensor.address, tensor.sub_format)
        dump_data_object_data[index] = dump_tensor


def convert_dump_data(dump_data: DumpData) -> DumpDataObj:
    """
    Convert dump_data to DumpDataObj
    @param dump_data:  DD.DumpData object
    @return: DumpDataObj object
    """
    dump_data_object = DumpDataObj(dump_data)
    build_dump_tensor(dump_data_object.output_data)
    build_dump_tensor(dump_data_object.input_data)
    dump_data_object.op_name = handle_op_name(dump_data_object.op_name)
    return dump_data_object


@convert_dump_data_object
def parse_dump_file(input_path: str, dump_version: int) -> DumpDataObj:
    """
    Parse dump fil
    :param input_path: the input file path
    :param dump_version: the dump version
    :return: DumpData
    """
    return DumpDataHandler(input_path).parse_dump_data(dump_version)


def convert_ndarray_to_bytes(array: np.ndarray) -> bytes:
    """
    convert ndarray to bytes
    @param array: ndarray
    @return:bytes
    """
    return array.tobytes()


def check_valid_timestamp(timestamp) -> bool:
    """
    Check if timestamp format is valid
    @param timestamp: timestamp from dump_file_path
    @return: True or False
    """
    return len(timestamp) == ConstManager.TIMESTAMP_LENGTH and timestamp.isdigit()


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


def _handle_csv_object(csv_object: any, mapping_file_path: str) -> dict:
    hash_to_file_name_map = {}
    for item in csv_object:
        if len(item) == 2:
            hash_to_file_name_map[item[0]] = item[1]
        else:
            log.print_error_log(
                'The content (%s) of the mapping file "%s" is invalid.' % (item, mapping_file_path))
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
        log.print_error_log('Failed to read csv object. The content of the mapping file "%s" is invalid.'
                            % mapping_file_path)
    except (OSError, SystemError, ValueError, TypeError, RuntimeError, MemoryError) as error:
        log.print_open_file_error(mapping_file_path, error)
    finally:
        pass
    return hash_to_file_name_map


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


def dump_path_contains_npy(dump_path: str):
    """
    check dump_file is npy file in dump path
    args: dump_path
    returns: bool
    """
    if dump_path and os.path.isfile(dump_path):
        return dump_path.endswith(ConstManager.NUMPY_SUFFIX)
    elif dump_path and os.path.isdir(dump_path):
        return has_npy_at_dir(dump_path)
    else:
        return False


def has_npy_at_dir(dump_path: str):
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


def get_op_type_from_file_name(dump_path: str):
    """
    get op_type from dump file name
    """
    dump_file_name = os.path.basename(dump_path).replace("*", "0")
    is_match, match = RegManager.match_group(RegManager.OFFLINE_DUMP_PATTERN, dump_file_name)
    if is_match:
        op_type_end_index = dump_file_name.find('.')
        return dump_file_name[:op_type_end_index]
    return ConstManager.NAN


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
        if csv_file:
            # write header to file
            writer = csv.writer(fp_write)
            writer.writerow(header_list)
            # write sorted result to file
            for item in sorted_result_line:
                writer.writerow(item[1])
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


def check_file_size(file_path: str, size_limit: int) -> None:
    try:
        file_size = os.path.getsize(file_path)
    except OSError as os_error:
        log.print_open_file_error(file_path, os_error)
        raise CompareError(CompareError.MSACCUCMP_OPEN_FILE_ERROR) from os_error
    if file_size > size_limit:
        log.print_warn_log(
            'The size (%d) of %s exceeds %dMB, it may task more time to run, please wait.'
            % (file_size, file_path, size_limit / 1024 / 1024))


def least_common_multiple(left: int, right: int) -> int:
    """
    Least common multiple, in this file, n could not zero
    :param left: One of the calculation parameters
    :param right: One of the calculation parameters
    :return: left, right Least common multiple
    """
    return (left * right) // math.gcd(left, right)


def ceiling_divide(left: int, right: int) -> int:
    """
    Ceiling divide, in this file, n could not zero
    :param left: One of the calculation parameters
    :param right: One of the calculation parameters
    :return: left, right Ceiling divide
    """
    return (left + right - 1) // right


def handle_op_name(file_op_name: str) -> (str, int):
    # filter field '_lxsliceX' and '_sgt_field'
    if ConstManager.FFTS_MANUAL_MODE_FIELD not in file_op_name \
            and ConstManager.SGT_FIELD not in file_op_name:
        return file_op_name
    # field '_lxsliceX' at the end of name
    if ConstManager.FFTS_MANUAL_MODE_FIELD in file_op_name:
        first_match = RegManager.get_matchs(
            RegManager.FFTS_MANUAL_FIELD_PATTERN, file_op_name)[0]
        file_op_name = \
            file_op_name[:first_match.start() - 1] if first_match.end() == first_match.endpos else file_op_name
    # filter field '_sgt_field'
    if ConstManager.SGT_FIELD in file_op_name:
        # field '_sgt_graph' in the name
        end_match = RegManager.get_matchs(
            RegManager.SGT_FLIED_PATTERN, file_op_name)[-1]
        file_op_name = file_op_name[end_match.end() + 1:] if end_match.end() != end_match.endpos else file_op_name
    return file_op_name
