#!/usr/bin/env python
# coding=utf-8
"""
Function:
This file mainly involves the common function.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2019-2021
"""

import os
import re
from enum import Enum

import csv
import numpy as np

import dump_data_pb2 as DD
import common
import log

from const_manager import ConstManager

from reg_manager import RegManager

from big_dump_data import DumpDataHandler

from compare_error import CompareError


class ShapeType(Enum):
    """
    The enum for shape type
    """
    Scalar = 0
    Vector = 1
    Matrix = 2
    Tensor = 3


class FusionRelation(Enum):
    """
    The enum for fusion relation
    """
    OneToOne = 0
    MultiToOne = 1
    OneToMulti = 2
    MultiToMulti = 3
    L1Fusion = 4


class PathType(Enum):
    """
    The enum for path type
    """
    All = 0
    File = 1
    Directory = 2


class DatasetAttr(Enum):
    """
    The enum for pytorch dump data attribute
    """
    DataType = 0
    DeviceType = 1
    FormatType = 2
    Type = 3
    Stride = 4


class DeviceType(Enum):
    """
    The enum for device type
    """
    GPU = 1
    NPU = 10
    CPU = 0


def deserialize_dump_data_to_array(tensor: any) -> any:
    """
    Deserialize dump data to array
    :param tensor: the dump data for input or output
    :return: the numpy array
    """
    return np.frombuffer(tensor.data, dtype=common.get_dtype_by_data_type(tensor.data_type))


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


def parse_dump_file(input_path: str, dump_version: int) -> DD.DumpData:
    """
    Parse dump fil
    :param input_path: the input file path
    :param dump_version: the dump version
    :return: DumpData
    """
    return DumpDataHandler(input_path).parse_dump_data(dump_version)


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
        return "NaN"
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
        return "NaN"


def dump_path_contains_npy(dump_path: str):
    """
    check dump_file is npy file in dump path
    args: dump_path
    returns: bool
    """
    if dump_path and os.path.isfile(dump_path):
        return dump_path.endswith(ConstManager.NUMPY_SUFFIX)
    elif dump_path and os.path.isdir(dump_path):
        return _has_npy_at_dir(dump_path)
    else:
        return False


def _has_npy_at_dir(dump_path: str):
    file_list = os.listdir(dump_path)
    for file_path in file_list:
        if str(file_path).endswith(ConstManager.NUMPY_SUFFIX):
            return True
    return False


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
    with open(result_file, 'r') as fp_read:
        table_header_info, header_list, origin_result_line = _get_header_and_data(csv_file, fp_read)
        sorted_result_line = sorted(origin_result_line, key=lambda s: s[0])
    _write_sorted_result(result_file, sorted_result_line, header_list, table_header_info, csv_file)
