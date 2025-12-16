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
This file mainly involves the dump function.
"""
import os
import time
import struct
import warnings
import ctypes
import tempfile
import json
from typing.io import BinaryIO

import numpy as np
from google.protobuf.message import DecodeError

from dump_parse.proto_dump_data import DumpData
from cmp_utils import path_check
from cmp_utils import log
from cmp_utils import common
from cmp_utils.constant.const_manager import ConstManager, DD
from cmp_utils.constant.compare_error import CompareError


class BigDumpDataParser:
    """
    The class for big dump data parser
    """
    warnings.filterwarnings("ignore")

    def __init__(self: any, dump_file_path: str) -> None:
        self.dump_file_path = dump_file_path
        self.file_size = 0
        self.header_length = 0
        self.dump_data = DumpData()
        self.dump_json_data = {}
        self.data_types = ['input', 'output', 'buffer', 'space']
        self.parse_dump_so = "libascend_dump_parser.so"

    @staticmethod
    def find_shared_library(lib_name: str, relative_path: str) -> str:
        """
        优先从LD_LIBRARY_PATH查找 so 文件，如果没有则从相对路径中查找。
        """
        # 从环境变量 LD_LIBRARY_PATH 中查找
        ld_paths = os.environ.get("LD_LIBRARY_PATH", "").split(":")
        for path in ld_paths:
            candidate = os.path.join(path, lib_name)
            if os.path.exists(candidate):
                return os.path.realpath(candidate)

        current_dir = os.path.dirname(os.path.realpath(__file__))
        candidate = os.path.realpath(os.path.join(current_dir, relative_path, lib_name))

        if os.path.exists(candidate):
            return candidate

        raise OSError(
            f"Shared library {lib_name} not found."
        )

    def parse(self: any):
        """
        Parse the dump file path by big dump data format
        :return: DumpData
        :exception when read or parse file error
        """
        self.check_argument_valid()
        try:
            with open(self.dump_file_path, 'rb') as dump_file:
                # read header length
                self._read_header_length(dump_file)
                self._parse_dump_to_json()
                self._parse_binary_to_json_data(dump_file)
                return self.dump_data.from_dict(self.dump_json_data)
        except (OSError, IOError) as io_error:
            log.print_error_log('Failed to read the dump file %r. %s'
                                % (self.dump_file_path, str(io_error)))
            raise CompareError(CompareError.MSACCUCMP_OPEN_FILE_ERROR) from io_error
        finally:
            pass

    def check_argument_valid(self: any) -> None:
        """
        check argument valid
        :exception when invalid
        """
        ret = path_check.check_path_valid(self.dump_file_path, True)
        if ret != CompareError.MSACCUCMP_NONE_ERROR:
            raise CompareError(ret)
        # get file size
        try:
            self.file_size = os.path.getsize(self.dump_file_path)
        except (OSError, IOError) as error:
            log.print_error_log('get the size of dump file %r failed. %s'
                                % (self.dump_file_path, str(error)))
            raise CompareError(CompareError.MSACCUCMP_DUMP_FILE_ERROR) from error
        finally:
            pass

        if self.file_size <= ConstManager.UINT64_SIZE:
            log.print_error_log(
                'The size of %r must be greater than %d, but the file size'
                ' is %d. Please check the dump file.'
                % (self.dump_file_path, ConstManager.UINT64_SIZE, self.file_size))
            raise CompareError(CompareError.MSACCUCMP_UNMATCH_STANDARD_DUMP_SIZE)
        if self.file_size > ConstManager.ONE_GB:
            log.print_warn_log(
                'The size (%d) of %r exceeds 1GB, it may task more time to run, please wait.'
                % (self.file_size, self.dump_file_path))

    def _read_header_length(self: any, dump_file: BinaryIO) -> None:
        # read header length
        header_length = dump_file.read(ConstManager.UINT64_SIZE)
        self.header_length = struct.unpack(ConstManager.UINT64_FMT, header_length)[0]
        # check header_length <= file_size - 8
        if self.header_length > self.file_size - ConstManager.UINT64_SIZE:
            log.print_warn_log(
                'The header content size (%d) of %r must be less than or'
                ' equal to %d (file size) - %d (header length).'
                ' Please check the dump file.'
                % (self.header_length, self.dump_file_path, self.file_size, ConstManager.UINT64_SIZE))
            raise CompareError(CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR)
        dump_file.read(self.header_length)

    def _parse_dump_to_json(self):
        # 读取 dump 二进制数据
        with open(self.dump_file_path, 'rb') as dump_file:
            binary_data = dump_file.read()
        # 加载 C 解析库
        try:
            self.parse_dump_so = self.find_shared_library(self.parse_dump_so, "../../../adump/lib64")
            ret = path_check.check_path_valid(self.parse_dump_so, True, False)
            if ret != CompareError.MSACCUCMP_NONE_ERROR:
                raise CompareError(ret)
            if path_check.is_group_and_others_writable(self.parse_dump_so):
                log.print_error_log(f"Failed to load {self.parse_dump_so}, this file is not safe. Please check.")
                raise CompareError(CompareError.MSACCUCMP_INVALID_FILE_ERROR)
            dump_parse_cdll = ctypes.CDLL(self.parse_dump_so)
        except (OSError, IOError) as e:
            log.print_error_log(f"Failed to load {self.parse_dump_so}:{e}")
            raise CompareError(CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR) from e
        data_ptr = ctypes.c_char_p(binary_data)

        # 使用临时文件替代落盘 JSON
        fd, tmp_json_path = tempfile.mkstemp(suffix=".json")
        # 关闭 fd，让 C 库写文件
        os.close(fd)
        try:
            res = dump_parse_cdll.ParseDumpProtoToJson(
                data_ptr, ctypes.c_size_t(len(binary_data)), tmp_json_path.encode('utf-8'))

            if res != 0 or not os.path.isfile(tmp_json_path):
                log.print_error_log(f"Parse dump file to json failed.")
                raise CompareError(CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR)
            # 从临时文件加载 JSON
            with open(tmp_json_path, 'r') as load_f:
                self.dump_json_data = json.load(load_f)
        finally:
            # 删除临时文件，避免落盘
            if os.path.exists(tmp_json_path):
                os.remove(tmp_json_path)

    def _parse_binary_to_json_data(self, dump_file: BinaryIO):
        used_size = self.header_length + ConstManager.UINT64_SIZE
        for data_type in self.data_types:
            for item in self.dump_json_data.get(data_type, []):
                size = int(item.get('size', 0))
                used_size += size
                if used_size > self.file_size:
                    log.print_error_log(f'The size of {self.dump_file_path} is invalid, please check the dump file.')
                    raise CompareError(CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR)
                item['data'] = dump_file.read(size)


class DumpDataHandler:
    """
    Handle dump data
    """

    def __init__(self: any, dump_file_path: str) -> None:
        self.dump_file_path = dump_file_path
        self.file_size = 0

    def check_argument_valid(self: any) -> None:
        """
        check argument valid
        :exception when invalid
        """
        ret = path_check.check_path_valid(self.dump_file_path, True)
        if ret != CompareError.MSACCUCMP_NONE_ERROR:
            raise CompareError(ret)
        # get file size
        try:
            self.file_size = os.path.getsize(self.dump_file_path)
        except (OSError, IOError) as error:
            log.print_error_log('get the size of dump file %r failed. %s'
                                % (self.dump_file_path, str(error)))
            raise CompareError(CompareError.MSACCUCMP_DUMP_FILE_ERROR) from error
        finally:
            pass
        if self.file_size == 0:
            message = 'Failed to parse dump file %r. The file size is zero. Please check the dump file.' \
                      % self.dump_file_path
            log.print_error_log(message)
            raise CompareError(CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR, message)
        if self.file_size > ConstManager.ONE_GB:
            log.print_warn_log(
                'The size (%d) of %r exceeds 1GB, it may task more time to run, please wait.'
                % (self.file_size, self.dump_file_path))

    def read_numpy_file(self: any) -> np.ndarray:
        """
        Read numpy file
        :return: numpy data
        """
        self.check_argument_valid()
        try:
            if self.dump_file_path.endswith(".txt"):
                numpy_data = np.loadtxt(self.dump_file_path)
            else:
                numpy_data = np.load(self.dump_file_path)
        except (OSError, SystemError, ValueError, TypeError, RuntimeError, MemoryError, DecodeError) as error:
            log.print_error_log('Failed to parse dump file "%r". Only data of the numpy format is supported. %s'
                                % (self.dump_file_path, str(error)))
            raise CompareError(CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR) from error
        finally:
            pass
        return numpy_data

    def parse_dump_data(self: any):
        """
        Parse dump file
        :param dump_version: the dump version
        :return: DumpData
        """
        self.check_argument_valid()
        if self.dump_file_path.endswith('.npy'):
            numpy_data = self.read_numpy_file()
            dump_data, _ = _convert_numpy_to_dump(numpy_data, only_header=False)
            return dump_data
        try:
            with open(self.dump_file_path, 'rb') as dump_file:
                file_content = dump_file.read()
        except (OSError, SystemError, ValueError, TypeError, RuntimeError, MemoryError) as error:
            message = 'Failed to open dump file %r. Please check the dump file. %s' \
                      % (self.dump_file_path, str(error))
            log.print_error_log(message)
            raise CompareError(CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR, message) from error
        finally:
            pass
        return BigDumpDataParser(self.dump_file_path).parse()


def _convert_numpy_to_dump(numpy_data: np.ndarray, only_header: bool):
    dump_data = DumpData()
    dump_data.version = '2.0'
    dump_data.dump_time = int(round(time.time() * ConstManager.TIME_LENGTH))
    output = dump_data.output.add()
    output.data_type = common.get_data_type_by_dtype(numpy_data.dtype)
    output.format = DD.FORMAT_RESERVED
    total_size = 1
    for dim in numpy_data.shape:
        output.shape.dim.append(dim)
        total_size *= dim
    # make output data
    struct_format = common.get_struct_format_by_data_type(output.data_type)
    data = struct.pack('%d%s' % (total_size, struct_format), *numpy_data.flatten())
    output.size = len(output.data)
    if not only_header:
        output.data = data
    return dump_data, data


def write_dump_data(numpy_data: np.ndarray, output_dump_path: str) -> None:
    """
    write numpy data to dump data file
    :param numpy_data: the numpy data
    :param output_dump_path: the output dump file path
    :exception when write file error
    """
    # make content of dump data header
    dump_data, data = _convert_numpy_to_dump(numpy_data, only_header=True)
    dump_data_ser = dump_data.SerializeToString()
    try:
        path_check.check_write_path_secure(output_dump_path)
        with os.fdopen(os.open(output_dump_path, ConstManager.WRITE_FLAGS,
                               ConstManager.WRITE_MODES), 'wb') as dump_file:
            # write the header length
            dump_file.write(struct.pack(ConstManager.UINT64_FMT, len(dump_data_ser)))
            # write the header content
            dump_file.write(dump_data_ser)
            # write output data
            dump_file.write(data)
    except IOError as io_error:
        log.print_error_log('Failed to write dump file %r. %s'
                            % (output_dump_path, str(io_error)))
        raise CompareError(CompareError.MSACCUCMP_WRITE_FILE_ERROR) from io_error
    finally:
        pass
