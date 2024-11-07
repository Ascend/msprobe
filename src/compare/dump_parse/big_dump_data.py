
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
"""
Function:
This file mainly involves the dump function.
"""
import os
import time
import struct
import warnings
from typing.io import BinaryIO

import numpy as np
from google.protobuf.message import DecodeError
import dump_data_pb2 as DD

from cmp_utils import path_check
from cmp_utils import log
from cmp_utils import common
from cmp_utils.constant.const_manager import ConstManager
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
        self.dump_data = None

    def parse(self: any) -> DD.DumpData:
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
                # read dump data proto
                self._read_dump_data(dump_file)
                self._check_size_match()
                # read tensor data
                self._read_input_data(dump_file)
                self._read_output_data(dump_file)
                self._read_buffer_data(dump_file)
                self._read_space_data(dump_file)
                return self.dump_data
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

    def _check_size_match(self: any) -> None:
        input_data_size = 0
        for item in self.dump_data.input:
            input_data_size += item.size
        output_data_size = 0
        for item in self.dump_data.output:
            output_data_size += item.size
        buffer_data_size = 0
        for item in self.dump_data.buffer:
            buffer_data_size += item.size
        space_data_size = 0
        for item in self.dump_data.space:
            space_data_size += item.size
        # check 8 + content size + sum(input.data) + sum(output.data)
        # + sum(buffer.data) equal to file size
        if self.header_length + ConstManager.UINT64_SIZE + input_data_size \
                + output_data_size + buffer_data_size + space_data_size != self.file_size:
            log.print_error_log(
                'The file size (%d) of %r is not equal to %d (header length)'
                ' + %d(the size of header content) '
                '+ %d(the sum of input data) + %d(the sum of output data) '
                '+ %d(the sum of buffer data) + %d(the sum of space data). Please check the dump file.'
                % (self.file_size, self.dump_file_path, ConstManager.UINT64_SIZE, self.header_length,
                   input_data_size, output_data_size, buffer_data_size, space_data_size))
            raise CompareError(
                CompareError.MSACCUCMP_UNMATCH_STANDARD_DUMP_SIZE)

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

    def _read_dump_data(self: any, dump_file: BinaryIO) -> None:
        content = dump_file.read(self.header_length)
        self.dump_data = DD.DumpData()
        try:
            self.dump_data.ParseFromString(content)
        except DecodeError as de_error:
            log.print_warn_log(
                'Failed to parse the serialized header content of %r. '
                'Please check the dump file. %s '
                % (self.dump_file_path, str(de_error)))
            raise CompareError(CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR) from de_error
        finally:
            pass

    def _read_input_data(self: any, dump_file: BinaryIO) -> None:
        for data_input in self.dump_data.input:
            data_input.data = dump_file.read(data_input.size)

    def _read_output_data(self: any, dump_file: BinaryIO) -> None:
        for data_output in self.dump_data.output:
            data_output.data = dump_file.read(data_output.size)

    def _read_buffer_data(self: any, dump_file: BinaryIO) -> None:
        for data_buffer in self.dump_data.buffer:
            data_buffer.data = dump_file.read(data_buffer.size)

    def _read_space_data(self: any, dump_file: BinaryIO) -> None:
        for data_space in self.dump_data.space:
            data_space.data = dump_file.read(data_space.size)


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

    def parse_dump_data(self: any, dump_version: int) -> DD.DumpData:
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

        # compatible with earlier versions
        if dump_version == ConstManager.OLD_DUMP_TYPE:
            return self._parse_by_old_version(file_content)

        try:
            if dump_version == ConstManager.PROTOBUF_DUMP_TYPE:
                dump_data = DD.DumpData()
                dump_data.ParseFromString(file_content)
                return dump_data
        except DecodeError as error:
            message = 'Failed to parse the dump file %r, type is protobuf type. Please check the dump file. %s' \
                      % (self.dump_file_path, str(error))
            log.print_error_log(message)
            raise CompareError(CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR, message) from error
        finally:
            pass

        return BigDumpDataParser(self.dump_file_path).parse()

    def _parse_by_old_version(self: any, file_content: any) -> DD.DumpData:
        dump_data = DD.DumpData()
        try:
            decoded_data = file_content.decode('utf-8', errors='ignore')
            parse_size = dump_data.ParseFromString(decoded_data)
        except (DecodeError, UnicodeDecodeError, TypeError):
            return BigDumpDataParser(self.dump_file_path).parse()
        finally:
            pass
        # if parse size is not equal to file size,
        # means the content cannot parse by protobuf
        if parse_size != self.file_size or dump_data.version == '2.0':
            return BigDumpDataParser(self.dump_file_path).parse()
        return dump_data


def _convert_numpy_to_dump(numpy_data: np.ndarray, only_header: bool) -> (DD.DumpData, bytes):
    dump_data = DD.DumpData()
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
