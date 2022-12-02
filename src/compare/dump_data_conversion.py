#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
"""
Function:
DumpDataConversion class. This class mainly involves the convert_data function.
"""

import os
import sys
import argparse
import time
import numpy as np
import utils
import big_dump_data
import log

from const_manager import ConstManager
from multi_convert_process import MultiConvertProcess
from reg_manager import RegManager
from compare_error import CompareError


class DumpDataConversion:
    """
    The class for dump data conversion
    """

    SUPPORT_TARGET = ('dump', 'numpy')
    SUPPORT_TYPE = ('caffe', 'tf', 'quant', 'offline')

    def __init__(self: any) -> None:
        parse = argparse.ArgumentParser()
        parse.add_argument("-target", dest="target",
                           help="<Required> the target after conversion, "
                                "numpy or dump", required=True)
        parse.add_argument("-type", dest="type",
                           help="<Required> the type for data, "
                                "caffe, tf, quant or offline",
                           required=True)
        parse.add_argument("-i", dest="input_path",
                           help="<Required> the input path", required=True)
        parse.add_argument("-o", dest="output_path", default="",
                           help="<Required> the output path", required=True)

        args, _ = parse.parse_known_args(sys.argv[1:])
        self.target = args.target
        self.input_path = os.path.realpath(args.input_path)
        self.output_path = os.path.realpath(args.output_path)
        self.type = args.type
        self.multi_process = MultiConvertProcess(self._convert_file, self.input_path, self.output_path)

    def check_arguments_valid(self: any) -> int:
        """
        Check arguments valid
        :return: error code
        """
        return_code = utils.check_path_valid(self.input_path, True)
        if return_code != CompareError.MSACCUCMP_NONE_ERROR:
            return return_code

        return_code = utils.check_output_path_valid(self.output_path, True)
        if return_code != CompareError.MSACCUCMP_NONE_ERROR:
            return return_code

        if self.target not in self.SUPPORT_TARGET:
            log.print_error_log('The target "%s" does not support, only supports dump and numpy.' % self.target)
            return CompareError.MSACCUCMP_INVALID_TARGET_ERROR
        if self.type not in self.SUPPORT_TYPE:
            log.print_error_log(
                'The type "%s" does not support, only supports tf, caffe, quant and offline.' % self.type)
            return CompareError.MSACCUCMP_INVALID_TYPE_ERROR
        # remove old error file list
        error_file_path = os.path.join(self.output_path, ConstManager.CONVERT_FAILED_FILE_LIST_NAME)
        if os.path.exists(error_file_path):
            os.remove(error_file_path)
        return return_code

    def convert_data(self: any) -> int:
        """
        Convert data between numpy and dump
        :return: error_code
        """
        # 1. check arguments valid
        return_code = self.check_arguments_valid()
        if return_code != CompareError.MSACCUCMP_NONE_ERROR:
            return return_code
        # 2. convert dump data one by one
        if os.path.isfile(self.input_path):
            ret, _ = self._convert_file(self.input_path)
            return ret
        return self.multi_process.process()

    def _get_op_name_from_path(self: any, file_path: str) -> str:
        layer_name = os.path.basename(file_path)
        if self.type == "caffe" or self.type == "tf":
            layer_name = self._get_standard_layer_name(layer_name)
        elif self.type == "quant":
            layer_name = self._get_quant_layer_name(layer_name)
        elif self.type == "offline":
            layer_name = self._get_offline_layer_name(layer_name)
        return layer_name

    def _get_standard_layer_name(self: any, layer_name: str) -> str:
        match = None
        if self.target == "numpy":
            _, match = RegManager.match_group(RegManager.STANDARD_DUMP_PATTERN, layer_name)
            if match is None:
                log.print_warn_log(
                    "Invalid file name: %s, only supports 'op_name.output_index.timestamp.pb'." % layer_name)
                raise CompareError(CompareError.MSACCUCMP_DUMP_FILE_ERROR)
        elif self.target == "dump":
            _, match = RegManager.match_group(RegManager.STANDARD_NUMPY_PATTERN, layer_name)
            if match is None:
                log.print_warn_log(
                    "Invalid file name: %s, only supports 'op_name.output_index.timestamp.npy'." % layer_name)
                raise CompareError(CompareError.MSACCUCMP_DUMP_FILE_ERROR)
        return match.group(1)

    def _get_quant_layer_name(self: any, layer_name: str) -> str:
        match = None
        if self.target == "numpy":
            _, match = RegManager.match_group(RegManager.QUANT_DUMP_PATTERN, layer_name)
            if match is None:
                log.print_warn_log(
                    "Invalid file name: %s, only supports 'op_name.output_index.timestamp.quant'." % layer_name)
                raise CompareError(CompareError.MSACCUCMP_DUMP_FILE_ERROR)
        elif self.target == "dump":
            _, match = RegManager.match_group(RegManager.STANDARD_NUMPY_PATTERN, layer_name)
            if match is None:
                log.print_warn_log(
                    "Invalid file name: %s, only supports 'op_name.output_index.timestamp.npy'." % layer_name)
                raise CompareError(CompareError.MSACCUCMP_DUMP_FILE_ERROR)
        return match.group(1)

    def _get_offline_layer_name(self: any, layer_name: str) -> str:
        match = None
        if self.target == "numpy":
            _, match = RegManager.match_group(RegManager.OFFLINE_DUMP_PATTERN, layer_name)
            if match is None:
                log.print_warn_log(
                    "Invalid file name: %s, only supports 'op_type.op_name.task_id.timestamp'." % layer_name)
                raise CompareError(CompareError.MSACCUCMP_DUMP_FILE_ERROR)
        elif self.target == "dump":
            _, match = RegManager.match_group(RegManager.OFFLINE_NUMPY_PATTERN, layer_name)
            if match is None:
                log.print_warn_log(
                    "Invalid file name: %s, only supports 'op_type.op_name.task_id.timestamp."
                    "output_index.npy/data/bin/txt'." % layer_name)
                raise CompareError(CompareError.MSACCUCMP_DUMP_FILE_ERROR)
        return match.group(1)

    def _save_tensor_to_file(self: any, tensor_list: list, name: str, input_path: str, tensor_type: str) -> None:
        if len(tensor_list) == 0:
            log.print_warn_log('There is no %s in "%s".' % (tensor_type, input_path))
            return
        for (index, tensor) in enumerate(tensor_list):
            log.print_info_log('Start to convert %s:%d of "%s" to numpy file.'
                                 % (tensor_type, index, input_path))
            array = tensor.data
            try:
                array = array.reshape(tensor.shape)
            except ValueError as error:
                log.print_error_log(
                    'Failed to convert %s:%d of "%s" to numpy file. %s' % (tensor_type, index, input_path, error))
                continue
            finally:
                pass
            if self.type == "caffe" or self.type == "tf" or self.type == "quant":
                file_name = "%s.npy" % name
            else:
                file_name = "%s.%s.%d.npy" % (name, tensor_type, index)
            output_dump_path = os.path.join(self.output_path, file_name)
            np.save(output_dump_path, array)
            log.print_info_log('The %s:%d of "%s" has been converted to file "%s".'
                                 % (tensor_type, index, input_path, output_dump_path))

    def _save_buffer_to_file(self: any, tensor_list: list, name: str, input_path: str) -> None:
        if len(tensor_list) == 0:
            log.print_warn_log('There is no buffer data in "%s".' % input_path)
            return
        for (index, tensor) in enumerate(tensor_list):
            buffer_type = ConstManager.BUFFER_TYPE_MAP.get(tensor.buffer_type)
            log.print_info_log('Start to convert %sbuffer:%d of "%s" to bin file.'
                                 % (buffer_type, index, input_path))
            file_name = "%s.%sbuffer.%s.bin" % (name, buffer_type, index)
            output_dump_path = os.path.join(self.output_path, file_name)
            try:
                with os.fdopen(os.open(output_dump_path, ConstManager.WRITE_FLAGS,
                                       ConstManager.WRITE_MODES), 'wb') as output_file:
                    output_file.write(tensor.data)
                    log.print_info_log('The %sbuffer:%d of "%s" has been converted to file "%s".'
                                         % (buffer_type, index, input_path, output_dump_path))
            except IOError as io_error:
                log.print_error_log('Failed to open "%s". %s ' % (output_dump_path, str(io_error)))
            finally:
                pass

    def _convert_file_exec(self: any, input_path: str) -> None:
        name = self._get_op_name_from_path(input_path)
        if self.target == "dump":
            log.print_info_log('Start to convert the numpy file "%s" to dump file.' % input_path)
            numpy_data = utils.read_numpy_file(input_path)
            file_name = ""
            if self.type == "caffe" or self.type == "tf":
                file_name = name + ".pb"
            elif self.type == "quant":
                file_name = name + ".quant"
            elif self.type == "offline":
                file_name = name
            output_dump_path = os.path.join(self.output_path, file_name)
            big_dump_data.write_dump_data(numpy_data, output_dump_path)
            log.print_info_log(
                'The file "%s" has been converted to file "%s".' % (input_path, output_dump_path))
        elif self.target == "numpy":
            dump_data = utils.parse_dump_file(input_path, ConstManager.OLD_DUMP_TYPE)
            self._save_tensor_to_file(dump_data.input_data, name, input_path, 'input')
            self._save_tensor_to_file(dump_data.output_data, name, input_path, 'output')
            self._save_buffer_to_file(dump_data.buffer, name, input_path)
        
    def _convert_file(self: any, input_path: str) -> (int, str):
        return_code = CompareError.MSACCUCMP_NONE_ERROR
        try:
            self._convert_file_exec(input_path)
        except CompareError as error:
            return_code = error.code
        except MemoryError:
            log.print_error_log('Failed to convert file "%s" by memory error.' % input_path)
            return_code = CompareError.MSACCUCMP_UNKNOWN_ERROR
        finally:
            pass
        return return_code, input_path


if __name__ == "__main__":
    log.print_deprecated_warning(sys.argv[0])
    START = time.time()
    CONVERSION = DumpDataConversion()
    try:
        RET = CONVERSION.convert_data()
    except CompareError as err:
        RET = err.code
    END = time.time()
    log.print_info_log("The dump data conversion was completed and took %.2f seconds." % (END - START))
    sys.exit(RET)
