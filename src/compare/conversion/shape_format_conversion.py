
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
"""
Function:
Shape Format Conversion class. This class mainly involves the convert_shape function.
"""

import argparse
import os
import sys
import re

import numpy as np
import dump_data_pb2 as DD

from cmp_utils import utils, utils_type, path_check
from cmp_utils import common
from cmp_utils import log
from cmp_utils.utils import safe_path_string
from cmp_utils.constant.const_manager import ConstManager
from cmp_utils.multi_process.multi_convert_process import MultiConvertProcess
from cmp_utils.reg_manager import RegManager
from cmp_utils.file_utils import FileUtils
from cmp_utils.constant.compare_error import CompareError
from format_manager.format_manager import FormatManager
from format_manager.format_manager import SrcToDest
from format_manager.format_manager import ShapeConversion
from conversion.tensor_conversion import TensorConversion
from dump_parse import dump, dump_utils, mapping
from dump_parse.dump_data_object import DumpDataObj, DumpTensor


def _check_shape_valid(shape_str: str) -> list:
    shape = []
    if shape_str:
        # check shape is valid
        shape_pattern = re.compile(RegManager.SUPPORT_SHAPE_PATTERN)
        match = shape_pattern.match(shape_str)
        if match is None:
            log.print_error_log('The shape(%r) is invalid. The supported formats are "1,3,224,224".' % shape_str)
            raise CompareError(CompareError.MSACCUCMP_INVALID_SHAPE_ERROR)
        split_list = shape_str.split(',')

        for dim_str in split_list:
            dim = int(dim_str)
            if dim > 0:
                shape.append(dim)
            else:
                log.print_error_log('The shape(%r) is invalid. Each dimension must be greater than 0.' % shape_str)
                raise CompareError(CompareError.MSACCUCMP_INVALID_SHAPE_ERROR)
    return shape


class ShapeConversionMain:
    """
    The class for format conversion
    """

    def __init__(self: any) -> None:
        parse = argparse.ArgumentParser()
        parse.add_argument("-i", dest="dump_file_path", default="", type=safe_path_string,
                           help="<Required> the dump file path",
                           required=True)
        parse.add_argument("-format", dest="format", default="",
                           help="<Required> the format to transfer",
                           required=True)
        parse.add_argument("-o", dest="output_path", default="", type=safe_path_string,
                           help="<Required> the output path", required=True)
        parse.add_argument('-tensor', dest="tensor", default="output",
                           help="<Optional> the tensor, input or output")
        parse.add_argument('-index', dest="index", default="0",
                           help="<Optional> the index for tensor")
        parse.add_argument(
            '-shape', dest="shape",
            help="<Optional> the shape for format transfer, currently only"
                 " used for FRACTAL_NZ conversion, shape format is "
                 "([0-9]+,)+[0-9]+, such as 1,3,224,224")
        parse.add_argument("-custom", dest="custom_path", default="", type=safe_path_string,
                           help="<Optional> user-defined path, including format conversion",
                           required=False)
        args, _ = parse.parse_known_args(sys.argv[1:])
        self.dump_file_path = os.path.realpath(args.dump_file_path)
        if os.path.islink(os.path.abspath(args.output_path)):
            log.print_error_log('The path "%r" is a softlink, not permitted.' % args.output_path)
            raise CompareError(CompareError.MSACCUCMP_INVALID_PATH_ERROR)
        self.output_path = os.path.realpath(args.output_path)
        format_str = args.format
        shape_str = args.shape
        self.manager = FormatManager(args.custom_path)
        index_str = args.index
        self.tensor = args.tensor
        if not RegManager.match_pattern(RegManager.NUMBER_PATTERN, index_str):
            log.print_only_support_error('tensor index', index_str, 'natural number')
            raise CompareError(CompareError.MSACCUCMP_INVALID_PARAM_ERROR)
        self.index = int(index_str)
        # check format is valid
        if format_str not in ConstManager.STRING_TO_FORMAT_MAP:
            log.print_error_log('The format "%r" is not supported. '
                                'Please check the format.' % format_str)
            raise CompareError(CompareError.MSACCUCMP_INVALID_PARAM_ERROR)
        self.format_to = ConstManager.STRING_TO_FORMAT_MAP.get(format_str)
        self.shape = _check_shape_valid(shape_str)

    def process(self: any) -> int:
        """
        Check arguments valid, then convert shape and slice data
        :return: error code
        """
        ret = self.check_arguments_valid()
        if ret != CompareError.MSACCUCMP_NONE_ERROR:
            return ret
        try:
            dump_data = dump_utils.parse_dump_file(self.dump_file_path, ConstManager.OLD_DUMP_TYPE)
        except CompareError as error:
            return error.code

        if self.tensor == ConstManager.INPUT:
            tensor_list = dump_data.input_data
        else:
            tensor_list = dump_data.output_data
        if self.index >= len(tensor_list):
            log.print_out_of_range_error('', self.tensor, self.index, '[0, %d)' % len(tensor_list))
            return CompareError.MSACCUCMP_INDEX_OUT_OF_BOUNDS_ERROR

        try:
            self._convert_format(tensor_list[self.index], self.index)
        except CompareError as error:
            return error.code
        return ret

    def check_arguments_valid(self: any) -> int:
        """
        Check arguments valid
        :return error code
        """
        self.manager.check_arguments_valid()

        # check tensor valid
        if self.tensor not in ConstManager.SUPPORT_DETAIL_TYPE:
            log.print_only_support_error('tensor', self.tensor, ConstManager.SUPPORT_DETAIL_TYPE)
            return CompareError.MSACCUCMP_INVALID_PARAM_ERROR

        # check dump file path is valid
        ret = path_check.check_path_valid(
            self.dump_file_path, True, False, path_check.PathType.File)
        if ret != CompareError.MSACCUCMP_NONE_ERROR:
            return ret

        # check output path is valid
        ret = path_check.check_output_path_valid(self.output_path, True)
        if ret != CompareError.MSACCUCMP_NONE_ERROR:
            return ret

        log.print_info_log('Start to transfer format for the dump file "%r".' % self.dump_file_path)
        log.print_info_log('The target format is %s for %r:%d.'
                           % (common.get_format_string(self.format_to), self.tensor, self.index))
        return CompareError.MSACCUCMP_NONE_ERROR

    def _convert_format(self: any, tensor: any, index: int) -> None:
        if tensor.tensor_format == self.format_to:
            log.print_info_log("There is no need to transfer format because the format (%s) is the same."
                               % common.get_format_string(tensor.tensor_format))
            return
        # deserialize dump data
        dump_data_array = tensor.data

        real_format = tensor.tensor_format
        # check the 4-dimension valid
        if tensor.tensor_format == DD.FORMAT_HWCN or tensor.tensor_format == DD.FORMAT_NCHW \
                or tensor.tensor_format == DD.FORMAT_NHWC:
            if len(tensor.shape) != ConstManager.FOUR_DIMS_LENGTH:
                log.print_warn_log(
                    'The format(%s) of the dump data is 4 dimensions, but the shape%s is not 4 dimensions but ND.'
                    % (common.get_format_string(tensor.tensor_format), utils.convert_shape_to_string(tensor.shape)))
                real_format = DD.FORMAT_ND
        log.print_info_log("Before transferring the format, the dump data is %s%s."
                           % (common.get_format_string(real_format),
                              utils.convert_shape_to_string(tensor.shape)))
        if real_format == DD.FORMAT_FRACTAL_NZ:
            utils.check_shape_valid_in_nz(self.shape, tensor.shape)

        dump_data_np = self.manager.execute_format_convert(
            SrcToDest(real_format, self.format_to, tensor.shape, self.shape),
            dump_data_array, {'group': common.get_sub_format(tensor)})
        log.print_info_log("After transferring the format, the dump data is %s%s."
                           % (common.get_format_string(self.format_to),
                              utils.convert_shape_to_string(dump_data_np.shape)))
        # save numpy data to file
        output_file_path = os.path.join(self.output_path, '%s.%s.%d.%s.npy'
                                        % (os.path.basename(self.dump_file_path), self.tensor, index,
                                           utils.get_string_from_list(dump_data_np.shape, 'x')))
        np.save(output_file_path, dump_data_np)
        log.print_info_log('The dump data for %r:%d has been saved to "%r".'
                           % (self.tensor, index, output_file_path))


class FormatConversionMain:
    """
    The class for format conversion
    """

    def __init__(self: any, arguments: any = None) -> None:
        self.path_str = arguments.dump_path
        if os.path.islink(os.path.abspath(arguments.output_path)):
            log.print_error_log('The path "%r" is a softlink, not permitted.' % arguments.output_path)
            raise CompareError(CompareError.MSACCUCMP_INVALID_PATH_ERROR)
        self.output_path = os.path.realpath(arguments.output_path)
        self.attr = {
            'dump_version': arguments.dump_version,
            'output_file_type': arguments.output_file_type
        }
        custom_script_path = arguments.custom_script_path if arguments.custom_script_path else ''
        self.manager = FormatManager(custom_script_path)
        self.one_file_info = None
        if ',' not in self.path_str and os.path.isfile(os.path.realpath(self.path_str)):
            tensor_type = ''
            index_str = '0'
            if arguments.input:
                tensor_type = ConstManager.INPUT
                index_str = arguments.input
            if arguments.output:
                tensor_type = ConstManager.OUTPUT
                index_str = arguments.output
            if not RegManager.match_pattern(RegManager.NUMBER_PATTERN, index_str):
                log.print_only_support_error('tensor index', index_str, 'natural number')
                raise CompareError(CompareError.MSACCUCMP_INVALID_PARAM_ERROR)
            self.index = int(index_str)
            shape = _check_shape_valid(arguments.shape)
            self.one_file_info = {'tensor': tensor_type, 'index': int(index_str), 'shape': shape}

        format_str = arguments.format
        # check format is valid
        if format_str not in ConstManager.STRING_TO_FORMAT_MAP:
            log.print_error_log('The format "%s" is not supported. Please check the format.' % format_str)
            raise CompareError(CompareError.MSACCUCMP_INVALID_PARAM_ERROR)
        self.format_to = ConstManager.STRING_TO_FORMAT_MAP.get(format_str)
        self.multi_process = None
        self.input_path = []

    def convert_format(self: any) -> int:
        """
        Convert dump data to numpy or bin file
        :return error_code
        """
        # 1. check arguments valid
        ret = self.check_arguments_valid()
        if ret != CompareError.MSACCUCMP_NONE_ERROR:
            log.print_error_log('Check arguments Failed. Please recheck the arguments')
        # 2. convert format for dump data
        if len(self.input_path) == 1 and os.path.isfile(self.input_path[0]):
            ret, _ = self._convert_format_for_one_file(self.input_path[0])
            if ret != CompareError.MSACCUCMP_NONE_ERROR:
                log.print_error_log('Failed to convert format for %s".' % self.input_path[0])
            return ret
        return self.multi_process.process()

    def check_arguments_valid(self: any) -> int:
        """
        Check arguments valid
        :return: error code
        """
        self.manager.check_arguments_valid()

        # check dump file path is valid
        self.input_path = path_check.get_path_list_for_str(self.path_str)
        self.multi_process = MultiConvertProcess(self._convert_format_for_one_file, self.input_path,
                                                 self.output_path)

        # check output path is valid
        ret = path_check.check_output_path_valid(self.output_path, True)
        if ret != CompareError.MSACCUCMP_NONE_ERROR:
            return ret

        log.print_info_log('The target format is "%s".'
                           % common.get_format_string(self.format_to))

        return CompareError.MSACCUCMP_NONE_ERROR

    def _save_to_file(self: any, *args: any) -> None:
        output_format, dump_data_np, tensor_type, index, dump_file_path, tensor, op_name = args
        if self.attr.get('output_file_type') == 'npy':
            file_name = '%s.%s.%d.%s.npy' % (os.path.basename(dump_file_path), tensor_type, index,
                                             utils.get_string_from_list(dump_data_np.shape, 'x'))
            file_name = FileUtils.handle_too_long_file_name(
                file_name, '.npy', os.path.join(self.output_path, ConstManager.MAPPING_FILE_NAME))
        elif self.attr.get('output_file_type') == 'msnpy':
            file_name = utils.make_msnpy_file_name(dump_file_path, op_name, tensor_type, index, output_format)
            file_name = FileUtils.handle_too_long_file_name(
                file_name, '.npy', os.path.join(self.output_path, ConstManager.MAPPING_FILE_NAME))
        else:
            file_name = '%s.%s.%d.%s.%s.%s.bin' % (os.path.basename(dump_file_path), tensor_type, index,
                                                   utils.get_string_from_list(dump_data_np.shape, '_'),
                                                   np.dtype(common.get_dtype_by_data_type(tensor.data_type)).name,
                                                   common.get_format_string(output_format))
            file_name = FileUtils.handle_too_long_file_name(
                file_name, '.bin', os.path.join(self.output_path, ConstManager.MAPPING_FILE_NAME))
        output_file_path = os.path.join(self.output_path, file_name)
        FileUtils.save_array_to_file(output_file_path, dump_data_np,
                                     self.attr.get('output_file_type') != 'bin', dump_data_np.shape)
        log.print_info_log(
            'The data of %s:%s has been saved to "%r".' % (tensor_type, index, output_file_path))

    def _get_format_and_shape(self: any, tensor: any, index: int, tensor_type: str,
                              dump_file_path: str) -> (int, DD.Shape):
        # check the 4-dimension valid
        real_format = tensor.tensor_format
        if tensor.tensor_format == DD.FORMAT_HWCN or tensor.tensor_format == DD.FORMAT_NCHW \
                or tensor.tensor_format == DD.FORMAT_NHWC:
            if len(tensor.shape) != ConstManager.FOUR_DIMS_LENGTH:
                log.print_warn_log(
                    'The format(%s) of the dump data is 4 dimensions, but the shape %s is not 4 dimensions,'
                    ' the real format is ND for %s:%d of "%r".'
                    % (common.get_format_string(tensor.tensor_format), utils.convert_shape_to_string(tensor.shape),
                       tensor_type, index, dump_file_path))
                real_format = DD.FORMAT_ND

        shape = []
        if self.one_file_info:
            shape = self.one_file_info.get('shape')
        if len(shape) == 0:
            shape = tensor.original_shape
        if real_format == DD.FORMAT_FRACTAL_NZ:
            utils.check_shape_valid_in_nz(shape, tensor.shape)

        return real_format, shape

    def _convert_format_for_one_tensor(self: any, *args: any) -> None:
        tensor, index, tensor_type, dump_file_path, op_name = args
        # deserialize dump data
        dump_data_array = tensor.data

        real_format, shape_to = self._get_format_and_shape(tensor, index, tensor_type, dump_file_path)

        log.print_info_log(
            "Before transferring the format, the data of %s:%d is %s%s."
            % (tensor_type, index, common.get_format_string(real_format),
               utils.convert_shape_to_string(tensor.shape)))
        src_to_dest = SrcToDest(real_format, self.format_to, tensor.shape, shape_to)
        if real_format == self.format_to:
            log.print_info_log(
                'There is no need to transfer format because the format (%s) '
                'is the same for %s:%d of "%r".'
                % (common.get_format_string(tensor.tensor_format), tensor_type,
                   index, dump_file_path))
            dump_data_np = ShapeConversion(self.manager).reshape(src_to_dest, dump_data_array)
        else:
            # convert shape
            dump_data_np = self.manager.execute_format_convert(
                src_to_dest, dump_data_array, {'group': common.get_sub_format(tensor)})
            log.print_info_log(
                "After transferring the format, the data of %s:%d is %s%s."
                % (tensor_type, index, common.get_format_string(self.format_to),
                   utils.convert_shape_to_string(dump_data_np.shape)))
        output_format = self.format_to
        # slice data
        try:
            if len(shape_to) > 0 and real_format != self.format_to:
                old_shape_str = utils.convert_shape_to_string(dump_data_np.shape)
                tensor_conversion = TensorConversion(None, self.manager, is_detail=False)
                dump_data_np = tensor_conversion.slice_data(dump_data_np, shape_to)
                log.print_info_log('The data of %s:%d has been sliced from %s to %s.' %
                                   (tensor_type, index, old_shape_str,
                                    utils.convert_shape_to_string(shape_to)))
        except (OSError, SystemError, ValueError, TypeError, RuntimeError,
                MemoryError, CompareError):
            log.print_error_log('Failed to slice data for %r from %s to %s.'
                                % (dump_file_path,
                                   utils.convert_shape_to_string(dump_data_np.shape),
                                   utils.convert_shape_to_string(shape_to)))
            output_format = tensor.tensor_format
        finally:
            pass
        # save numpy data to file
        self._save_to_file(output_format, dump_data_np, tensor_type, index, dump_file_path, tensor, op_name)

    def _save_file_for_convert_failed_tensor(self: any, *args: any) -> None:
        tensor, dump_path, tensor_type, index, op_name = args
        if self.attr.get('output_file_type') == 'msnpy':
            file_name = utils.make_msnpy_file_name(dump_path, op_name, tensor_type, index, tensor.tensor_format)
            file_name = FileUtils.handle_too_long_file_name(
                file_name, '.npy', os.path.join(self.output_path, ConstManager.MAPPING_FILE_NAME))
            output_file_path = os.path.join(self.output_path, file_name)
            FileUtils.save_array_to_file(output_file_path, tensor.data,
                                         self.attr.get('output_file_type') != 'bin',
                                         tensor.shape)
            log.print_info_log('The data of %s:%d has been parsed into "%r".'
                               % (tensor_type, index, output_file_path))

    def _convert_format_for_tensor(self: any, tensor_list: list, dump_file_path: str, tensor_type: str,
                                   op_name: str) -> (int, str):
        ret = CompareError.MSACCUCMP_NONE_ERROR
        msg = ""
        for index, tensor in enumerate(tensor_list):
            try:
                self._convert_format_for_one_tensor(tensor, index, tensor_type, dump_file_path, op_name)
            except (OSError, SystemError, ValueError, TypeError, RuntimeError,
                    MemoryError) as error:
                log.print_error_log('Failed to convert format for %s:%d of "%r". %s'
                                    % (tensor_type, index, dump_file_path, error))
                ret = CompareError.MSACCUCMP_UNKNOWN_ERROR
                msg += ',%s:%d' % (tensor_type, index)
                self._save_file_for_convert_failed_tensor(tensor, dump_file_path, tensor_type, index, op_name)
            except CompareError as error:
                log.print_error_log('Failed to convert format for %s:%d of "%r"'
                                    % (tensor_type, index, dump_file_path))
                ret = error.code
                msg += ',%s:%d' % (tensor_type, index)
                self._save_file_for_convert_failed_tensor(tensor, dump_file_path, tensor_type, index, op_name)
            finally:
                pass
        return ret, msg

    def _convert_format_for_input_or_output(self: any, dump_data: DumpDataObj, dump_file_path: str) -> None:
        if self.one_file_info.get('tensor') == ConstManager.INPUT:
            tensor_list = dump_data.input_data
        else:
            tensor_list = dump_data.output_data
        if self.one_file_info.get('index') >= len(tensor_list):
            log.print_error_log(
                'The %s index (%d) is out of range [0, %d). Please check the index.'
                % (self.one_file_info.get('tensor'), self.one_file_info.get('index'),
                   len(tensor_list)))
            raise CompareError(CompareError.MSACCUCMP_INDEX_OUT_OF_BOUNDS_ERROR)
        self._convert_format_for_one_tensor(
            tensor_list[self.one_file_info.get('index')], self.one_file_info.get('index'),
            self.one_file_info.get('tensor'), dump_file_path, dump_data.op_name)

    def _convert_format_for_tensor_list(self: any, dump_data: DumpDataObj, dump_file_path: str) -> (int, str):
        ret = CompareError.MSACCUCMP_NONE_ERROR
        msg = dump_file_path
        ret_input, msg_input = self._convert_format_for_tensor(dump_data.input_data, dump_file_path, ConstManager.INPUT,
                                                               dump_data.op_name)
        if ret_input != CompareError.MSACCUCMP_NONE_ERROR:
            ret = ret_input
        msg += msg_input
        ret_output, msg_output = self._convert_format_for_tensor(dump_data.output_data, dump_file_path,
                                                                 ConstManager.OUTPUT, dump_data.op_name)
        if ret_output != CompareError.MSACCUCMP_NONE_ERROR:
            ret = ret_output
        msg += msg_output
        return ret, msg

    def _convert_format_for_one_file(self: any, dump_file_path: str) -> (int, str):
        ret = CompareError.MSACCUCMP_NONE_ERROR
        msg = dump_file_path
        log.print_info_log('Start to transfer format for the dump file "%r".' % dump_file_path)
        try:
            dump_data = dump_utils.parse_dump_file(dump_file_path, self.attr.get('dump_version'))

        except CompareError as error:
            return error.code, msg

        try:
            if self.one_file_info and self.one_file_info.get('tensor'):
                self._convert_format_for_input_or_output(dump_data, dump_file_path)
            else:
                ret, msg = self._convert_format_for_tensor_list(dump_data, dump_file_path)
        except (OSError, SystemError, ValueError, TypeError, RuntimeError, MemoryError) as error:
            log.print_error_log('Failed to convert format for "%r". %s' % (dump_file_path, error))
            ret = CompareError.MSACCUCMP_UNKNOWN_ERROR
        except CompareError as error:
            ret = error.code
        return ret, msg
