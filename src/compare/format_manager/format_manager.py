
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
"""
Function:
FormatManager class.
This class mainly involves the execute_format_convert function.
"""
import os
import sys
import importlib
from functools import reduce

import numpy as np
import dump_data_pb2 as DD

from cmp_utils import utils, utils_type, path_check
from cmp_utils import log
from cmp_utils import common
from cmp_utils.reg_manager import RegManager
from cmp_utils.constant.const_manager import ConstManager
from cmp_utils.constant.compare_error import CompareError


class SrcToDest:
    """
    The class for src to dest
    """

    def __init__(self: any, src_format: int, dest_format: int, src_shape: any, dest_shape: any) -> None:
        self.src_shape = src_shape
        self.dest_shape = dest_shape
        self.src_format = src_format
        self.dest_format = dest_format

    def get_src_format(self: any) -> int:
        """
        Get src format
        :return: src format
        """
        return self.src_format

    def get_src_shape(self: any) -> any:
        """
        Get src shape
        :return: src shape
        """
        return self.src_shape


class FormatManager:
    """
    The class for format conversion manager
    """

    BUILT_IN_FORMAT_CONVERT_DIR_NAME = "builtin_format_convert"
    CUSTOM_FORMAT_CONVERT_DIR_NAME = "format_convert"
    CONVERT_FUNC_NAME = 'convert'
    CONVERT_ARG_COUNT = 3
    TO_FRACTAL_Z_FUNC_ARG_COUNT = 4

    def __init__(self: any, custom_path: str) -> None:
        self.custom_path = custom_path
        self.built_in_support_format = []
        self.custom_support_format = []
        self.support_format_map = {}

    @staticmethod
    def _add_format_file_to_list(file_path: str, support_format_list: list) -> bool:
        if os.path.isfile(file_path):
            _, match = RegManager.match_group(RegManager.FORMAT_CONVERT_FILE_NAME_PATTERN,
                                              os.path.basename(file_path))
            if match is not None:
                support_format_list.append(match.group(1))
                return True
            log.print_warn_log("The file '%s' does not match 'convert_{src_format}_to_{dest_format}.py in '%s', "
                               "please check the file." % (os.path.basename(file_path),
                                                           os.path.dirname(file_path)))
        return False

    def execute_format_convert(self: any, src_to_dest: SrcToDest, data: any, args: dict) -> np.ndarray:
        """
        Format convert
        :param src_to_dest: the format and shape for src and dest
        :param data: the data to convert
        :param args: the argument contain group conv
        :return: the array after conversion
        """
        src_format_str = common.get_format_string(src_to_dest.src_format)
        dest_format_str = common.get_format_string(src_to_dest.dest_format)
        module_name = 'convert_%s_to_%s' % (src_format_str, dest_format_str)
        if module_name not in self.support_format_map:
            log.print_warn_log("Not supported from %s to %s." % (src_format_str, dest_format_str))
            raise CompareError(CompareError.MSACCUCMP_INVALID_FORMAT_ERROR)

        # call convert function
        try:
            if args.get("group") < 2:
                new_array = self.support_format_map.get(module_name)(
                    src_to_dest.src_shape, src_to_dest.dest_shape, data)
            else:
                new_array = self.support_format_map.get(module_name)(
                    src_to_dest.src_shape, src_to_dest.dest_shape, data, args.get("group"))
        except Exception as err:
            log.print_error_log("Failed to execute '%s' in '%s'. %s"
                                % (self.CONVERT_FUNC_NAME, module_name, str(err)))
            raise CompareError(CompareError.MSACCUCMP_INVALID_CONVERT_FUNC_ERROR) from err
        finally:
            pass

        # check the return value is ndarray
        self._check_return_value_valid(new_array, module_name)

        return new_array

    def check_arguments_valid(self: any) -> None:
        """
        Check argument valid
        """
        self._make_support_format()
        self._make_support_format_map()

    def _make_support_format_by_path(self: any, dir_path: str, support_format_list: list) -> None:
        if not os.path.exists(dir_path):
            log.print_warn_log("There is no '%s' in '%s', please check the custom path."
                               % (self.CUSTOM_FORMAT_CONVERT_DIR_NAME, os.path.dirname(dir_path)))
            return
        one_match = False
        for item in os.listdir(dir_path):
            if self._add_format_file_to_list(os.path.join(dir_path, item), support_format_list):
                one_match = True
        if not one_match:
            log.print_warn_log("There is no legal 'convert_{src_format}_to_{dest_format}.py' file in '%s', "
                               "please check the path." % dir_path)

    def _make_support_format(self: any) -> None:
        """
        Make support format list for custom and built in
        """
        dir_path = os.path.join(os.path.dirname(__file__), self.BUILT_IN_FORMAT_CONVERT_DIR_NAME)
        self._make_support_format_by_path(dir_path, self.built_in_support_format)
        if self.custom_path:
            ret = path_check.check_path_valid(
                self.custom_path, True, False, path_check.PathType.Directory)
            if ret != CompareError.MSACCUCMP_NONE_ERROR:
                raise CompareError(ret)

            path_check.check_path_all_file_exec_valid(self.custom_path)
            dir_path = os.path.join(self.custom_path, self.CUSTOM_FORMAT_CONVERT_DIR_NAME)
            self._make_support_format_by_path(dir_path, self.custom_support_format)
            sys.path.append(self.custom_path)

    def _get_module(self: any, format_name: str, dir_name: str, module_type: str) -> (bool, any):
        if dir_name is self.BUILT_IN_FORMAT_CONVERT_DIR_NAME:
            format_module = importlib.import_module('%s.%s.%s' % ("format_manager",
                                                                  dir_name, format_name))
        if dir_name is self.CUSTOM_FORMAT_CONVERT_DIR_NAME:
            format_module = importlib.import_module('%s.%s' % (dir_name, format_name))

        # check exist convert attr
        if not hasattr(format_module, self.CONVERT_FUNC_NAME):
            log.print_warn_log("[%s] The file '%s' has no attribute '%s'. Please check the file."
                               % (module_type, str(format_module.__file__), self.CONVERT_FUNC_NAME))
            return False, format_module
        return True, format_module

    def _get_function(self: any, format_module: any, module_type: str) -> (bool, any):
        format_func = getattr(format_module, self.CONVERT_FUNC_NAME)
        # check convert is function
        if not callable(format_func):
            log.print_warn_log("[%s] The '%s' in %s is not function. Please check the file." %
                               (module_type, self.CONVERT_FUNC_NAME, str(format_module.__file__)))
            return False, ''

        # check argument count of convert
        if format_func.__code__.co_argcount != self.CONVERT_ARG_COUNT \
                and 'to_FRACTAL_Z' not in format_module.__name__:
            log.print_warn_log("[%s] The argument count (%d) of '%s' in %s is not %d. Please check the file." %
                               (module_type, format_func.__code__.co_argcount, self.CONVERT_FUNC_NAME,
                                str(format_module.__file__), self.CONVERT_ARG_COUNT))
            return False, ''
        if format_func.__code__.co_argcount != self.TO_FRACTAL_Z_FUNC_ARG_COUNT \
                and 'to_FRACTAL_Z' in format_module.__name__:
            log.print_warn_log("[%s] The argument count (%d) of '%s' in %s is not %d. Please check the file." %
                               (module_type, format_func.__code__.co_argcount, self.CONVERT_FUNC_NAME,
                                str(format_module.__file__), self.TO_FRACTAL_Z_FUNC_ARG_COUNT))
            return False, ''
        return True, format_func

    def _make_support_format_map_by_list(self: any, format_list: list, dir_name: str, module_type: str) -> None:
        for format_name in format_list:
            get_ok, format_module = self._get_module(format_name, dir_name, module_type)
            if not get_ok:
                continue
            get_ok, format_func = self._get_function(format_module, module_type)
            if not get_ok:
                continue
            self.support_format_map[format_name] = format_func

    def _make_support_format_map(self: any) -> None:
        """
        Make support format map
        """
        self._make_support_format_map_by_list(self.built_in_support_format, self.BUILT_IN_FORMAT_CONVERT_DIR_NAME,
                                              ConstManager.BUILTIN)
        self._make_support_format_map_by_list(self.custom_support_format, self.CUSTOM_FORMAT_CONVERT_DIR_NAME,
                                              ConstManager.CUSTOM)
        if not self.support_format_map:
            log.print_error_log("There is no support format conversion.")
            raise CompareError(CompareError.MSACCUCMP_INVALID_CONVERT_FUNC_ERROR)

    def _check_return_value_valid(self: any, value: object, format_name: str) -> None:
        if not isinstance(value, np.ndarray):
            log.print_error_log(
                "The return value of '%s' in '%s' is not numpy.ndarray. Please check the return value."
                % (self.CONVERT_FUNC_NAME, format_name))
            raise CompareError(CompareError.MSACCUCMP_INVALID_CONVERT_FUNC_ERROR)


class ShapeConversion:
    """
    The class for shape conversion
    """

    def __init__(self: any, format_manager: FormatManager) -> None:
        self.format_manager = format_manager

    @staticmethod
    def reshape(src_to_dest: SrcToDest, array: any) -> any:
        """
        Reshape the array
        :param src_to_dest:the format and shape for src and dest
        :param array: the data array
        :return: the numpy array
        """
        shape = []
        size = 1
        for dim in src_to_dest.src_shape:
            shape.append(dim)
            size *= dim
        if 0 in array.shape:
            return array.reshape(shape)
        if size != len(array):
            log.print_error_log("The length(%d) is not match with the shape %s."
                                % (len(array), utils.convert_shape_to_string(src_to_dest.src_shape)))
            raise CompareError(CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR)
        return array.reshape(shape)

    @staticmethod
    def get_convert_fractal_nz_to_nd_dest_shape_length(src_to_dest):
        """
        Get convert FRACTAL_NZ to ND dest shape.
        :param src_to_dest: the format and shape for src and dest
        :return: the length of dest shape.
        """
        axis_h = reduce(lambda x, y: x * y, src_to_dest.dest_shape[:-2])
        axis_c1 = src_to_dest.src_shape[-4]
        axis_no = src_to_dest.src_shape[-3]
        axis_ni = src_to_dest.src_shape[-2]
        axis_c0 = src_to_dest.src_shape[-1]
        return axis_h * axis_c1 * axis_no * axis_ni * axis_c0

    def convert_shape(self: any, src_to_dest: SrcToDest, array: any, args: dict) -> any:
        """
        Check shape valid, then convert shape
        :param src_to_dest: the format and shape for src and dest
        :param array: the data to convert
        :param args: the argument contain group conv
        :return: the numpy array
        """
        # from format is ND, no need to convert
        if src_to_dest.src_format == DD.FORMAT_ND:
            return self.reshape(src_to_dest, array)
        # if format and shape is not match, return
        if src_to_dest.src_format == DD.FORMAT_NCHW \
                or src_to_dest.src_format == DD.FORMAT_HWCN \
                or src_to_dest.src_format == DD.FORMAT_NHWC:
            if len(src_to_dest.src_shape) != ConstManager.FOUR_DIMS_LENGTH:
                return self.reshape(src_to_dest, array)
        # src and dest format are the same, no need to convert
        if src_to_dest.src_format == src_to_dest.dest_format:
            return self.reshape(src_to_dest, array)
        # if convert FRACTAL_NZ to ND and the array is not equal to dest shape, no need to convert.
        if src_to_dest.src_format == DD.FORMAT_FRACTAL_NZ \
                and src_to_dest.dest_format == DD.FORMAT_ND \
                and len(src_to_dest.dest_shape) > 2:
            if len(array) != self.get_convert_fractal_nz_to_nd_dest_shape_length(src_to_dest):
                log.print_warn_log(
                    "Cannot convert FRACTAL_NZ to ND, the reason is the length "
                    "of array is not equal to the length of dest shape.")
                return self.reshape(src_to_dest, array)
        if src_to_dest.src_format == DD.FORMAT_FRACTAL_NZ:
            utils.check_shape_valid_in_nz(src_to_dest.dest_shape, src_to_dest.src_shape, is_convert_mode=False)
        return self.format_manager.execute_format_convert(src_to_dest, array, args)
