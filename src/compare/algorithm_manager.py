#!/usr/bin/env python
# coding=utf-8
"""
Function:
AlgorithmManager class.
This class mainly involves the compare function.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2019-2021
"""
import os
import re
import sys
import time
import importlib
import log
import utils
import numpy as np

from algorithm_parameter import AlgorithmParameter

from const_manager import ConstManager

from reg_manager import RegManager

from compare_error import CompareError
from file_utils import FileUtils


class AlgorithmManager:
    """
    The class for algorithm manager
    """

    def __init__(self: any, custom_script_path: str, select_algorithm: object, algorithm_options: str) -> None:
        self.custom_path = custom_script_path
        self.built_in_support_algorithm = []
        self.custom_support_algorithm = []
        self._make_support_algorithm()
        self.select_algorithm_list = self._parse_selection_algorithm(select_algorithm)
        self.algorithm_param = self._parse_algorithm_argument(algorithm_options)
        self.support_algorithm_map = self._make_select_algorithm_map()

    @staticmethod
    def _check_value_invalid(value: object, parameter: str) -> None:
        if (isinstance(value, str) and not value) or (isinstance(value, list) and len(value) != 2):
            log.print_error_log('The algorithm argument (%s) is invalid, just supports '
                                '"algorithm_name:name1=value1,name2=value2;".' % parameter)
            raise CompareError(CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    @staticmethod
    def _update_algorithm_param(algorithm_name: str, algorithm_param: dict, param_map: dict) -> None:
        if RegManager.match_pattern(RegManager.BUILTIN_ALGORITHM_INDEX_PATTERN, algorithm_name):
            algorithm_name = ConstManager.BUILT_IN_ALGORITHM[int(algorithm_name)]
        if algorithm_name in algorithm_param:
            algorithm_param[algorithm_name] = dict(algorithm_param.get(algorithm_name), **param_map)
        else:
            algorithm_param[algorithm_name] = param_map

    @staticmethod
    def _get_function(algorithm_module: any, module_type: str) -> (bool, any):
        algorithm_func = getattr(algorithm_module, ConstManager.COMPARE_FUNC_NAME)
        # check compare is function
        if not callable(algorithm_func):
            log.print_warn_log("[%s] The '%s' in %s is not function. Please check the file." %
                               (module_type, ConstManager.COMPARE_FUNC_NAME, str(algorithm_module.__file__)))
            return False, ''

        # check argument count of compare
        if algorithm_func.__code__.co_argcount != ConstManager.COMPARE_ARGUMENT_COUNT:
            log.print_warn_log("[%s] The argument count (%d) of '%s' in %s is not %d. Please check the file." %
                               (module_type, algorithm_func.__code__.co_argcount, ConstManager.COMPARE_FUNC_NAME,
                                str(algorithm_module.__file__), ConstManager.COMPARE_ARGUMENT_COUNT))
            return False, ''
        return True, algorithm_func

    @staticmethod
    def _add_algorithm_file_to_list(file_path: str, support_algorithm_list: list) -> bool:
        if os.path.isfile(file_path):
            file_name_pattern = re.compile(ConstManager.ALGORITHM_FILE_NAME_PATTERN)
            match = file_name_pattern.match(os.path.basename(file_path))
            if match is not None:
                support_algorithm_list.append(match.group(1))
                return True
            log.print_warn_log("The file '%s' does not match 'alg_{algorithm_name}.py'"
                               " in '%s', please check the file." % (os.path.basename(file_path),
                                                                     os.path.dirname(file_path)))
        return False

    @staticmethod
    def _check_data_size_valid(my_output_dump_data: any, ground_truth_dump_data: any, args: dict) -> None:
        my_output_dump_data_size = len(my_output_dump_data)
        if my_output_dump_data_size == 0 and args.get('my_output_dump_file') is not None:
            msg = 'The dump data size is 0 in %s.' % args.get('my_output_dump_file')
            log.print_warn_log(msg)
        ground_truth_dump_data_size = len(ground_truth_dump_data)
        if ground_truth_dump_data_size == 0 and args.get('ground_truth_dump_file') is not None:
            msg = 'The dump data size is 0 in %s.' % args.get('ground_truth_dump_file')
            log.print_warn_log(msg)

        if my_output_dump_data_size != ground_truth_dump_data_size \
                and args.get('my_output_dump_file') is not None \
                and args.get('ground_truth_dump_file') is not None:
            msg = "The my output dump data size (%d) in '%s' does not match the ground truth dump data size (%d) " \
                  "in '%s'." % (my_output_dump_data_size, args.get('my_output_dump_file'),
                                ground_truth_dump_data_size, args.get('ground_truth_dump_file'))
            log.print_warn_log(msg)
            raise CompareError(CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR, msg)

    @staticmethod
    def _check_return_value_valid(value: object, select_algorithm: str) -> None:
        # check the return value is string
        if not isinstance(value, str):
            err_msg = "The return value (%s) of '%s' in '%s' is not string. Please check the return value." \
                      % (value, ConstManager.COMPARE_FUNC_NAME, select_algorithm)
            raise CompareError(ConstManager.COMPARE_FUNC_NAME, err_msg)

    def compare(self: any, my_output_dump_data: any, ground_truth_dump_data: any, args: dict) -> (list, list):
        """
        Compare the my output dump data and the ground truth dump data by select algorithm
        :param my_output_dump_data: the my output dump data to compare
        :param ground_truth_dump_data: the ground truth dump data to compare
        :param args: the algorithm parameter
        :return the result list and compare_fail_message
        """
        self._check_data_size_valid(my_output_dump_data, ground_truth_dump_data, args)
        result = []
        error_msg = []
        for select_algorithm, compare_func in self.support_algorithm_map.items():
            if my_output_dump_data.dtype == np.bool_ and ground_truth_dump_data.dtype == np.bool_ and \
                    select_algorithm not in ConstManager.BOOL_ALGORITHM:
                result.append(ConstManager.NAN)
                error_msg += ["Algorithm %s does not support Boolean types." % select_algorithm]
            else:
                alg_args = self._make_algorithm_param(select_algorithm, args)
                # call compare function
                alg_result, alg_error_msg = self._call_compare_function(
                    compare_func, my_output_dump_data, ground_truth_dump_data, alg_args, select_algorithm)
                result.append(alg_result)
                if alg_error_msg:
                    error_msg += [alg_error_msg]
        return result, error_msg

    def get_result_title(self: any) -> list:
        """
        Get algorithm name list
        :return: the list
        """
        return list(self.support_algorithm_map.keys())

    def make_nan_result(self: any) -> list:
        """
        Make nan result for compare algorithm
        :return: the list, result is nan
        """
        return [ConstManager.NAN] * len(self.support_algorithm_map)

    def _add_algorithm_name_to_list(self: any, algorithm_name: str, select_algorithm_list: list) -> None:
        if not algorithm_name:
            return
        if RegManager.match_pattern(RegManager.BUILTIN_ALGORITHM_INDEX_PATTERN, algorithm_name):
            if ConstManager.BUILT_IN_ALGORITHM[int(algorithm_name)] not in select_algorithm_list:
                select_algorithm_list.append(ConstManager.BUILT_IN_ALGORITHM[int(algorithm_name)])
            return
        if algorithm_name not in self.built_in_support_algorithm + self.custom_support_algorithm:
            log.print_warn_log("The '%s' does not supported in builtin or custom. Please check the "
                               "algorithm name or algorithm index." % algorithm_name)
            return
        if algorithm_name not in select_algorithm_list:
            select_algorithm_list.append(algorithm_name)

    def _make_all_algorithm_list(self: any, select_algorithm: str) -> list:
        select_algorithm_list = []
        if select_algorithm.lower() == 'all':
            select_algorithm_list = ConstManager.BUILT_IN_ALGORITHM
            for item in self.custom_support_algorithm:
                if item not in self.custom_support_algorithm:
                    select_algorithm_list.append(item)
            return select_algorithm_list
        return select_algorithm_list

    def _parse_selection_algorithm(self: any, select_algorithm: object) -> list:
        if not select_algorithm:
            log.print_error_log('There is no algorithm to select. Please select at least one algorithm.')
            raise CompareError(CompareError.MSACCUCMP_INVALID_PARAM_ERROR)
        if isinstance(select_algorithm, list):
            select_algorithm = ','.join(str(x) for x in select_algorithm)

        select_algorithm_list = self._make_all_algorithm_list(select_algorithm)
        if select_algorithm_list:
            return select_algorithm_list
        for item in select_algorithm.split(','):
            self._add_algorithm_name_to_list(item.strip(), select_algorithm_list)
        if not select_algorithm_list:
            log.print_error_log(
                "The algorithm in '%s' does not supported. Just supports %s. Please check the "
                "select algorithm name." % (select_algorithm,
                                            self.built_in_support_algorithm + self.custom_support_algorithm))
            raise CompareError(CompareError.MSACCUCMP_INVALID_ALGORITHM_ERROR)
        return select_algorithm_list

    def _change_str_to_map(self: any, algorithm_param_info: list, algorithm_args: str) -> dict:
        param_map = {}
        for param_str in algorithm_param_info:
            if param_str.strip():
                param = param_str.split('=')
                self._check_value_invalid(param, algorithm_args)
                self._check_value_invalid(param[0].strip(), algorithm_args)
                self._check_value_invalid(param[1].strip(), algorithm_args)
                param_map[param[0].strip()] = param[1].strip()
        return param_map

    def _parse_algorithm_argument(self: any, algorithm_options: str) -> dict:
        algorithm_param = {}
        if not algorithm_options:
            return algorithm_param

        for algorithm_args in algorithm_options.split(';'):
            if algorithm_args:
                algorithm_param_info = algorithm_args.split(':')
                self._check_value_invalid(algorithm_param_info, algorithm_args)
                algorithm_name = algorithm_param_info[0].strip()
                self._check_value_invalid(algorithm_name, algorithm_args)
                param_map = self._change_str_to_map(algorithm_param_info[1].split(','), algorithm_args)
                self._update_algorithm_param(algorithm_name, algorithm_param, param_map)
        return algorithm_param

    def _make_support_algorithm(self: any) -> None:
        """
        Make support algorithm list for custom and built in
        """
        dir_path = os.path.join(os.path.dirname(__file__), ConstManager.BUILT_IN_ALGORITHM_DIR_NAME)
        self._make_support_algorithm_by_path(dir_path, self.built_in_support_algorithm)
        if self.custom_path:
            ret = utils.check_path_valid(
                self.custom_path, True, False, utils.PathType.Directory)
            if ret != CompareError.MSACCUCMP_NONE_ERROR:
                raise CompareError(ret)
            dir_path = os.path.join(self.custom_path, ConstManager.CUSTOM_ALGORITHM_DIR_NAME)
            self._make_support_algorithm_by_path(dir_path, self.custom_support_algorithm)

    def _get_module(self: any, algorithm_name: str) -> (bool, any, str):
        if algorithm_name in self.custom_support_algorithm:
            sys.path.append(self.custom_path)
            algorithm_module = importlib.import_module('%s.alg_%s' %
                                                       (ConstManager.CUSTOM_ALGORITHM_DIR_NAME, algorithm_name))
            module_type = ConstManager.CUSTOM
        elif algorithm_name in self.built_in_support_algorithm:
            algorithm_module = importlib.import_module('%s.alg_%s' %
                                                       (ConstManager.BUILT_IN_ALGORITHM_DIR_NAME, algorithm_name))
            module_type = ConstManager.BUILTIN
        else:
            return False, '', ''
        # check exist compare attr
        if not hasattr(algorithm_module, ConstManager.COMPARE_FUNC_NAME):
            log.print_warn_log("[%s] The file '%s' has no attribute '%s'. Please check the file."
                               % (module_type, str(algorithm_module.__file__), ConstManager.COMPARE_FUNC_NAME))
            return False, algorithm_module, module_type
        return True, algorithm_module, module_type

    def _make_select_algorithm_map(self: any) -> dict:
        """
        Make support algorithm map by select algorithm list
        """
        support_algorithm_map = {}
        for select_algorithm in self.select_algorithm_list:
            get_ok, algorithm_module, module_type = self._get_module(select_algorithm)
            if not get_ok:
                continue
            get_ok, algorithm_func = self._get_function(algorithm_module, module_type)
            if not get_ok:
                continue
            support_algorithm_map[select_algorithm] = algorithm_func
        if not support_algorithm_map:
            log.print_error_log(
                "The algorithm in %s does not supported. Just supports %s. Please check the "
                "select algorithm name." % (self.select_algorithm_list,
                                            self.built_in_support_algorithm + self.custom_support_algorithm))
            raise CompareError(CompareError.MSACCUCMP_INVALID_ALGORITHM_ERROR)
        return support_algorithm_map

    def _make_support_algorithm_by_path(self: any, dir_path: str, support_algorithm_list: list) -> None:
        if not os.path.exists(dir_path):
            log.print_warn_log("There is no '%s' in '%s', please check the custom path."
                               % (ConstManager.CUSTOM_ALGORITHM_DIR_NAME, os.path.dirname(dir_path)))
            return
        one_match = False
        for item in os.listdir(dir_path):
            if self._add_algorithm_file_to_list(os.path.join(dir_path, item), support_algorithm_list):
                one_match = True
        if not one_match:
            log.print_warn_log("There is no legal 'alg_{algorithm_name}.py' file in '%s', "
                               "please check the path." % dir_path)

    def _make_algorithm_param(self: any, algorithm_name: str, args: dict) -> AlgorithmParameter:
        alg_arg_map = args
        if algorithm_name in self.algorithm_param:
            alg_arg_map = dict(self.algorithm_param.get(algorithm_name), **args)
        alg_args = AlgorithmParameter()
        alg_args.__dict__ = alg_arg_map
        return alg_args

    def _call_compare_function(self: any, *args: any) -> (str, str):
        compare_func, my_output_dump_data, ground_truth_dump_data, alg_args, algorithm_name = args
        alg_result = []
        alg_error_msg = ''
        try:
            if compare_func:
                alg_result, alg_error_msg = compare_func(my_output_dump_data, ground_truth_dump_data, alg_args)
                self._check_return_value_valid(alg_result, algorithm_name)
                self._check_return_value_valid(alg_error_msg, algorithm_name)
        except Exception as ex:
            alg_error_msg = "Failed to execute '%s' in '%s'. %s" \
                            % (ConstManager.COMPARE_FUNC_NAME, algorithm_name, str(ex))
            log.print_warn_log(alg_error_msg)
            alg_result = ConstManager.NAN
        finally:
            pass
        return alg_result, alg_error_msg


class AlgorithmManagerMain:
    """
    The class for algorithm manager main
    """

    def __init__(self: any, args: any) -> None:
        self.my_output_dump_file_path = os.path.realpath(args.my_dump_path)
        self.ground_truth_dump_file_path = os.path.realpath(args.golden_dump_path)
        self.manager = AlgorithmManager(args.custom_script_path, args.algorithm, args.algorithm_options)
        if args.output_path:
            self.output_path = os.path.realpath(args.output_path)

    def check_arguments_valid(self: any) -> None:
        """
        Check arguments valid, if invalid, throw exception
        """
        ret = utils.check_path_valid(self.my_output_dump_file_path, True, False,
                                     utils.PathType.File)
        if ret != CompareError.MSACCUCMP_NONE_ERROR:
            raise CompareError(ret)
        ret = utils.check_path_valid(self.ground_truth_dump_file_path, True, False,
                                     utils.PathType.File)
        if ret != CompareError.MSACCUCMP_NONE_ERROR:
            raise CompareError(ret)

    def process(self: any, save_result: bool = False) -> int:
        """
        Do compare for two numpy file
        :param save_result: is save result
        :return: VectorComparisonErrorCode
        """
        self.check_arguments_valid()
        log.print_info_log("The my output dump file is %s." % self.my_output_dump_file_path)
        log.print_info_log("The ground truth file is %s." % self.ground_truth_dump_file_path)
        try:
            self._process_exec(save_result)
        except CompareError as error:
            return error.code
        finally:
            pass
        return CompareError.MSACCUCMP_NONE_ERROR

    def _process_exec(self: any, save_result: bool) -> None:
        my_output_dump_data = utils.read_numpy_file(self.my_output_dump_file_path)
        ground_truth_dump_data = utils.read_numpy_file(self.ground_truth_dump_file_path)
        self._check_shape_valid(my_output_dump_data, ground_truth_dump_data)
        result, error_msg = self.manager.compare(
            my_output_dump_data.flatten(), ground_truth_dump_data.flatten(),
            {'my_output_dump_file': self.my_output_dump_file_path,
             'ground_truth_dump_file': self.ground_truth_dump_file_path,
             'shape_type': utils.get_shape_type(my_output_dump_data.shape)})
        self._print_result(result, error_msg, save_result)

    def _print_result(self: any, result: list, error_msg: list, save_result: bool) -> None:
        header = self.manager.get_result_title()
        title = ''
        line = ''
        for (algorithm_index, value) in enumerate(result):
            while len(line) < len(title):
                line += ConstManager.SPACE
            while len(title) < len(line):
                title += ConstManager.SPACE
            title += str(header[algorithm_index]) + ConstManager.SPACE
            line += str(value) + ConstManager.SPACE
        log.print_info_log(title)
        log.print_info_log(line)
        if error_msg:
            log.print_info_log(str(error_msg))
        if save_result:
            content = ""
            for index, data in enumerate(result):
                content += "%s: %s\n" % (header[index], data)
            file_name = 'file_result_%s.txt' % time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
            summary_file_path = os.path.join(self.output_path, file_name)
            FileUtils.save_file(summary_file_path, content)
            log.print_info_log('The file compare result have been written to "%s".' % summary_file_path)

    def _check_shape_valid(self: any, my_output_dump_data: any, ground_truth_dump_data: any) -> None:
        my_output_shape = my_output_dump_data.shape
        ground_truth_shape = ground_truth_dump_data.shape
        if my_output_shape != ground_truth_shape:
            log.print_error_log("My output shape %s in '%s' does not match the ground truth shape %s in '%s'."
                                % (my_output_shape, self.my_output_dump_file_path, ground_truth_shape,
                                   self.ground_truth_dump_file_path))
            raise CompareError(CompareError.MSACCUCMP_INVALID_SHAPE_ERROR)
