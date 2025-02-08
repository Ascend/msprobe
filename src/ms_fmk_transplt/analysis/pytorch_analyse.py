#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import argparse
import os.path
import shutil
import sys
from pathlib import Path

from analysis.unsupported_api_analysis import UnsupportedApiAnalyzer
from analysis.third_party_analysis import ThirdPartyAnalyzer
from analysis.dynamic_shape_analysis import DynamicShapeAnalyzer
from analysis.affinity_api_analysis import AffinityApiAnalyzer
from utils import trans_utils as utils
from utils import transplant_logger as translog


class PyTorchAnalyse:
    def __init__(self):
        self.input_path = ''
        self.output_path = ''
        self.py_file_counts = 0
        self.analyse_dict = {
            'third_party': ThirdPartyAnalyzer,
            'torch_apis': UnsupportedApiAnalyzer,
            'dynamic_shape': DynamicShapeAnalyzer,
            'affinity_apis': AffinityApiAnalyzer
        }
        self.dynamic_shape_analysis_package = 'dynamic_shape_analysis/msft_dynamic_analysis'

    @staticmethod
    def __parse_command():
        description = 'Pytorch Analyse powered by MindStudio\nCopyright (c) Huawei Technologies Co., Ltd. 2022-2024'
        parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
        parser.add_argument('-i', '--input', required=True, metavar='(DIR, FILE)',
                            help='Input path or file. (required)')
        parser.add_argument('-o', '--output', required=True, default='', metavar='DIR', help='Output path. (required)')
        parser.add_argument('-v', '--version', required=True,
                            choices=['1.11.0', '2.1.0', '2.2.0', '2.3.1', '2.4.0', '2.5.1'],
                            help='Target pytorch version of output. (required)')
        parser.add_argument('-m', '--mode', default='torch_apis',
                            choices=['third_party', 'torch_apis', 'dynamic_shape', 'affinity_apis'],
                            help='The way the script is analyzed. Only support torch_apis,'
                                 'third_party and dynamic_shape currently.')
        parser.add_argument('-api', '--api-files', nargs='*', metavar='FILE',
                            help='The unsupported op list file path output by the third-party analyse.')
        parser.add_argument('-env', '--env-path', nargs='*', type=str, help='env path of the input project.')
        return parser.parse_args()

    @staticmethod
    def __check_file_valid(args):
        if len(args.api_files) > utils.MAX_INPUT_FILE_COUNT:
            raise ValueError(f'The count of api files cannot exceed {utils.MAX_INPUT_FILE_COUNT}.')
        for file_path in args.api_files:
            input_info = utils.InputInfo(max_file_size=utils.MAX_PYTHON_FILE_SIZE, file_name='unsupported api file')
            utils.check_input_file_valid(file_path, input_info)
            real_path = os.path.realpath(file_path)
            if not real_path.endswith('.csv'):
                raise ValueError('The unsupported api file %s should be a csv file!' % file_path)
            utils.check_api_file_valid(real_path)

    @staticmethod
    def __check_env_path_valid(args):
        env_path = args.env_path
        if len(env_path) > utils.MAX_INPUT_FILE_COUNT:
            raise ValueError(f'The count of env paths cannot exceed {utils.MAX_INPUT_FILE_COUNT}.')
        for path in env_path:
            input_info = utils.InputInfo(file_name='env path', is_dir=True)
            utils.check_input_file_valid(path, input_info)
            if not utils.check_is_subdirectory(args.input, path):
                raise ValueError('env path %s should be a subdirectory of Input %s' % (path, args.input))

    def main(self):
        args = self.__parse_command()
        ret = 0
        result_dict = {}
        try:
            self.__check_param_valid(args)
            self.__check_input_valid(args)
            self.__check_output_valid(args)
            if args.api_files:
                self.__check_file_valid(args)
            if args.env_path:
                self.__check_env_path_valid(args)
            if args.mode in ['third_party', 'affinity_apis']:
                if not utils.IS_JEDI_INSTALLED:
                    raise ModuleNotFoundError("%s analysis must have jedi installed" % args.mode.replace('_', ' '))
            if args.mode == 'dynamic_shape':
                self.__copy_project()
                self.__init_logger()
                pytorch_analysis = self.analyse_dict.get(args.mode)(self.output_path, self.output_path, args.version)
            else:
                self.__init_logger()
                pytorch_analysis = self.analyse_dict.get(args.mode)(self.input_path, self.output_path, args.version,
                                                                    args.api_files)
            translog.info('PyTorch analysis start working now, please wait for a moment.')
            env_path = pytorch_analysis.package_env_path_set
            if args.env_path:
                env_path = list(str(Path(env_path_value)) for env_path_value in args.env_path)
                pytorch_analysis.package_env_path_set = set(env_path)
            global_visitor = self.__get_global_visitor(env_path) if utils.IS_JEDI_INSTALLED else None
            pytorch_analysis.init_global_visitor(global_visitor)
            pytorch_analysis.set_py_file_counts(self.py_file_counts)
            pytorch_analysis.run()
            result_dict = pytorch_analysis.result_dict
            self.__operation_in_different_args_mode(args, result_dict)
        except KeyboardInterrupt:
            translog.error('User canceled.')
            ret = 1
        except BaseException as exp:
            translog.error(f"An error occurred: {exp}")
            ret = 1
        finally:
            if args.mode == 'third_party' and utils.IS_JEDI_INSTALLED:
                utils.clear_parso_cache()
        if ret != 0:
            translog.error('Analyse run failed!')
        else:
            translog.info('Analyse run succeeded, welcome to the next use.')
            analysis_rel_path = os.path.basename(self.output_path)
            utils.get_analysis_result_statistics(result_dict, analysis_rel_path)
        self.__set_report_files_permission(0o440)
        return ret

    def __operation_in_different_args_mode(self, args, result_dict):
        if args.mode == 'dynamic_shape':
            self.__copy_analysis_pack()
        if args.mode == 'third_party' and result_dict.get('full_unsupported_results.csv', 0) > 0:
            translog.info(
                "The path to the 'full_unsupported_results.csv' file can be used as the -api parameter "
                "to analyze the full unsupported api information in the script.")

    def __check_param_valid(self, args):
        if utils.islink(args.input):
            raise utils.SoftlinkCheckException("Input path doesn't support soft link.")

        # check input path
        self.input_path = os.path.realpath(args.input)
        if not os.path.exists(self.input_path):
            raise ValueError('Input %s does not exist!' % args.input)

        if not os.access(self.input_path, os.R_OK):
            raise PermissionError('Input %s is not readable!' % args.input)

        if not utils.check_path_length_valid(self.input_path):
            raise ValueError('The real path or file name of input is too long.')

        utils.check_path_pattern_valid(self.input_path)

        if not utils.check_path_owner_consistent(self.input_path):
            utils.user_interactive_confirm(
                'The input path is insecure because it does not belong to you. Do you want to continue?')

        # check output path
        output_path = os.path.realpath(args.output)
        if utils.islink(args.output):
            raise utils.SoftlinkCheckException("Output path doesn't support soft link.")

        if not os.path.isdir(output_path):
            raise ValueError('Output %s is not a valid directory!' % args.output)

        if not os.access(output_path, os.W_OK):
            raise PermissionError('Output %s is not writeable!' % args.output)

        if not utils.check_path_length_valid(output_path):
            raise ValueError('The real path or file name of output is too long.')

        if not utils.check_path_owner_consistent(output_path):
            utils.user_interactive_confirm(
                'The output path is insecure because it does not belong to you. Do you want to continue?')

        if utils.check_is_subdirectory(args.input, args.output):
            raise ValueError('Output %s should not be a subdirectory of Input %s' % (args.output, args.input))

    def __check_input_valid(self, args):
        translog.info("Start to check input path...")
        if os.path.isfile(self.input_path):
            raise utils.InputCheckException('The input path must be a directory.')
        output_free_size = shutil.disk_usage(os.path.realpath(args.output)).free
        self.py_file_counts = utils.walk_input_path(self.input_path, output_free_size)
        if not self.py_file_counts:
            raise utils.InputCheckException('There are no valid python files in the folder.')

    def __check_output_valid(self, args):
        output_path = os.path.realpath(args.output)
        if os.path.isdir(self.input_path):
            self.output_path = os.path.join(output_path, os.path.split(self.input_path)[1] + '_analysis')
        if os.path.exists(self.output_path):
            utils.user_interactive_confirm('The output directory already exists. Do you want to overwrite?')
            self.__set_report_files_permission(0o640)
            utils.remove_path(self.output_path)

    def __copy_project(self):
        shutil.copytree(self.input_path, self.output_path + '/', symlinks=True)
        utils.change_mode(self.output_path)

    def __copy_analysis_pack(self):
        function_pack_dir = os.path.join(os.path.dirname(__file__), self.dynamic_shape_analysis_package)
        base_pack_dir = os.path.basename(self.dynamic_shape_analysis_package)
        if os.path.isdir(self.output_path):
            dst_path = os.path.join(self.output_path, base_pack_dir)
        else:
            return
        shutil.rmtree(dst_path, ignore_errors=True)
        shutil.copytree(function_pack_dir, dst_path)
        utils.change_mode(dst_path)
        translog.info(f"Package {base_pack_dir} has been copy to the output dir, "
                      f"please add {os.path.dirname(dst_path)} to PYTHONPATH before run net.")

    def __init_logger(self):
        log_file = os.path.join(self.output_path, 'pytorch_analysis.txt')
        if os.path.exists(log_file):
            utils.remove_path(log_file)
        translog.init_logging_file(log_file)

    def __set_report_files_permission(self, permission):
        report_files = ['pytorch_analysis.txt', 'unsupported_op.csv']
        report_files.extend(f'pytorch_analysis.txt.{idx}' for idx in range(1, translog.BACKUP_COUNT + 1))
        for filename in report_files:
            file_path = os.path.join(self.output_path, filename)
            if not os.path.isfile(file_path):
                continue
            os.chmod(file_path, permission)

    def __get_global_visitor(self, env_path):
        from global_analysis import GlobalReferenceVisitor

        utils.refresh_parso_cache()
        global_reference_visitor = GlobalReferenceVisitor(self.input_path, sys_path=env_path)
        return global_reference_visitor


if __name__ == '__main__':
    sys.exit(PyTorchAnalyse().main())
