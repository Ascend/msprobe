#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

import argparse
import os.path
import shutil
import sys

from .unsupported_api_analysis import UnsupportedApiAnalyzer
from .third_party_analysis import ThirdPartyAnalyzer
from ..utils import trans_utils as utils
from ..utils import transplant_logger as translog


class PyTorchAnalyse:
    def __init__(self):
        self.input_path = ''
        self.output_path = ''
        self.py_file_counts = 0
        self.analyse_dict = {
            'third_party': ThirdPartyAnalyzer,
            'torch_apis': UnsupportedApiAnalyzer
        }

    @staticmethod
    def __parse_command():
        parser = argparse.ArgumentParser()
        parser.add_argument('-i', '--input', required=True, metavar='(DIR, FILE)', help='Input path or file')
        parser.add_argument('-o', '--output', required=True, default='', metavar='DIR', help='Output path')
        parser.add_argument('-v', '--version', default='1.8.1', choices=['1.5.0', '1.8.1', '1.11.0'],
                            help='Target pytorch version of output')
        parser.add_argument('-m', '--mode', default='torch_apis', choices=['third_party', 'torch_apis'],
                            help='The way the script is analyzed. Only support torch_apis and third_party currently')
        return parser.parse_args()

    def main(self):
        args = self.__parse_command()
        ret = 0
        try:
            self.__check_param_valid(args)
            self.__check_input_valid(args)
            self.__check_output_valid(args)
            self.__init_logger()
            translog.info('PyTorch analysis start working now, please wait for a moment.')
            pytorch_analysis = self.analyse_dict.get(args.mode)(self.input_path, self.output_path, args.version)
            if args.mode == 'third_party':
                pytorch_analysis.init_global_visitor(self.__get_global_visitor())
            pytorch_analysis.set_py_file_counts(self.py_file_counts)
            pytorch_analysis.run()
        except KeyboardInterrupt:
            translog.error('User canceled.')
            ret = 1
        except BaseException as exp:
            translog.error(exp)
            ret = 1
        finally:
            if args.mode == 'third_party' and utils.IS_JEDI_INSTALLED:
                utils.clear_parso_cache()
        if ret != 0:
            translog.error('Analyse run fail!')
        else:
            translog.info('Analyse run success, welcome to the next use.')
        self.__set_report_files_permission(0o440)
        return ret

    def __check_param_valid(self, args):
        if utils.islink(args.input):
            raise utils.SoftlinkCheckException("Input path doesn't support soft link.")

        self.input_path = os.path.realpath(args.input)

        # check input path
        if not utils.check_path_length_valid(self.input_path):
            raise ValueError('The real path or file name of input is too long.')
        utils.check_path_pattern_valid(self.input_path)

        if not os.path.exists(self.input_path):
            raise ValueError('Input %s does not exist!' % args.input)

        if not os.access(self.input_path, os.R_OK):
            raise PermissionError('Input %s is not readable!' % args.input)

        if not utils.check_path_owner_consistent(self.input_path):
            utils.user_interactive_confirm(
                'The input path is insecure because it does not belong to you. Do you want to continue?')

        # check output path
        output_path = os.path.realpath(args.output)
        if not utils.check_path_length_valid(output_path):
            raise ValueError('The real path or file name of output is too long.')

        if not os.path.isdir(output_path):
            raise ValueError('Output %s is not a valid directory!' % args.output)

        if not os.access(output_path, os.W_OK):
            raise PermissionError('Output %s is not writeable!' % args.output)

        if not utils.check_path_owner_consistent(output_path):
            utils.user_interactive_confirm(
                'The output path is insecure because it does not belong to you. Do you want to continue?')

        if utils.check_is_subdirectory(args.input, args.output):
            raise ValueError('Output %s should not be a subdirectory of Input %s' % (args.output, args.input))

    def __check_input_valid(self, args):
        translog.info("Start to check input path...")
        if os.path.isfile(self.input_path):
            if not self.input_path.endswith('.py'):
                raise utils.InputCheckException('The input file is not a python file.')
            return
        output_free_size = shutil.disk_usage(os.path.realpath(args.output)).free
        self.py_file_counts = utils.walk_input_path(self.input_path, output_free_size)
        if not self.py_file_counts:
            raise utils.InputCheckException('There are no valid python files in the folder.')

    def __check_output_valid(self, args):
        output_path = os.path.realpath(args.output)
        if os.path.isfile(self.input_path):
            self.output_path = os.path.join(output_path,
                                            os.path.splitext(os.path.basename(self.input_path))[0] + '_analysis')
        if os.path.isdir(self.input_path):
            self.output_path = os.path.join(output_path, os.path.split(self.input_path)[1] + '_analysis')
        if os.path.exists(self.output_path):
            utils.user_interactive_confirm('The output directory already exists. Do you want to overwrite?')
            self.__set_report_files_permission(0o640)
            utils.remove_path(self.output_path)

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

    def __get_global_visitor(self):
        if not utils.IS_JEDI_INSTALLED:
            raise ModuleNotFoundError("third party analysis must have jedi installed")
        from ..global_analysis import GlobalReferenceVisitor

        utils.refresh_parso_cache()
        global_reference_visitor = GlobalReferenceVisitor(self.input_path, sys_path=[])
        return global_reference_visitor


if __name__ == '__main__':
    sys.exit(PyTorchAnalyse().main())
