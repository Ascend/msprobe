#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.

import argparse
import os
import shutil
import sys

from utils import trans_utils as utils
from utils import transplant_logger as translog
from transfer.transplant import Transplant
from transfer.rules.rule_getter import rule_getter

TRANSPLANT_OUTPUT_DIR_NAME = 'transplant_result_file'


class MsFmkTransplt(object):
    TRANSPLANT_OUTPUT_PATH_SUFFIX = '_msft'

    def __init__(self):
        self.input = ''
        self.output = ''
        self.feature_switch = ['normal']
        self.rule_list = []
        self.py_file_counts = 0
        self.transplant_file_output = ''
        self.package_env_path_set = None

    @staticmethod
    def __check_distributed_rule_param_valid(args):
        if not hasattr(args, 'main'):
            return
        utils.check_input_file_valid(args.main,
                                     utils.InputInfo(max_file_size=utils.MAX_PYTHON_FILE_SIZE, file_name='Main file'))
        main_file = os.path.realpath(args.main)
        if not main_file.endswith('.py'):
            raise ValueError('Main file %s should be a python file!' % args.main)
        if not utils.check_is_subdirectory(args.input, args.main):
            raise ValueError('Main file %s is not in Input %s' % (args.main, args.input))
        if not args.target_model:
            raise ValueError('Target model variable name is not set!')
        utils.check_model_name_valid(args.target_model)

    @staticmethod
    def __distributed_parser(subparsers):
        distributed_parser = subparsers.add_parser('distributed',
                                                   help='This option is required only if you want to transplant '
                                                        'a single GPU script to a distributed NPU script. '
                                                        'Ensure that your code is a single GPU script. '
                                                        'This option contains the -m and -t parameters. '
                                                        'If you want to view the specific information of '
                                                        'the parameters, please execute the command '
                                                        '"./pytorch_gpu2npu.sh distributed -h"')
        distributed_parser.add_argument('-m', '--main', default='', metavar='FILE', required=True,
                                        help='The entry python file of the project, for example, train.py main.py.')
        distributed_parser.add_argument('-t', '--target_model', metavar='model', default='model',
                                        help='The variable name of the target model, for example, '
                                             '"model=LeNet() model", "self.model=LeNet() self.model"')

    def main(self):
        args = self.__parse_command()
        ret = 0
        result_dict = {}
        try:
            self.__para_check_valid(args)
            self.__check_output_valid(args)
            self.__check_input_valid(args)
            self.__copy_project()
            self.__check_transplant_file_output_valid()
            self.__init_custom_para(args)
            self.__init_logger()
            translog.info('Initialing rules...')
            self.__init_rules(args)
            translog.info('MsFmkTransplt start working now, please wait for a moment.')
            transplant = Transplant(self.output, self.rule_list, args, self.transplant_file_output)
            transplant.set_py_file_counts(self.py_file_counts)
            transplant.init_global_visitor(self.__get_global_visitor())
            transplant.run()
            if args.modelarts:
                self.__copy_function_pack('ascend_modelarts_function')
            result_dict = transplant.transplant_result_statistics
        except KeyboardInterrupt:
            translog.error("User canceled.")
            ret = 1
        except BaseException as exp:
            translog.error(f"An error occurred: {exp}")
            ret = 1
        finally:
            if utils.IS_JEDI_INSTALLED:
                utils.clear_parso_cache()

        if ret != 0:
            translog.error('MsFmkTransplt run failed!')
        else:
            translog.info('MsFmkTransplt run succeeded, welcome to the next use.')
            analysis_rel_path = os.path.basename(self.output)
            utils.get_analysis_result_statistics(result_dict, analysis_rel_path)
        self.__set_report_files_permission(0o440)
        return ret

    def __get_global_visitor(self):
        global_reference_visitor = None
        if utils.IS_JEDI_INSTALLED:
            utils.refresh_parso_cache()
            from global_analysis import GlobalReferenceVisitor
            global_reference_visitor = GlobalReferenceVisitor(self.input, self.package_env_path_set)
        else:
            translog.warning('Since jedi is not correctly installed, global analysis will not take effect. You '
                             'can install it via pip.')
        return global_reference_visitor

    def __set_report_files_permission(self, permission):
        output_dir = os.path.dirname(self.transplant_file_output) if os.path.isfile(self.transplant_file_output) \
            else self.transplant_file_output
        report_files = [
            'msFmkTranspltlog.txt', 'unsupported_api.csv', 'change_list.csv', 'unknown_api.csv', 'cuda_op_list.csv',
            'api_performance_advice.csv', 'api_precision_advice.csv'
        ]
        report_files.extend(f'msFmkTranspltlog.txt.{idx}' for idx in range(1, translog.BACKUP_COUNT + 1))
        for filename in report_files:
            file_path = os.path.join(output_dir, filename)
            if not os.path.isfile(file_path):
                continue
            os.chmod(file_path, permission)

    def __para_check_valid(self, args):
        if utils.islink(args.input):
            raise utils.SoftlinkCheckException("Input path doesn't support soft link.")

        input_path = os.path.realpath(args.input)
        output = os.path.realpath(args.output)

        if not utils.check_path_length_valid(input_path):
            raise ValueError('The real path or file name of input is too long.')

        utils.check_path_pattern_valid(input_path)

        if not os.path.exists(input_path):
            raise ValueError('Input %s does not exist!' % args.input)

        if not os.access(input_path, os.R_OK):
            raise PermissionError('Input %s is not readable!' % args.input)

        if not utils.check_path_owner_consistent(input_path):
            utils.user_interactive_confirm(
                'The input path is insecure because it does not belong to you. Do you want to continue?')

        if not utils.check_path_length_valid(output):
            raise ValueError('The real path or file name of output is too long.')

        utils.check_path_pattern_valid(output)

        if utils.islink(args.output):
            raise utils.SoftlinkCheckException("Output path doesn't support soft link.")

        if not os.path.isdir(output):
            raise ValueError('Output %s is not a valid directory!' % args.output)

        if not os.access(output, os.W_OK):
            raise PermissionError('Output %s is not writeable!' % args.output)

        if not utils.check_path_owner_consistent(output):
            utils.user_interactive_confirm(
                'The output path is insecure because it does not belong to you. Do you want to continue?')

        if utils.check_is_subdirectory(args.input, args.output):
            raise ValueError('Output %s should not be a subdirectory of Input %s' % (args.output, args.input))

        self.__check_distributed_rule_param_valid(args)

    def __parse_command(self):
        description = 'Pytorch GPU2NPU powered by MindStudio\nCopyright (c) Huawei Technologies Co., Ltd. 2022-2025'
        parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
        parser.add_argument('-i', '--input', required=True, metavar='(DIR, FILE)',
                            help='Input path or file. (required)')
        parser.add_argument('-o', '--output', required=True, default='', metavar='DIR', help='Output path. (required)')
        parser.add_argument('-s', '--specify-device', dest='specify_device', action='store_true',
                            help='This option is required only if you want to use the DEVICE_ID'
                                 'environment variable to specify the running device.')
        parser.add_argument('-v', '--version', required=True,
                            choices=['1.11.0', '2.1.0', '2.2.0', '2.3.1', '2.4.0', '2.5.1'],
                            help='Target pytorch version of output. (required)')
        parser.add_argument('-m', '--modelarts', action='store_true',
                            help='Convert to a ModelArts-compatible project.')
        subparsers = parser.add_subparsers(help='commands')
        self.__distributed_parser(subparsers)
        return parser.parse_args()

    def __copy_project(self):
        translog.info("Start to copy files...")
        shutil.copytree(self.input, self.output, symlinks=True)
        utils.change_mode(self.output)

    def __init_custom_para(self, args):
        if args.specify_device:
            self.feature_switch.append('specify_device')
        if hasattr(args, 'main'):
            shell_file_path = self.output if os.path.isdir(self.output) else os.path.dirname(self.output)
            utils.generate_distributed_shell_file(shell_file_path)
            self.feature_switch.append('distributed')
        if args.version != '1.11.0':
            self.feature_switch.append('2.1.0')

    def __copy_function_pack(self, pack_name):
        function_pack_dir = os.path.join(os.path.dirname(__file__), "transfer", "adapter", pack_name)
        if os.path.isdir(self.output):
            dst_path = os.path.join(self.output, pack_name)
        else:
            return
        shutil.rmtree(dst_path, ignore_errors=True)
        shutil.copytree(function_pack_dir, dst_path)
        utils.change_mode(dst_path)
        translog.info(f"Package {pack_name} has been copy to the output dir, "
                      f"please add {os.path.dirname(dst_path)} to PYTHONPATH before run net.")

    def __init_logger(self):
        log_file = os.path.join(self.transplant_file_output, 'msFmkTranspltlog.txt')
        if os.path.exists(log_file):
            utils.remove_path(log_file)
        translog.init_logging_file(log_file)
        utils.change_mode(log_file)

    def __init_rules(self, args):
        self.rule_list = rule_getter.get_builtin_rule(self.feature_switch, args)

    def __check_output_valid(self, args):
        self.input = os.path.realpath(args.input)
        self.package_env_path_set = utils.search_package_env_path(self.input)
        if hasattr(args, 'main'):
            project_suffix = '_msft_multi'
        else:
            project_suffix = '_msft'
        self.output = os.path.join(args.output, os.path.split(self.input)[1] + project_suffix)
        if os.path.exists(self.output):
            utils.user_interactive_confirm('The output directory already exists. Do you want to overwrite?')
            utils.remove_path(self.output)

    def __check_input_valid(self, args):
        translog.info("Start to check input path...")
        if os.path.isfile(args.input):
            raise utils.InputCheckException('The input path must be a directory.')
        output_free_size = shutil.disk_usage(os.path.realpath(args.output)).free
        self.py_file_counts = utils.walk_input_path(os.path.realpath(args.input), output_free_size)
        if not self.py_file_counts:
            raise utils.InputCheckException('There are no valid python files in the folder.')

    def __check_transplant_file_output_valid(self):
        self.transplant_file_output = os.path.join(self.output, TRANSPLANT_OUTPUT_DIR_NAME)
        if os.path.exists(self.transplant_file_output):
            utils.user_interactive_confirm("The transplant result file output directory "
                                           "'transplant_result_file' already exists. Do you want to overwrite?")
            self.__set_report_files_permission(0o640)
            utils.remove_path(self.transplant_file_output)


if __name__ == '__main__':
    sys.exit(MsFmkTransplt().main())
