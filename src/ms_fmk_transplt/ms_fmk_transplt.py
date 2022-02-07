#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.

import argparse
import os
import shutil
import sys
import transplant_logger as translog
from transplant import Transplant
import trans_utils as utils


class MsFmkTransplt(object):
    TRANSPLANT_OUTPUT_PATH_SUFFIX = '_msft'

    def __init__(self):
        self.input = ''
        self.output = ''
        self.custom_rule_file = ''
        self.feature_switch = ['normal']
        self.rule_list = []
        self.py_file_counts = 0

    def __para_check_valid(self, args):
        if os.path.islink(args.input):
            raise utils.SoftlinkCheckException("Input path doesn't support soft link.")

        input_path = os.path.realpath(args.input)
        output = os.path.realpath(args.output)

        if not os.path.exists(input_path):
            raise ValueError('Input %s does not exist!' % args.input)

        if not os.access(input_path, os.R_OK):
            raise PermissionError('Input %s is not readable!' % args.input)

        if not self.__check_path_owner_consistent(input_path):
            utils.user_interactive_confirm(
                'This input path is insecure because it does not belong to you. Do you want to continue?')

        if not os.path.isdir(output):
            raise ValueError('Output %s is not a valid directory!' % args.output)

        if not os.access(output, os.W_OK):
            raise PermissionError('Output %s is not writeable!' % args.output)

        if not self.__check_path_owner_consistent(output):
            utils.user_interactive_confirm(
                'This output path is insecure because it does not belong to you. Do you want to continue?')

        if self.__check_is_subdirectory(args.input, args.output):
            raise ValueError('Output %s should not be a subdirectory of Input %s' % (args.output, args.input))

        self.__check_custom_rule_param_valid(args)
        self.__check_distributed_rule_param_valid(args)

    @staticmethod
    def __check_custom_rule_param_valid(args):
        if not args.rule:
            return
        rule = os.path.realpath(args.rule)
        if not os.path.exists(rule):
            raise ValueError('Custom rule file %s does not exist!' % args.rule)

        if not os.access(rule, os.R_OK):
            raise PermissionError('Custom rule file %s is not readable!' % args.rule)

    @staticmethod
    def __check_distributed_rule_param_valid(args):
        if not hasattr(args, 'main'):
            return
        if os.path.islink(args.main):
            raise utils.SoftlinkCheckException("Main file path doesn't support soft link.")
        main_file = os.path.realpath(args.main)
        if not main_file.endswith('.py'):
            raise ValueError('Main file %s should be a python file!' % args.main)
        if not os.path.exists(main_file):
            raise ValueError('Main file %s does not exist!' % args.main)
        if not MsFmkTransplt.__check_is_subdirectory(args.input, args.main):
            if os.path.isdir(args.input):
                raise ValueError('Main file %s is not in Input %s' % (args.main, args.input))
            if os.path.isfile(args.input):
                raise ValueError('Main file %s should be the input file %s' % (args.main, args.input))
        if not os.access(main_file, os.R_OK):
            raise PermissionError('Main file %s is not readable!' % args.main)
        if not args.target_model:
            raise ValueError('Target model variable is not set!')

    def __parse_command(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-i', '--input', required=True, metavar='(DIR, FILE)', help='Input path or file')
        parser.add_argument('-o', '--output', required=True, default='', metavar='DIR', help='Output path')
        parser.add_argument('-r', '--rule', default='', metavar='FILE', help='Custom rules file path')
        parser.add_argument('-s', '--specify-device', dest='specify_device', action='store_true',
                            help='Use specified device which is set by environment variable NPU_CALCULATE_DEVICE')
        parser.add_argument('-sim', '--similar', action='store_true',
                            help='Replaces certain unsupported APIs with functionally similar ones. '
                                 'Note that this may result in accuracy loss and performance degradation')
        parser.add_argument('-a', '--amp_model', metavar='model', default='', help='The variable name of the '
                                                                                   'amp target model')
        subparsers = parser.add_subparsers(help='commands')
        self.__distributed_parser(subparsers)
        return parser.parse_args()

    @staticmethod
    def __distributed_parser(subparsers):
        distributed_parser = subparsers.add_parser('distributed',
                                                   help='Specified this argument only when you want to transplant '
                                                        'a single GPU script to a distributed NPU script')
        distributed_parser.add_argument('-m', '--main', default='', metavar='FILE', required=True,
                                        help='The entry python file of the project')
        distributed_parser.add_argument('-t', '--target_model', metavar='model', default='model',
                                        help='The variable name of the target model')

    def __init_default_para(self):
        translog.info("Start to copy files...")
        if os.path.isfile(self.input):
            shutil.copyfile(self.input, self.output)
        if os.path.isdir(self.input):
            shutil.copytree(self.input, self.output, symlinks=True)
        utils.change_mode(self.output)

    def __init_custom_para(self, args):
        self.custom_rule_file = args.rule if args.rule else ""
        if args.specify_device:
            self.feature_switch.append('specify_device')
        if args.similar:
            self.feature_switch.append('similar')
        if hasattr(args, 'main'):
            utils.generate_distributed_shell_file(self.output if os.path.isdir(self.output) else
                                                  os.path.dirname(self.output))
            self.feature_switch.append('distributed')

    def __copy_function_pack(self):
        function_pack_dir = os.path.join(os.path.dirname(__file__), 'ascend_function')
        if os.path.isdir(self.output):
            dst_path = os.path.join(self.output, 'ascend_function')
        elif os.path.isfile(self.output):
            dst_path = os.path.join(os.path.dirname(self.output), 'ascend_function')
        else:
            return
        shutil.rmtree(dst_path, ignore_errors=True)
        shutil.copytree(function_pack_dir, dst_path)
        utils.change_mode(dst_path)
        translog.info(f"Package ascend_function has been copy to the output dir, "
                      f"please add {os.path.dirname(dst_path)} to PYTHONPATH before run net.")

    def __init_self_para(self, args):
        self.__init_default_para()
        self.__init_custom_para(args)

    def __init_logger(self):
        log_file = 'msFmkTranspltlog.txt'
        if os.path.isfile(self.input):
            log_file = os.path.join(os.path.dirname(self.output), 'msFmkTranspltlog.txt')
        if os.path.isdir(self.input):
            log_file = os.path.join(self.output, 'msFmkTranspltlog.txt')
        if os.path.exists(log_file):
            utils.remove_path(log_file)
        translog.init_logging_file(log_file)
        utils.change_mode(log_file)

    def __init_rules(self, args):
        self.rule_list = utils.get_builtin_rule(self.feature_switch, args)
        if self.custom_rule_file:
            self.rule_list = utils.get_custom_rule(self.custom_rule_file, self.rule_list)

    def __check_output_valid(self, args):
        self.input = os.path.realpath(args.input)
        if os.path.isfile(self.input):
            self.output = os.path.join(args.output, os.path.split(self.input)[1])
        if os.path.isdir(self.input):
            self.output = os.path.join(args.output, os.path.split(self.input)[1] + '_msft')
        if os.path.islink(self.output):
            raise utils.SoftlinkCheckException(f"The output path {self.output} shouldn't be a soft link.")
        if os.path.exists(self.output):
            utils.user_interactive_confirm('The output directory already exists. Do you want to overwrite?')
            utils.remove_path(self.output)

    def __check_input_valid(self, args):
        translog.info("Start to check input path...")
        if os.path.isfile(args.input):
            if not args.input.endswith('.py'):
                raise utils.InputCheckException('The input file is not a python file.')
            return
        self.py_file_counts = utils.walk_input_path(os.path.realpath(args.input), os.path.realpath(args.output))
        if not self.py_file_counts:
            raise utils.InputCheckException('There are no python files in the folder.')

    @staticmethod
    def __check_path_owner_consistent(path):
        try:
            import pwd
            file_owner = pwd.getpwuid(os.stat(path).st_uid).pw_name
            return file_owner == os.getlogin()
        except ImportError:
            return True

    @staticmethod
    def get_main_file(args):
        if not hasattr(args, 'main'):
            return ''
        if os.path.isfile(args.input):
            return os.path.basename(args.main)
        return os.path.relpath(args.main, args.input)

    @staticmethod
    def __check_is_subdirectory(path_may_be_parent, path_may_be_child):
        path_may_be_parent = os.path.realpath(path_may_be_parent)
        path_may_be_child = os.path.realpath(path_may_be_child)
        if path_may_be_parent[0] != path_may_be_child[0]:
            return False
        commonpath = os.path.commonpath([path_may_be_parent, path_may_be_child])
        return commonpath == path_may_be_parent

    def main(self):
        args = self.__parse_command()

        try:
            self.__para_check_valid(args)
            self.__check_output_valid(args)
            self.__check_input_valid(args)
            self.__init_self_para(args)
            self.__init_logger()
            translog.info('Initialing rules...')
            self.__init_rules(args)
            translog.info('MsFmkTransplt start working now, please wait for a moment.')
            transplant = Transplant(self.output, self.rule_list, self.get_main_file(args))
            transplant.set_py_file_counts(self.py_file_counts)
            transplant.run()
            if args.similar:
                self.__copy_function_pack()
        except SystemExit:
            return 1
        except BaseException as exp:
            translog.error(exp)
            return 1

        return 0


if __name__ == '__main__':
    ret = MsFmkTransplt().main()
    if ret != 0:
        translog.error('MsFmkTransplt run fail!')
        sys.exit(ret)
    translog.info('MsFmkTransplt run success, welcome to the next use.')
