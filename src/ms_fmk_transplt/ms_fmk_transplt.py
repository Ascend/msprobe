#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.

import argparse
import os
import shutil
import sys

import pytorch_gpu2npu.utils.trans_utils as utils
import pytorch_gpu2npu.utils.transplant_logger as translog
from transplant import Transplant


class MsFmkTransplt(object):
    TRANSPLANT_OUTPUT_PATH_SUFFIX = '_msft'

    def __init__(self):
        self.input = ''
        self.output = ''
        self.custom_rule_file = ''
        self.feature_switch = ['normal']
        self.rule_list = []
        self.py_file_counts = 0

    @staticmethod
    def __check_custom_rule_param_valid(args):
        if not args.rule:
            return
        if os.path.islink(args.rule):
            raise utils.SoftlinkCheckException("Custom rule file doesn't support soft link.")
        rule = os.path.realpath(args.rule)

        if not utils.check_path_length_valid(rule):
            raise ValueError('The real path or file name of custom rule file is too long.')

        if not os.path.exists(rule):
            raise ValueError('Custom rule file %s does not exist!' % args.rule)

        if not (os.path.isfile(rule) and rule.endswith('.json')):
            raise ValueError('Custom rule file %s should be a json file!' % args.rule)

        if not os.access(rule, os.R_OK):
            raise PermissionError('Custom rule file %s is not readable!' % args.rule)

        if not utils.check_path_owner_consistent(rule):
            utils.user_interactive_confirm(
                'Custom rule file is insecure because it does not belong to you. Do you want to continue?')

        if os.path.getsize(rule) >= utils.MAX_SIZE_OF_RULE_FILE:
            raise ValueError('Custom rule file is too large.')

    @staticmethod
    def __check_distributed_rule_param_valid(args):
        if not hasattr(args, 'main'):
            return
        if os.path.islink(args.main):
            raise utils.SoftlinkCheckException("Main file path doesn't support soft link.")
        main_file = os.path.realpath(args.main)
        if not utils.check_path_length_valid(main_file):
            raise ValueError('The real path or file name of main file is too long.')
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
            raise ValueError('Target model variable name is not set!')
        utils.check_model_name_valid(args.target_model)

    @staticmethod
    def __distributed_parser(subparsers):
        distributed_parser = subparsers.add_parser('distributed',
                                                   help='This option is required only if you want to transplant '
                                                        'a single GPU script to a distributed NPU script. '
                                                        'Ensure that your code is a single GPU script.')
        distributed_parser.add_argument('-m', '--main', default='', metavar='FILE', required=True,
                                        help='The entry python file of the project, for example, train.py main.py.')
        distributed_parser.add_argument('-t', '--target_model', metavar='model', default='model',
                                        help='The variable name of the target model, for example, '
                                             '"model=LeNet() model", "self.model=LeNet() self.model"')

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
        ret = 0
        try:
            self.__para_check_valid(args)
            self.__check_output_valid(args)
            self.__check_input_valid(args)
            self.__copy_project()
            self.__init_custom_para(args)
            self.__init_logger()
            translog.info('Initialing rules...')
            self.__init_rules(args)
            translog.info('MsFmkTransplt start working now, please wait for a moment.')
            transplant = Transplant(self.output, self.rule_list, args)
            transplant.set_py_file_counts(self.py_file_counts)
            if hasattr(args, 'main'):
                transplant.init_global_visitor(self.__get_global_visitor())
            transplant.run()
            if args.similar:
                self.__copy_function_pack('ascend_function')
            if args.modelarts:
                self.__copy_function_pack('ascend_modelarts_function')
        except BaseException as exp:
            translog.error(exp)
            ret = 1
        finally:
            if hasattr(args, 'main') and utils.IS_JEDI_INSTALLED:
                utils.clear_parso_cache()

        if ret != 0:
            translog.error('MsFmkTransplt run fail!')
        else:
            translog.info('MsFmkTransplt run success, welcome to the next use.')
        self.__set_report_files_permission(0o440)
        return ret

    def __get_global_visitor(self):
        global_reference_visitor = None
        if utils.IS_JEDI_INSTALLED:
            utils.refresh_parso_cache()
            from pytorch_gpu2npu.global_analysis import GlobalReferenceVisitor
            global_reference_visitor = GlobalReferenceVisitor(self.input)
        else:
            translog.warning('Since jedi is not correctly installed, global analysis will not take effect. You '
                             'can install it via pip.')
        return global_reference_visitor

    def __set_report_files_permission(self, permission):
        output_dir = os.path.dirname(self.output) if os.path.isfile(self.output) else self.output
        report_files = ['msFmkTranspltlog.txt', 'unsupported_op.csv', 'change_list.csv']
        report_files.extend(f'msFmkTranspltlog.txt.{idx}' for idx in range(1, translog.BACKUP_COUNT + 1))
        for filename in report_files:
            file_path = os.path.join(output_dir, filename)
            if not os.path.isfile(file_path):
                continue
            os.chmod(file_path, permission)

    def __para_check_valid(self, args):
        if os.path.islink(args.input):
            raise utils.SoftlinkCheckException("Input path doesn't support soft link.")

        input_path = os.path.realpath(args.input)
        output = os.path.realpath(args.output)

        if not utils.check_path_length_valid(input_path):
            raise ValueError('The real path or file name of input is too long.')

        if not os.path.exists(input_path):
            raise ValueError('Input %s does not exist!' % args.input)

        if not os.access(input_path, os.R_OK):
            raise PermissionError('Input %s is not readable!' % args.input)

        if not utils.check_path_owner_consistent(input_path):
            utils.user_interactive_confirm(
                'The input path is insecure because it does not belong to you. Do you want to continue?')

        if not utils.check_path_length_valid(output):
            raise ValueError('The real path or file name of output is too long.')

        if not os.path.isdir(output):
            raise ValueError('Output %s is not a valid directory!' % args.output)

        if not os.access(output, os.W_OK):
            raise PermissionError('Output %s is not writeable!' % args.output)

        if not utils.check_path_owner_consistent(output):
            utils.user_interactive_confirm(
                'The output path is insecure because it does not belong to you. Do you want to continue?')

        if self.__check_is_subdirectory(args.input, args.output):
            raise ValueError('Output %s should not be a subdirectory of Input %s' % (args.output, args.input))

        if args.amp_model:
            utils.check_model_name_valid(args.amp_model)

        if args.version not in ['1.5.0', '1.8.1']:
            raise ValueError('Pytorch version only support 1.5.0 and 1.8.1 currently.')

        self.__check_custom_rule_param_valid(args)
        self.__check_distributed_rule_param_valid(args)

    def __parse_command(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-i', '--input', required=True, metavar='(DIR, FILE)', help='Input path or file')
        parser.add_argument('-o', '--output', required=True, default='', metavar='DIR', help='Output path')
        parser.add_argument('-r', '--rule', default='', metavar='FILE', help='Custom rules file path')
        parser.add_argument('-s', '--specify-device', dest='specify_device', action='store_true',
                            help='This option is required only if you want to use the DEVICE_ID'
                                 'environment variable to specify the running device.')
        parser.add_argument('-sim', '--similar', action='store_true',
                            help='Replaces certain unsupported APIs with functionally similar ones. '
                                 'Note that this may result in accuracy loss and performance degradation')
        parser.add_argument('-a', '--amp_model', metavar='model', default='',
                            help='This option is required only if you want to convert torch.cuda.amp to apex.amp')
        parser.add_argument('-v', '--version', default='1.5.0',
                            help='Target pytorch version of output. Only support 1.5.0 and 1.8.1 currently')
        parser.add_argument('-m', '--modelarts', action='store_true',
                            help='Convert to a ModelArts-compatible project.')
        subparsers = parser.add_subparsers(help='commands')
        self.__distributed_parser(subparsers)
        return parser.parse_args()

    def __copy_project(self):
        translog.info("Start to copy files...")
        if os.path.isfile(self.input):
            shutil.copy2(self.input, self.output)
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
            shell_file_path = self.output if os.path.isdir(self.output) else os.path.dirname(self.output)
            utils.generate_distributed_shell_file(shell_file_path)
            self.feature_switch.append('distributed')

    def __copy_function_pack(self, pack_name):
        function_pack_dir = os.path.join(os.path.dirname(__file__), pack_name)
        if os.path.isdir(self.output):
            dst_path = os.path.join(self.output, pack_name)
        elif os.path.isfile(self.output):
            dst_path = os.path.join(os.path.dirname(self.output), pack_name)
        else:
            return
        shutil.rmtree(dst_path, ignore_errors=True)
        shutil.copytree(function_pack_dir, dst_path)
        utils.change_mode(dst_path)
        translog.info(f"Package {pack_name} has been copy to the output dir, "
                      f"please add {os.path.dirname(dst_path)} to PYTHONPATH before run net.")

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
            if hasattr(args, 'main'):
                project_suffix = '_msft_multi'
            else:
                project_suffix = '_msft'
            self.output = os.path.join(args.output, os.path.split(self.input)[1] + project_suffix)
        if os.path.exists(self.output):
            utils.user_interactive_confirm('The output directory already exists. Do you want to overwrite?')
            self.__set_report_files_permission(0o640)
            utils.remove_path(self.output)

    def __check_input_valid(self, args):
        translog.info("Start to check input path...")
        if os.path.isfile(args.input):
            if not args.input.endswith('.py'):
                raise utils.InputCheckException('The input file is not a python file.')
            return
        output_free_size = shutil.disk_usage(os.path.realpath(args.output)).free
        self.py_file_counts = utils.walk_input_path(os.path.realpath(args.input), output_free_size)
        if not self.py_file_counts:
            raise utils.InputCheckException('There are no valid python files in the folder.')


if __name__ == '__main__':
    sys.exit(MsFmkTransplt().main())
