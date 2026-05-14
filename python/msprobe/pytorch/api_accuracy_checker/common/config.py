#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

import os
from collections import namedtuple
from msprobe.core.common.file_utils import load_yaml, check_file_or_directory_path
from msprobe.core.common.utils import is_int
from msprobe.pytorch.dump.pt_config import ACCCheckConfig


AccCheckConfig = namedtuple('AccCheckConfig', ['forward_content', 'backward_content',
                                               'result_csv_path', 'details_csv_path',
                                         'save_error_data', 'is_continue_acc_check', 'real_data_path', 'white_list',
                                         'black_list', 'error_data_path'])


class Config:
    def __init__(self, yaml_file):
        check_file_or_directory_path(yaml_file, False)
        config = load_yaml(yaml_file)
        self.config = {key: self.validate(key, value) for key, value in config.items()}

    def __getattr__(self, item):
        return self.config[item]

    def __str__(self):
        return '\n'.join(f"{key}={value}" for key, value in self.config.items())

    @staticmethod
    def validate(key, value):
        validators = {
            'white_list': list,
            'black_list': list,
            'error_data_path': str,
            'precision': int,
            'quantization_api_list': list
        }
        if key not in validators:
            raise ValueError(f"{key} must be one of {validators.keys()}")
        if not isinstance(value, validators.get(key)):
            raise ValueError(f"{key} must be {validators[key].__name__} type")
        if key == 'precision' and not is_int(value):
            raise ValueError("precision must be an integer")
        if key == 'precision' and (value < 0 or value > 20):
            raise ValueError("precision must be greater than or equal to 0 and less than 21")
        if key == 'white_list':
            ACCCheckConfig.check_filter_list_config(key, value)
        if key == 'black_list':
            ACCCheckConfig.check_filter_list_config(key, value)
        if key == 'error_data_path':
            ACCCheckConfig.check_error_data_path_config(value)
        if key == 'quantization_api_list':
            ACCCheckConfig.check_filter_list_config(key, value)
        return value


cur_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
yaml_path = os.path.join(cur_path, "config.yaml")
msCheckerConfig = Config(yaml_path)


class CheckerConfig:
    def __init__(self, task_config=None):
        self.white_list = msCheckerConfig.white_list
        self.black_list = msCheckerConfig.black_list
        self.error_data_path = msCheckerConfig.error_data_path

        if task_config:
            self.load_config(task_config)

    def load_config(self, task_config):
        self.white_list = task_config.white_list
        self.black_list = task_config.black_list
        self.error_data_path = task_config.error_data_path
    

    def get_acc_check_config(self, **config_params):
        return AccCheckConfig(
            forward_content=config_params.get('forward_content'),
            backward_content=config_params.get('backward_content'),
            result_csv_path=config_params.get('result_csv_path'),
            details_csv_path=config_params.get('details_csv_path'),
            save_error_data=config_params.get('save_error_data'),
            is_continue_acc_check=config_params.get('is_continue_acc_check'),
            real_data_path=config_params.get('real_data_path'),
            white_list=self.white_list.copy() if self.white_list else [],
            black_list=self.black_list.copy() if self.black_list else [],
            error_data_path=config_params.get('error_data_path')
        )
