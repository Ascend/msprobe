#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

from transfer.common_rules.common_rule import BaseInsertGlobalRule
from transfer.modelarts.path_wrapper_rule import ModelArtsPathWrapperRule
from transfer.modelarts.file_handler_api import FILE_HANDLER_API


def get_modelarts_rule():
    add_import_rule = BaseInsertGlobalRule(
        insert_content=['from ascend_modelarts_function import ModelArtsPathManager'])
    return [ModelArtsPathWrapperRule(FILE_HANDLER_API, add_import_rule), add_import_rule]
