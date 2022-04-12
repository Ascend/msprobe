#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

from pytorch_gpu2npu.common_rules.common_rule import BaseInsertGlobalRule
from pytorch_gpu2npu.modelarts.path_wrapper_converter import ModelArtsPathWrapperRule
from pytorch_gpu2npu.modelarts.pytorch_api import FILE_HANDLER_API


def get_modelarts_rule():
    add_import_rule = BaseInsertGlobalRule(
        insert_content=['from ascend_modelarts_function import ModelArtsPathManager'])
    return [ModelArtsPathWrapperRule(FILE_HANDLER_API, add_import_rule), add_import_rule]
