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

from ..common_rules.common_rule import BaseInsertGlobalRule
from .path_wrapper_rule import ModelArtsPathWrapperRule
from .file_handler_api import FILE_HANDLER_API


def get_modelarts_rule():
    add_import_rule = BaseInsertGlobalRule(
        insert_content=['from ascend_modelarts_function import ModelArtsPathManager'])
    return [ModelArtsPathWrapperRule(FILE_HANDLER_API, add_import_rule), add_import_rule]
