#!/usr/bin/env python3
# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
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

from unittest import TestCase
from msprobe.core.compare.layer_mapping.data_scope_parser import DumpDataItem
from msprobe.core.common.utils import CompareException


class TestDataScopeParser(TestCase):

    def test_check_stack_valid_invalid_stack_type(self):
        stack_info_string = "conv1.Conv2d.forward.input"
        with self.assertRaises(CompareException) as context:
            DumpDataItem.check_stack_valid(stack_info_string)
        self.assertEqual(context.exception.code, CompareException.INVALID_DATA_ERROR)

    def test_check_stack_valid_invalid_stack_info(self):
        stack_info_list = ["conv1.Conv2d.forward.input", 1]
        with self.assertRaises(CompareException) as context:
            DumpDataItem.check_stack_valid(stack_info_list)
        self.assertEqual(context.exception.code, CompareException.INVALID_DATA_ERROR)
