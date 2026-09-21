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

import json
import os
import shutil
from unittest import TestCase
from unittest.mock import patch, MagicMock, mock_open

from msprobe.core.common.exceptions import ParseJsonException
from msprobe.pytorch.common.parse_json import parse_data_name_with_pattern
from msprobe.pytorch.common.parse_json import parse_json_info_forward_backward


class TestParseJson(TestCase):

    def setUp(self):
        self.base_test_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        self.input_dir = os.path.join(self.base_test_dir, 'resources')
        self.output_path = os.path.abspath(os.path.join(self.base_test_dir, 'test_output'))

        os.makedirs(self.output_path, mode=0o700, exist_ok=True)
        self.has_error = False

    def tearDown(self) -> None:
        shutil.rmtree(self.output_path, ignore_errors=True)

    def test_parse_data_name_with_pattern_exception(self):
        invalid_data_name = "matmul.unforward"
        pattern = "forward"
        json_path = "test/test.json"

        with self.assertRaises(ParseJsonException):
            parse_data_name_with_pattern(invalid_data_name, pattern, json_path)

    def test_parse_json_info_forward_backward_no_dump_data_exception(self):
        no_dump_data_json = os.path.join(self.input_dir, 'common', 'test_no_dump_data.json')
        with self.assertRaises(ParseJsonException):
            parse_json_info_forward_backward(no_dump_data_json)

    def test_parse_json_info_forward_backward(self):
        backward_json = os.path.join(self.input_dir, 'common', 'test_backward.json')
        golden_result = ({}, {"matmul": 1}, "./")
        result = parse_json_info_forward_backward(backward_json)
        self.assertEqual(result, golden_result)

    def test_parse_json_info_forward_backward_no_fb_data_exception(self):
        no_fb_data_json = os.path.join(self.input_dir, 'common', 'test_no_fb_data.json')
        with self.assertRaises(ParseJsonException):
            parse_json_info_forward_backward(no_fb_data_json)
