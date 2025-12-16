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
import unittest
from unittest.mock import patch, MagicMock

import pytest

from mscmp_advisor import _check_input_file, check_safe_string, check_string_length, check_file_size, parse_input_nodes
from cmp_utils.constant.compare_error import CompareError
from cmp_utils.constant.const_manager import ConstManager



class TestUtilsMethods(unittest.TestCase):
    
    @patch('cmp_utils.log.print_error_log')
    @patch('cmp_utils.path_check.check_exec_file_valid')
    def test_valid_file(self, mock_check_exec_file_valid, mock_print_error_log):
        # 模拟 check_exec_file_valid 返回正常值
        mock_check_exec_file_valid.return_value = CompareError.MSACCUCMP_NONE_ERROR
        input_file = 'test.txt'
        file_type = '.txt'
        try:
            _check_input_file(input_file, file_type)
        except CompareError:
            self.fail("_check_input_file() raised CompareError unexpectedly!")
        # 断言没有打印错误日志
        mock_print_error_log.assert_not_called()

    @patch('cmp_utils.log.print_error_log')
    @patch('cmp_utils.path_check.check_exec_file_valid')
    def test_invalid_file_type(self, mock_check_exec_file_valid, mock_print_error_log):
        input_file = 'test.jpg'
        file_type = '.txt'
        with pytest.raises(CompareError) as error:
            _check_input_file(input_file, file_type)
        self.assertEqual(error.value.code, CompareError.MSACCUCMP_INVALID_TYPE_ERROR)
        # 断言打印了错误日志
        mock_print_error_log.assert_called_once()

    @patch('cmp_utils.log.print_error_log')
    @patch('cmp_utils.path_check.check_exec_file_valid')
    def test_invalid_exec_file(self, mock_check_exec_file_valid, mock_print_error_log):
        # 模拟 check_exec_file_valid 返回错误值
        mock_check_exec_file_valid.return_value = CompareError.MSACCUCMP_OPEN_FILE_ERROR
        input_file = 'test.txt'
        file_type = '.txt'
        with pytest.raises(CompareError) as error:
            _check_input_file(input_file, file_type)
        self.assertEqual(error.value.code, CompareError.MSACCUCMP_OPEN_FILE_ERROR)
        # 断言打印了错误日志
        mock_print_error_log.assert_not_called()
    
    @patch('os.path.getsize')
    @patch('cmp_utils.log.print_error_log')
    def test_file_size_within_limit(self, mock_print_error_log, mock_getsize):
        # 模拟文件大小在限制范围内
        mock_getsize.return_value = 10 * 1024 * 1024  # 10MB
        input_file = 'test_file.txt'
        try:
            check_file_size(input_file)
        except CompareError:
            self.fail("check_file_size() raised CompareError unexpectedly!")
        # 断言没有打印错误日志
        mock_print_error_log.assert_not_called()

    @patch('os.path.getsize')
    @patch('cmp_utils.log.print_error_log')
    def test_file_size_exceeds_limit(self, mock_print_error_log, mock_getsize):
        # 模拟文件大小超过限制
        mock_getsize.return_value = 200 * 1024 * 1024  # 200MB
        input_file = 'test_file.txt'
        with pytest.raises(CompareError) as error:
            check_file_size(input_file)
        self.assertEqual(error.value.code, CompareError.MSACCUCMP_INVALID_FILE_ERROR)
        # 断言打印了错误日志
        mock_print_error_log.assert_called_once()

    @patch('os.path.getsize')
    @patch('cmp_utils.log.print_error_log')
    def test_os_error(self, mock_print_error_log, mock_getsize):
        # 模拟 os.path.getsize 抛出 OSError
        mock_getsize.side_effect = OSError('Mocked OSError')
        input_file = 'test_file.txt'
        with pytest.raises(CompareError) as error:
            check_file_size(input_file)
        self.assertEqual(error.value.code, CompareError.MSACCUCMP_OPEN_FILE_ERROR)
        # 断言打印了错误日志
        mock_print_error_log.assert_called_once()

    def test_string_length_exceeds_limit(self):
        # 测试字符串长度超过限制的情况
        # 构造一个长度超过 MAX_STRING_LENGTH 的字符串
        MAX_STRING_LENGTH = 1024
        s = 'a' * (MAX_STRING_LENGTH + 1)
        with pytest.raises(CompareError) as error:
            check_string_length(s)
        self.assertEqual(error.value.code, CompareError.MSACCUCMP_INVALID_FILE_ERROR)
        
    def test_parse_input_nodes(self):
        input_nodes = ""
        self.assertEqual(parse_input_nodes(input_nodes), [])
        input_nodes = "conv2"
        self.assertEqual(parse_input_nodes(input_nodes), ["conv2"])


if __name__ == '__main__':
    unittest.main()