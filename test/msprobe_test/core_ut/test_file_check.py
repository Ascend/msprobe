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

from unittest import TestCase
from unittest.mock import patch, MagicMock

from msprobe.core.common.log import logger
from msprobe.core.common.const import FileCheckConst
from msprobe.core.common.exceptions import FileCheckException
from msprobe.core.common.file_utils import (check_link,
                                            check_path_length,
                                            check_path_exists,
                                            check_path_readability,
                                            check_path_writability,
                                            check_path_executable,
                                            check_other_user_writable,
                                            check_path_pattern_valid,
                                            check_file_size,
                                            check_common_file_size,
                                            check_file_suffix,
                                            check_path_type)


class TestFileCheckUtil(TestCase):
    @patch.object(logger, "error")
    def test_check_link(self, mock_logger_error):
        with patch("msprobe.core.common.file_utils.os.path.abspath", return_value="link_path"), \
             patch("msprobe.core.common.file_utils.os.path.islink", return_value=True):
            with self.assertRaises(FileCheckException) as context:
                check_link("link_path")
        self.assertEqual(str(context.exception),
                         FileCheckException.err_strs.get(FileCheckException.SOFT_LINK_ERROR))
        mock_logger_error.assert_called_once()
        self.assertIn("link_path is a soft link.", mock_logger_error.call_args.args[0])

    @patch.object(logger, "error")
    def test_check_path_length(self, mock_logger_error):
        path = "P" * (FileCheckConst.DIRECTORY_LENGTH + 1)
        with self.assertRaises(FileCheckException) as context:
            check_path_length(path)
        self.assertEqual(str(context.exception),
                         FileCheckException.err_strs.get(FileCheckException.ILLEGAL_PATH_ERROR))
        mock_logger_error.assert_called_with("The file path length exceeds limit.")

        path = "P" * (FileCheckConst.FILE_NAME_LENGTH + 1)
        with self.assertRaises(FileCheckException) as context:
            check_path_length(path)
        self.assertEqual(str(context.exception),
                         FileCheckException.err_strs.get(FileCheckException.ILLEGAL_PATH_ERROR))
        mock_logger_error.assert_called_with("The file path length exceeds limit.")

        path = "P" * (FileCheckConst.FILE_NAME_LENGTH - 5)
        with self.assertRaises(FileCheckException) as context:
            check_path_length(path, name_length=FileCheckConst.FILE_NAME_LENGTH - 6)
        self.assertEqual(str(context.exception),
                         FileCheckException.err_strs.get(FileCheckException.ILLEGAL_PATH_ERROR))
        mock_logger_error.assert_called_with("The file path length exceeds limit.")

    @patch.object(logger, "error")
    def test_check_path_exists(self, mock_logger_error):
        with patch("msprobe.core.common.file_utils.os.path.exists", return_value=False):
            with self.assertRaises(FileCheckException) as context:
                check_path_exists("file_path")
            self.assertEqual(str(context.exception),
                             FileCheckException.err_strs.get(FileCheckException.ILLEGAL_PATH_ERROR))
            mock_logger_error.assert_called_with("The file path file_path does not exist.")

    @patch.object(logger, "error")
    def test_check_path_readability(self, mock_logger_error):
        path = "file_path"
        with patch("msprobe.core.common.file_utils.os.access", return_value=False):
            with self.assertRaises(FileCheckException) as context:
                check_path_readability(path)
            self.assertEqual(str(context.exception),
                             FileCheckException.err_strs.get(FileCheckException.FILE_PERMISSION_ERROR))
            mock_logger_error.assert_called_with(f"The file path {path} is not readable.")

        mock_access = MagicMock()
        mock_access.return_value = True
        with patch("msprobe.core.common.file_utils.os.access", new=mock_access):
            check_path_readability(path)
        self.assertEqual(mock_access.call_args[0], (path, os.R_OK))

    @patch.object(logger, "error")
    def test_check_path_writability(self, mock_logger_error):
        path = "file_path"
        with patch("msprobe.core.common.file_utils.os.access", return_value=False):
            with self.assertRaises(FileCheckException) as context:
                check_path_writability(path)
            self.assertEqual(str(context.exception),
                             FileCheckException.err_strs.get(FileCheckException.FILE_PERMISSION_ERROR))
            mock_logger_error.assert_called_with(f"The file path {path} is not writable.")

        mock_access = MagicMock()
        mock_access.return_value = True
        with patch("msprobe.core.common.file_utils.os.access", new=mock_access):
            check_path_writability(path)
        self.assertEqual(mock_access.call_args[0], (path, os.W_OK))

    @patch.object(logger, "error")
    def test_check_path_executable(self, mock_logger_error):
        path = "file_path"
        with patch("msprobe.core.common.file_utils.os.access", return_value=False):
            with self.assertRaises(FileCheckException) as context:
                check_path_executable(path)
            self.assertEqual(str(context.exception),
                             FileCheckException.err_strs.get(FileCheckException.FILE_PERMISSION_ERROR))
            mock_logger_error.assert_called_with(f"The file path {path} is not executable.")

        mock_access = MagicMock()
        mock_access.return_value = True
        with patch("msprobe.core.common.file_utils.os.access", new=mock_access):
            check_path_executable(path)
        self.assertEqual(mock_access.call_args[0], (path, os.X_OK))

    @patch.object(logger, "error")
    def test_check_other_user_writable(self, mock_logger_error):
        class TestStat:
            def __init__(self, mode):
                self.st_mode = mode

        path = "file_path"
        mock_stat = TestStat(0o002)
        with patch("msprobe.core.common.file_utils.os.stat", return_value=mock_stat):
            with self.assertRaises(FileCheckException) as context:
                check_other_user_writable(path)
            self.assertEqual(str(context.exception),
                             FileCheckException.err_strs.get(FileCheckException.FILE_PERMISSION_ERROR))
            mock_logger_error.assert_called_with(f"The file path {path} may be insecure "
                                                 "because other users have write permissions. ")

    @patch.object(logger, "error")
    def test_check_path_pattern_valid(self, mock_logger_error):
        path = "path"
        mock_re_match = MagicMock()
        mock_re_match.return_value = False
        with patch("msprobe.core.common.file_utils.re.match", new=mock_re_match):
            with self.assertRaises(FileCheckException) as context:
                check_path_pattern_valid(path)
            self.assertEqual(str(context.exception),
                             FileCheckException.err_strs.get(FileCheckException.ILLEGAL_PATH_ERROR))
        mock_logger_error.assert_called_with(f"The file path {path} contains special characters.")
        mock_re_match.assert_called_with(FileCheckConst.FILE_VALID_PATTERN, path)

    @patch.object(logger, "error")
    def test_check_file_size(self, mock_logger_error):
        file_path = os.path.realpath(__file__)
        file_size = os.path.getsize(file_path)
        max_size = file_size
        with self.assertRaises(FileCheckException) as context:
            check_file_size(file_path, max_size)
        self.assertEqual(str(context.exception),
                         FileCheckException.err_strs.get(FileCheckException.FILE_TOO_LARGE_ERROR))
        mock_logger_error.assert_called_with(f"The size ({file_size}) of {file_path} exceeds ({max_size}) bytes, "
                                             f"tools not support.")

    def test_check_common_file_size(self):
        mock_check_file_size = MagicMock()
        with patch("msprobe.core.common.file_utils.os.path.isfile", return_value=True), \
             patch("msprobe.core.common.file_utils.check_file_size", new=mock_check_file_size):
            for suffix, max_size in FileCheckConst.FILE_SIZE_DICT.items():
                check_common_file_size(suffix)
                mock_check_file_size.assert_called_with(suffix, max_size)

    @patch.object(logger, "error")
    def test_check_file_suffix(self, mock_logger_error):
        file_path = "file_path"
        suffix = "suffix"
        with self.assertRaises(FileCheckException) as context:
            check_file_suffix(file_path, suffix)
        self.assertEqual(str(context.exception),
                         FileCheckException.err_strs.get(FileCheckException.INVALID_FILE_ERROR))
        mock_logger_error.assert_called_with(f"{file_path} should be one of the following types of files: {[suffix]}!")

    @patch.object(logger, "error")
    def test_check_path_type(self, mock_logger_error):
        file_path = "file_path"

        with patch("msprobe.core.common.file_utils.os.path.isfile", return_value=False), \
             patch("msprobe.core.common.file_utils.os.path.isdir", return_value=True):
            with self.assertRaises(FileCheckException) as context:
                check_path_type(file_path, FileCheckConst.FILE)
            self.assertEqual(str(context.exception),
                             FileCheckException.err_strs.get(FileCheckException.INVALID_FILE_ERROR))
        mock_logger_error.assert_called_with(f"The {file_path} should be a file!")

        with patch("msprobe.core.common.file_utils.os.path.isfile", return_value=True), \
             patch("msprobe.core.common.file_utils.os.path.isdir", return_value=False):
            with self.assertRaises(FileCheckException) as context:
                check_path_type(file_path, FileCheckConst.DIR)
            self.assertEqual(str(context.exception),
                             FileCheckException.err_strs.get(FileCheckException.INVALID_FILE_ERROR))
        mock_logger_error.assert_called_with(f"The {file_path} should be a dictionary!")
