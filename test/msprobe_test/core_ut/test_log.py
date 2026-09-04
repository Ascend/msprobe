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

from unittest import TestCase
from unittest.mock import patch, MagicMock

from msprobe.core.common.log import BaseLogger, logger


class TestLog(TestCase):
    @patch("msprobe.core.common.log.print")
    def test__print_log(self, mock_print):
        logger._print_log("level", "msg")
        self.assertIn("[level] msg", mock_print.call_args[0][0])
        self.assertEqual("\n", mock_print.call_args[1].get("end"))

        logger._print_log("level", "msg", end="end")
        self.assertIn("[level] msg", mock_print.call_args[0][0])
        self.assertEqual("end", mock_print.call_args[1].get("end"))

    @patch.object(BaseLogger, "_print_log")
    def test_print_info_log(self, mock__print_log):
        logger.info("\n\n\ninfo_msg")
        mock__print_log.assert_called_with("INFO", "___info_msg")

    @patch.object(BaseLogger, "_print_log")
    def test_print_warn_log(self, mock__print_log):
        logger.warning("\n\n\nwarn_msg")
        mock__print_log.assert_called_with("WARNING", "___warn_msg")

    @patch.object(BaseLogger, "_print_log")
    def test_print_error_log(self, mock__print_log):
        logger.error("\n\n\nerror_msg")
        mock__print_log.assert_called_with("ERROR", "___error_msg")

    @patch.object(BaseLogger, "error")
    def test_error_log_with_exp(self, mock_error):
        with self.assertRaises(Exception) as context:
            logger.error_log_with_exp("msg", Exception("Exception"))
        self.assertEqual(str(context.exception), "Exception")
        mock_error.assert_called_with("msg")

    @patch.object(BaseLogger, "get_rank")
    def test_on_rank_0(self, mock_get_rank):
        mock_func = MagicMock()
        func_rank_0 = logger.on_rank_0(mock_func)

        mock_get_rank.return_value = 1
        func_rank_0()
        mock_func.assert_not_called()

        mock_get_rank.return_value = 0
        func_rank_0()
        mock_func.assert_called()

        mock_func = MagicMock()
        func_rank_0 = logger.on_rank_0(mock_func)
        mock_get_rank.return_value = None
        func_rank_0()
        mock_func.assert_called()

    @patch.object(BaseLogger, "get_rank")
    def test_info_on_rank_0(self, mock_get_rank):
        mock_print = MagicMock()
        with patch("msprobe.core.common.log.print", new=mock_print):
            mock_get_rank.return_value = 0
            logger.info_on_rank_0("msg")
            self.assertIn("[INFO] msg", mock_print.call_args[0][0])

            mock_get_rank.return_value = 1
            logger.info_on_rank_0("msg")
            mock_print.assert_called_once()

    @patch.object(BaseLogger, "get_rank")
    def test_error_on_rank_0(self, mock_get_rank):
        mock_print = MagicMock()
        with patch("msprobe.core.common.log.print", new=mock_print):
            mock_get_rank.return_value = 0
            logger.error_on_rank_0("msg")
            self.assertIn("[ERROR] msg", mock_print.call_args[0][0])

            mock_get_rank.return_value = 1
            logger.error_on_rank_0("msg")
            mock_print.assert_called_once()

    @patch.object(BaseLogger, "get_rank")
    def test_warning_on_rank_0(self, mock_get_rank):
        mock_print = MagicMock()
        with patch("msprobe.core.common.log.print", new=mock_print):
            mock_get_rank.return_value = 0
            logger.warning_on_rank_0("msg")
            self.assertIn("[WARNING] msg", mock_print.call_args[0][0])

            mock_get_rank.return_value = 1
            logger.warning_on_rank_0("msg")
            mock_print.assert_called_once()
