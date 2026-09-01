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
import sys
from unittest import TestCase
from unittest.mock import patch, MagicMock

from msprobe.core.common.log import BaseLogger, logger


class TestLog(TestCase):
    @patch.object(BaseLogger, "_output")
    def test__print_log(self, mock_output):
        logger._print_log("level", "msg")
        self.assertIn("[level] msg", mock_output.call_args[0][0])
        self.assertEqual("\n", mock_output.call_args[1].get("end"))

        logger._print_log("level", "msg", end="end")
        self.assertIn("[level] msg", mock_output.call_args[0][0])
        self.assertEqual("end", mock_output.call_args[1].get("end"))

    @patch.object(BaseLogger, "_output")
    def test__print_log_routes_to_output(self, mock_output):
        logger._print_log("level", "msg")
        self.assertEqual(mock_output.call_count, 1)
        self.assertIn("[level] msg", mock_output.call_args[0][0])
        self.assertEqual("\n", mock_output.call_args[1].get("end"))

    @patch.object(BaseLogger, "_output")
    def test_raw_routes_to_output(self, mock_output):
        logger.raw("raw_msg___")
        mock_output.assert_called_with("raw_msg")

    @staticmethod
    def _fake_tqdm_module(instances):
        fake_cls = MagicMock()
        fake_cls._instances = instances
        fake_module = MagicMock()
        fake_module.tqdm = fake_cls
        return fake_module, fake_cls

    def test__output_given_active_bar_in_owner_process_then_use_tqdm_write(self):
        bar = MagicMock()
        fake_module, fake_cls = self._fake_tqdm_module([bar])
        with patch.dict(sys.modules, {"tqdm": fake_module}), \
                patch("msprobe.core.common.log.print") as mock_print:
            BaseLogger._output("msg")
        fake_cls.write.assert_called_once_with("msg", end="\n")
        mock_print.assert_not_called()
        bar.clear.assert_not_called()

    def test__output_given_forked_child_with_inherited_bar_then_clear_without_redraw(self):
        bar = MagicMock()
        fake_module, fake_cls = self._fake_tqdm_module([bar])
        with patch.dict(sys.modules, {"tqdm": fake_module}), \
                patch("msprobe.core.common.log._INIT_PID", os.getpid() - 1), \
                patch("msprobe.core.common.log.print") as mock_print:
            BaseLogger._output("msg")
        bar.clear.assert_called_once_with(nolock=True)
        fake_cls.write.assert_not_called()
        mock_print.assert_called_once_with("msg", end="\n")

    def test__output_given_clear_raises_then_fallback_to_print(self):
        bar = MagicMock()
        bar.clear.side_effect = Exception("clear failed")
        fake_module, _ = self._fake_tqdm_module([bar])
        with patch.dict(sys.modules, {"tqdm": fake_module}), \
                patch("msprobe.core.common.log._INIT_PID", os.getpid() - 1), \
                patch("msprobe.core.common.log.print") as mock_print:
            BaseLogger._output("msg", end="")
        mock_print.assert_called_once_with("msg", end="")

    def test__output_given_no_active_bar_then_plain_print(self):
        fake_module, fake_cls = self._fake_tqdm_module([])
        with patch.dict(sys.modules, {"tqdm": fake_module}), \
                patch("msprobe.core.common.log.print") as mock_print:
            BaseLogger._output("msg")
        fake_cls.write.assert_not_called()
        mock_print.assert_called_once_with("msg", end="\n")

    def test__output_given_tqdm_unavailable_then_plain_print(self):
        with patch.dict(sys.modules, {"tqdm": None}), \
                patch("msprobe.core.common.log.print") as mock_print:
            BaseLogger._output("msg")
        mock_print.assert_called_once_with("msg", end="\n")

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
        mock_output = MagicMock()
        with patch.object(BaseLogger, "_output", new=mock_output):
            mock_get_rank.return_value = 0
            logger.info_on_rank_0("msg")
            self.assertIn("[INFO] msg", mock_output.call_args[0][0])

            mock_get_rank.return_value = 1
            logger.info_on_rank_0("msg")
            mock_output.assert_called_once()

    @patch.object(BaseLogger, "get_rank")
    def test_error_on_rank_0(self, mock_get_rank):
        mock_output = MagicMock()
        with patch.object(BaseLogger, "_output", new=mock_output):
            mock_get_rank.return_value = 0
            logger.error_on_rank_0("msg")
            self.assertIn("[ERROR] msg", mock_output.call_args[0][0])

            mock_get_rank.return_value = 1
            logger.error_on_rank_0("msg")
            mock_output.assert_called_once()

    @patch.object(BaseLogger, "get_rank")
    def test_warning_on_rank_0(self, mock_get_rank):
        mock_output = MagicMock()
        with patch.object(BaseLogger, "_output", new=mock_output):
            mock_get_rank.return_value = 0
            logger.warning_on_rank_0("msg")
            self.assertIn("[WARNING] msg", mock_output.call_args[0][0])

            mock_get_rank.return_value = 1
            logger.warning_on_rank_0("msg")
            mock_output.assert_called_once()
