"""CliLogo 单元测试 — 测试 logo 渲染与条件逻辑。"""
import io
import os
import sys
import unittest
from unittest.mock import patch

from msprobe.core.common.logo import CliLogo


class TestCliLogo(unittest.TestCase):

    def setUp(self):
        self.logo = CliLogo()

    # ── _render_simple ──

    def test_render_simple_contains_mindstudio(self):
        result = self.logo._render_simple()
        self.assertIn("MindStudio", result)
        self.assertIn("=================================================================", result)

    # ── _render_colored ──

    def test_render_colored_contains_ansi(self):
        result = self.logo._render_colored()
        self.assertIn("\033[", result)
        self.assertIn("MindStudio", result)

    # ── _should_use_color_logo ──

    def test_should_use_color_logo_with_valid_term(self):
        with patch.object(sys, "stderr") as mock_stderr, \
             patch.dict(os.environ, {"TERM": "xterm-256color"}, clear=False):
            mock_stderr.isatty.return_value = True
            self.assertTrue(self.logo._should_use_color_logo())

    def test_should_use_color_logo_dumb_term(self):
        with patch.object(sys, "stderr") as mock_stderr, \
             patch.dict(os.environ, {"TERM": "dumb"}, clear=False):
            mock_stderr.isatty.return_value = True
            self.assertFalse(self.logo._should_use_color_logo())

    def test_should_use_color_logo_not_tty(self):
        with patch.object(sys, "stderr") as mock_stderr:
            mock_stderr.isatty.return_value = False
            self.assertFalse(self.logo._should_use_color_logo())

    def test_should_use_color_logo_no_term(self):
        with patch.object(sys, "stderr") as mock_stderr:
            mock_stderr.isatty.return_value = True
            with patch.dict(os.environ, {}, clear=True):
                self.assertFalse(self.logo._should_use_color_logo())

    # ── _should_skip_logo ──

    def test_should_skip_logo_with_help(self):
        test_argv = ["msprobe", "-h"]
        with patch.object(sys, "argv", test_argv):
            self.assertTrue(self.logo._should_skip_logo())

    def test_should_skip_logo_with_help_long(self):
        test_argv = ["msprobe", "--help"]
        with patch.object(sys, "argv", test_argv):
            self.assertTrue(self.logo._should_skip_logo())

    def test_should_not_skip_logo_without_help(self):
        test_argv = ["msprobe", "acc_check", "-api_info", "x.json"]
        with patch.object(sys, "argv", test_argv):
            self.assertFalse(self.logo._should_skip_logo())

    # ── print_logo ──

    @patch("msprobe.core.common.logo.CliLogo._should_skip_logo")
    @patch("msprobe.core.common.logo.CliLogo._should_use_color_logo")
    def test_print_logo_simple_when_no_color(self, mock_color, mock_skip):
        mock_skip.return_value = False
        mock_color.return_value = False
        buf = io.StringIO()
        with patch.object(sys, "stderr", buf):
            self.logo.print_logo()
        output = buf.getvalue()
        self.assertIn("MindStudio", output)
        self.assertNotIn("\033[", output)

    @patch("msprobe.core.common.logo.CliLogo._should_skip_logo")
    @patch("msprobe.core.common.logo.CliLogo._should_use_color_logo")
    def test_print_logo_colored(self, mock_color, mock_skip):
        mock_skip.return_value = False
        mock_color.return_value = True
        buf = io.StringIO()
        with patch.object(sys, "stderr", buf):
            self.logo.print_logo()
        output = buf.getvalue()
        self.assertIn("MindStudio", output)
        self.assertIn("\033[", output)

    @patch("msprobe.core.common.logo.CliLogo._should_skip_logo")
    def test_print_logo_skipped(self, mock_skip):
        mock_skip.return_value = True
        buf = io.StringIO()
        with patch.object(sys, "stderr", buf):
            self.logo.print_logo()
        self.assertEqual(buf.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
