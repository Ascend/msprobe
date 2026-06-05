"""acc_check_cli 单元测试 — 测试预检 CLI 入口函数。"""
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from msprobe.core.common.const import Const
from msprobe.core.acc_check.acc_check_cli import (
    _detect_framework_from_api_info,
    acc_check_cli,
    multi_acc_check_cli,
)


def _make_api_info(framework: str) -> str:
    """Create a temp api_info json file and return its path."""
    data = {"framework": framework}
    fd, path = tempfile.mkstemp(suffix=".json", text=True)
    with os.fdopen(fd, "w") as f:
        json.dump(data, f)
    return path


class TestDetectFrameworkFromApiInfo(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, "_tmp_path") and os.path.exists(self._tmp_path):
            os.unlink(self._tmp_path)

    def test_pytorch_framework(self):
        self._tmp_path = _make_api_info("pytorch")
        result = _detect_framework_from_api_info(self._tmp_path)
        self.assertEqual(result, Const.PT_FRAMEWORK)

    def test_pt_alias(self):
        self._tmp_path = _make_api_info("pt")
        result = _detect_framework_from_api_info(self._tmp_path)
        self.assertEqual(result, Const.PT_FRAMEWORK)

    def test_mindspore_framework(self):
        self._tmp_path = _make_api_info("mindspore")
        result = _detect_framework_from_api_info(self._tmp_path)
        self.assertEqual(result, Const.MS_FRAMEWORK)

    def test_ms_alias(self):
        self._tmp_path = _make_api_info("ms")
        result = _detect_framework_from_api_info(self._tmp_path)
        self.assertEqual(result, Const.MS_FRAMEWORK)

    def test_mt_alias(self):
        # Const.MT_FRAMEWORK is "mindtorch" — the code uses .lower()
        self._tmp_path = _make_api_info(Const.MT_FRAMEWORK)
        result = _detect_framework_from_api_info(self._tmp_path)
        self.assertEqual(result, Const.MS_FRAMEWORK)

    def test_missing_framework_key(self):
        fd, path = tempfile.mkstemp(suffix=".json", text=True)
        self._tmp_path = path
        with os.fdopen(fd, "w") as f:
            json.dump({"other": "value"}, f)
        with self.assertRaises(ValueError):
            _detect_framework_from_api_info(path)

    def test_empty_framework(self):
        self._tmp_path = _make_api_info("")
        with self.assertRaises(ValueError):
            _detect_framework_from_api_info(self._tmp_path)

    def test_unsupported_framework(self):
        self._tmp_path = _make_api_info("tensorflow")
        with self.assertRaises(ValueError):
            _detect_framework_from_api_info(self._tmp_path)

    def test_file_not_found(self):
        from msprobe.core.common.file_utils import FileCheckException
        with self.assertRaises(FileCheckException):
            _detect_framework_from_api_info("/nonexistent/path.json")

    def test_invalid_json(self):
        fd, path = tempfile.mkstemp(suffix=".json", text=True)
        self._tmp_path = path
        with os.fdopen(fd, "w") as f:
            f.write("not valid json")
        with self.assertRaises(ValueError):
            _detect_framework_from_api_info(path)

    def test_empty_path_raises(self):
        from msprobe.core.common.file_utils import FileCheckException
        with self.assertRaises(FileCheckException):
            _detect_framework_from_api_info("")


class TestAccCheckCli(unittest.TestCase):

    @patch("msprobe.core.acc_check.acc_check_cli.argparse.ArgumentParser.print_help")
    def test_no_api_info_prints_help(self, mock_print_help):
        acc_check_cli([])
        mock_print_help.assert_called_once()

    @patch("msprobe.core.acc_check.acc_check_cli._detect_framework_from_api_info")
    @patch("msprobe.core.acc_check.acc_check_cli.argparse.ArgumentParser.parse_args")
    def test_pt_framework_dispatches(self, mock_parse, mock_detect):
        """PT 分支验证正确调用 acc_check_command。"""
        mock_detect.return_value = Const.PT_FRAMEWORK
        mock_parse.return_value = MagicMock()
        mock_acc_check = MagicMock()
        mock_acc_check.acc_check_command = MagicMock()

        with patch.dict("sys.modules", {
            "msprobe.pytorch": MagicMock(),
            "msprobe.pytorch.api_accuracy_checker": MagicMock(),
            "msprobe.pytorch.api_accuracy_checker.acc_check": MagicMock(),
            "msprobe.pytorch.api_accuracy_checker.acc_check.acc_check": mock_acc_check,
        }):
            acc_check_cli(["-api_info", "/tmp/dump.json"])
            mock_acc_check.acc_check_command.assert_called_once()

    @patch("msprobe.core.acc_check.acc_check_cli._detect_framework_from_api_info")
    @patch("msprobe.core.acc_check.acc_check_cli.argparse.ArgumentParser.parse_args")
    def test_ms_framework_dispatches(self, mock_parse, mock_detect):
        """MS 分支验证正确调用 api_checker_main。"""
        mock_detect.return_value = Const.MS_FRAMEWORK
        mock_parse.return_value = MagicMock()
        mock_main = MagicMock()

        with patch.dict("sys.modules", {
            "mindspore": MagicMock(),
            "msprobe.mindspore": MagicMock(__path__=[]),
            "msprobe.mindspore.api_accuracy_checker": MagicMock(),
            "msprobe.mindspore.api_accuracy_checker.cmd_parser": MagicMock(),
            "msprobe.mindspore.api_accuracy_checker.main": mock_main,
        }):
            acc_check_cli(["-api_info", "/tmp/dump.json"])
            mock_main.api_checker_main.assert_called_once()


class TestMultiAccCheckCli(unittest.TestCase):

    @patch("msprobe.core.acc_check.acc_check_cli.argparse.ArgumentParser.print_help")
    def test_no_api_info_prints_help(self, mock_print_help):
        multi_acc_check_cli([])
        mock_print_help.assert_called_once()

    @patch("msprobe.core.acc_check.acc_check_cli._detect_framework_from_api_info")
    @patch("msprobe.core.acc_check.acc_check_cli.argparse.ArgumentParser.parse_args")
    def test_pt_framework_dispatches(self, mock_parse, mock_detect):
        """multi PT 分支验证调用 run_parallel_ut。"""
        mock_detect.return_value = Const.PT_FRAMEWORK
        mock_parse.return_value = MagicMock()
        mock_multi = MagicMock()

        with patch.dict("sys.modules", {
            "msprobe.pytorch": MagicMock(),
            "msprobe.pytorch.api_accuracy_checker": MagicMock(),
            "msprobe.pytorch.api_accuracy_checker.acc_check": MagicMock(),
            "msprobe.pytorch.api_accuracy_checker.acc_check.acc_check": MagicMock(),
            "msprobe.pytorch.api_accuracy_checker.acc_check.multi_acc_check": mock_multi,
        }):
            multi_acc_check_cli(["-api_info", "/tmp/dump.json"])
            mock_multi.run_parallel_ut.assert_called_once()

    @patch("msprobe.core.acc_check.acc_check_cli._detect_framework_from_api_info")
    @patch("msprobe.core.acc_check.acc_check_cli.argparse.ArgumentParser.parse_args")
    def test_ms_framework_dispatches(self, mock_parse, mock_detect):
        """multi MS 分支验证调用 mul_api_checker_main。"""
        mock_detect.return_value = Const.MS_FRAMEWORK
        mock_parse.return_value = MagicMock()
        mock_main = MagicMock()

        with patch.dict("sys.modules", {
            "mindspore": MagicMock(),
            "msprobe.mindspore": MagicMock(__path__=[]),
            "msprobe.mindspore.api_accuracy_checker": MagicMock(),
            "msprobe.mindspore.api_accuracy_checker.cmd_parser": MagicMock(),
            "msprobe.mindspore.api_accuracy_checker.main": mock_main,
        }):
            multi_acc_check_cli(["-api_info", "/tmp/dump.json"])
            mock_main.mul_api_checker_main.assert_called_once()


if __name__ == "__main__":
    unittest.main()
