import os
import re
import sys
import unittest
from unittest.mock import MagicMock, patch

from msprobe.core.dump.common_config import CommonConfig
from msprobe.core.dump.debugger.precision_debugger import BasePrecisionDebugger
from msprobe.pytorch.dump.debugger.precision_debugger import PrecisionDebugger
from msprobe.pytorch.dump.pt_config import StatisticsConfig

from msprobe.core.common.const import Const
from msprobe.core.common.exceptions import MsprobeException, FileCheckException
from msprobe.core.common.file_utils import FileChecker
from msprobe.core.common.utils import get_real_step_or_rank


class Args:
    def __init__(self, config_path=None, task=None, dump_path=None, level=None, model=None):
        self.config_path = config_path
        self.task = task
        self.dump_path = dump_path
        self.level = level
        self.model = model


class TestPrecisionDebugger(unittest.TestCase):
    json_config = {
        "task": "statistics",
        "dump_path": "/absolute_path",
        "rank": [],
        "step": [],
        "level": "L1",
        "async_dump": False,
    }

    statistics_common_config = CommonConfig(json_config)
    statistics_task_config = StatisticsConfig(json_config)

    def test_init(self):
        step = get_real_step_or_rank([0, 1, "3-5"], Const.STEP)
        self.assertListEqual(step, [0, 1, 3, 4, 5])

    def test_check_input_params(self):
        args = Args(config_path=1)
        with self.assertRaises(MsprobeException) as context:
            PrecisionDebugger._check_input_params(args.config_path, args.task, args.dump_path, args.level)
        self.assertEqual(context.exception.code, MsprobeException.INVALID_PARAM_ERROR)

        args = Args(config_path="/")
        with self.assertRaises(FileCheckException) as context:
            PrecisionDebugger._check_input_params(args.config_path, args.task, args.dump_path, args.level)
        self.assertEqual(context.exception.code, FileCheckException.INVALID_FILE_ERROR)

        args = Args(task=1)
        with self.assertRaises(MsprobeException) as context:
            PrecisionDebugger._check_input_params(args.config_path, args.task, args.dump_path, args.level)
        self.assertEqual(context.exception.code, MsprobeException.INVALID_PARAM_ERROR)

        args = Args(dump_path=1)
        with self.assertRaises(MsprobeException) as context:
            PrecisionDebugger._check_input_params(args.config_path, args.task, args.dump_path, args.level)
        self.assertEqual(context.exception.code, MsprobeException.INVALID_PARAM_ERROR)

        args = Args(level=1)
        with self.assertRaises(MsprobeException) as context:
            PrecisionDebugger._check_input_params(args.config_path, args.task, args.dump_path, args.level)
        self.assertEqual(context.exception.code, MsprobeException.INVALID_PARAM_ERROR)

    def test_start_statistics(self):
        PrecisionDebugger._instance = None
        with patch.object(BasePrecisionDebugger, "_parse_config_path",
                          return_value=(self.statistics_common_config, self.statistics_task_config)):
            debugger = PrecisionDebugger(dump_path="./dump_path")
        debugger.service = MagicMock()
        debugger.config = MagicMock()
        debugger.task = 'statistics'
        debugger.start()
        debugger.service.start.assert_called_once()

    def test_stop_statistics(self):
        PrecisionDebugger._instance = None
        debugger = PrecisionDebugger(dump_path="./dump_path")
        debugger.service = MagicMock()
        debugger.task = ''
        debugger._maybe_reload_config = MagicMock()
        debugger.stop()
        debugger.service.stop.assert_called_once()
        debugger._maybe_reload_config.assert_called_once_with()

    def test_step_statistics(self):
        PrecisionDebugger._instance = None
        debugger = PrecisionDebugger(dump_path="./dump_path")
        debugger.service = MagicMock()
        debugger.task = ''
        debugger._maybe_reload_config = MagicMock()
        debugger.step()
        debugger.service.step.assert_called_once()
        debugger._maybe_reload_config.assert_called_once_with()

    def test_start_statistics_with_reload(self):
        PrecisionDebugger._instance = None
        with patch.object(BasePrecisionDebugger, "_parse_config_path",
                          return_value=(self.statistics_common_config, self.statistics_task_config)):
            debugger = PrecisionDebugger(dump_path="./dump_path")
        debugger.service = MagicMock()
        debugger.config = MagicMock()
        debugger.task = 'statistics'
        debugger._maybe_reload_config = MagicMock()
        debugger.start()
        debugger.service.start.assert_called_once()
        debugger._maybe_reload_config.assert_called_once_with()

    # ========== _resolve_module_path tests ==========

    def test_resolve_valid_single_module(self):
        result = PrecisionDebugger._resolve_module_path("os")
        self.assertIs(result, sys.modules["os"])

    def test_resolve_valid_nested_module(self):
        result = PrecisionDebugger._resolve_module_path("os.path")
        self.assertIs(result, os.path)

    def test_resolve_already_loaded_module(self):
        result = PrecisionDebugger._resolve_module_path("sys")
        self.assertIs(result, sys.modules["sys"])

    def test_resolve_empty_string_raises(self):
        with self.assertRaises(MsprobeException) as ctx:
            PrecisionDebugger._resolve_module_path("")
        self.assertEqual(ctx.exception.code, MsprobeException.INVALID_CHAR_ERROR)

    def test_resolve_non_string_raises(self):
        with self.assertRaises(MsprobeException) as ctx:
            PrecisionDebugger._resolve_module_path(123)
        self.assertEqual(ctx.exception.code, MsprobeException.INVALID_CHAR_ERROR)

    def test_resolve_path_traversal_raises(self):
        for malicious in ["../etc/passwd", "os;rm -rf /", "__import__('subprocess')"]:
            with self.assertRaises(MsprobeException, msg=f"Should reject: {malicious}"):
                PrecisionDebugger._resolve_module_path(malicious)

    def test_resolve_overly_long_string_raises(self):
        long_path = "a" * (Const.MAX_MODULE_PATH_LEN + 1)
        with self.assertRaises(MsprobeException):
            PrecisionDebugger._resolve_module_path(long_path)

    def test_resolve_numeric_start_raises(self):
        with self.assertRaises(MsprobeException):
            PrecisionDebugger._resolve_module_path("123module")

    def test_resolve_max_length_accepted(self):
        valid_path = "a" * Const.MAX_MODULE_PATH_LEN
        with self.assertRaises((MsprobeException, ImportError)):
            PrecisionDebugger._resolve_module_path(valid_path)

    # ========== _load_custom_api_from_yaml tests ==========

    @staticmethod
    def _create_debugger_without_init():
        obj = PrecisionDebugger.__new__(PrecisionDebugger)
        obj._custom_api_auto_registered = set()
        obj._custom_api_pending = []
        return obj

    def test_load_yaml_dir_not_exist(self):
        debugger = self._create_debugger_without_init()
        with patch.object(FileChecker, "common_check",
                          side_effect=FileCheckException(FileCheckException.INVALID_FILE_ERROR)):
            result = debugger._load_custom_api_from_yaml()
        self.assertEqual(result, [])

    def test_load_yaml_dir_is_symlink(self):
        debugger = self._create_debugger_without_init()
        with patch.object(FileChecker, "common_check",
                          side_effect=FileCheckException(FileCheckException.SOFT_LINK_ERROR)):
            result = debugger._load_custom_api_from_yaml()
        self.assertEqual(result, [])

    def test_load_yaml_file_not_readable(self):
        debugger = self._create_debugger_without_init()
        with patch.object(FileChecker, "common_check", return_value="/fake/api_dump"), \
             patch("msprobe.pytorch.dump.debugger.precision_debugger.load_yaml",
                   side_effect=FileCheckException(FileCheckException.INVALID_FILE_ERROR)):
            result = debugger._load_custom_api_from_yaml()
        self.assertEqual(result, [])

    def test_load_yaml_content_not_dict(self):
        debugger = self._create_debugger_without_init()
        with patch.object(FileChecker, "common_check", return_value="/fake/api_dump"), \
             patch("msprobe.pytorch.dump.debugger.precision_debugger.load_yaml",
                   return_value=["not", "a", "dict"]):
            result = debugger._load_custom_api_from_yaml()
        self.assertEqual(result, [])

    def test_load_yaml_empty_dict(self):
        debugger = self._create_debugger_without_init()
        with patch.object(FileChecker, "common_check", return_value="/fake/api_dump"), \
             patch("msprobe.pytorch.dump.debugger.precision_debugger.load_yaml",
                   return_value={}):
            result = debugger._load_custom_api_from_yaml()
        self.assertEqual(result, [])

    def test_load_yaml_valid_entries(self):
        debugger = self._create_debugger_without_init()
        with patch.object(FileChecker, "common_check", return_value="/fake/api_dump"), \
             patch("msprobe.pytorch.dump.debugger.precision_debugger.load_yaml",
                   return_value={
                       "torch.nn": ["Conv2d", "Linear"],
                       "torch.nn.functional": "relu",
                   }):
            result = debugger._load_custom_api_from_yaml()
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], {"module": "torch.nn", "api": "Conv2d",
                                      "prefix": "torch.nn"})
        self.assertEqual(result[1], {"module": "torch.nn", "api": "Linear",
                                      "prefix": "torch.nn"})
        self.assertEqual(result[2], {"module": "torch.nn.functional", "api": "relu",
                                      "prefix": "torch.nn.functional"})

    def test_load_yaml_filter_invalid_module_path(self):
        debugger = self._create_debugger_without_init()
        with patch.object(FileChecker, "common_check", return_value="/fake/api_dump"), \
             patch("msprobe.pytorch.dump.debugger.precision_debugger.load_yaml",
                   return_value={
                       "../etc/passwd": ["evil"],
                       "torch.nn": ["Conv2d"],
                       "os;rm -rf": ["bad"],
                   }):
            result = debugger._load_custom_api_from_yaml()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["module"], "torch.nn")

    def test_load_yaml_filter_invalid_api_name(self):
        debugger = self._create_debugger_without_init()
        with patch.object(FileChecker, "common_check", return_value="/fake/api_dump"), \
             patch("msprobe.pytorch.dump.debugger.precision_debugger.load_yaml",
                   return_value={
                       "torch.nn": ["Conv2d", "exec()", "a;print(1)"],
                   }):
            result = debugger._load_custom_api_from_yaml()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["api"], "Conv2d")

    def test_load_yaml_filter_non_string_module_path(self):
        debugger = self._create_debugger_without_init()
        with patch.object(FileChecker, "common_check", return_value="/fake/api_dump"), \
             patch("msprobe.pytorch.dump.debugger.precision_debugger.load_yaml",
                   return_value={
                       123: ["test"],
                       None: ["test"],
                       "torch.nn": ["Conv2d"],
                   }):
            result = debugger._load_custom_api_from_yaml()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["module"], "torch.nn")

    def test_load_yaml_api_list_not_list_or_str(self):
        debugger = self._create_debugger_without_init()
        with patch.object(FileChecker, "common_check", return_value="/fake/api_dump"), \
             patch("msprobe.pytorch.dump.debugger.precision_debugger.load_yaml",
                   return_value={
                       "torch.nn": 123,
                       "torch.nn.functional": ["relu"],
                   }):
            result = debugger._load_custom_api_from_yaml()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["api"], "relu")

    def test_load_yaml_module_path_too_long(self):
        debugger = self._create_debugger_without_init()
        long_module = "a" * (Const.MAX_MODULE_PATH_LEN + 1)
        with patch.object(FileChecker, "common_check", return_value="/fake/api_dump"), \
             patch("msprobe.pytorch.dump.debugger.precision_debugger.load_yaml",
                   return_value={
                       long_module: ["test"],
                       "torch.nn": ["Conv2d"],
                   }):
            result = debugger._load_custom_api_from_yaml()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["module"], "torch.nn")

    def test_load_yaml_api_name_too_long(self):
        debugger = self._create_debugger_without_init()
        long_api = "a" * (Const.MAX_API_NAME_LEN + 1)
        with patch.object(FileChecker, "common_check", return_value="/fake/api_dump"), \
             patch("msprobe.pytorch.dump.debugger.precision_debugger.load_yaml",
                   return_value={
                       "torch.nn": [long_api, "Conv2d"],
                   }):
            result = debugger._load_custom_api_from_yaml()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["api"], "Conv2d")

    # ========== Validation regex tests ==========

    def test_module_path_pattern_valid(self):
        valid = ["torch", "torch.nn", "torch.nn.functional",
                 "my_module", "_private", "Module123", "a.b.c.d"]
        for p in valid:
            self.assertIsNotNone(
                re.match(Const.PY_MODULE_PATH_PATTERN, p),
                f"Should accept: {p}"
            )

    def test_module_path_pattern_invalid(self):
        invalid = ["../etc/passwd", "os;rm -rf /", "exec()", "",
                   "__import__('os')", "module path", "123start"]
        for p in invalid:
            self.assertIsNone(
                re.match(Const.PY_MODULE_PATH_PATTERN, p),
                f"Should reject: {p}"
            )

    def test_api_name_pattern_valid(self):
        valid = ["conv2d", "_forward", "relu", "BatchNorm2d", "a1b2"]
        for n in valid:
            self.assertIsNotNone(
                re.match(Const.PY_IDENTIFIER_PATTERN, n),
                f"Should accept: {n}"
            )

    def test_api_name_pattern_invalid(self):
        invalid = ["exec()", "x;print(1)", "a b", "", "123", "a.b"]
        for n in invalid:
            self.assertIsNone(
                re.match(Const.PY_IDENTIFIER_PATTERN, n),
                f"Should reject: {n}"
            )
