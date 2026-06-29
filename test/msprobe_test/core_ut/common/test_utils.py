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

import json
import os
import tempfile
import unittest
from unittest import TestCase
from unittest.mock import MagicMock, mock_open, patch
import numpy as np
from pathlib import Path
import torch

from msprobe.core.common.const import Const
from msprobe.core.common.file_utils import (
    FileCheckConst,
    FileCheckException,
    check_file_or_directory_path,
    check_file_size,
    get_file_content_bytes,
    get_json_contents,
    save_json,
)
from msprobe.core.common.log import logger
from msprobe.core.common.exceptions import MsprobeException
from msprobe.core.common.utils import (CompareException,
                                       check_compare_param,
                                       _check_json,
                                       check_json_file,
                                       check_regex_prefix_format_valid,
                                       set_dump_path,
                                       get_dump_mode,
                                       get_real_step_or_rank,
                                       get_step_or_rank_from_string,
                                       get_stack_construct_by_dump_json_path,
                                       check_seed_all,
                                       safe_get_value,
                                       MsprobeBaseException,
                                       check_str_param,
                                       is_json_file,
                                       detect_framework_by_dump_json,
                                       is_save_variable_valid,
                                       get_file_type,
                                       check_dump_json_key,
                                       is_torchrec_available,
                                       is_jagged_tensor,
                                       is_keyed_jagged_tensor,
                                       is_keyed_tensor)
from msprobe.core.common.decorator import recursion_depth_decorator
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor, KeyedTensor


class TestUtils(TestCase):

    def test_check_file_or_directory_path(self):
        class TestFileChecker:
            file_path = ""
            path_type = ""
            ability = ""
            file_suffix = None
            checked = False

            def __init__(self, file_path, path_type, ability=None, file_suffix=None):
                TestFileChecker.file_path = file_path
                TestFileChecker.path_type = path_type
                TestFileChecker.ability = ability
                TestFileChecker.file_suffix = file_suffix

            def common_check(self):
                TestFileChecker.checked = True

        file_path = os.path.realpath(__file__)
        dirname = os.path.dirname(file_path)

        with patch("msprobe.core.common.file_utils.FileChecker", new=TestFileChecker):
            check_file_or_directory_path(file_path, isdir=False, is_strict=False, file_suffix=None)
        self.assertTrue(TestFileChecker.checked)
        self.assertEqual(TestFileChecker.file_path, file_path)
        self.assertEqual(TestFileChecker.path_type, FileCheckConst.FILE)
        self.assertEqual(TestFileChecker.ability, FileCheckConst.READ_ABLE)

        TestFileChecker.checked = False
        with patch("msprobe.core.common.file_utils.FileChecker", new=TestFileChecker):
            check_file_or_directory_path(dirname, isdir=True, is_strict=False, file_suffix=None)
        self.assertTrue(TestFileChecker.checked)
        self.assertEqual(TestFileChecker.file_path, dirname)
        self.assertEqual(TestFileChecker.path_type, FileCheckConst.DIR)
        self.assertEqual(TestFileChecker.ability, FileCheckConst.WRITE_ABLE)

    @patch.object(logger, "error")
    def test_check_compare_param(self, mock_error):
        params = {
            "npu_path": "npu_path.json",
            "bench_path": "bench_path.json",
            "stack_path": "stack_path.json",
            "npu_dump_data_dir": "npu_dump_data_dir",
            "bench_dump_data_dir": "bench_dump_data_dir"
        }

        call_args = [
            ("npu_path.json", False),
            ("bench_path.json", False),
            ("npu_dump_data_dir", True),
            ("bench_dump_data_dir", True),
            ("output_path", True),
            ("npu_path.json", False),
            ("bench_path.json", False),
            ("stack_path.json", False),
            ("output_path", True)
        ]

        with self.assertRaises(CompareException) as context:
            check_compare_param("npu_path", "output_path", dump_mode=Const.ALL, stack_mode=False)
        self.assertEqual(context.exception.code, CompareException.INVALID_PARAM_ERROR)
        mock_error.assert_called_with("Invalid input parameter 'input_param', "
                                      "the expected type dict but got <class 'str'>.")

        mock_check_file_or_directory_path = MagicMock()
        mock__check_json = MagicMock()
        with patch("msprobe.core.common.utils.FileOpen", mock_open(read_data="")), \
                patch("msprobe.core.common.utils._check_json", mock__check_json), \
                patch("msprobe.core.common.utils.check_file_or_directory_path", mock_check_file_or_directory_path):
            check_compare_param(params, "output_path", dump_mode=Const.ALL, stack_mode=False)
            check_compare_param(params, "output_path", dump_mode=Const.MD5, stack_mode=True)
        for i in range(len(call_args)):
            self.assertEqual(mock_check_file_or_directory_path.call_args_list[i][0], call_args[i])
        self.assertEqual(len(mock__check_json.call_args[0]), 2)
        self.assertEqual(mock__check_json.call_args[0][1], "stack_path.json")

    @patch.object(logger, "error")
    def test__check_json(self, mock_error):
        class TestOpen:
            def __init__(self, string):
                self.string = string

            def readline(self):
                return self.string

            def seek(self, begin, end):
                self.string = str(begin) + "_" + str(end)

        with self.assertRaises(CompareException) as context:
            _check_json(TestOpen(""), "test.json")
        self.assertEqual(context.exception.code, CompareException.INVALID_DUMP_FILE)
        mock_error.assert_called_with("dump file test.json have empty line!")

        handler = TestOpen("jons file\n")
        _check_json(handler, "test.json")
        self.assertEqual(handler.string, "0_0")

    @patch("msprobe.core.common.utils._check_json")
    def test_check_json_file(self, _mock_check_json):
        input_param = {
            "npu_path": "npu_path",
            "bench_path": "bench_path",
            "stack_path": "stack_path"
        }
        check_json_file(input_param, "npu_json", "bench_json", "stack_json")
        self.assertEqual(_mock_check_json.call_args_list[0][0], ("npu_json", "npu_path"))
        self.assertEqual(_mock_check_json.call_args_list[1][0], ("bench_json", "bench_path"))
        self.assertEqual(_mock_check_json.call_args_list[2][0], ("stack_json", "stack_path"))

    @patch.object(logger, "error")
    def test_check_file_size(self, mock_error):
        with patch("msprobe.core.common.utils.os.path.getsize", return_value=120):
            with self.assertRaises(FileCheckException) as context:
                check_file_size("input_file", 100)
        self.assertEqual(context.exception.code, FileCheckException.FILE_TOO_LARGE_ERROR)
        mock_error.assert_called_with("The size (120) of input_file exceeds (100) bytes, tools not support.")

    def test_check_regex_prefix_format_valid(self):
        prefix = "A" * 21
        with self.assertRaises(ValueError) as context:
            check_regex_prefix_format_valid(prefix)
        self.assertEqual(str(context.exception), f"Maximum length of prefix is {Const.REGEX_PREFIX_MAX_LENGTH}, "
                                                 f"while current length is {len(prefix)}")

        prefix = "(prefix)"
        with self.assertRaises(ValueError) as context:
            check_regex_prefix_format_valid(prefix)
        self.assertEqual(str(context.exception), f"prefix contains invalid characters, "
                                                 f"prefix pattern {Const.REGEX_PREFIX_PATTERN}")

    @patch.object(logger, "error")
    def test_set_dump_path(self, mock_error):
        input_param = {
            "npu_path": None,
            "bench_path": "bench_path"
        }

        with self.assertRaises(CompareException) as context:
            set_dump_path(input_param)
        self.assertEqual(context.exception.code, CompareException.INVALID_PATH_ERROR)
        mock_error.assert_called_with(
            "npu_path and bench_path must both end with dump.json or debug.json. "
            "npu_path: None, bench_path: bench_path")

    @patch.object(logger, "error")
    def test_get_dump_mode(self, mock_error):
        input_param = {
            "npu_json_path": None,
            "bench_json_path": "bench_path"
        }
        npu_json = {
            "task": Const.TENSOR,
            "dump_data_dir": "dump_data_dir",
            "data": {"api": "value"}
        }

        input_param["npu_json_path"] = "npu_path"
        with patch("msprobe.core.common.utils.load_json", return_value=npu_json), \
                patch("msprobe.core.common.utils.get_file_type", return_value=Const.DUMP_JSON_FILE):
            dump_mode = get_dump_mode(input_param)
        self.assertEqual(dump_mode, Const.ALL)

        npu_json["task"] = Const.STATISTICS
        with patch("msprobe.core.common.utils.load_json", return_value=npu_json), \
                patch("msprobe.core.common.utils.md5_find", return_value=True), \
                patch("msprobe.core.common.utils.get_file_type", return_value=Const.DUMP_JSON_FILE):
            dump_mode = get_dump_mode(input_param)
        self.assertEqual(dump_mode, Const.MD5)

        npu_json["task"] = Const.EXCEPTION_DUMP
        with patch("msprobe.core.common.utils.load_json", return_value=npu_json), \
                patch("msprobe.core.common.utils.get_file_type", return_value=Const.DUMP_JSON_FILE):
            with self.assertRaises(CompareException) as context:
                dump_mode = get_dump_mode(input_param)
            self.assertEqual(context.exception.code, CompareException.INVALID_TASK_ERROR)
            mock_error.assert_called_with("Compare applies only to task is tensor or statistics")

    def test_get_file_type(self):
        # 测试有效的 file_path (dump.json)
        file_path = 'path/to/dump.json'
        expected_file_type = Const.DUMP_JSON_FILE
        self.assertEqual(get_file_type(file_path), expected_file_type)

        # 测试有效的 file_path (debug.json)
        file_path = 'path/to/debug.json'
        expected_file_type = Const.DEBUG_JSON_FILE
        self.assertEqual(get_file_type(file_path), expected_file_type)

        # 测试无效的 file_path
        file_path = 'path/to/unknown.json'
        with self.assertRaises(CompareException) as context:
            get_file_type(file_path)
        self.assertEqual(context.exception.code, CompareException.INVALID_PATH_ERROR)

        # 测试非字符串类型的 file_path
        file_path = 12345  # 非字符串类型
        with self.assertRaises(CompareException) as context:
            get_file_type(file_path)
        self.assertEqual(context.exception.code, CompareException.INVALID_PATH_ERROR)

    @patch('msprobe.core.common.file_utils.get_file_content_bytes')
    def test_get_json_contents_should_raise_exception(self, mock_get_file_content_bytes):
        mock_get_file_content_bytes.return_value = 'not a dict'
        with self.assertRaises(FileCheckException) as ce:
            get_json_contents('')
        self.assertEqual(ce.exception.code, FileCheckException.INVALID_FILE_ERROR)

    def test_get_json_contents_should_return_json_obj(self):
        test_dict = {"key": "value"}
        file_name = 'test.json'

        fd = os.open(file_name, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
        with os.fdopen(fd, 'w') as f:
            json.dump(test_dict, f)
        self.assertEqual(get_json_contents(file_name), test_dict)
        os.remove(file_name)

    def test_get_file_content_bytes(self):
        fd = os.open('test.txt', os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
        with os.fdopen(fd, 'w') as f:
            f.write("Hello, World!")
        self.assertEqual(get_file_content_bytes('test.txt'), b"Hello, World!")
        os.remove('test.txt')

    def test_get_real_step_or_rank(self):
        with self.assertRaises(MsprobeException) as context:
            get_real_step_or_rank([], "invalid_obj")
        self.assertEqual(context.exception.code, MsprobeException.INVALID_PARAM_ERROR)
        result = get_real_step_or_rank(None, "step")
        self.assertEqual(result, [])
        with self.assertRaises(MsprobeException) as context:
            get_real_step_or_rank("not_a_list", "step")
        self.assertEqual(context.exception.code, MsprobeException.INVALID_PARAM_ERROR)
        with self.assertRaises(MsprobeException) as context:
            get_real_step_or_rank([-1, 1, 2], "step")
        self.assertEqual(context.exception.code, MsprobeException.INVALID_PARAM_ERROR)
        with self.assertRaises(MsprobeException) as context:
            get_real_step_or_rank([1, 2, 3.5], "step")
        self.assertEqual(context.exception.code, MsprobeException.INVALID_PARAM_ERROR)
        with self.assertRaises(MsprobeException) as context:
            get_real_step_or_rank([True, 1, 2], "step")
        self.assertEqual(context.exception.code, MsprobeException.INVALID_PARAM_ERROR)
        with self.assertRaises(MsprobeException) as context:
            get_real_step_or_rank([10000000], "step")
        self.assertEqual(context.exception.code, MsprobeException.INVALID_PARAM_ERROR)
        result = get_real_step_or_rank([1, 10, 50], "step")
        self.assertEqual(result, [1, 10, 50])

    def test_get_step_or_rank_from_string(self):
        with self.assertRaises(MsprobeException) as context:
            get_step_or_rank_from_string("1-4-5", "step")
        self.assertEqual(context.exception.code, MsprobeException.INVALID_PARAM_ERROR)
        with self.assertRaises(MsprobeException) as context:
            get_step_or_rank_from_string("!-,", "step")
        self.assertEqual(context.exception.code, MsprobeException.INVALID_PARAM_ERROR)
        with self.assertRaises(MsprobeException) as context:
            get_step_or_rank_from_string("5-3", "step")
        self.assertEqual(context.exception.code, MsprobeException.INVALID_PARAM_ERROR)
        with self.assertRaises(MsprobeException) as context:
            get_step_or_rank_from_string("5-100000000000000", "step")
        self.assertEqual(context.exception.code, MsprobeException.INVALID_PARAM_ERROR)
        result = get_real_step_or_rank(["1-5", 10], "rank")
        self.assertEqual(result, [1, 2, 3, 4, 5, 10])
        result = get_real_step_or_rank([10, "1-3", 3], "step")
        self.assertEqual(result, [1, 2, 3, 10])

    def test_get_stack_construct_by_dump_json_path_when_dump_json_path_is_none_then_fail(self):
        dump_json_path = None
        with self.assertRaises(CompareException) as context:
            stack, construct = get_stack_construct_by_dump_json_path(dump_json_path)
        self.assertEqual(context.exception.code, CompareException.INVALID_PATH_ERROR)

    def test_get_stack_construct_by_dump_json_path_when_dump_json_path_invalid_then_fail(self):
        dump_json_path = "./abc/dump.json"
        framework = Const.MS_FRAMEWORK
        with self.assertRaises(FileCheckException) as context:
            stack, construct = get_stack_construct_by_dump_json_path(dump_json_path)
        self.assertEqual(context.exception.code, FileCheckException.ILLEGAL_PATH_ERROR)

    def test_get_stack_construct_by_dump_json_path_valid_paths_then_pass(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stack_json_path = os.path.join(temp_dir, 'stack.json')
            construct_json_path = os.path.join(temp_dir, 'construct.json')
            dump_json_path = os.path.join(temp_dir, 'dump.json')

            save_json(stack_json_path, {'stack_key': 'stack_value'})
            save_json(construct_json_path, {'construct_key': 'construct_value'})
            save_json(dump_json_path, {'dump_key': 'dump_value'})

            stack, construct = get_stack_construct_by_dump_json_path(dump_json_path)

            self.assertEqual(stack, {'stack_key': 'stack_value'})
            self.assertEqual(construct, {'construct_key': 'construct_value'})

    @patch.object(logger, "error")
    def test_recursion_depth_decorator(self, mock_error):
        # 测试递归深度限制函数
        recursion_list = [[]]
        temp_list = recursion_list[0]
        for _ in range(Const.MAX_DEPTH):
            temp_list.append([])
            temp_list = temp_list[0]
        temp_list.append(0)
        call_record = []

        @recursion_depth_decorator("test func_info")
        def recursion_func(test_list, call_record):
            call_record.append(1)
            if isinstance(test_list, list):
                recursion_func(test_list[0], call_record)

        with self.assertRaises(MsprobeException) as context:
            recursion_func(recursion_list, call_record)
        # 执行超过限制的递归函数会触发异常、且函数成功调用次数等于限制次数
        self.assertEqual(context.exception.code, MsprobeException.RECURSION_LIMIT_ERROR)
        mock_error.assert_called_with("call test func_info exceeds the recursion limit.")
        self.assertEqual(len(call_record), Const.MAX_DEPTH)

    def test_check_seed_all(self):
        with self.assertRaises(MsprobeException) as context:
            check_seed_all(-1, True, True)
        self.assertEqual(context.exception.code, MsprobeException.INVALID_PARAM_ERROR)
        with self.assertRaises(MsprobeException) as context:
            check_seed_all(Const.MAX_SEED_VALUE + 1, True, True)
        self.assertEqual(context.exception.code, MsprobeException.INVALID_PARAM_ERROR)
        with self.assertRaises(MsprobeException) as context:
            check_seed_all("1", True, True)
        self.assertEqual(context.exception.code, MsprobeException.INVALID_PARAM_ERROR)
        with self.assertRaises(MsprobeException) as context:
            check_seed_all(True, True, True)
        self.assertEqual(context.exception.code, MsprobeException.INVALID_PARAM_ERROR)
        with self.assertRaises(MsprobeException) as context:
            check_seed_all(True, 1, True)
        self.assertEqual(context.exception.code, MsprobeException.INVALID_PARAM_ERROR)
        with self.assertRaises(MsprobeException) as context:
            check_seed_all(1, True, "test")
        self.assertEqual(context.exception.code, MsprobeException.INVALID_PARAM_ERROR)

    def test_safe_get_value_dict_valid_key_index(self):
        # Test valid key and index in a dictionary
        dict_container = {'a': [1, 2, 3], 'b': [4, 5, 6]}
        self.assertEqual(safe_get_value(dict_container, 1, 'dict_container', key='a'), 2)

    def test_safe_get_value_invalid_key(self):
        # Test invalid key in dictionary
        dict_container = {'a': [1, 2, 3], 'b': [4, 5, 6]}
        with self.assertRaises(MsprobeBaseException) as context:
            safe_get_value(dict_container, 1, 'dict_container', key='invalid_key')
        self.assertEqual(context.exception.code, MsprobeBaseException.INVALID_OBJECT_TYPE_ERROR)

    def test_safe_get_value_valid_key_invalid_index(self):
        # Test invalid index in dictionary[key]
        dict_container = {'a': [1, 2, 3], 'b': [4, 5, 6]}
        with self.assertRaises(MsprobeBaseException) as context:
            safe_get_value(dict_container, 5, 'dict_container', key='a')
        self.assertEqual(context.exception.code, MsprobeBaseException.INDEX_OUT_OF_BOUNDS_ERROR)

    def test_safe_get_value_list_valid_index(self):
        # Test valid index in a list
        list_container = [10, 20, 30]
        self.assertEqual(safe_get_value(list_container, 1, 'list_container'), 20)

    def test_safe_get_value_list_index_out_of_bounds(self):
        # Test index out of bounds in a list
        list_container = [10, 20, 30]
        with self.assertRaises(MsprobeBaseException) as context:
            safe_get_value(list_container, 10, 'list_container')
        self.assertEqual(context.exception.code, MsprobeBaseException.INDEX_OUT_OF_BOUNDS_ERROR)

    def test_safe_get_value_tuple_valid_index(self):
        # Test valid index in a tuple
        tuple_container = (100, 200, 300)
        self.assertEqual(safe_get_value(tuple_container, 2, 'tuple_container'), 300)

    def test_safe_get_value_array_valid_index(self):
        # Test valid index in a numpy array
        array_container = np.array([1000, 2000, 3000])
        self.assertEqual(safe_get_value(array_container, 0, 'array_container'), 1000)

    def test_safe_get_value_unsupported_container_type(self):
        # Test unsupported container type (e.g., a string)
        with self.assertRaises(MsprobeBaseException) as context:
            safe_get_value("unsupported_type", 0, 'string_container')
        self.assertEqual(context.exception.code, MsprobeBaseException.INVALID_OBJECT_TYPE_ERROR)

    def test_valid_str_param(self):
        valid_param = "valid_string_without_special_chars"
        check_str_param(valid_param)

    def test_invalid_str_param(self):
        invalid_param = "invalid$tring&with^special*chars()"
        with self.assertRaises(MsprobeBaseException) as context:
            check_str_param(invalid_param)
        self.assertEqual(context.exception.code, MsprobeBaseException.INVALID_CHAR_ERROR)

    def test_is_json_file(self):
        file_path_true = 'step/rank/stack.json'
        file_path_false = 1
        self.assertTrue(is_json_file(file_path_true))
        self.assertFalse(is_json_file(file_path_false))


class TestDetectFrameworkByDumpJson(unittest.TestCase):

    @patch('msprobe.core.common.utils.load_json')
    def test_valid_pytorch_framework(self, mock_load_json):
        mock_load_json.return_value = {"framework": Const.PT_FRAMEWORK}

        result = detect_framework_by_dump_json("dummy_path")

        self.assertEqual(result, Const.PT_FRAMEWORK)

    @patch('msprobe.core.common.utils.load_json')
    def test_valid_mindspore_framework(self, mock_load_json):
        mock_load_json.return_value = {"framework": Const.MS_FRAMEWORK}

        result = detect_framework_by_dump_json("dummy_path")

        self.assertEqual(result, Const.MS_FRAMEWORK)

    def test_detect_framework_in_file(self):
        self.current_dir = Path(__file__).parent
        file_path = self.current_dir / "test_dump_file/pt_dump_no_framework.json"
        result = detect_framework_by_dump_json(file_path)
        self.assertEqual(result, Const.PT_FRAMEWORK)

        self.current_dir = Path(__file__).parent
        file_path = self.current_dir / "test_dump_file/ms_dump_no_framework.json"
        result = detect_framework_by_dump_json(file_path)
        self.assertEqual(result, Const.MS_FRAMEWORK)

    @patch("msprobe.core.common.utils.logger")
    def test_detect_framework_exception(self, mock_logger):
        self.current_dir = Path(__file__).parent
        file_path = self.current_dir / "test_dump_file/dump_no_pt_no_ms.json"
        with self.assertRaises(CompareException) as context:
            result = detect_framework_by_dump_json(file_path)
        self.assertEqual(context.exception.code, CompareException.INVALID_PARAM_ERROR)
        mock_logger.error.assert_called_once_with(f"{file_path} must be based on the MindSpore or PyTorch framework.")


class TestIsSaveVariableValid(unittest.TestCase):
    def setUp(self):
        self.valid_special_types = (int, float, str, bool)

    @patch.object(Const, "DUMP_MAX_DEPTH", 5)
    def test_is_save_variable_valid_DepthExceeded_ReturnsFalse(self):
        # 构造深度 = 阈值 + 1
        nested = [0] * 3
        for _ in range(Const.DUMP_MAX_DEPTH + 1):  # 注意 +1，确保“超过”阈值
            nested = [nested]
        self.assertFalse(is_save_variable_valid(nested, self.valid_special_types))

    def test_is_save_variable_valid_ValidSpecialTypes_ReturnsTrue(self):
        for valid_type in self.valid_special_types:
            self.assertTrue(is_save_variable_valid(valid_type(0), self.valid_special_types))

    def test_is_save_variable_valid_ListWithValidElements_ReturnsTrue(self):
        self.assertTrue(is_save_variable_valid([1, 2, 3], self.valid_special_types))

    def test_is_save_variable_valid_ListWithInvalidElement_ReturnsFalse(self):
        self.assertFalse(is_save_variable_valid([1, "test", [1, slice(1)]], self.valid_special_types))

    def test_is_save_variable_valid_TupleWithValidElements_ReturnsTrue(self):
        self.assertTrue(is_save_variable_valid((1, 2, 3), self.valid_special_types))

    def test_is_save_variable_valid_TupleWithInvalidElement_ReturnsFalse(self):
        self.assertFalse(is_save_variable_valid((1, "test", [1, slice(1)]), self.valid_special_types))

    def test_is_save_variable_valid_DictWithValidElements_ReturnsTrue(self):
        self.assertTrue(is_save_variable_valid({"a": 1, "b": "test"}, self.valid_special_types))

    def test_is_save_variable_valid_DictWithInvalidKey_ReturnsFalse(self):
        self.assertFalse(is_save_variable_valid({1: "test"}, self.valid_special_types))

    def test_is_save_variable_valid_DictWithInvalidValue_ReturnsFalse(self):
        self.assertFalse(is_save_variable_valid({"a": [1, slice(1)]}, self.valid_special_types))


class TestCheckDumpJsonKey(unittest.TestCase):
    def test_valid_input(self):
        json_data = {
            "task": "tensor",
            "data": {"api1": "value1"}
        }
        task, api_data = check_dump_json_key(json_data, "NPU")
        self.assertEqual(task, "tensor")
        self.assertEqual(api_data, {"api1": "value1"})

    @patch("msprobe.core.common.utils.logger")
    def test_missing_task(self, mock_logger):
        json_data = {
            "data": {"api1": "value1"}
        }
        with self.assertRaises(CompareException) as context:
            check_dump_json_key(json_data, "bench")
        self.assertEqual(context.exception.code, CompareException.INVALID_TASK_ERROR)
        mock_logger.error.assert_called_once_with(
            "Task for bench is empty, please check."
        )

    @patch("msprobe.core.common.utils.logger")
    def test_missing_data(self, mock_logger):
        json_data = {
            "task": "tensor"
        }
        with self.assertRaises(CompareException) as context:
            check_dump_json_key(json_data, "npu")
        self.assertEqual(context.exception.code, CompareException.INVALID_DATA_ERROR)
        mock_logger.error.assert_called_once_with(
            "Missing 'data' in dump.json, please check dump.json of npu."
        )

    @patch("msprobe.core.common.utils.logger")
    def test_wrong_data_type(self, mock_logger):
        json_data = {
            "task": "tensor",
            "data": [1]
        }
        with self.assertRaises(CompareException) as context:
            check_dump_json_key(json_data, "npu")
        self.assertEqual(context.exception.code, CompareException.INVALID_DATA_ERROR)
        mock_logger.error.assert_called_once_with(
            "Invalid type for 'data': expected a dict. Please check dump.json of npu."
        )


class TestTorchrecUtils(unittest.TestCase):
    """Tests for torchrec related utility functions."""

    def test_is_torchrec_available(self):
        result = is_torchrec_available()
        self.assertIsInstance(result, bool)

    def test_is_jagged_tensor_with_non_tensor_data(self):
        non_tensor_data = {"key": "value"}
        result = is_jagged_tensor(non_tensor_data)
        self.assertFalse(result)

    def test_is_keyed_jagged_tensor_with_non_tensor_data(self):
        non_tensor_data = [1, 2, 3]
        result = is_keyed_jagged_tensor(non_tensor_data)
        self.assertFalse(result)

    def test_is_keyed_tensor_with_non_tensor_data(self):
        non_tensor_data = "string_data"
        result = is_keyed_tensor(non_tensor_data)
        self.assertFalse(result)

    def test_is_jagged_tensor_with_valid_jagged_tensor(self):
        values = torch.tensor([1, 2, 3, 4, 5])
        lengths = torch.tensor([2, 3])
        jt = JaggedTensor(values=values, lengths=lengths)
        result = is_jagged_tensor(jt)
        self.assertTrue(result)

    def test_is_keyed_jagged_tensor_with_valid_keyed_jagged_tensor(self):
        kjt = KeyedJaggedTensor(
            keys=["featA", "featB"],
            values=torch.tensor([1, 2, 3, 4, 5]),
            lengths=torch.tensor([2, 3]),
            stride=1
        )
        result = is_keyed_jagged_tensor(kjt)
        self.assertTrue(result)

    def test_is_keyed_tensor_with_valid_keyed_tensor(self):
        user_dim = 2
        item_dim = 5
        combined_tensor = torch.randn([3, user_dim + item_dim])
        kt = KeyedTensor(
            keys=["user", "item"],
            values=combined_tensor,
            length_per_key=[user_dim, item_dim]
        )
        result = is_keyed_tensor(kt)
        self.assertTrue(result)

    @patch("msprobe.core.common.utils._has_torchrec", False)
    def test_is_jagged_tensor_when_torchrec_unavailable(self):
        values = torch.tensor([1, 2, 3, 4, 5])
        lengths = torch.tensor([3, 2])
        jt = JaggedTensor(values=values, lengths=lengths)
        result = is_jagged_tensor(jt)
        self.assertFalse(result)
