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
import shutil
import threading
import unittest
from unittest.mock import patch, MagicMock

from msprobe.infer.offline.compare.msquickcmp.common.utils import (
    check_exec_script_file,
    check_file_or_directory_path,
    check_input_bin_file_path,
    check_file_size_valid,
    check_input_args,
    check_convert_is_valid_used,
    check_locat_is_valid,
    check_device_param_valid,
    AccuracyCompareException,
    ACCURACY_COMPARISON_INVALID_PATH_ERROR,
    ACCURACY_COMPARISON_INVALID_RIGHT_ERROR,
    ACCURACY_COMPARISON_INVALID_DATA_ERROR,
    INVALID_CHARS,
    MAX_DEVICE_ID,
    _check_colon_exist,
    _check_content_split_length,
    _check_shape_number,
    check_max_size_param_valid,
    get_input_path,
    InputShapeError,
    get_shape_to_directory_name,
    get_dump_data_path, get_batch_index,
    DynamicArgumentEnum,
    BATCH_SCENARIO_OP_NAME,
    get_mbatch_op_name,
    get_batch_index_from_name,
    get_data_len_by_shape,
    load_npy_from_buffer,
    find_om_files
)


class TestCheckFunctions1(unittest.TestCase):

    @patch('msprobe.infer.offline.compare.msquickcmp.common.utils.logger')
    @patch('msprobe.infer.offline.compare.msquickcmp.common.utils.os.path.exists')
    @patch('msprobe.infer.offline.compare.msquickcmp.common.utils.os.access')
    def test_check_exec_script_file_not_exist(self, mock_access, mock_exists, mock_logger):
        mock_exists.return_value = False
        with self.assertRaises(AccuracyCompareException) as cm:
            check_exec_script_file('fake_script.sh')
        self.assertEqual(cm.exception.error_info, ACCURACY_COMPARISON_INVALID_PATH_ERROR)
        mock_logger.error.assert_called_once()

    @patch('msprobe.infer.offline.compare.msquickcmp.common.utils.logger')
    @patch('msprobe.infer.offline.compare.msquickcmp.common.utils.os.path.exists')
    @patch('msprobe.infer.offline.compare.msquickcmp.common.utils.os.access')
    def test_check_exec_script_file_no_execute_right(self, mock_access, mock_exists, mock_logger):
        mock_exists.return_value = True
        mock_access.return_value = False
        with self.assertRaises(AccuracyCompareException) as cm:
            check_exec_script_file('fake_script.sh')
        self.assertEqual(cm.exception.error_info, ACCURACY_COMPARISON_INVALID_RIGHT_ERROR)
        mock_logger.error.assert_called_once()

    @patch('msprobe.infer.offline.compare.msquickcmp.common.utils.logger')
    @patch('msprobe.infer.offline.compare.msquickcmp.common.utils.os.path.isdir')
    @patch('msprobe.infer.offline.compare.msquickcmp.common.utils.os.access')
    def test_check_file_or_directory_path_dir_invalid(self, mock_access, mock_isdir, mock_logger):
        mock_isdir.return_value = False
        with self.assertRaises(AccuracyCompareException):
            check_file_or_directory_path('/some/path', isdir=True)
        mock_logger.error.assert_called_once()

    @patch('msprobe.infer.offline.compare.msquickcmp.common.utils.logger')
    @patch('msprobe.infer.offline.compare.msquickcmp.common.utils.os.path.isdir')
    @patch('msprobe.infer.offline.compare.msquickcmp.common.utils.os.access')
    def test_check_file_or_directory_path_dir_no_write(self, mock_access, mock_isdir, mock_logger):
        mock_isdir.return_value = True
        mock_access.return_value = False
        with self.assertRaises(AccuracyCompareException):
            check_file_or_directory_path('/some/path', isdir=True)
        mock_logger.error.assert_called_once()

    @patch('msprobe.infer.offline.compare.msquickcmp.common.utils.logger')
    @patch('msprobe.infer.offline.compare.msquickcmp.common.utils.os.path.isfile')
    @patch('msprobe.infer.offline.compare.msquickcmp.common.utils.os.access')
    def test_check_file_or_directory_path_file_invalid(self, mock_access, mock_isfile, mock_logger):
        mock_isfile.return_value = False
        with self.assertRaises(AccuracyCompareException):
            check_file_or_directory_path('/some/file', isdir=False)
        mock_logger.error.assert_called_once()

    @patch('msprobe.infer.offline.compare.msquickcmp.common.utils.logger')
    @patch('msprobe.infer.offline.compare.msquickcmp.common.utils.os.path.isfile')
    @patch('msprobe.infer.offline.compare.msquickcmp.common.utils.os.access')
    def test_check_file_or_directory_path_file_no_read(self, mock_access, mock_isfile, mock_logger):
        mock_isfile.return_value = True
        mock_access.return_value = False
        with self.assertRaises(AccuracyCompareException):
            check_file_or_directory_path('/some/file', isdir=False)
        mock_logger.error.assert_called_once()

    @patch('msprobe.infer.offline.compare.msquickcmp.common.utils.check_file_or_directory_path')
    @patch('msprobe.infer.offline.compare.msquickcmp.common.utils.get_input_path')
    @patch('msprobe.infer.offline.compare.msquickcmp.common.utils.os.path.realpath')
    def test_check_input_bin_file_path_bin_and_dir(self, mock_realpath, mock_get_input_path, mock_check_path):
        def side_effect_realpath(arg):
            return arg

        mock_realpath.side_effect = side_effect_realpath

        mock_check_path.return_value = None

        mock_get_input_path.side_effect = lambda path, arr: arr.append(path + "/dummy.bin")

        input_str = "/path/file1.bin,/path/dir1"
        result = check_input_bin_file_path(input_str)
        self.assertIn("/path/file1.bin", result)
        self.assertTrue(any(p.endswith("dummy.bin") for p in result))

    @patch('msprobe.infer.offline.compare.msquickcmp.common.utils.logger')
    @patch('msprobe.infer.offline.compare.msquickcmp.common.utils.os.stat')
    def test_check_file_size_valid_too_large(self, mock_stat, mock_logger):
        mock_stat.return_value.st_size = 1024
        size_max = 100
        with self.assertRaises(AccuracyCompareException) as cm:
            check_file_size_valid('somefile', size_max)
        self.assertEqual(cm.exception.error_info, ACCURACY_COMPARISON_INVALID_DATA_ERROR)
        mock_logger.error.assert_called_once()

    @patch('msprobe.infer.offline.compare.msquickcmp.common.utils.logger')
    def test_check_input_args_invalid_char(self, mock_logger):
        with self.assertRaises(AccuracyCompareException):
            check_input_args(list(INVALID_CHARS))
        mock_logger.error.assert_called_once()

    @patch('msprobe.infer.offline.compare.msquickcmp.common.utils.logger')
    def test_check_convert_is_valid_used_invalid(self, mock_logger):
        with self.assertRaises(AccuracyCompareException):
            check_convert_is_valid_used(False, True, "")
        mock_logger.error.assert_called_once()

        with self.assertRaises(AccuracyCompareException):
            check_convert_is_valid_used(False, False, "custom")
        mock_logger.error.assert_called()

    @patch('msprobe.infer.offline.compare.msquickcmp.common.utils.logger')
    def test_check_locat_is_valid_invalid(self, mock_logger):
        with self.assertRaises(AccuracyCompareException):
            check_locat_is_valid(False, True)
        mock_logger.error.assert_called_once()

    @patch('msprobe.infer.offline.compare.msquickcmp.common.utils.logger')
    def test_check_device_param_valid_invalid(self, mock_logger):
        with self.assertRaises(AccuracyCompareException):
            check_device_param_valid("abc")
        mock_logger.error.assert_called_once()

        with self.assertRaises(AccuracyCompareException):
            check_device_param_valid(str(MAX_DEVICE_ID + 1))
        mock_logger.error.assert_called()


class TestCheckFunctions2(unittest.TestCase):

    @patch('msprobe.infer.offline.compare.msquickcmp.common.utils.logger')
    @patch('msprobe.infer.offline.compare.msquickcmp.common.utils.get_shape_not_match_message')
    def test_check_colon_exist_no_colon(self, mock_get_msg, mock_logger):
        mock_get_msg.return_value = "error msg"
        with self.assertRaises(AccuracyCompareException):
            _check_colon_exist("abc")

        mock_logger.error.assert_called_once_with("error msg")
        mock_get_msg.assert_called_once_with(InputShapeError.FORMAT_NOT_MATCH, "abc")

    def test_check_colon_exist_with_colon(self):
        _check_colon_exist("a:b")

    @patch('msprobe.infer.offline.compare.msquickcmp.common.utils.logger')
    @patch('msprobe.infer.offline.compare.msquickcmp.common.utils.get_shape_not_match_message')
    def test_check_content_split_length_empty_second(self, mock_get_msg, mock_logger):
        mock_get_msg.return_value = "error msg"
        with self.assertRaises(AccuracyCompareException):
            _check_content_split_length(["something", ""])
        mock_logger.error.assert_called_once_with("error msg")
        mock_get_msg.assert_called_once_with(InputShapeError.VALUE_TYPE_NOT_MATCH, "")

    def test_check_content_split_length_non_empty(self):
        _check_content_split_length(["foo", "bar"])

    @patch('msprobe.infer.offline.compare.msquickcmp.common.utils.logger')
    @patch('msprobe.infer.offline.compare.msquickcmp.common.utils.get_shape_not_match_message')
    def test_check_shape_number_invalid(self, mock_get_msg, mock_logger):
        mock_get_msg.return_value = "error msg"
        with self.assertRaises(AccuracyCompareException):
            _check_shape_number("invalid_shape")

        mock_logger.error.assert_called_once_with("error msg")
        mock_get_msg.assert_called_once_with(InputShapeError.VALUE_TYPE_NOT_MATCH, "invalid_shape")

    def test_check_shape_number_valid(self):
        valid_shape = "123"
        with patch('msprobe.infer.offline.compare.msquickcmp.common.utils.DIM_PATTERN', r'^\d+$'):
            _check_shape_number(valid_shape)

    @patch('msprobe.infer.offline.compare.msquickcmp.common.utils.logger')
    def test_check_max_size_param_valid_negative(self, mock_logger):
        with self.assertRaises(AccuracyCompareException):
            check_max_size_param_valid(-1)

        mock_logger.error.assert_called_once()
        err_msg = mock_logger.error.call_args[0][0]
        self.assertIn("max_cmp_size", err_msg)

    def test_check_max_size_param_valid_zero_and_positive(self):
        check_max_size_param_valid(0)
        check_max_size_param_valid(100)

    @patch('msprobe.infer.offline.compare.msquickcmp.common.utils.os.walk')
    def test_get_input_path(self, mock_walk):
        mock_walk.return_value = [
            ('/root', ('subdir',), ('file1.bin', 'file2.txt')),
            ('/root/subdir', (), ('file3.bin',)),
        ]

        bin_file_path_array = []
        get_input_path('/root', bin_file_path_array)

        expected_paths = ['/root/file1.bin', '/root/subdir/file3.bin']
        self.assertEqual(bin_file_path_array, expected_paths)


class TestYourFunctions(unittest.TestCase):
    @patch("msprobe.infer.offline.compare.msquickcmp.common.utils.os")
    @patch("msprobe.infer.offline.compare.msquickcmp.common.utils.shutil")
    @patch("msprobe.infer.offline.compare.msquickcmp.common.utils.logger")
    def test_get_dump_data_path_is_net_output_true(self, mock_logger, mock_shutil, mock_os):
        mock_os.listdir.return_value = ['12_423_246_4352', 'abc']
        mock_os.path.isdir.side_effect = lambda x: '12_423_246_4352' in x or 'abc' in x
        mock_os.path.join.side_effect = lambda a, b: f"{a}/{b}"
        mock_os.walk.return_value = [
            ('/dump/abc', [], ['file1', 'file2']),
        ]

        dump_data_path, file_is_exist = get_dump_data_path("/dump", is_net_output=True)
        self.assertEqual(dump_data_path, "/dump/abc")
        self.assertTrue(file_is_exist)

    @patch("msprobe.infer.offline.compare.msquickcmp.common.utils.os")
    @patch("msprobe.infer.offline.compare.msquickcmp.common.utils.shutil")
    @patch("msprobe.infer.offline.compare.msquickcmp.common.utils.logger")
    def test_get_dump_data_path_is_net_output_false(self, mock_logger, mock_shutil, mock_os):
        mock_os.path.isdir.side_effect = lambda x: '123456' in x or 'xyz' in x
        mock_os.path.join.side_effect = lambda a, b: f"{a}/{b}"

        def listdir_side_effect(path):
            if path == "/dump":
                return ['123456', 'xyz']
            elif 'modelB' in path:
                return ['fileB1']
            elif 'modelA' in path:
                return ['fileA1']
            return []

        mock_os.listdir.side_effect = listdir_side_effect
        mock_os.walk.return_value = [
            ('/dump/123456/modelA', [], ['fileA1']),
            ('/dump/123456/modelB', [], ['fileB1']),
        ]
        mock_shutil.move = MagicMock()
        dump_data_path, file_is_exist = get_dump_data_path("/dump", is_net_output=False, model_name="modelB")
        self.assertEqual(dump_data_path, "/dump/123456/modelB")
        self.assertTrue(file_is_exist)
        mock_shutil.move.assert_called()

    @patch("msprobe.infer.offline.compare.msquickcmp.common.utils.os")
    @patch("msprobe.infer.offline.compare.msquickcmp.common.utils.logger")
    def test_get_dump_data_path_no_dump_data_dir(self, mock_logger, mock_os):
        mock_os.listdir.return_value = ['abc', 'xyz']
        mock_os.path.isdir.return_value = True
        mock_os.path.join.side_effect = lambda a, b: f"{a}/{b}"

        with self.assertRaises(AccuracyCompareException):
            get_dump_data_path("/dump", is_net_output=False)

    def test_get_shape_to_directory_name(self):
        input_shape = "1:3,5;7"
        expected = "1-3_5-7"
        self.assertEqual(get_shape_to_directory_name(input_shape), expected)

    @patch("msprobe.infer.offline.compare.msquickcmp.common.utils.os")
    @patch("msprobe.infer.offline.compare.msquickcmp.common.utils.get_batch_index_from_name")
    def test_get_batch_index_found(self, mock_get_batch_index_from_name, mock_os):
        mock_os.walk.return_value = [
            ("/dump/path", [], ["ascend_mbatch_batch_123", "otherfile"])
        ]
        mock_get_batch_index_from_name.return_value = "123"
        result = get_batch_index("/dump/path")
        self.assertEqual(result, "123")

    @patch("msprobe.infer.offline.compare.msquickcmp.common.utils.os")
    def test_get_batch_index_not_found(self, mock_os):
        mock_os.walk.return_value = [
            ("/dump/path", [], ["file1", "file2"])
        ]
        result = get_batch_index("/dump/path")
        self.assertEqual(result, "")

    @patch("msprobe.infer.offline.compare.msquickcmp.common.utils.get_batch_index")
    def test_get_mbatch_op_name_batch_scenario(self, mock_get_batch_index):
        mock_get_batch_index.return_value = "10"
        om_parser = MagicMock()
        om_parser.get_dynamic_scenario_info.return_value = ("", DynamicArgumentEnum.DYM_BATCH)
        op_name = "op"
        npu_dump_data_path = "/path"
        expected = BATCH_SCENARIO_OP_NAME.format(op_name, "10")
        result = get_mbatch_op_name(om_parser, op_name, npu_dump_data_path)
        self.assertEqual(result, expected)

    def test_get_mbatch_op_name_other_scenario(self):
        om_parser = MagicMock()
        om_parser.get_dynamic_scenario_info.return_value = ("", "OTHER_SCENARIO")
        op_name = "op"
        npu_dump_data_path = "/path"
        result = get_mbatch_op_name(om_parser, op_name, npu_dump_data_path)
        self.assertEqual(result, op_name)

    def test_get_batch_index_from_name(self):
        name = "ascend_mbatch_batch_1234xyz"
        result = get_batch_index_from_name(name)
        self.assertEqual(result, "1234")

    @patch("msprobe.infer.offline.compare.msquickcmp.common.utils.logger")
    def test_get_data_len_by_shape_valid(self, mock_logger):
        shape = [2, 3, 4]
        result = get_data_len_by_shape(shape)
        self.assertEqual(result, 24)

    @patch("msprobe.infer.offline.compare.msquickcmp.common.utils.logger")
    def test_get_data_len_by_shape_invalid(self, mock_logger):
        shape = [2, -1, 4]
        result = get_data_len_by_shape(shape)
        self.assertEqual(result, -1)
        mock_logger.warning.assert_called_once_with("please check your input shape, one dim in shape is -1.")


class TestParseInputShapeToList(unittest.TestCase):
    @patch('msprobe.infer.offline.compare.msquickcmp.common.utils._check_colon_exist')
    @patch('msprobe.infer.offline.compare.msquickcmp.common.utils.dym_shape_range_interaction', return_value=True)
    def test_valid_shape(self, mock_interact, mock_check):
        from msprobe.infer.offline.compare.msquickcmp.common.utils import parse_input_shape_to_list
        input_shape = "tensor1:1,2;tensor2:3,4"
        result, _ = parse_input_shape_to_list(input_shape)
        self.assertEqual(result, [[1, 2], [3, 4]])

    @patch('msprobe.infer.offline.compare.msquickcmp.common.utils._check_colon_exist')
    def test_invalid_shape_format(self, mock_check):
        from msprobe.infer.offline.compare.msquickcmp.common.utils import parse_input_shape_to_list, \
            AccuracyCompareException
        with patch('msprobe.infer.offline.compare.msquickcmp.common.utils.logger.error'), \
                patch('msprobe.infer.offline.compare.msquickcmp.common.utils.get_shape_not_match_message',
                      return_value="error"):
            with self.assertRaises(AccuracyCompareException):
                parse_input_shape_to_list("tensor1-1,2")

    @patch('msprobe.infer.offline.compare.msquickcmp.common.utils._check_colon_exist')
    def test_dim_not_digit(self, mock_check):
        from msprobe.infer.offline.compare.msquickcmp.common.utils import parse_input_shape_to_list
        with self.assertRaises(ValueError):
            parse_input_shape_to_list("tensor1:1,a")


class TestDymShapeRangeInteraction(unittest.TestCase):
    @patch('builtins.input', return_value='y')
    def test_yes_input(self, mock_input):
        from msprobe.infer.offline.compare.msquickcmp.common.utils import dym_shape_range_interaction
        self.assertTrue(dym_shape_range_interaction("prompt"))

    @patch('builtins.input', side_effect=Exception("Input error"))
    def test_input_exception(self, mock_input):
        from msprobe.infer.offline.compare.msquickcmp.common.utils import dym_shape_range_interaction
        self.assertFalse(dym_shape_range_interaction("prompt"))


class TestParseDymShapeRange(unittest.TestCase):
    @patch('msprobe.infer.offline.compare.msquickcmp.common.utils._check_colon_exist')
    @patch('msprobe.infer.offline.compare.msquickcmp.common.utils._check_shape_number')
    @patch('msprobe.infer.offline.compare.msquickcmp.common.utils._check_content_split_length')
    @patch('msprobe.infer.offline.compare.msquickcmp.common.utils.dym_shape_range_interaction', return_value=True)
    @patch('msprobe.infer.offline.compare.msquickcmp.common.utils.logger.info')
    def test_parse_valid_range(self, mock_log, mock_interact, mock_len, mock_shape, mock_colon):
        from msprobe.infer.offline.compare.msquickcmp.common.utils import parse_dym_shape_range
        result = parse_dym_shape_range("tensor1:1,2~3")
        self.assertIsInstance(result, list)


class TestParseArgValue(unittest.TestCase):
    @patch('msprobe.infer.offline.compare.msquickcmp.common.utils.parse_value_by_comma')
    def test_parse(self, mock_parse):
        from msprobe.infer.offline.compare.msquickcmp.common.utils import parse_arg_value
        mock_parse.return_value = [1, 2]
        result = parse_arg_value("1,2;3,4")
        self.assertEqual(result, [[1, 2], [1, 2]])


class TestParseValueByComma(unittest.TestCase):
    def test_valid_values(self):
        from msprobe.infer.offline.compare.msquickcmp.common.utils import parse_value_by_comma
        result = parse_value_by_comma("1,2,3")
        self.assertEqual(result, [1, 2, 3])

    def test_invalid_values(self):
        from msprobe.infer.offline.compare.msquickcmp.common.utils import parse_value_by_comma, AccuracyCompareException
        with patch('msprobe.infer.offline.compare.msquickcmp.common.utils.logger.error'):
            with self.assertRaises(AccuracyCompareException):
                parse_value_by_comma("1,a")


class TestExecuteCommand(unittest.TestCase):
    @patch('msprobe.infer.offline.compare.msquickcmp.common.utils.subprocess.Popen')
    def test_command_success(self, mock_popen):
        from msprobe.infer.offline.compare.msquickcmp.common.utils import execute_command
        mock_proc = MagicMock()
        mock_proc.poll.side_effect = [None, 0]
        mock_proc.stdout.readline.side_effect = [b'output\n', b'']
        mock_proc.returncode = 0
        mock_popen.return_value = mock_proc
        with patch('msprobe.infer.offline.compare.msquickcmp.common.utils.logger.info'):
            execute_command(['echo', 'test'])


class TestHandleGroundTruthFiles(unittest.TestCase):
    @patch('msprobe.infer.offline.compare.msquickcmp.common.utils.get_batch_index', return_value=1)
    @patch('msprobe.infer.offline.compare.msquickcmp.common.utils.shutil.copy')
    @patch('msprobe.infer.offline.compare.msquickcmp.common.utils.os.walk')
    def test_handle(self, mock_walk, mock_copy, mock_batch):
        from msprobe.infer.offline.compare.msquickcmp.common.utils import handle_ground_truth_files
        parser_mock = MagicMock()
        parser_mock.get_dynamic_scenario_info.return_value = (None, 'DYM_BATCH')
        mock_walk.return_value = [('/root', [], ['x.bin'])]
        with patch('msprobe.infer.offline.compare.msquickcmp.common.utils.BATCH_SCENARIO_OP_NAME', '{}_{}'):
            handle_ground_truth_files(parser_mock, '/npu', '/golden')


class TestLoadNpyFromBuffer(unittest.TestCase):
    @patch("numpy.frombuffer")
    def test_load_npy_success(self, mock_frombuffer):
        fake_data = b'1234'
        mock_array = MagicMock()
        mock_array.reshape.return_value = "reshaped_array"
        mock_frombuffer.return_value = mock_array
        result = load_npy_from_buffer(fake_data, dtype='int8', shape=(2, 2))
        mock_frombuffer.assert_called_once_with(fake_data, dtype='int8')
        mock_array.reshape.assert_called_once_with((2, 2))
        self.assertEqual(result, "reshaped_array")

    @patch("numpy.frombuffer", side_effect=ValueError("Invalid buffer"))
    def test_load_npy_exception(self, mock_frombuffer):
        result = load_npy_from_buffer(b'invalid', dtype='float32', shape=(3, 3))
        self.assertIsNone(result)


base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_utils_msquickcmp_find_om_files')


class TestFindOmFiles(unittest.TestCase):
    def setUp(self):
        os.makedirs(base_dir, mode=0o750, exist_ok=True)
        self.lock = threading.Lock()

    def tearDown(self):
        if os.path.exists(base_dir):
            shutil.rmtree(base_dir)

    def test_find_om_files(self):
        file1 = os.path.join(base_dir, 'test1.om')
        file2 = os.path.join(base_dir, 'test2.txt')
        file3 = os.path.join(base_dir, 'test3.om')
        with open(file1, 'w') as f:
            f.write('1')
        with open(file2, 'w') as f:
            f.write('21')
        with open(file3, 'w') as f:
            f.write('3')

        result = find_om_files(base_dir)

        self.assertEqual(len(result), 2)
