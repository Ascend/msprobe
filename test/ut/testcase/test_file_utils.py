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

import re
import time
import unittest
from unittest import mock
import struct
import os
import numpy as np
import pytest

from cmp_utils.constant.compare_error import CompareError
from cmp_utils import file_utils
from dump_parse import dump_utils
from dump_parse.proto_dump_data import DumpData, OpInput, OpOutput
from cmp_utils.constant.const_manager import DD

class TestUtilsMethods(unittest.TestCase):
    def test_load_json_file_case2(self):
        file_util = file_utils.FileUtils()
        with pytest.raises(CompareError) as err:
            with mock.patch("os.path.getsize", return_value=1000000000000000):
                with mock.patch('builtins.open', side_effect=None):
                    with mock.patch('json.load', side_effect=IOError):
                        file_util.load_json_file("/test")

        self.assertEqual(err.value.args[0],
                         CompareError.MSACCUCMP_PARSER_JSON_FILE_ERROR)

    def test_read_csv_case1(self):
        file_util = file_utils.FileUtils()
        ret = file_util.read_csv("/homt/test.bin")
        self.assertEqual(ret, [])

    def test_read_csv_case2(self):
        file_util = file_utils.FileUtils()
        with pytest.raises(CompareError) as err:
            with mock.patch('builtins.open', side_effect=IOError):
                ret = file_util.read_csv("/home/test.csv")
        self.assertEqual(err.value.args[0],
                         CompareError.MSACCUCMP_OPEN_FILE_ERROR)

    def test_read_csv_case3(self):
        file_util = file_utils.FileUtils()
        output_content = ['word1 end!',
                          'word2 end!']
        with mock.patch('builtins.open', side_effect=None), \
             mock.patch('os.path.getsize', return_value=1024):
            with mock.patch('csv.reader', return_value=output_content):
                ret = file_util.read_csv("/home/test.csv")
        self.assertEqual(ret, ['word1 end!', 'word2 end!'])

    def test_save_file_case1(self):
        file_util = file_utils.FileUtils()
        with pytest.raises(CompareError) as err:
            with mock.patch('os.open', side_effect=IOError) as open_file, \
                    mock.patch('os.fdopen'):
                open_file.write = None
                file_util.save_file("/test", "result")
        self.assertEqual(err.value.args[0],
                         CompareError.MSACCUCMP_WRITE_FILE_ERROR)

    def test_save_file_case2(self):
        file_util = file_utils.FileUtils()
        with mock.patch('os.open', side_effect=None) as open_file, \
                mock.patch('os.fdopen'):
            open_file.write = None
            file_util.save_file("/test", "result")

    def test_save_array_to_file_with_write_then_success(self):
        file_util = file_utils.FileUtils()
        test_array = np.array([[1, 2], [3, 4]])
        test_path = 'test_save_array_to_file.npy'
        file_util.save_array_to_file(test_path, test_array, np_save=True, shape=None)
        self.assertTrue(os.path.exists(test_path))
        loaded_array = np.load(test_path)
        self.assertTrue(np.array_equal(test_array, loaded_array))

    def test_list_file_with_pattern_case1(self):
        overflow_file_util = file_utils.OverflowFileUtils()
        with pytest.raises(CompareError) as err:
            with mock.patch("os.path.exists", return_value=False):
                overflow_file_util._list_file_with_pattern("/home", '', '', None)
        self.assertEqual(err.value.args[0],
                         CompareError.MSACCUCMP_INVALID_PATH_ERROR)

    def test_list_file_with_pattern_case2(self):
        def gen_info_func(name, dir_path, match):
            return name

        overflow_file_util = file_utils.OverflowFileUtils()
        files = (item for item in [["path_root", "folders", ["Opdebug.Node_OpDebug.1.1234567891234567", "test2"]],
                                   ["path_root1", "folders1", ["test3", "test4"]]])
        with mock.patch("os.path.exists", return_value=True):
            with mock.patch("os.walk", return_value=files):
                ret = overflow_file_util.\
                    _list_file_with_pattern("/home",
                                            overflow_file_util.DUMP_FILE_PATTERN,
                                            '',
                                            gen_info_func)
        self.assertEqual(ret, {'Opdebug.Node_OpDebug.1.1234567891234567'
                               :'Opdebug.Node_OpDebug.1.1234567891234567'})

    def test_list_file_with_pattern_case3(self):
        overflow_file_util = file_utils.OverflowFileUtils()
        files = (item for item in [["path_root", "folders", ["Opdebug.Node_OpDebug.1.1234567891234567", "test2"]],
                                   ["path_root1", "folders1", ["test3", "test4"]]])
        with mock.patch("os.path.exists", return_value=True):
            with mock.patch("os.walk", return_value=files):
                ret = overflow_file_util.\
                    _list_file_with_pattern("/home",
                                            overflow_file_util.DUMP_FILE_PATTERN,
                                            r'extern_pattern',
                                            None)
        self.assertEqual(ret, {})

    def test_list_file_with_pattern_case4(self):
        overflow_file_util = file_utils.OverflowFileUtils()
        files = (item for item in [["path_root", "folders", ["Node_OpDebug.1.1234567891234567", "test2"]],
                                   ["path_root1", "folders1", ["test3", "test4"]]])
        with mock.patch("os.path.exists", return_value=True):
            with mock.patch("os.walk", return_value=files):
                ret = overflow_file_util.\
                    _list_file_with_pattern("/home",
                                            overflow_file_util.DUMP_FILE_PATTERN,
                                            r'extern_pattern',
                                            None)
        self.assertEqual(ret, {})


    def test_parse_mapping_csv_case1(self):
        overflow_file_util = file_utils.OverflowFileUtils()
        mapping_file_content = (item for item in [['123124178524', 'Opdebug.Node_OpDebug.1.25.1614717262233160'],
                                                  ['241314144178', 'Node_OpDebug.1.1234567afa891234567']])

        with mock.patch("os.listdir", return_value='mapping.csv'):
            with mock.patch("cmp_utils.file_utils.OverflowFileUtils.read_csv",
                            return_value=mapping_file_content):
                with mock.patch("os.path.isfile", return_value=True):
                    ret = overflow_file_util.\
                        parse_mapping_csv("/home",
                                          overflow_file_util.DUMP_FILE_PATTERN)

        self.assertEqual(len(ret), 1)
        DumpFileDescObj = ret['123124178524']
        self.assertEqual(DumpFileDescObj.op_type,'Opdebug')
        self.assertEqual(DumpFileDescObj.op_name, 'Node_OpDebug')
        self.assertEqual(DumpFileDescObj.task_id, 1)
        self.assertEqual(DumpFileDescObj.stream_id, 25)
        self.assertEqual(DumpFileDescObj.timestamp, 1614717262233160)
        self.assertEqual(DumpFileDescObj.file_path, '/home/123124178524')

    def test_parse_mapping_csv_case2(self):
        overflow_file_util = file_utils.OverflowFileUtils()
        mapping_file_content = (item for item in [['123128524', 'Opdebug.Node_OpDebug.1.25.161233160'],
                                                  ['241314178', 'Node_OpDebug.1.1234567a1234567']])
        files = (item for item in [["path_root", "folder", ["mapping.csv", "test2"]],
                                   ["path_root1", "folder1", ["test3", "test4"]]])

        with mock.patch("os.walk", return_value=files):
            with mock.patch("cmp_utils.file_utils.OverflowFileUtils.read_csv",
                            return_value=mapping_file_content):
                with mock.patch("os.path.isfile", return_value=False):
                    ret = overflow_file_util.\
                        parse_mapping_csv("/home", overflow_file_util.DUMP_FILE_PATTERN, '')
        self.assertEqual(ret, {})

    def test_list_dump_files_case(self):
        file_desc = {
            "file_path": "/test/Opdebug",
            "timestamp": int("161233160")
        }
        dump_attr = {
            "op_name": 'Node_OpDebug',
            "op_type": 'Opdebug',
            "task_id": int('11'),
            "stream_id": '25'
        }
        overflow_file_util = file_utils.OverflowFileUtils()
        DumpFileDescObj = file_utils.DumpFileDesc(file_desc, dump_attr)
        with mock.patch("cmp_utils.file_utils.OverflowFileUtils._list_file_with_pattern",
                        return_value = {}):
            with mock.patch("cmp_utils.file_utils.OverflowFileUtils.parse_mapping_csv",
                            return_value={'241314178': DumpFileDescObj}):
                ret = overflow_file_util. \
                    list_dump_files("/home", overflow_file_util.DUMP_FILE_PATTERN)
        self.assertEqual(ret,  [DumpFileDescObj])

    def test_list_parsed_dump_files_case1(self):
        overflow_file_util = file_utils.OverflowFileUtils()
        dump_data = DumpData()
        dump_data.version = '1.0'
        dump_data.op_name = 'Node_OpDebug'
        dump_data.dump_time = int(round(time.time() * 1000))
        op_output = dump_data.output.add()
        op_output.data_type = DD.DT_FLOAT16
        op_output.format = DD.FORMAT_NCHW
        length = 1
        for dim in [1, 3, 4, 4]:
            op_output.shape.dim.append(dim)
            length *= dim
        data_list = np.arange(length)
        origin_numpy = np.array(data_list, np.float16)
        dump_data = dump_utils.convert_dump_data(dump_data)
        file_desc = {
            "file_path": "/home/Opdebug.Node_OpDebug.1.25.161233160",
            "timestamp": int("161233160")
        }
        dump_attr = {
            "op_name": 'Node_OpDebug',
            "op_type": 'Opdebug',
            "task_id": int('1'),
            "stream_id": '25'
        }
        dump_file_obj = file_utils.DumpFileDesc(file_desc, dump_attr)

        op_output.data = struct.pack('%de' % length, *origin_numpy)
        with mock.patch('dump_parse.dump_utils.parse_dump_file',
                        return_value=dump_data):
            with mock.patch('os.path.basename', return_value='Opdebug.Node_OpDebug.1.25.161233160'):
                parsed_dump_file_obj = overflow_file_util.list_parsed_dump_files('/home', dump_file_obj)

        parsed_dump_desc = list(parsed_dump_file_obj.values())[0]
        self.assertEqual(parsed_dump_desc.op_type, 'Opdebug')
        self.assertEqual(parsed_dump_desc.op_name, 'Node_OpDebug')
        self.assertEqual(parsed_dump_desc.task_id, 1)
        self.assertEqual(parsed_dump_desc.stream_id, 25)
        self.assertEqual(parsed_dump_desc.timestamp, 161233160)
        self.assertEqual(parsed_dump_desc.file_path, '/home/Opdebug.Node_OpDebug.1.25.161233160.output.0.npy')
        self.assertEqual(parsed_dump_desc.type, 'output')
        self.assertEqual(parsed_dump_desc.idx, 0)
        self.assertEqual(parsed_dump_desc.format, 'NCHW')

    def test_gen_parsed_debug_file_info(self):
        overflow_file_util = file_utils.OverflowFileUtils()
        pattern = overflow_file_util.PARSED_DEBUG_FILE_PATTERN
        re_pattern = re.compile(pattern)
        debugfile_name = 'Opdebug.Node_OpDebug.1.25.161233160.output.1.json'
        match = re_pattern.match(debugfile_name)
        DumpFileDescObj = overflow_file_util._gen_parsed_debug_file_info(debugfile_name, '/test', match)

        self.assertEqual(DumpFileDescObj.op_type,'Opdebug')
        self.assertEqual(DumpFileDescObj.op_name, 'Node_OpDebug')
        self.assertEqual(DumpFileDescObj.task_id, 1)
        self.assertEqual(DumpFileDescObj.stream_id, 25)
        self.assertEqual(DumpFileDescObj.timestamp, 161233160)
        self.assertEqual(DumpFileDescObj.file_path, '/test/Opdebug.Node_OpDebug.1.25.161233160.output.1.json')
        self.assertEqual(DumpFileDescObj.type, 'output')
        self.assertEqual(DumpFileDescObj.idx, 1)

    def test_list_parsed_debug_files(self):
        overflow_file_util = file_utils.OverflowFileUtils()
        with mock.patch("cmp_utils.file_utils.OverflowFileUtils.list_parsed_debug_files",
                        return_value={}):
            ret = overflow_file_util. \
                list_parsed_debug_files("/home", '')
        self.assertEqual(ret, {})

    def test_FileDesc_func(self):
        file_desc = {
            "file_path": "/home/test_file",
            "timestamp": 12414512452
        }
        file_desc_cls_obj = file_utils.FileDesc(file_desc)

        ret = file_desc_cls_obj.get_file_path()
        self.assertEqual(ret, "/home/test_file")

        ret = file_desc_cls_obj.get_file_time()
        self.assertEqual(ret, 12414512452)
