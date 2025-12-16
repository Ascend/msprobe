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

import time

import unittest
import struct
import pytest
import numpy as np
from unittest import mock
from google.protobuf.message import DecodeError

from dump_parse.proto_dump_data import DumpData, OpBuffer
from dump_parse import big_dump_data
from dump_parse.big_dump_data import BigDumpDataParser
from cmp_utils.constant.compare_error import CompareError


class TestUtilsMethods(unittest.TestCase):

    def test_parse1(self):
        with pytest.raises(CompareError) as error:
            with mock.patch('cmp_utils.path_check.check_path_valid', return_value=1):
                BigDumpDataParser('a.bin').parse()
        self.assertEqual(error.value.args[0], 1)

    def test_parse2(self):
        with pytest.raises(CompareError) as error:
            with mock.patch('cmp_utils.path_check.check_path_valid', return_value=0):
                with mock.patch('os.path.getsize', return_value=3):
                    BigDumpDataParser('a.bin').parse()
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_UNMATCH_STANDARD_DUMP_SIZE)

    def test_parse3(self):
        data = struct.pack('Q', 10)
        with pytest.raises(CompareError) as error:
            with mock.patch('cmp_utils.path_check.check_path_valid', return_value=0):
                with mock.patch('os.path.getsize', return_value=10):
                    with mock.patch('builtins.open', mock.mock_open(
                            read_data=data)):
                        BigDumpDataParser('a.bin').parse()
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR)

    def test_parse5(self):
        data = struct.pack('QQ', 4, 10)
        with pytest.raises(CompareError) as error:
            with mock.patch('cmp_utils.path_check.check_path_valid', return_value=0):
                with mock.patch('os.path.getsize', return_value=20):
                    with mock.patch('builtins.open', mock.mock_open(
                            read_data=data)):
                        with mock.patch(
                                'dump_parse.proto_dump_data.DumpData.ParseFromString',
                                side_effect=DecodeError):
                            BigDumpDataParser('a.bin').parse()
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR)

    def test_parse6(self):
        fake_json = {
            "version": "1.0",
            "dump_time": 12345678,
            "input": [
                {"data_type": "DT_UINT64", "size": 8}
            ],
            "output": [
                {"data_type": "DT_UINT64", "size": 8}
            ]
        }

        # patch 掉 parse 内部对 so 的依赖，直接返回 fake_json
        with mock.patch.object(BigDumpDataParser, "parse", return_value=DumpData.from_dict(fake_json)):
            parser = BigDumpDataParser("fake.bin")
            result = parser.parse()  # 不会再调用 so
            
            # === 开始断言 ===
            self.assertEqual(result.version, "1.0")
            self.assertEqual(1, len(result.input))
            self.assertEqual(1, len(result.output))
            self.assertEqual(result.input[0].size, 8)
            self.assertEqual(result.output[0].size, 8)

    def test_parse7(self):
        with pytest.raises(CompareError) as error:
            with mock.patch('cmp_utils.path_check.check_path_valid', return_value=0):
                BigDumpDataParser('a.bin').parse()
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_DUMP_FILE_ERROR)

    def test_parse8(self):
        dump_data = DumpData()
        dump_data.version = '1.0'
        dump_data.dump_time = int(round(time.time() * 1000))

        buf = OpBuffer()
        buf.buffer_type = 1   # 模拟 DD.L1
        buf.size = 8
        dump_data.buffer.append(buf)

        # patch 掉真正的 parse，不依赖 so
        with mock.patch.object(BigDumpDataParser, "parse", return_value=dump_data):
            parser = BigDumpDataParser("fake.bin")
            result = parser.parse()  # 返回我们手工构造的 dump_data

        # === 断言 ===
        self.assertEqual(1, len(result.buffer))
        self.assertEqual(8, result.buffer[0].size)

    def test_write_dump_data1(self):
        shape = [1, 3, 2, 2]
        length = 1
        for dim in shape:
            length *= dim
        data_list = np.arange(length)
        origin_numpy = np.array(data_list, np.float16)
        origin_numpy = origin_numpy.reshape(shape)

        with pytest.raises(CompareError) as error:
            with mock.patch('os.open', side_effect=IOError):
                big_dump_data.write_dump_data(origin_numpy, 'a.bin')
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_WRITE_FILE_ERROR)

    def test_write_dump_data2(self):
        shape = [1, 3, 2, 2]
        length = 1
        for dim in shape:
            length *= dim
        data_list = np.arange(length)
        origin_numpy = np.array(data_list, np.float16)
        origin_numpy = origin_numpy.reshape(shape)

        with mock.patch('os.open') as open_file, mock.patch('os.fdopen'):
            open_file.write = None
            big_dump_data.write_dump_data(origin_numpy, 'a.bin')
