import unittest

import struct
import pytest
import numpy as np
import utils
import time
import dump_data_pb2 as DD
from unittest import mock
from dump_parse import big_dump_data
from dump_parse.big_dump_data import BigDumpDataParser
from cmp_utils.constant.compare_error import CompareError
from google.protobuf.message import DecodeError


class TestUtilsMethods(unittest.TestCase):

    def test_parse1(self):
        with pytest.raises(utils.CompareError) as error:
            with mock.patch('utils.check_path_valid', return_value=1):
                BigDumpDataParser('a.bin').parse()
        self.assertEqual(error.value.args[0], 1)

    def test_parse2(self):
        with pytest.raises(utils.CompareError) as error:
            with mock.patch('utils.check_path_valid', return_value=0):
                with mock.patch('os.path.getsize', return_value=3):
                    BigDumpDataParser('a.bin').parse()
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_UNMATCH_STANDARD_DUMP_SIZE)

    def test_parse3(self):
        data = struct.pack('Q', 10)
        with pytest.raises(utils.CompareError) as error:
            with mock.patch('utils.check_path_valid', return_value=0):
                with mock.patch('os.path.getsize', return_value=10):
                    with mock.patch('builtins.open', mock.mock_open(
                            read_data=data)):
                        BigDumpDataParser('a.bin').parse()
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR)

    def test_parse4(self):
        data = struct.pack('QQ', 4, 10)
        with pytest.raises(utils.CompareError) as error:
            with mock.patch('utils.check_path_valid', return_value=0):
                with mock.patch('os.path.getsize', return_value=20):
                    with mock.patch('builtins.open', mock.mock_open(
                            read_data=data)):
                        BigDumpDataParser('a.bin').parse()
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_UNMATCH_STANDARD_DUMP_SIZE)

    def test_parse5(self):
        data = struct.pack('QQ', 4, 10)
        with pytest.raises(utils.CompareError) as error:
            with mock.patch('utils.check_path_valid', return_value=0):
                with mock.patch('os.path.getsize', return_value=20):
                    with mock.patch('builtins.open', mock.mock_open(
                            read_data=data)):
                        with mock.patch(
                                'dump_data_pb2.DumpData.ParseFromString',
                                side_effect=DecodeError):
                            BigDumpDataParser('a.bin').parse()
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR)

    def test_parse6(self):
        dump_data = DD.DumpData()
        dump_data.version = '1.0'
        dump_data.dump_time = int(round(time.time() * 1000))
        output = dump_data.output.add()
        output.data_type = DD.DT_UINT64
        output.size = 8
        input = dump_data.input.add()
        input.data_type = DD.DT_UINT64
        input.size = 8
        dump_data_ser = dump_data.SerializeToString()
        struct_format = 'Q' + str(len(dump_data_ser)) + 'sQQ'
        data = struct.pack(
            struct_format, len(dump_data_ser), dump_data_ser, 13, 55)
        with mock.patch('utils.check_path_valid', return_value=0):
            with mock.patch('os.path.getsize', return_value=48):
                with mock.patch('builtins.open', mock.mock_open(
                        read_data=data)):
                    result = BigDumpDataParser('a.bin').parse()
        self.assertEqual(1, len(result.input))
        self.assertEqual(1, len(result.output))
        self.assertEqual(8, result.output[0].size)
        self.assertEqual(8, result.input[0].size)

    def test_parse7(self):
        with pytest.raises(utils.CompareError) as error:
            with mock.patch('utils.check_path_valid', return_value=0):
                BigDumpDataParser('a.bin').parse()
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_DUMP_FILE_ERROR)

    def test_parse8(self):
        dump_data = DD.DumpData()
        dump_data.version = '1.0'
        dump_data.dump_time = int(round(time.time() * 1000))
        buffer = dump_data.buffer.add()
        buffer.buffer_type = DD.L1
        buffer.size = 8
        dump_data_ser = dump_data.SerializeToString()
        struct_format = 'Q' + str(len(dump_data_ser)) + 'sQ'
        data = struct.pack(
            struct_format, len(dump_data_ser), dump_data_ser, 88)
        with mock.patch('utils.check_path_valid', return_value=0):
            with mock.patch('os.path.getsize', return_value=32):
                with mock.patch('builtins.open', mock.mock_open(
                        read_data=data)):
                    result = BigDumpDataParser('a.bin').parse()
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

        with pytest.raises(utils.CompareError) as error:
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
