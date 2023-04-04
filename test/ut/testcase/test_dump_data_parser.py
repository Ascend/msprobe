import time
import unittest
import pytest
from src.compare.dump_parse import dump_data_parser as DP
from unittest import mock
import dump_data_pb2 as DD
import struct
import numpy as np
from cmp_utils.constant.compare_error import CompareError
import utils


class TestUtilsMethods(unittest.TestCase):
    def test_check_arguments_valid1(self):
        arguments = mock.Mock()
        arguments.dump_path = '/home/CONV2.pron.1.1234567891234567'
        arguments.dump_version = 2
        arguments.output_file_type = "npy"
        arguments.output_path = ""
        with pytest.raises(utils.CompareError) as error:
            with mock.patch('utils.check_path_valid', return_value=1):
                DP.DumpDataParser(arguments).check_arguments_valid()
        self.assertEqual(error.value.args[0], 1)

    def test_check_arguments_valid2(self):
        arguments = mock.Mock()
        arguments.dump_path = '/home/CONV2.pron.1.1234567891234567'
        arguments.dump_version = 2
        arguments.output_file_type = "npy"
        arguments.output_path = ""
        with pytest.raises(utils.CompareError) as error:
            with mock.patch('utils.check_path_valid', return_value=0):
                with mock.patch("utils.check_output_path_valid", return_value=1):
                    DP.DumpDataParser(arguments).check_arguments_valid()
        self.assertEqual(error.value.args[0], 1)

    def test_parse_dump_data1(self):
        arguments = mock.Mock()
        arguments.dump_path = '/home/CONV2.pron.1.1234567891234567'
        arguments.dump_version = 2
        arguments.output_file_type = "npy"
        arguments.output_path = ""
        dump_data = DD.DumpData()
        dump_data.version = '1.0'
        dump_data.dump_time = int(round(time.time() * 1000))
        buffer = dump_data.buffer.add()
        buffer.buffer_type = DD.L1
        buffer.size = 8
        buffer.data = struct.pack('Q', 35)
        dump_data = utils.convert_dump_data(dump_data)
        with mock.patch('utils.check_path_valid',
                        return_value=CompareError.MSACCUCMP_NONE_ERROR):
            with mock.patch('utils.check_output_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR):
                with mock.patch('utils.parse_dump_file',
                                return_value=dump_data):
                    with mock.patch('os.open',
                                    side_effect=OSError) as open_file, \
                            mock.patch('os.fdopen'):
                        with mock.patch("os.path.isfile", return_value=True):
                            open_file.write = None
                            ret = DP.DumpDataParser(arguments).parse_dump_data()
        self.assertEqual(ret, CompareError.MSACCUCMP_WRITE_FILE_ERROR)

    def test_main_parse_dump_data2(self):
        arguments = mock.Mock()
        arguments.dump_path = '/home/CONV2.pron.1.1234567891234567'
        arguments.dump_version = 2
        arguments.output_file_type = "npy"
        arguments.output_path = ""
        dump_data = DD.DumpData()
        dump_data.version = '1.0'
        dump_data.dump_time = int(round(time.time() * 1000))
        buffer = dump_data.buffer.add()
        buffer.buffer_type = DD.L1
        buffer.size = 8
        buffer.data = struct.pack('Q', 35)
        dump_data = utils.convert_dump_data(dump_data)
        with mock.patch('utils.check_path_valid',
                        return_value=CompareError.MSACCUCMP_NONE_ERROR):
            with mock.patch('utils.check_output_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR):
                with mock.patch('utils.parse_dump_file',
                                return_value=dump_data):
                    with mock.patch('os.open') as open_file, \
                            mock.patch('os.fdopen'):
                        with mock.patch("os.path.isfile", return_value=True):
                            open_file.write = None
                            ret = DP.DumpDataParser(arguments).parse_dump_data()
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_main_parse_dump_data3(self):
        arguments = mock.Mock()
        arguments.dump_path = '/home/CONV2.pron.1.1234567891234567'
        arguments.dump_version = 2
        arguments.output_file_type = "npy"
        arguments.output_path = ""
        dump_data = DD.DumpData()
        dump_data.version = '1.0'
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
        op_output.data = struct.pack('%de' % length, *origin_numpy)
        dump_data = utils.convert_dump_data(dump_data)
        with mock.patch('utils.check_path_valid',
                        return_value=CompareError.MSACCUCMP_NONE_ERROR):
            with mock.patch('utils.check_output_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR):
                with mock.patch('utils.parse_dump_file', return_value=dump_data):
                    with mock.patch('numpy.save'):
                        with mock.patch("os.path.isfile", return_value=True):
                            ret = DP.DumpDataParser(arguments).parse_dump_data()
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_main_parse_dump_data4(self):
        arguments = mock.Mock()
        arguments.dump_path = '/home/CONV2.pron.1.1234567891234567'
        arguments.dump_version = 2
        arguments.output_file_type = "npy"
        arguments.output_path = ""
        dump_data = DD.DumpData()
        dump_data.version = '1.0'
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
        op_output.data = struct.pack('%de' % length, *origin_numpy)
        dump_data = utils.convert_dump_data(dump_data)
        with mock.patch('utils.check_path_valid',
                        return_value=CompareError.MSACCUCMP_NONE_ERROR):
            with mock.patch('utils.check_output_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR):
                with mock.patch('utils.parse_dump_file',
                                return_value=dump_data):
                    with mock.patch('numpy.save',
                                    side_effect=ValueError):
                        with mock.patch("os.path.isfile", return_value=True):
                            ret = DP.DumpDataParser(arguments).parse_dump_data()
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_main_parse_dump_data5(self):
        arguments = mock.Mock()
        arguments.dump_path = '/home/CONV2.pron.1.1234567891234567'
        arguments.dump_version = 2
        arguments.output_file_type = "msnpy"
        arguments.output_path = ""
        dump_data = DD.DumpData()
        dump_data.version = '1.0'
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
        op_output.data = struct.pack('%de' % length, *origin_numpy)
        dump_data = utils.convert_dump_data(dump_data)
        with mock.patch('utils.check_path_valid',
                        return_value=CompareError.MSACCUCMP_NONE_ERROR):
            with mock.patch('utils.check_output_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR):
                with mock.patch('utils.parse_dump_file',
                                return_value=dump_data):
                    with mock.patch('numpy.save'):
                        with mock.patch("os.path.isfile", return_value=True):
                            ret = DP.DumpDataParser(arguments).parse_dump_data()
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_main_parse_dump_data_with_multi_file(self):
        arguments = mock.Mock()
        arguments.dump_path = '/home/CONV2.pron1.1.1234567891234567,/home/CONV2.pron2.1.1234567891234567'
        arguments.dump_version = 2
        arguments.output_file_type = "msnpy"
        arguments.output_path = ""
        dump_data = DD.DumpData()
        dump_data.version = '1.0'
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
        op_output.data = struct.pack('%de' % length, *origin_numpy)
        dump_data = utils.convert_dump_data(dump_data)
        with mock.patch('utils.check_path_valid',
                        return_value=CompareError.MSACCUCMP_NONE_ERROR):
            with mock.patch('utils.check_output_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR):
                with mock.patch('utils.parse_dump_file', return_value=dump_data):
                    with mock.patch('numpy.save'):
                        with mock.patch("os.path.isfile", return_value=True), \
                                mock.patch('os.path.getsize', return_value=1000):
                            ret = DP.DumpDataParser(arguments).parse_dump_data()
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_save_op_debug_to_file1(self):
        arguments = mock.Mock()
        arguments.dump_path = '/home/Opdebug.Node_OpDebug.1.1234567891234567'
        arguments.dump_version = 2
        arguments.output_file_type = "npy"
        arguments.output_path = ""
        dump_data = DD.DumpData()
        dump_data.version = '1.0'
        dump_data.dump_time = int(round(time.time() * 1000))
        op_output = dump_data.output.add()
        op_output.data_type = DD.DT_FLOAT16
        op_output.format = DD.FORMAT_NCHW
        op_output.data = struct.pack('Q', 10)
        dump_data = utils.convert_dump_data(dump_data)
        with mock.patch('utils.check_path_valid',
                        return_value=CompareError.MSACCUCMP_NONE_ERROR):
            with mock.patch('utils.check_output_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR):
                with mock.patch('utils.parse_dump_file',
                                return_value=dump_data):
                    with mock.patch("os.path.isfile", return_value=True):
                        ret = DP.DumpDataParser(arguments).parse_dump_data()
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR)

    def test_save_op_debug_to_file2(self):
        arguments = mock.Mock()
        arguments.dump_path = '/home/Opdebug.Node_OpDebug.1.1234567891234567'
        arguments.dump_version = 2
        arguments.output_file_type = "npy"
        arguments.output_path = ""
        dump_data = DD.DumpData()
        dump_data.version = '1.0'
        dump_data.dump_time = int(round(time.time() * 1000))
        op_output = dump_data.output.add()
        op_output.data_type = DD.DT_FLOAT16
        op_output.format = DD.FORMAT_NCHW
        zero_bytes = self._make_uint64_data(2048)
        op_output.data = struct.pack('%dQ' % len(zero_bytes), *zero_bytes)
        dump_data = utils.convert_dump_data(dump_data)
        with mock.patch('utils.check_path_valid',
                        return_value=CompareError.MSACCUCMP_NONE_ERROR):
            with mock.patch('utils.check_output_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR):
                with mock.patch('utils.parse_dump_file',
                                return_value=dump_data):
                    with mock.patch('os.open',
                                    side_effect=OSError) as open_file, \
                            mock.patch('os.fdopen'):
                        with mock.patch("os.path.isfile", return_value=True):
                            open_file.write = None
                            ret = DP.DumpDataParser(arguments).parse_dump_data()
        self.assertEqual(ret, CompareError.MSACCUCMP_WRITE_FILE_ERROR)

    def test_save_op_debug_to_file3(self):
        arguments = mock.Mock()
        arguments.dump_path = '/home/Opdebug.Node_OpDebug.1.1234567891234567'
        arguments.dump_version = 2
        arguments.output_file_type = "npy"
        arguments.output_path = ""
        dump_data = DD.DumpData()
        dump_data.version = '1.0'
        dump_data.dump_time = int(round(time.time() * 1000))
        op_output = dump_data.output.add()
        op_output.data_type = DD.DT_FLOAT16
        op_output.format = DD.FORMAT_NCHW
        zero_bytes = self._make_uint64_data(2048)
        op_output.data = struct.pack('%dQ' % len(zero_bytes), *zero_bytes)
        dump_data = utils.convert_dump_data(dump_data)
        with mock.patch('utils.check_path_valid',
                        return_value=CompareError.MSACCUCMP_NONE_ERROR):
            with mock.patch('utils.check_output_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR):
                with mock.patch('utils.parse_dump_file',
                                return_value=dump_data):
                    with mock.patch('os.open') as open_file, \
                            mock.patch('os.fdopen'):
                        with mock.patch("os.path.isfile", return_value=True):
                            open_file.write = None
                            ret = DP.DumpDataParser(arguments).parse_dump_data()
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_save_op_debug_to_file4(self):
        arguments = mock.Mock()
        arguments.dump_path = '/home/Opdebug.Node_OpDebug.1.1234567891244567'
        arguments.dump_version = 2
        arguments.output_file_type = "npy"
        arguments.output_path = ""
        dump_data = DD.DumpData()
        dump_data.version = '1.0'
        dump_data.dump_time = int(round(time.time() * 1000))
        op_output = dump_data.output.add()
        op_output.data_type = DD.DT_FLOAT16
        op_output.format = DD.FORMAT_NCHW
        overflow_data = self._make_overflow_data_new_version(88)
        op_output.data = struct.pack('6i11Q', *overflow_data)
        dump_data = utils.convert_dump_data(dump_data)
        with mock.patch('utils.check_path_valid',
                        return_value=CompareError.MSACCUCMP_NONE_ERROR):
            with mock.patch('utils.check_output_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR):
                with mock.patch('utils.parse_dump_file',
                                return_value=dump_data):
                    with mock.patch('os.open') as open_file, \
                            mock.patch('os.fdopen'):
                        with mock.patch("os.path.isfile", return_value=True):
                            open_file.write = None
                            ret = DP.DumpDataParser(arguments).parse_dump_data()
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    @staticmethod
    def _make_uint64_data(size):
        count = int(size / 8)
        data = [0] * count
        return data

    @staticmethod
    def _make_overflow_data_new_version(size):
        count = int(size / 8)
        data = [0x5a5a5a5a, 0, 1, 1, 0, 88]
        data += [0] * count
        return data
