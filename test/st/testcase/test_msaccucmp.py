import unittest

import pytest
import msaccucmp
import struct
from src.compare.cmp_utils import utils
import time
import numpy as np
from unittest import mock
import dump_data_pb2 as DD
from src.compare.cmp_utils.constant.compare_error import CompareError
import json
from cmp_utils import file_utils


class TestUtilsMethods(unittest.TestCase):

    def test_main1(self):
        args = ['aaa.py', 'compare', '-m', '/home/left.npy', '-g',
                '/home/right.npy']
        dump_data = np.arange(2)
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.isfile', return_value=True):
                    with mock.patch('src.compare.cmp_utils.utils.check_path_valid', return_value=CompareError.MSACCUCMP_NONE_ERROR):
                        with mock.patch("src.compare.cmp_utils.utils.read_numpy_file", return_value=dump_data):
                            msaccucmp.main()
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_NONE_ERROR)

    def test_main2(self):
        args = ['aaa.py', 'compare', '-m', '/home/left.npy', '-g',
                '/home/right.npy']
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('src.compare.cmp_utils.utils.check_path_valid', return_value=CompareError.MSACCUCMP_NONE_ERROR):
                    with mock.patch('os.path.isdir', return_value=True):
                        with mock.patch('os.listdir',
                                        side_effect=[['alg_MaxAbsoluteError.py'], ['alg_MaxAbsoluteError.py'],
                                                     ["mapping.csv", '23423125315', "Add.0.1223242.npy"]]):
                            msaccucmp.main()
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_DUMP_FILE_ERROR)

    def test_main3(self):
        args = ['aaa.py', 'convert', '-d', '/home/left.bin']
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                ret = msaccucmp.main()
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_PATH_ERROR)

    def test_main4(self):
        args = ['aaa.py', 'convert', '-d', '/home/left.bin', '-f', 'NCHW']
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                with mock.patch("src.compare.cmp_utils.utils.check_path_valid", return_value=0):
                    with mock.patch("os.path.isfile", return_value=True):
                        with mock.patch('os.path.getsize', return_value=100):
                            msaccucmp.main()
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR)

    def test_main5(self):
        args = ['aaa.py']
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                msaccucmp.main()
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_main6(self):
        args = ['aaa.py', 'compare', "-s", ',,,', '-m', '/home/left.bin', '-g',
                '/home/right.bin']
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                msaccucmp.main()
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_main7(self):
        args = ['aaa.py', 'compare', "-s", ',,,', '-op', 'prob', '-m', '/home/left.bin', '-g',
                '/home/right.bin', '-f', '/home/a.json']
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                msaccucmp.main()
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_main_compare1(self):
        args = ['aaa.py', 'compare', '-m', '/home/left.bin', '-g',
                '/home/right.bin', '-o', '1']
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                msaccucmp.main()
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_main_compare2(self):
        args = ['aaa.py', 'compare', '-m', '/home/left.bin', '-g',
                '/home/right.bin', '-i', '1']
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                msaccucmp.main()
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_main_compare3(self):
        args = ['aaa.py', 'compare', '-m', '/home/left.bin', '-g',
                '/home/right.bin', '-op', 'aaa']
        with pytest.raises(SystemExit) as error:
            with mock.patch('os.path.isfile', return_value=False):
                with mock.patch('sys.argv', args):
                    msaccucmp.main()
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_PATH_ERROR)

    def test_main_compare4(self):
        args = ['aaa.py', 'compare', '-m', '/home/left.bin', '-g',
                '/home/right.bin', '-op', 'aaa', '-i', '1']
        with pytest.raises(SystemExit) as error:
            with mock.patch('os.path.isfile', return_value=False):
                with mock.patch('sys.argv', args):
                    msaccucmp.main()
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_PATH_ERROR)

    def test_main_compare5(self):
        args = ['aaa.py', 'compare', '-m', '/home/left.bin', '-g',
                '/home/right.bin', '-op', 'aaa', '-o', '1']
        with pytest.raises(SystemExit) as error:
            with mock.patch('os.path.isfile', return_value=False):
                with mock.patch('sys.argv', args):
                    msaccucmp.main()
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_PATH_ERROR)

    def test_main_compare6(self):
        args = ['aaa.py', 'compare', '-m', '/home/left', '-g',
                '/home/right', '-cf', '/home/demo/xx.json']
        with pytest.raises(SystemExit) as error:
            with mock.patch('os.path.isfile', return_value=False):
                with mock.patch('sys.argv', args):
                    msaccucmp.main()
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_PATH_ERROR)

    def test_main_convert1(self):
        args = ['aaa.py', 'convert', '-d', '/home/left.bin', '-i', '1']
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                msaccucmp.main()
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_main_convert2(self):
        args = ['aaa.py', 'convert', '-d', '/home/left.bin', '-o', '1']
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                msaccucmp.main()
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_main_convert3(self):
        args = ['aaa.py', 'convert', '-d', '/home/left.bin', '-s', '1']
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                msaccucmp.main()
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_main_convert4(self):
        args = ['aaa.py', 'convert', '-d', '/home/left.bin', '-c', '1']
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                msaccucmp.main()
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_main_dump_data_parser1(self):
        args = ['aaa.py', 'convert', '-d',
                '/home/CONV2.pron.1.1234567891234567']
        with mock.patch('sys.argv', args):
            with pytest.raises(SystemExit) as error:
                with mock.patch('src.compare.cmp_utils.utils.check_path_valid', return_value=1):
                    msaccucmp.main()
        self.assertEqual(error.value.code, 1)

    def test_main_dump_data_parser2(self):
        args = ['aaa.py', 'convert', '-d',
                '/home/CONV2.pron.1.1234567891234567']
        with mock.patch('sys.argv', args):
            with pytest.raises(SystemExit) as error:
                with mock.patch('src.compare.cmp_utils.utils.check_path_valid',
                                return_value=CompareError.MSACCUCMP_NONE_ERROR):
                    with mock.patch('src.compare.cmp_utils.utils.check_output_path_valid',
                                    return_value=1):
                        msaccucmp.main()
        self.assertEqual(error.value.code, 1)

    def test_main_dump_data_parser3(self):
        dump_data = DD.DumpData()
        dump_data.version = '1.0'
        dump_data.dump_time = int(round(time.time() * 1000))
        buffer = dump_data.buffer.add()
        buffer.buffer_type = DD.L1
        buffer.size = 8
        buffer.data = struct.pack('Q', 35)
        args = ['aaa.py', 'convert', '-d',
                '/home/CONV2.pron.1.1234567891234567']
        with mock.patch('sys.argv', args):
            with pytest.raises(SystemExit) as error:
                with mock.patch('src.compare.cmp_utils.utils.check_path_valid',
                                return_value=CompareError.MSACCUCMP_NONE_ERROR):
                    with mock.patch('src.compare.cmp_utils.utils.check_output_path_valid',
                                    return_value=CompareError.MSACCUCMP_NONE_ERROR):
                        with mock.patch('src.compare.cmp_utils.utils.parse_dump_file',
                                        return_value=dump_data):
                            with mock.patch('os.open',
                                            side_effect=OSError) as open_file, \
                                    mock.patch('os.fdopen'):
                                with mock.patch("os.path.isfile", return_value=True):
                                    open_file.write = None
                                    msaccucmp.main()
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_WRITE_FILE_ERROR)

    def test_main_dump_data_parser4(self):
        dump_data = DD.DumpData()
        dump_data.version = '1.0'
        dump_data.dump_time = int(round(time.time() * 1000))
        buffer = dump_data.buffer.add()
        buffer.buffer_type = DD.L1
        buffer.size = 8
        buffer.data = struct.pack('Q', 35)
        args = ['aaa.py', 'convert', '-d',
                '/home/CONV2.pron.1.1234567891234567']
        with mock.patch('sys.argv', args):
            with pytest.raises(SystemExit) as error:
                with mock.patch('src.compare.cmp_utils.utils.check_path_valid',
                                return_value=CompareError.MSACCUCMP_NONE_ERROR):
                    with mock.patch('src.compare.cmp_utils.utils.check_output_path_valid',
                                    return_value=CompareError.MSACCUCMP_NONE_ERROR):
                        with mock.patch('src.compare.cmp_utils.utils.parse_dump_file',
                                        return_value=dump_data):
                            with mock.patch('os.open') as open_file, \
                                    mock.patch('os.fdopen'):
                                with mock.patch("os.path.isfile", return_value=True):
                                    open_file.write = None
                                    msaccucmp.main()
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_NONE_ERROR)

    def test_main_dump_data_parser5(self):
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
        args = ['aaa.py', 'convert', '-d',
                '/home/CONV2.pron.1.1234567891234567']
        with mock.patch('sys.argv', args):
            with pytest.raises(SystemExit) as error:
                with mock.patch('src.compare.cmp_utils.utils.check_path_valid',
                                return_value=CompareError.MSACCUCMP_NONE_ERROR):
                    with mock.patch('src.compare.cmp_utils.utils.check_output_path_valid',
                                    return_value=CompareError.MSACCUCMP_NONE_ERROR):
                        with mock.patch('src.compare.cmp_utils.utils.parse_dump_file',
                                        return_value=dump_data):
                            with mock.patch('numpy.save'):
                                with mock.patch("os.path.isfile", return_value=True):
                                    msaccucmp.main()
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_NONE_ERROR)

    def test_main_dump_data_parser6(self):
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
        args = ['aaa.py', 'convert', '-d',
                '/home/CONV2.pron.1.1234567891234567']
        with mock.patch('sys.argv', args):
            with pytest.raises(SystemExit) as error:
                with mock.patch('src.compare.cmp_utils.utils.check_path_valid',
                                return_value=CompareError.MSACCUCMP_NONE_ERROR):
                    with mock.patch('src.compare.cmp_utils.utils.check_output_path_valid',
                                    return_value=CompareError.MSACCUCMP_NONE_ERROR):
                        with mock.patch('src.compare.cmp_utils.utils.parse_dump_file',
                                        return_value=dump_data):
                            with mock.patch('numpy.save',
                                            side_effect=ValueError):
                                with mock.patch("os.path.isfile", return_value=True):
                                    msaccucmp.main()
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_NONE_ERROR)

    def test_main_dump_data_parser7(self):
        dump_data = DD.DumpData()
        dump_data.version = '1.0'
        dump_data.dump_time = int(round(time.time() * 1000))
        op_output = dump_data.output.add()
        op_output.data_type = DD.DT_RESOURCE
        op_output.format = DD.FORMAT_NCHW
        length = 1
        for dim in [8]:
            op_output.shape.dim.append(dim)
            length *= dim
        data_list = np.arange(length)
        origin_numpy = np.array(data_list, np.float16)
        op_output.data = struct.pack('%de' % length, *origin_numpy)
        args = ['aaa.py', 'convert', '-d',
                '/home/CONV2.pron.1.1234567891234567']
        with mock.patch('sys.argv', args):
            with pytest.raises(SystemExit) as error:
                with mock.patch('src.compare.cmp_utils.utils.check_path_valid',
                                return_value=CompareError.MSACCUCMP_NONE_ERROR):
                    with mock.patch('src.compare.cmp_utils.utils.check_output_path_valid',
                                    return_value=CompareError.MSACCUCMP_NONE_ERROR):
                        with mock.patch('src.compare.cmp_utils.utils.parse_dump_file',
                                        return_value=dump_data):
                            with mock.patch('numpy.save'):
                                with mock.patch("os.path.isfile", return_value=True):
                                    msaccucmp.main()
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_NONE_ERROR)

    def test_main_parse_dump_data5(self):
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
        args = ['aaa.py', 'convert', '-d',
                '/home/CONV2.pron.1.1234567891234567', '-t', 'msnpy']
        with mock.patch('sys.argv', args):
            with pytest.raises(SystemExit) as error:
                with mock.patch('src.compare.cmp_utils.utils.check_path_valid',
                                return_value=CompareError.MSACCUCMP_NONE_ERROR):
                    with mock.patch('src.compare.cmp_utils.utils.check_output_path_valid',
                                    return_value=CompareError.MSACCUCMP_NONE_ERROR):
                        with mock.patch('src.compare.cmp_utils.utils.parse_dump_file',
                                        return_value=dump_data):
                            with mock.patch('numpy.save'):
                                with mock.patch("os.path.isfile", return_value=True):
                                    msaccucmp.main()
        self.assertEqual(error.value.code, CompareError.MSACCUCMP_NONE_ERROR)

    def test_main_parse_dump_data_with_multi_file(self):
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
        args = ['aaa.py', 'convert', '-d',
                '/home/CONV2.pron1.1.1234567891234567,/home/CONV2.pron2.1.1234567891234567', '-t', 'msnpy']
        with mock.patch('sys.argv', args):
            with pytest.raises(SystemExit) as error:
                with mock.patch('src.compare.cmp_utils.utils.check_path_valid',
                                return_value=CompareError.MSACCUCMP_NONE_ERROR):
                    with mock.patch('src.compare.cmp_utils.utils.check_output_path_valid',
                                    return_value=CompareError.MSACCUCMP_NONE_ERROR):
                        with mock.patch('src.compare.cmp_utils.utils.parse_dump_file', return_value=dump_data):
                            with mock.patch('numpy.save'):
                                with mock.patch("os.path.isfile", return_value=True), \
                                     mock.patch('os.path.getsize', return_value=1000):
                                    msaccucmp.main()
        self.assertEqual(error.value.code, CompareError.MSACCUCMP_NONE_ERROR)

    def test_main_dump_data_parser_opdebug1(self):
        dump_data = DD.DumpData()
        dump_data.version = '1.0'
        dump_data.dump_time = int(round(time.time() * 1000))
        op_output = dump_data.output.add()
        op_output.data_type = DD.DT_FLOAT16
        op_output.format = DD.FORMAT_NCHW
        op_output.data = struct.pack('Q', 10)
        args = ['aaa.py', 'convert', '-d',
                '/home/Opdebug.Node_OpDebug.1.1234567891234567']
        with mock.patch('sys.argv', args):
            with pytest.raises(SystemExit) as error:
                with mock.patch('src.compare.cmp_utils.utils.check_path_valid',
                                return_value=CompareError.MSACCUCMP_NONE_ERROR):
                    with mock.patch('src.compare.cmp_utils.utils.check_output_path_valid',
                                    return_value=CompareError.MSACCUCMP_NONE_ERROR):
                        with mock.patch('src.compare.cmp_utils.utils.parse_dump_file',
                                        return_value=dump_data):
                            with mock.patch("os.path.isfile", return_value=True):
                                msaccucmp.main()
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR)

    def test_main_dump_data_parser_opdebug2(self):
        dump_data = DD.DumpData()
        dump_data.version = '1.0'
        dump_data.dump_time = int(round(time.time() * 1000))
        op_output = dump_data.output.add()
        op_output.data_type = DD.DT_FLOAT16
        op_output.format = DD.FORMAT_NCHW
        zero_bytes = self._make_uint64_data(2048)
        op_output.data = struct.pack('%dQ' % len(zero_bytes), *zero_bytes)
        args = ['aaa.py', 'convert', '-d',
                '/home/Opdebug.Node_OpDebug.1.1234567891234567']
        with mock.patch('sys.argv', args):
            with pytest.raises(SystemExit) as error:
                with mock.patch('src.compare.cmp_utils.utils.check_path_valid',
                                return_value=CompareError.MSACCUCMP_NONE_ERROR):
                    with mock.patch('src.compare.cmp_utils.utils.check_output_path_valid',
                                    return_value=CompareError.MSACCUCMP_NONE_ERROR):
                        with mock.patch('src.compare.cmp_utils.utils.parse_dump_file',
                                        return_value=dump_data):
                            with mock.patch('os.open',
                                            side_effect=OSError) as open_file, \
                                    mock.patch('os.fdopen'):
                                with mock.patch("os.path.isfile", return_value=True):
                                    open_file.write = None
                                    msaccucmp.main()
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_WRITE_FILE_ERROR)

    def test_main_dump_data_parser_opdebug3(self):
        dump_data = DD.DumpData()
        dump_data.version = '1.0'
        dump_data.dump_time = int(round(time.time() * 1000))
        op_output = dump_data.output.add()
        op_output.data_type = DD.DT_FLOAT16
        op_output.format = DD.FORMAT_NCHW
        zero_bytes = self._make_uint64_data(2048)
        op_output.data = struct.pack('%dQ' % len(zero_bytes), *zero_bytes)
        args = ['aaa.py', 'convert', '-d',
                '/home/Opdebug.Node_OpDebug.1.1234567891234567']
        with mock.patch('sys.argv', args):
            with pytest.raises(SystemExit) as error:
                with mock.patch('src.compare.cmp_utils.utils.check_path_valid',
                                return_value=CompareError.MSACCUCMP_NONE_ERROR):
                    with mock.patch('src.compare.cmp_utils.utils.check_output_path_valid',
                                    return_value=CompareError.MSACCUCMP_NONE_ERROR):
                        with mock.patch('src.compare.cmp_utils.utils.parse_dump_file',
                                        return_value=dump_data):
                            with mock.patch('os.open') as open_file, \
                                    mock.patch('os.fdopen'):
                                with mock.patch("os.path.isfile", return_value=True):
                                    open_file.write = None
                                    msaccucmp.main()
        self.assertEqual(error.value.code, CompareError.MSACCUCMP_NONE_ERROR)
        
    def test_main_dump_data_parser_opdebug_new_format(self):
        dump_data = DD.DumpData()
        dump_data.version = '1.0'
        dump_data.dump_time = int(round(time.time() * 1000))
        op_output = dump_data.output.add()
        op_output.data_type = DD.DT_FLOAT16
        op_output.format = DD.FORMAT_NCHW
        overflow_data = self._make_overflow_data_new_version(88)
        op_output.data = struct.pack('6i11Q', *overflow_data)
        args = ['aaa.py', 'convert', '-d',
                '/home/Opdebug.Node_OpDebug.1.1234567891234567']
        with mock.patch('sys.argv', args):
            with pytest.raises(SystemExit) as error:
                with mock.patch('src.compare.cmp_utils.utils.check_path_valid', return_value=CompareError.MSACCUCMP_NONE_ERROR), \
                     mock.patch('utils.check_output_path_valid', return_value=CompareError.MSACCUCMP_NONE_ERROR), \
                     mock.patch('src.compare.cmp_utils.utils.parse_dump_file', return_value=dump_data), \
                     mock.patch('os.open') as open_file, mock.patch('os.fdopen'), \
                     mock.patch("os.path.isfile", return_value=True):
                    open_file.write = None
                    msaccucmp.main()
        self.assertEqual(error.value.code, CompareError.MSACCUCMP_NONE_ERROR)

    def test_mapping_error_parameter1(self):
        args = ['aaa.py', 'compare', '-m', '/home/left.bin', '-g',
                '/home/right.bin', "-map"]
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                msaccucmp.main()
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_mapping_error_parameter2(self):
        args = ['aaa.py', 'compare', '-m', '/home/left.bin', '-g',
                '/home/right.bin', "-f", "/home/a.json", "-map", "-op", "name"]
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                msaccucmp.main()
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_mapping_error_parameter3(self):
        args = ['aaa.py', 'compare', '-m', '/home/left.bin', '-g',
                '/home/right.bin', '-f', "/home/a.json", "-map", "-i", "1"]
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                msaccucmp.main()
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_mapping_success_param(self):
        args = ['aaa.py', 'compare', '-m', '/home/left.bin', '-g',
                '/home/right.bin', '-f', "/home/a.json", "-map"]
        dump_data = DD.DumpData()
        dump_data.input.append(self._make_op_input(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data.output.append(self._make_op_output(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=True), \
                     mock.patch('os.access', return_value=True), \
                     mock.patch('os.remove'), \
                     mock.patch('os.listdir',
                                side_effect=[['alg_MaxAbsoluteError.py'],
                                             ['ReduceMeanD.conv1conv1_relu.6.4.1613727240764749'],
                                             ['input.4.1613727240736566.pb',
                                              'trans_Cast_1167.4.1613727241293941.pb'],
                                             ['convert_NC1HWC0_to_NCHW.py']]), \
                     mock.patch('os.path.isdir', return_value=True), \
                     mock.patch('os.path.isfile', return_value=True), \
                     mock.patch('src.compare.cmp_utils.utils.parse_dump_file', return_value=dump_data):
                    with mock.patch("os.path.getsize", return_value=100):
                        with mock.patch('builtins.open',
                                        mock.mock_open(read_data=self._make_input_json().encode('utf-8'))):
                            print("***********start**************")
                            msaccucmp.main()
                            print("************end*************")
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_NONE_ERROR)

    def test_batch_compare(self):
        args = ['aaa.py', 'compare', '-m', '/home/left.bin', '-g',
                '/home/right.bin', '-f', "/home/a.json", "-op", "data"]
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                with mock.patch("src.compare.cmp_utils.utils.check_hdf5_file_valid", return_value=False):
                    with mock.patch("os.path.isfile", return_value=False):
                        with mock.patch("os.path.exists", return_value=False):
                            with mock.patch("src.compare.cmp_utils.utils.check_path_valid", return_value=CompareError.MSACCUCMP_NONE_ERROR):
                                msaccucmp.main()
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_msaccucmp_alg_help(self):
        args = ['aaa.py', 'compare', "--help", '-alg', '1', '2', '3']
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                msaccucmp.main()
        self.assertEqual(error.value.args[0], 0)

    def test_msaccucmp_alg_help2(self):
        args = ['aaa.py', 'compare', "--help", '-alg', '1', '2', '9']
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                msaccucmp.main()
        self.assertEqual(error.value.args[0], 0)

    def test_msaccucmp_help(self):
        args = ['aaa.py', 'compare', "--help"]
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                msaccucmp.main()
        self.assertEqual(error.value.args[0], 0)

    def test_check_range_effect1(self):
        args = ['aaa.py', 'compare', "-r", ',,', '-m', '/home/left.bin', '-g',
                '/home/right.bin']
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                msaccucmp.main()
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_check_range_effect2(self):
        args = ['aaa.py', 'compare', "-r", ',,', '-op', 'prob', '-m', '/home/left.bin', '-g',
                '/home/right.bin', '-f', '/home/a.json']
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                msaccucmp.main()
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_check_range_effect3(self):
        args = ['aaa.py', 'compare', "-s", ',,', '-m', '/home/left.bin', '-g',
                '/home/right.bin']
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                msaccucmp.main()
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_check_range_effect2(self):
        args = ['aaa.py', 'compare', "-s", ',,', '-op', 'prob', '-m', '/home/left.bin', '-g',
                '/home/right.bin', '-f', '/home/a.json']
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                msaccucmp.main()
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_check_max_line_effect1(self):
        args = ['aaa.py', 'compare', '-op', 'prob', '-m', '/home/left.bin', '-g',
                '/home/right.bin', '-f', '', '--max_line', '100']
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                with mock.patch("src.compare.cmp_utils.utils.check_path_valid", return_value=0):
                    msaccucmp.main()
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_check_max_line_effect2(self):
        args = ['aaa.py', 'compare', '-op', 'prob', '-m', '/home/left.bin', '-g',
                '/home/right.bin', '-f', '', '--max_line', '10000000']
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                with mock.patch("src.compare.cmp_utils.utils.check_path_valid", return_value=0):
                    msaccucmp.main()
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    @staticmethod
    def _make_op_output(dd_format, shape):
        op_output = DD.OpOutput()
        op_output.data_type = DD.DT_FLOAT
        op_output.format = dd_format
        length = 1
        if shape is None:
            length = 20
        else:
            for dim in shape:
                op_output.shape.dim.append(dim)
                length *= dim
        data_list = np.arange(length)
        origin_numpy = np.array(data_list, np.float16)
        op_output.data = struct.pack('f' * length, *origin_numpy)
        return op_output

    @staticmethod
    def _make_op_input(dd_format, shape):
        op_input = DD.OpInput()
        op_input.data_type = DD.DT_FLOAT
        op_input.format = dd_format
        length = 1
        if shape is None:
            length = 20
        else:
            for dim in shape:
                op_input.shape.dim.append(dim)
                length *= dim
        data_list = np.arange(length)
        origin_numpy = np.array(data_list, np.float16)
        op_input.data = struct.pack('f' * length, *origin_numpy)
        return op_input

    @staticmethod
    def _make_input_json():
        return json.dumps({'name': 'resnet50', 'graph': [
            {'name': 'merge1', 'op':
                [{'name': 'data',
                  'type': 'Input',
                  "attr": [
                      {"key": "xxx",
                       "value": 'xxx'},
                  ],
                  'output_desc': [
                      {'attr': [
                          {'key': 'xxx',
                           'value': 'xxx'},
                          {'key': 'origin_format',
                           'value': {'s': 'NCHW'}},
                          {'key': 'origin_shape',
                           'value': {"list": {"val_type": 1,
                                              "i": [1, 3, 4, 4]}}},
                      ]},
                  ]
                  },
                 {'name': 'dynamic_const_471', 'type': 'Const', "attr": [
                     {"key": "_datadump_original_op_names",
                      "value": {"list": {"val_type": 1}}}]},
                 {'name': 'dynamic_const_387', 'type': 'Const', "attr": [
                     {"key": "_datadump_original_op_names",
                      "value": {"list": {"val_type": 1}}}]},
                 {'name': 'conv1conv1_relu',
                  'type': 'Relu',
                  "attr": [
                      {"key": "_datadump_original_op_names",
                       "value": {"list": {"val_type": 1,
                                          "s": ["scale_conv1", "conv1",
                                                "bn_conv1", "conv1_relu"]}}}
                  ],
                  "input": [
                      "data:0",
                      "dynamic_const_471:0",
                      "dynamic_const_387:0"
                  ],
                  'output_desc': [
                      {'attr': [
                          {'key': '_datadump_origin_name',
                           'value': {'s': 'conv1_relu'}},
                          {'key': '_datadump_origin_output_index',
                           'value': {'i': 0}},
                          {'key': '_datadump_origin_format',
                           'value': {'s': 'NCHW'}},
                          {'key': 'origin_shape',
                           'value': {"list": {"val_type": 1,
                                              "i": [1, 3, 4, 4]}}},
                      ]},
                  ]
                  },

                 ],
             },
        ]})

    def test_pytorch_main1(self):
        args = ['aaa.py', 'compare', '-m', '/home/left.h5', '-g',
                '/home/right.h5']
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.isfile', return_value=True):
                    msaccucmp.main()
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_PATH_ERROR)

    def test_pytorch_main2(self):
        args = ['aaa.py', 'compare', '-m', '/home/left.h5', '-g',
                '/home/right.h5']
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('src.compare.cmp_utils.utils.check_path_valid',
                                return_value=CompareError.MSACCUCMP_NONE_ERROR):
                    with mock.patch('src.compare.cmp_utils.utils.check_output_path_valid',
                                    return_value=CompareError.MSACCUCMP_NONE_ERROR):
                        with mock.patch("os.path.isfile", return_value=True):
                            with mock.patch("hdf5_parser.Hdf5Parser.open_file",
                                            return_value=CompareError.MSACCUCMP_NONE_ERROR):
                                with mock.patch('os.open',
                                                side_effect=OSError) as open_file, \
                                        mock.patch('os.fdopen'):
                                    open_file.write = None
                                    msaccucmp.main()
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_OPEN_FILE_ERROR)

    def test_main_overflow_case1(self):
        args = ['aaa.py', 'overflow', '-d', '/home/left.bin', '-out', '/home/output', '-n', '1']
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                msaccucmp.main()
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_PATH_ERROR)

    def test_main_overflow_case2(self):
        args = ['aaa.py', 'overflow', '-d', '/home/left.bin', '-out', '/home/output2', '-n', '1']
        with pytest.raises(SystemExit) as err:
            with mock.patch('sys.argv', args):
                with mock.patch('overflow.overflow_analyse.OverflowAnalyse.check_argument',
                                return_value=CompareError.MSACCUCMP_NONE_ERROR):
                    with mock.patch('overflow.overflow_analyse.OverflowAnalyse.analyse',
                                    return_value=CompareError.MSACCUCMP_NONE_ERROR):
                        msaccucmp.main()
        self.assertEqual(err.value.args[0], CompareError.MSACCUCMP_NONE_ERROR)

    def test_main_overflow_case3(self):
        args = ['aaa.py', 'overflow', '-d', '/home/left.bin', '-out', '/home/output3', '-n', '1']
        debug_files = (item for item in [["/home/", " ", ["Opdebug.Node_OpDebug.1.1234567891234567", "test2"]],
                                         ["path_root1", "folders1", ["test3", "test4"]]])
        file_desc = {
            "file_path": "/test/Opdebug.Node_OpDebug.1.25.161233160.output.0.json",
            "timestamp": int("161233160")
        }
        dump_attr = {
            "op_name": 'Node_OpDebug',
            "op_type": 'Opdebug',
            "task_id": int('11'),
            "stream_id": '25'
        }
        anchor = {
            "anchor_type": 'output',
            "anchor_idx": '0',
            "format": 'NCHW'
        }
        json_txt = {
            'AI Core':
                {
                    'task_id': 1,
                    'stream_id': 25,
                    'status': 253
                },
            'L2 Atomic Add':
                {
                    'task_id': 2,
                    'stream_id': 35,
                    'status': 0
                }
        }
        dump_attr = {
            "op_name": 'cov2d',
            "op_type": 'convolution',
            "task_id": int('1'),
            "stream_id": '25'
        }
        parsed_debug_file_desc = file_utils.ParsedDumpFileDesc(file_desc, dump_attr, anchor)
        file_desc = {
            "file_path": "/test/convolution.cov2d.1.25.161233160",
            "timestamp": int("161233160")
        }
        dump_file_desc = file_utils.DumpFileDesc(file_desc, dump_attr)
        file_desc = {
            "file_path": "/test/convolution.cov2d.1.25.161233160.output.0.npy",
            "timestamp": int("161233160")
        }
        parsed_dump_file_desc = file_utils.ParsedDumpFileDesc(file_desc, dump_attr, anchor)
        np_summary_result = '[Shape: (3, 4)] [Dtype: int64] [Max: 6] [Min: 1] [Mean: 3.5]'
        with pytest.raises(SystemExit) as err:
            with mock.patch('sys.argv', args):
                with mock.patch('overflow.overflow_analyse.OverflowAnalyse.check_argument',
                                return_value=CompareError.MSACCUCMP_NONE_ERROR):
                    with mock.patch('os.path.realpath', return_value='/home'):
                        with mock.patch('os.walk', return_value=debug_files):
                            with mock.patch('os.path.basename', return_value='Opdebug.Node_OpDebug.1.25.161233160'):
                                with mock.patch('overflow.overflow_analyse.OverflowAnalyse._parse_overflow_file',
                                                return_value=''):
                                    with mock.patch('cmp_utils.file_utils.OverflowFileUtils.list_parsed_debug_files',
                                                    side_effect=iter([{},
                                                                      {'Opdebug.Node_OpDebug.1.25.161233160.output'
                                                                       '.0.json': parsed_debug_file_desc}])):
                                        with mock.patch('cmp_utils.file_utils.OverflowFileUtils.load_json_file',
                                                        return_value=json_txt):
                                            with mock.patch(
                                                    'overflow_analyse.OverflowAnalyse._find_dump_files_by_task_id',
                                                    return_value=dump_file_desc):
                                                with mock.patch(
                                                        'overflow_analyse.OverflowAnalyse._get_parsed_dump_file',
                                                        return_value=[parsed_dump_file_desc]):
                                                    with mock.patch(
                                                            'overflow_analyse.OverflowAnalyse.npy_data_summary',
                                                            return_value=np_summary_result):
                                                        with mock.patch('cmp_utils.file_utils.FileUtils.save_file',
                                                                        return_value=True):
                                                            msaccucmp.main()
        self.assertEqual(err.value.args[0], CompareError.MSACCUCMP_NONE_ERROR)

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

    @staticmethod
    def _make_input_json2():
        return json.dumps({
            'name': 'resnet50',
            'graph': [
                {
                    'name': 'merge1',
                    'op': [
                        {
                            'name': 'data',
                            'type': 'Input',
                            "attr": [
                                {"key": "xxx",
                                 "value": 'xxx'},
                            ],
                            'output_desc': [
                                {
                                    'attr': [
                                        {
                                            'key': 'xxx',
                                            'value': 'xxx'
                                        },
                                        {
                                            'key': 'origin_format',
                                            'value': {'s': 'NCHW'}
                                        },
                                        {
                                            'key': 'origin_shape',
                                            'value': {"list": {"val_type": 1,
                                                               "i": [1, 3, 4, 4]}}
                                        },
                                    ]
                                },
                            ],
                            'dtype': 'DT_FLOAT'
                        },
                        {
                            'name': 'conv1conv1_relu',
                            'type': 'Relu',
                            "attr": [
                                {
                                    "key": "_datadump_original_op_names",
                                    "value": {"list": {"val_type": 1,
                                                       "s": ["scale_conv1", "conv1",
                                                             "bn_conv1", "conv1_relu"]}}
                                }
                            ],
                            "input": [
                                "data:0"
                            ],
                            'output_desc': [
                                {
                                    'attr':
                                        [
                                            {
                                                'key': '_datadump_origin_name',
                                                'value': {'s': 'conv1_relu'}
                                            },
                                            {
                                                'key': '_datadump_origin_output_index',
                                                'value': {'i': 0}
                                            },
                                            {
                                                'key': '_datadump_origin_format',
                                                'value': {'s': 'NCHW'}
                                            },
                                            {
                                                'key': 'origin_shape',
                                                'value': {"list": {"val_type": 1, "i": [1, 3, 4, 4]}}
                                            },
                                        ]
                                },
                            ]
                        },
                        {
                            'name': 'TestNode_act_quant',
                            'type': 'quant',
                            'attr': [],
                            'input': [
                                'conv1conv1_relu:0'
                            ],
                            'output_desc': [
                                {
                                    'dtype': 'DT_INT8',
                                }
                            ]
                        },
                        {
                            'name': 'middle',
                            'type': 'middle',
                            'attr': [],
                            'input': [
                                'TestNode_act_quant:0'
                            ],
                            'output_desc': [
                                {
                                    'dtype': 'DT_INT8',
                                }
                            ]
                        },
                        {
                            'name': 'TestNode_dequant',
                            'type': 'dequant',
                            'attr': [],
                            'input': [
                                'middle:0',
                            ],
                            'output_desc': [
                                {
                                    'dtype': 'DT_FLOAT16',
                                }
                            ]
                        },
                        {
                            'name': 'output',
                            'type': 'Out',
                            'attr': [],
                            'input': [
                                'TestNode_dequant:0'
                            ],
                            'output_desc': [
                                {
                                    'dtype': 'DT_FLOAT'
                                }
                            ]
                        }
                    ],
                },
            ]
        })

    def test_quant_compare(self):
        args = ['aaa.py', 'compare', '-m', '/home/left.bin', '-g',
                '/home/right.bin', '-f', "/home/a.json"]
        dump_data = DD.DumpData()
        dump_data.input.append(self._make_op_input(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data.output.append(self._make_op_output(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=True), \
                     mock.patch('os.access', return_value=True), \
                     mock.patch('os.remove'), \
                     mock.patch('os.listdir',
                                side_effect=[['alg_MaxAbsoluteError.py'],
                                             ['ReduceMeanD.conv1conv1_relu.6.4.1613727240764749'],
                                             ['input.4.1613727240736566.pb',
                                              'trans_Cast_1167.4.1613727241293941.pb'],
                                             ['convert_NC1HWC0_to_NCHW.py']]), \
                     mock.patch('os.path.isdir', return_value=True), \
                     mock.patch('os.path.isfile', return_value=True), \
                     mock.patch('os.path.getsize', return_value=100), \
                     mock.patch('src.compare.cmp_utils.utils.parse_dump_file', return_value=dump_data):
                    with mock.patch('builtins.open',
                                    mock.mock_open(read_data=self._make_input_json2().encode('utf-8'))):
                        print("***********start**************")
                        msaccucmp.main()
                        print("************end*************")
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR)


if __name__ == '__main__':
    unittest.main()
