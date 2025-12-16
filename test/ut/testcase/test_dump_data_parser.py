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
import struct
import sys

import unittest
import pytest
from unittest import mock
import numpy as np

from cmp_utils.constant.compare_error import CompareError
from cmp_utils.constant.const_manager import ConstManager
from dump_parse import dump_data_parser as DP
from dump_parse import dump, dump_utils, mapping
from dump_parse.proto_dump_data import DumpData, OpInput, OpOutput
from cmp_utils.constant.const_manager import DD


class TestUtilsMethods(unittest.TestCase):
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
    def _fake_arguments(dump_path='/home/CONV2.pron.1.1234567891234567'):
        arguments = mock.Mock()
        arguments.dump_path = dump_path
        arguments.dump_version = 2
        arguments.output_file_type = "npy"
        arguments.output_path = ""
        return arguments

    @staticmethod
    def _base_fake_data(op_output_data, data_type=DD.DT_FLOAT16):
        dump_data = DumpData()
        dump_data.version = '1.0'
        dump_data.dump_time = int(round(time.time() * 1000))
        op_output = dump_data.output.add()
        op_output.data_type = data_type
        op_output.format = DD.FORMAT_NCHW
        op_output.data = op_output_data

        numpy_dtype = ConstManager.DATA_TYPE_TO_DTYPE_MAP.get(data_type, {}).get('dtype', np.float32)
        numpy_dtype_size = np.dtype(numpy_dtype).itemsize  # float32 -> 4, float16 -> 2
        op_output.shape.dim.append(len(op_output_data) // numpy_dtype_size)

        dump_data = dump_utils.convert_dump_data(dump_data)
        return dump_data

    @staticmethod
    def _fake_fp16_dump_data(data_type=DD.DT_FLOAT16):
        dump_data = DumpData()
        dump_data.dump_time = int(round(time.time() * 1000))
        dump_data.version = '1.0'
        op_output = dump_data.output.add()
        op_output.data_type = data_type
        op_output.format = DD.FORMAT_NCHW
        length = 1
        for dim in [1, 3, 4, 4]:
            op_output.shape.dim.append(dim)
            length *= dim
        data_list = np.arange(length)
        origin_numpy = np.array(data_list, np.float16)
        op_output.data = struct.pack('%de' % length, *origin_numpy)
        dump_data = dump_utils.convert_dump_data(dump_data)
        return dump_data

    @staticmethod
    def _fake_uint8_dump_data():
        dump_data = DumpData()
        dump_data.version = '1.0'
        dump_data.dump_time = int(round(time.time() * 1000))
        buffer = dump_data.buffer.add()
        buffer.buffer_type = DD.L1
        buffer.size = 8
        buffer.data = struct.pack('Q', 35)

        space = dump_data.space.add()
        space.size = 8
        space.data = struct.pack('Q', 35)
        dump_data = dump_utils.convert_dump_data(dump_data)
        return dump_data

    @staticmethod
    def _base_mock_run(run_func, dump_data):
        non_error = CompareError.MSACCUCMP_NONE_ERROR
        with mock.patch('cmp_utils.path_check.check_path_valid', return_value=non_error):
            with mock.patch("cmp_utils.path_check.check_output_path_valid", return_value=non_error):
                with mock.patch('dump_parse.dump_utils.parse_dump_file', return_value=dump_data):
                    with mock.patch("os.path.isfile", return_value=True), mock.patch('os.fdopen'):
                        return run_func()

    def test_check_arguments_valid_check_path_valid(self):
        arguments = self._fake_arguments()
        with pytest.raises(CompareError) as error:
            with mock.patch('cmp_utils.path_check.check_path_valid', return_value=1):
                DP.DumpDataParser(arguments).check_arguments_valid()
        self.assertEqual(error.value.args[0], 1)

    def test_check_arguments_valid_check_output_path_valid(self):
        arguments = self._fake_arguments()
        with pytest.raises(CompareError) as error:
            with mock.patch('cmp_utils.path_check.check_path_valid', return_value=0):
                with mock.patch("cmp_utils.path_check.check_output_path_valid", return_value=1):
                    DP.DumpDataParser(arguments).check_arguments_valid()
        self.assertEqual(error.value.args[0], 1)

    def test_output_path_argument_when_is_softlink_then_raise_error(self):
        arguments = self._fake_arguments()
        with self.assertRaises(CompareError) as error:
            with mock.patch("os.path.islink", return_value=True):
                DP.DumpDataParser(arguments)
        self.assertEqual(str(error.exception), "3")

    def test_parse_dump_data_uint8_pass(self):
        arguments = self._fake_arguments()
        dump_data = self._fake_uint8_dump_data()
        with mock.patch('os.open') as open_file:
            open_file.write = None
            ret = self._base_mock_run(DP.DumpDataParser(arguments).parse_dump_data, dump_data)
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_parse_dump_data_uint8_os_error(self):
        arguments = self._fake_arguments()
        dump_data = self._fake_uint8_dump_data()
        with mock.patch('os.open', side_effect=OSError) as open_file:
            open_file.write = None
            ret = self._base_mock_run(DP.DumpDataParser(arguments).parse_dump_data, dump_data)
        self.assertEqual(ret, CompareError.MSACCUCMP_WRITE_FILE_ERROR)

    def test_parse_dump_data_fp16_pass(self):
        arguments = self._fake_arguments()
        dump_data = self._fake_fp16_dump_data()
        with mock.patch('numpy.save'):
            ret = self._base_mock_run(DP.DumpDataParser(arguments).parse_dump_data, dump_data)
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_parse_dump_data_fp16_numpy_error(self):
        arguments = self._fake_arguments()
        dump_data = self._fake_fp16_dump_data()
        with mock.patch('numpy.save', side_effect=ValueError):
            ret = self._base_mock_run(DP.DumpDataParser(arguments).parse_dump_data, dump_data)
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_parse_dump_data_fp16_msnpy_pass(self):
        arguments = self._fake_arguments()
        arguments.output_file_type = "msnpy"
        dump_data = self._fake_fp16_dump_data()
        with mock.patch('numpy.save', side_effect=ValueError):
            ret = self._base_mock_run(DP.DumpDataParser(arguments).parse_dump_data, dump_data)
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_main_parse_dump_data_bfloat16_pass(self):
        try:
            from ml_dtypes import bfloat16
        except ModuleNotFoundError:
            ml_dtypes = type(sys)('ml_dtypes')
            ml_dtypes.bfloat16 = np.float16
            sys.modules['ml_dtypes'] = ml_dtypes

        arguments = self._fake_arguments()
        dump_data = self._fake_fp16_dump_data(data_type=DD.DT_BF16)
        with mock.patch('numpy.save'):
            ret = self._base_mock_run(DP.DumpDataParser(arguments).parse_dump_data, dump_data)
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_main_parse_dump_data_with_multi_file(self):
        arguments = self._fake_arguments()
        arguments.output_file_type = "msnpy"
        dump_data = self._fake_fp16_dump_data()
        with mock.patch('numpy.save'):
            with mock.patch('os.path.getsize', return_value=1000):
                ret = self._base_mock_run(DP.DumpDataParser(arguments).parse_dump_data, dump_data)
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_save_op_debug_to_file_pass(self):
        arguments = self._fake_arguments(dump_path='/home/Opdebug.Node_OpDebug.1.1234567891234567')
        zero_bytes = self._make_uint64_data(2048)
        op_output_data = struct.pack('%dQ' % len(zero_bytes), *zero_bytes)
        dump_data = self._base_fake_data(op_output_data)
        with mock.patch('os.open') as open_file:
            open_file.write = None
            ret = self._base_mock_run(DP.DumpDataParser(arguments).parse_dump_data, dump_data)
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_save_op_debug_to_file_invalid_dump_data(self):
        arguments = self._fake_arguments(dump_path='/home/Opdebug.Node_OpDebug.1.1234567891234567')
        op_output_data = struct.pack('Q', 10)
        dump_data = self._base_fake_data(op_output_data)
        ret = self._base_mock_run(DP.DumpDataParser(arguments).parse_dump_data, dump_data)
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR)

    def test_save_op_debug_to_file_os_error(self):
        arguments = self._fake_arguments(dump_path='/home/Opdebug.Node_OpDebug.1.1234567891234567')
        zero_bytes = self._make_uint64_data(2048)
        op_output_data = struct.pack('%dQ' % len(zero_bytes), *zero_bytes)
        dump_data = self._base_fake_data(op_output_data)
        with mock.patch('os.open', side_effect=OSError) as open_file:
            open_file.write = None
            ret = self._base_mock_run(DP.DumpDataParser(arguments).parse_dump_data, dump_data)
        self.assertEqual(ret, CompareError.MSACCUCMP_WRITE_FILE_ERROR)

    def test_save_op_debug_to_file_overflow_data(self):
        arguments = self._fake_arguments(dump_path='/home/Opdebug.Node_OpDebug.1.1234567891234567')
        overflow_data = self._make_overflow_data_new_version(88)
        op_output_data = struct.pack('6i11Q', *overflow_data)
        dump_data = self._base_fake_data(op_output_data)
        with mock.patch('os.open') as open_file:
            open_file.write = None
            ret = self._base_mock_run(DP.DumpDataParser(arguments).parse_dump_data, dump_data)
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_when_parse_invalid_thread_id_then_raise_error(self):
        with mock.patch('cmp_utils.path_check.check_path_valid', return_value=CompareError.MSACCUCMP_NONE_ERROR), \
                mock.patch('dump_parse.dump_utils.parse_dump_file',
                           return_value=mock.Mock(get_ffts_mode=True, output_data=None)):
            arguments = self._fake_arguments(dump_path='/home/Opdebug.OpDebug.a.invalid_thread_id')
            dump_data = self._fake_uint8_dump_data()
            parser = DP.DumpDataParser(arguments)

            dump_path = '/home/Opdebug.OpDebug.a.invalid_thread_id'

            with pytest.raises(CompareError) as error:
                parser._parse_one_file_exec(dump_path)
            self.assertEqual(error.value.args[0],
                             CompareError.MSACCUCMP_INVALID_PATH_ERROR)
