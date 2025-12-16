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

import unittest
import struct
import pytest

import numpy as np
from unittest import mock

from cmp_utils.constant.compare_error import CompareError
from cmp_utils import common
from conversion import shape_format_conversion
from dump_parse import dump_utils
from dump_parse.proto_dump_data import DumpData, OpInput, OpOutput, Shape
from cmp_utils.constant.const_manager import DD


class TestUtilsMethods(unittest.TestCase):
    def test_process(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-format', 'FRACTAL_Z', '-o',
                '/home']
        data_str, dump_data = self._make_dump_data_ser(DD.FORMAT_HWCN, [32, 64, 3, 1],
                                                       DD.DT_FLOAT16)
        dump_data = dump_utils.convert_dump_data(dump_data)
        with mock.patch('sys.argv', args):
            with mock.patch('cmp_utils.path_check.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR), \
                    mock.patch('os.path.getsize', return_value=len(data_str)), \
                    mock.patch('os.path.isdir', return_value=False):
                with mock.patch('numpy.save'):
                    with mock.patch('dump_parse.dump_utils.parse_dump_file', return_value=dump_data):
                        main = shape_format_conversion.ShapeConversionMain()
                        ret = main.process()
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_process1(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-format', 'HWXS', '-o',
                '/home']
        with pytest.raises(CompareError) as error:
            with mock.patch('sys.argv', args):
                main = shape_format_conversion.ShapeConversionMain()
                ret = main.process()
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_process2(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-format', 'NCHW', '-o',
                '/home']
        with mock.patch('sys.argv', args):
            with mock.patch('cmp_utils.path_check.check_path_valid', side_effect=[0, 2]):
                main = shape_format_conversion.ShapeConversionMain()
                ret = main.process()
        self.assertEqual(ret, 2)

    def test_process3(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-format', 'NHWC', '-o',
                '/home']
        with mock.patch('sys.argv', args):
            with mock.patch('cmp_utils.path_check.check_path_valid', return_value=3):
                main = shape_format_conversion.ShapeConversionMain()
                ret = main.process()
        self.assertEqual(ret, 3)

    def test_process4(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-format', 'NCHW', '-o',
                '/home', '-shape', '4,5,6,7']
        with mock.patch('sys.argv', args):
            with mock.patch('cmp_utils.path_check.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR):
                with mock.patch('os.path.getsize', return_value=100):
                    with mock.patch('dump_parse.nano_dump_data.NanoDumpDataHandler.check_is_nano_dump_format',
                                    return_value=False):
                        main = shape_format_conversion.ShapeConversionMain()
                        ret = main.process()
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR)

    def test_process5(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-format', 'NCHW', '-o',
                '/home', '-shape', '4,5,5,1']
        with mock.patch('sys.argv', args):
            with mock.patch('cmp_utils.path_check.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR):
                with mock.patch('os.path.getsize', return_value=100):
                    with mock.patch('dump_parse.nano_dump_data.NanoDumpDataHandler.check_is_nano_dump_format',
                                    return_value=False):
                        main = shape_format_conversion.ShapeConversionMain()
                        ret = main.process()
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR)

    def test_process6(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-format', 'NCHW', '-o',
                '/home', '-shape', '1,3,4,4']
        dump_data = DumpData()
        dump_data.output.append(
            self._make_op_output(DD.FORMAT_NC1HWC0, [1, 2, 4, 4, 2]))
        dump_data = dump_utils.convert_dump_data(dump_data)
        with mock.patch('sys.argv', args):
            with mock.patch('cmp_utils.path_check.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR):
                with mock.patch('dump_parse.dump_utils.parse_dump_file',
                                return_value=dump_data):
                    with mock.patch('numpy.save'):
                        main = shape_format_conversion.ShapeConversionMain()
                        ret = main.process()
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_process7(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-format', 'NCHW', '-o',
                '/home']
        dump_data = DumpData()
        dump_data.output.append(
            self._make_op_output(DD.FORMAT_NC1HWC0, [1, 2, 4, 4, 2]))
        dump_data = dump_utils.convert_dump_data(dump_data)
        with mock.patch('sys.argv', args):
            with mock.patch('cmp_utils.path_check.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR):
                with mock.patch('dump_parse.dump_utils.parse_dump_file',
                                return_value=dump_data):
                    with mock.patch('numpy.save'):
                        main = shape_format_conversion.ShapeConversionMain()
                        ret = main.process()
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)


    def test_process8(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-format', 'NCHW', '-o',
                '/home', '-index', 'xx']
        with pytest.raises(CompareError) as error:
            with mock.patch('sys.argv', args):
                shape_format_conversion.ShapeConversionMain()
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_process9(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-format', 'NCHW', '-o',
                '/home', '-tensor', 'xxxx']
        with mock.patch('sys.argv', args):
            main = shape_format_conversion.ShapeConversionMain()
            ret = main.process()
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_process10(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-format', 'NCHW', '-o',
                '/home', '-index', '10', '-tensor', 'input']
        dump_data = DumpData()
        dump_data.output.append(
            self._make_op_output(DD.FORMAT_NC1HWC0, [1, 2, 4, 4, 2]))
        dump_data = dump_utils.convert_dump_data(dump_data)
        with mock.patch('sys.argv', args):
            with mock.patch('cmp_utils.path_check.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR):
                with mock.patch('dump_parse.dump_utils.parse_dump_file',
                                return_value=dump_data):
                    main = shape_format_conversion.ShapeConversionMain()
                    ret = main.process()
        self.assertEqual(ret,
                         CompareError.MSACCUCMP_INDEX_OUT_OF_BOUNDS_ERROR)

    def test_process11(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-format', 'NCHW', '-o',
                '/home']
        with mock.patch('sys.argv', args):
            with mock.patch('cmp_utils.path_check.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR):
                with mock.patch('dump_parse.dump_utils.parse_dump_file',
                                side_effect=CompareError(2)):
                    main = shape_format_conversion.ShapeConversionMain()
                    ret = main.process()
        self.assertEqual(ret, 2)

    def test_process12(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-format', 'NCHW', '-o',
                '/home', '-index', '-1', '-tensor', 'output']
        dump_data = DumpData()
        dump_data.output.append(
            self._make_op_output(DD.FORMAT_NC1HWC0, [1, 2, 4, 4, 2]))
        dump_data = dump_utils.convert_dump_data(dump_data)
        with pytest.raises(CompareError) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('cmp_utils.path_check.check_path_valid',
                                return_value=CompareError.MSACCUCMP_NONE_ERROR):
                    with mock.patch('dump_parse.dump_utils.parse_dump_file',
                                    return_value=dump_data):
                        with mock.patch('numpy.save'):
                            main = shape_format_conversion.ShapeConversionMain()
                            ret = main.process()
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_process13(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-format', 'NCHW', '-o',
                '/home']
        dump_data = DumpData()
        dump_data.output.append(
            self._make_op_output(DD.FORMAT_FRACTAL_NZ,
                                 [1, 2, 4, 4 * 16, 2 * 16]))
        dump_data = dump_utils.convert_dump_data(dump_data)
        with mock.patch('sys.argv', args):
            with mock.patch('cmp_utils.path_check.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR):
                with mock.patch('dump_parse.dump_utils.parse_dump_file',
                                return_value=dump_data):
                    with mock.patch('numpy.save'):
                        main = shape_format_conversion.ShapeConversionMain()
                        ret = main.process()
        self.assertEqual(ret,
                         CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_process14(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-format', 'NCHW', '-o',
                '/home', '-shape', '3,2']
        dump_data = DumpData()
        dump_data.output.append(
            self._make_op_output(DD.FORMAT_FRACTAL_NZ,
                                 [1, 2, 4, 4 * 16, 2 * 16]))
        dump_data = dump_utils.convert_dump_data(dump_data)
        with mock.patch('sys.argv', args):
            with mock.patch('cmp_utils.path_check.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR):
                with mock.patch('dump_parse.dump_utils.parse_dump_file',
                                return_value=dump_data):
                    with mock.patch('numpy.save'):
                        main = shape_format_conversion.ShapeConversionMain()
                        ret = main.process()
        self.assertEqual(ret,
                         CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_process15(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-format', 'NCHW', '-o',
                '/home', '-shape', '2,64,32']
        dump_data = DumpData()
        dump_data.output.append(
            self._make_op_output(DD.FORMAT_FRACTAL_NZ,
                                 [1, 1, 1, 4 * 16, 2 * 16]))
        dump_data = dump_utils.convert_dump_data(dump_data)
        with mock.patch('sys.argv', args):
            with mock.patch('cmp_utils.path_check.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR):
                with mock.patch('dump_parse.dump_utils.parse_dump_file',
                                return_value=dump_data):
                    with mock.patch('numpy.save'):
                        main = shape_format_conversion.ShapeConversionMain()
                        ret = main.process()
        self.assertEqual(ret,
                         CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_process16(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-format', 'NCHW', '-o',
                '/home', '-shape', '1,64,32']
        dump_data = DumpData()
        dump_data.output.append(
            self._make_op_output(DD.FORMAT_FRACTAL_NZ,
                                 [1, 1, 1, 4 * 16, 2 * 16]))
        dump_data = dump_utils.convert_dump_data(dump_data)
        with mock.patch('sys.argv', args):
            with mock.patch('cmp_utils.path_check.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR):
                with mock.patch('dump_parse.dump_utils.parse_dump_file',
                                return_value=dump_data):
                    with mock.patch('numpy.save'):
                        main = shape_format_conversion.ShapeConversionMain()
                        ret = main.process()
        self.assertEqual(ret,
                         CompareError.MSACCUCMP_NONE_ERROR)

    def test_process17(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-format', 'NCHW', '-o',
                '/home', '-shape', '1,64,32']
        dump_data = DumpData()
        dump_data.output.append(
            self._make_op_output(DD.FORMAT_NCHW,
                                 [1, 1, 4 * 16, 2 * 16]))
        dump_data = dump_utils.convert_dump_data(dump_data)
        with mock.patch('sys.argv', args):
            with mock.patch('cmp_utils.path_check.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR):
                with mock.patch('dump_parse.dump_utils.parse_dump_file',
                                return_value=dump_data):
                    with mock.patch('numpy.save'):
                        main = shape_format_conversion.ShapeConversionMain()
                        ret = main.process()
        self.assertEqual(ret,
                         CompareError.MSACCUCMP_NONE_ERROR)

    def test_process18(self):
        arguments = mock.Mock()
        arguments.dump_path = "/home/left.bin"
        arguments.output_path = "/home"
        arguments.dump_version = 2.0
        arguments.output_file_type = "npy"
        arguments.output = '0'
        arguments.format = "NCDHW"
        arguments.custom_script_path = ''
        arguments.shape = ""
        dump_data = DumpData()
        dump_data.output.append(
            self._make_op_output(DD.FORMAT_NDC1HWC0, [1, 4, 16, 2, 2, 16], [1, 256, 4, 2, 2]))
        dump_data = dump_utils.convert_dump_data(dump_data)
        with mock.patch('cmp_utils.path_check.check_path_valid',
                        return_value=CompareError.MSACCUCMP_NONE_ERROR):
            with mock.patch('dump_parse.dump_utils.parse_dump_file', return_value=dump_data), \
                    mock.patch('os.path.isfile', return_value=True), \
                    mock.patch('os.chmod'):
                with mock.patch('numpy.save'):
                    main = shape_format_conversion.FormatConversionMain(arguments)
                    ret = main.convert_format()
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_process19(self):
        arguments = mock.Mock()
        arguments.dump_path = "/home/left.bin"
        arguments.output_path = "/home"
        arguments.dump_version = 2.0
        arguments.output_file_type = "npy"
        arguments.output = '0'
        arguments.format = "NCHW"
        arguments.custom_script_path = ''
        arguments.shape = ""
        dump_data = DumpData()
        dump_data.output.append(
            self._make_op_output(DD.FORMAT_FRACTAL_Z, [36, 4, 16, 16], [64, 64, 3, 3]))
        dump_data = dump_utils.convert_dump_data(dump_data)
        with mock.patch('cmp_utils.path_check.check_path_valid',
                        return_value=CompareError.MSACCUCMP_NONE_ERROR):
            with mock.patch('dump_parse.dump_utils.parse_dump_file', return_value=dump_data), \
                    mock.patch('os.path.isfile', return_value=True), \
                    mock.patch('os.chmod'):
                with mock.patch('numpy.save'):
                    main = shape_format_conversion.FormatConversionMain(arguments)
                    ret = main.convert_format()
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_process20(self):
        arguments = mock.Mock()
        arguments.dump_path = "/home/left.bin"
        arguments.output_path = "/home"
        arguments.dump_version = 2.0
        arguments.output_file_type = "npy"
        arguments.output = '0'
        arguments.format = "HWCN"
        arguments.custom_script_path = ''
        arguments.shape = ""
        dump_data = DumpData()
        dump_data.output.append(
            self._make_op_output(DD.FORMAT_FRACTAL_Z, [49, 4, 16, 16], [7, 7, 3, 64]))
        dump_data = dump_utils.convert_dump_data(dump_data)
        with mock.patch('cmp_utils.path_check.check_path_valid',
                        return_value=CompareError.MSACCUCMP_NONE_ERROR):
            with mock.patch('dump_parse.dump_utils.parse_dump_file', return_value=dump_data), \
                    mock.patch('os.path.isfile', return_value=True), \
                    mock.patch('os.chmod'):
                with mock.patch('numpy.save'):
                    main = shape_format_conversion.FormatConversionMain(arguments)
                    ret = main.convert_format()
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_formatConversionMain1(self):
        arguments = mock.Mock()
        arguments.dump_path = "/home"
        arguments.output_path = "/result"
        arguments.dump_version = 2.0
        arguments.output_file_type = "npy"
        arguments.input = "test"
        arguments.output = "test"
        arguments.shape = (1, 3, 224, 224)
        arguments.format = "NCHW"
        arguments.custom_script_path = False
        with pytest.raises(CompareError) as error:
            with mock.patch("os.path.isfile", return_value=True):
                shape_format_conversion.FormatConversionMain(arguments)
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_formatConversionMain2(self):
        arguments = mock.Mock()
        arguments.dump_path = "/home"
        arguments.output_path = "/result"
        arguments.dump_version = 2.0
        arguments.output_file_type = "npy"
        arguments.input = "12"
        arguments.output = "13"
        arguments.shape = "1,3,224,224"
        arguments.format = "NCHW"
        with mock.patch("os.path.isfile", return_value=True):
            shape_format_conversion.FormatConversionMain(arguments)

    def test_formatConversionMain3(self):
        arguments = mock.Mock()
        arguments.dump_path = "/home"
        arguments.output_path = "/result"
        arguments.dump_version = 2.0
        arguments.output_file_type = "npy"
        arguments.input = "12"
        arguments.output = "13"
        arguments.shape = "1,3,224,224"
        arguments.format = "ABC"
        with pytest.raises(CompareError) as error:
            with mock.patch("os.path.isfile", return_value=True):
                shape_format_conversion.FormatConversionMain(arguments)
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_convert_format_for_one_tensor(self):
        arguments = mock.Mock()
        arguments.dump_path = "/home"
        arguments.output_path = "/result"
        arguments.dump_version = 2.0
        arguments.output_file_type = "bin"
        arguments.input = "12"
        arguments.output = "13"
        arguments.shape = "1,3,224,224"
        arguments.format = "NCHW"
        arguments.custom_script_path = "/home"
        format_main = shape_format_conversion.FormatConversionMain(arguments)
        format_main.manager = mock.Mock
        format_main.manager.execute_format_convert = mock.Mock(return_value=np.array([1, 2, 3, 4]))
        format_main._save_to_file = mock.Mock()
        tensor = mock.Mock()
        tensor.data_type = "int32"
        tensor.tensor_format = 0
        tensor.shape = [1, 2, 2, 2, 2]
        tensor.original_shape = [1, 2, 3, 4, 5]
        index = 0
        tensor_type = "input"
        dump_file_path = "/home/a.npy"
        with mock.patch("dump_parse.dump_data_object._deserialize_dump_data_to_array", return_value=[1, 2]):
            with mock.patch("cmp_utils.common.get_format_string", return_value=tensor.format):
                with mock.patch("cmp_utils.common.get_sub_format", return_value=1):
                    format_main._convert_format_for_one_tensor(tensor, index, tensor_type, dump_file_path, 'bbb')

    def test_convert_format_for_tensor1(self):
        arguments = mock.Mock()
        arguments.dump_path = "/home"
        arguments.output_path = "/result"
        arguments.dump_version = 2.0
        arguments.output_file_type = "bin"
        arguments.input = "12"
        arguments.output = "13"
        arguments.shape = "1,3,224,224"
        arguments.format = "NCHW"
        arguments.custom_script_path = "/home"
        format_main = shape_format_conversion.FormatConversionMain(arguments)
        format_main._convert_format_for_one_tensor = mock.Mock(side_effect=OSError)
        tensor_list = [None]
        dump_file_path = "/home"
        tensor_type = "input"
        ret, msg = format_main._convert_format_for_tensor(tensor_list, dump_file_path, tensor_type, 'aa')
        self.assertEqual(ret, CompareError.MSACCUCMP_UNKNOWN_ERROR)

    def test_convert_format_for_tensor2(self):
        arguments = mock.Mock()
        arguments.dump_path = "/home"
        arguments.output_path = "/result"
        arguments.dump_version = 2.0
        arguments.output_file_type = "bin"
        arguments.input = "12"
        arguments.output = "13"
        arguments.shape = "1,3,224,224"
        arguments.format = "NCHW"
        arguments.custom_script_path = "/home"
        format_main = shape_format_conversion.FormatConversionMain(arguments)
        format_main._convert_format_for_one_tensor = mock.Mock(side_effect=CompareError(1))
        tensor_list = [None]
        dump_file_path = "/home"
        tensor_type = "input"
        ret, msg = format_main._convert_format_for_tensor(tensor_list, dump_file_path, tensor_type, 'xxx')
        self.assertEqual(ret, 1)

    def test_convert_format_for_one_file1(self):
        arguments = mock.Mock()
        arguments.dump_path = "/home"
        arguments.output_path = "/result"
        arguments.dump_version = 2.0
        arguments.output_file_type = "bin"
        arguments.input = "12"
        arguments.output = "13"
        arguments.shape = "1,3,224,224"
        arguments.format = "NCHW"
        arguments.custom_script_path = "/home"
        dump_data = mock.Mock
        dump_data.input_data = [None]
        dump_data.output_data = [None]
        format_main = shape_format_conversion.FormatConversionMain(arguments)
        format_main.one_file_info = {'tensor': "input", 'index': 2, 'shape': "1,3,224,224"}
        dump_file_path = "/home"
        with mock.patch("dump_parse.dump_utils.parse_dump_file", return_value=dump_data):
            ret, msg = format_main._convert_format_for_one_file(dump_file_path)
        self.assertEqual(ret, CompareError.MSACCUCMP_INDEX_OUT_OF_BOUNDS_ERROR)

    def test_convert_format_for_one_file2(self):
        arguments = mock.Mock()
        arguments.dump_path = "/home"
        arguments.output_path = "/result"
        arguments.dump_version = 2.0
        arguments.output_file_type = "bin"
        arguments.input = "12"
        arguments.output = "13"
        arguments.shape = "1,3,224,224"
        arguments.format = "NCHW"
        arguments.custom_script_path = "/home"
        dump_data = mock.Mock
        self.build_dump_data_object(dump_data)
        dump_data.input_data = [None]
        dump_data.output_data = [None]
        format_main = shape_format_conversion.FormatConversionMain(arguments)
        format_main._convert_format_for_one_tensor = mock.Mock(side_effect=OSError)
        format_main.one_file_info = {'tensor': "input", 'index': 0, 'shape': "1,3,224,224"}
        dump_file_path = "/home"
        with mock.patch("dump_parse.dump_utils.parse_dump_file", return_value=dump_data):
            ret, msg = format_main._convert_format_for_one_file(dump_file_path)
        self.assertEqual(ret, CompareError.MSACCUCMP_UNKNOWN_ERROR)

    def test_convert_format_for_one_file3(self):
        arguments = mock.Mock()
        arguments.dump_path = "/home"
        arguments.output_path = "/result"
        arguments.dump_version = 2.0
        arguments.output_file_type = "bin"
        arguments.input = "12"
        arguments.output = "13"
        arguments.shape = "1,3,224,224"
        arguments.format = "NCHW"
        arguments.custom_script_path = "/home"
        dump_data = mock.Mock
        self.build_dump_data_object(dump_data)
        dump_data.input_data = [None]
        dump_data.output_data = [None]
        format_main = shape_format_conversion.FormatConversionMain(arguments)
        format_main._convert_format_for_tensor = mock.Mock(return_value=[1, "/demo"])
        format_main.one_file_info = {'tensor': None, 'index': 0, 'shape': "1,3,224,224"}
        dump_file_path = "/home"
        with mock.patch("dump_parse.dump_utils.parse_dump_file", return_value=dump_data):
            ret, msg = format_main._convert_format_for_one_file(dump_file_path)
        self.assertEqual(ret, 1)

    def test_convert_format(self):
        arguments = mock.Mock()
        arguments.dump_path = "/home"
        arguments.output_path = "/result"
        arguments.dump_version = 2.0
        arguments.output_file_type = "bin"
        arguments.input = "12"
        arguments.output = "13"
        arguments.shape = "1,3,224,224"
        arguments.format = "NCHW"
        arguments.custom_script_path = "/home"
        format_main = shape_format_conversion.FormatConversionMain(arguments)
        format_main.check_arguments_valid = mock.Mock()
        format_main.input_path = ['xxx/aaa']
        format_main._convert_format_for_one_file = mock.Mock(return_value=[1, "ok"])
        with mock.patch("os.path.isfile", return_value=True):
            ret = format_main.convert_format()
        self.assertEqual(ret, 1)

    @staticmethod
    def _make_numpy_array(shape_from):
        count = 1
        for dim in shape_from:
            count *= dim
        return np.arange(count).flatten()

    @staticmethod
    def _make_shape(dim_list):
        shape = Shape()
        for dim in dim_list:
            shape.dim.append(dim)
        return shape

    @staticmethod
    def _make_op_output(dd_format, shape, dim_list=None):
        op_output = OpOutput()
        op_output.data_type = DD.DT_FLOAT16
        op_output.format = dd_format
        if dim_list:
            for dim in dim_list:
                op_output.original_shape.dim.append(dim)
        length = 1
        for dim in shape:
            op_output.shape.dim.append(dim)
            length *= dim
        data_list = np.arange(length)
        origin_numpy = np.array(data_list, np.float16)
        op_output.data = struct.pack('e' * length, *origin_numpy)
        return op_output

    @staticmethod
    def _make_dump_data_ser(dd_format, shape, data_type):
        dump_data = DumpData()
        op_output = OpOutput()
        op_output.data_type = data_type
        op_output.format = dd_format
        length = 1
        for dim in shape:
            op_output.shape.dim.append(dim)
            length *= dim
        data_list = np.arange(length)
        origin_numpy = np.array(data_list,
                                common.get_dtype_by_data_type(data_type))
        op_output.data = struct.pack(
            common.get_struct_format_by_data_type(data_type) * length,
            *origin_numpy)
        dump_data.output.append(op_output)

        op_input = OpInput()
        op_input.data_type = data_type
        op_input.format = dd_format
        length = 1
        for dim in shape:
            op_input.shape.dim.append(dim)
            length *= dim
        data_list = np.arange(length)
        origin_numpy = np.array(data_list,
                                common.get_dtype_by_data_type(data_type))
        op_input.data = struct.pack(
            common.get_struct_format_by_data_type(data_type) * length,
            *origin_numpy)
        dump_data.input.append(op_input)
        data_str = dump_data.SerializeToString()
        return data_str, dump_data

    @staticmethod
    def build_dump_data_object(dump_data):
        dump_data.version = None
        dump_data.op_name = None
        dump_data.dump_time = None
        dump_data.buffer = None
        dump_data.attr = None
        dump_data.input = []
        dump_data.output = []


if __name__ == '__main__':
    unittest.main()
