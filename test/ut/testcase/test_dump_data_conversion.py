import time
import struct

import unittest
from unittest import mock
import numpy as np
import pytest
import dump_data_pb2 as DD

from src.compare.cmp_utils.constant.compare_error import CompareError
from src.compare.dump_parse import dump_data_conversion


class TestUtilsMethods(unittest.TestCase):

    def test_check_arguments_valid1(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-target', 'xxx', '-o',
                '/home', '-type', 'tf']
        with mock.patch('sys.argv', args):
            with mock.patch('src.compare.cmp_utils.utils.check_path_valid',
                            return_value=CompareError.MSACCUCMP_INVALID_PATH_ERROR):
                main = dump_data_conversion.DumpDataConversion()
                ret = main.check_arguments_valid()
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_PATH_ERROR)

    def test_check_arguments_valid2(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-target', 'xxx', '-o',
                '/home', '-type', 'tf']
        with mock.patch('sys.argv', args):
            with mock.patch('src.compare.cmp_utils.utils.check_path_valid',
                            side_effect=[CompareError.MSACCUCMP_NONE_ERROR,
                                         CompareError.MSACCUCMP_INVALID_SHAPE_ERROR]):
                main = dump_data_conversion.DumpDataConversion()
                ret = main.check_arguments_valid()
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_SHAPE_ERROR)

    def test_check_arguments_valid3(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-target', 'xxx', '-o',
                '/home', '-type', 'tf']
        with mock.patch('sys.argv', args):
            with mock.patch('src.compare.cmp_utils.utils.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR):
                main = dump_data_conversion.DumpDataConversion()
                ret = main.check_arguments_valid()
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_TARGET_ERROR)

    def test_check_arguments_valid4(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-target', 'numpy', '-o',
                '/home', '-type', 'tfe']
        with mock.patch('sys.argv', args):
            with mock.patch('src.compare.cmp_utils.utils.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR):
                main = dump_data_conversion.DumpDataConversion()
                ret = main.check_arguments_valid()
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_TYPE_ERROR)

    def test_check_arguments_valid5(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-target', 'numpy', '-o',
                '/home', '-type', 'tf']
        with mock.patch('sys.argv', args):
            with mock.patch('src.compare.cmp_utils.utils.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR):
                with mock.patch('os.path.exists', return_value=True):
                    with mock.patch('os.remove'):
                        main = dump_data_conversion.DumpDataConversion()
                        ret = main.check_arguments_valid()
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_convert_data1(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-target', 'numpy', '-o',
                '/home', '-type', 'tf']
        with mock.patch('sys.argv', args):
            main = dump_data_conversion.DumpDataConversion()
            main.check_arguments_valid = mock.Mock(return_value=CompareError.MSACCUCMP_UNKNOWN_ERROR)
            ret = main.convert_data()
        self.assertEqual(ret, 1)

    def test_convert_data2(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-target', 'numpy', '-o',
                '/home', '-type', 'tf']
        with mock.patch('sys.argv', args):
            with mock.patch("os.path.isfile", return_value=True):
                main = dump_data_conversion.DumpDataConversion()
                main.check_arguments_valid = mock.Mock(return_value=CompareError.MSACCUCMP_NONE_ERROR)
                main._convert_file = mock.Mock(return_value=[CompareError.MSACCUCMP_NONE_ERROR, "ok"])
                ret = main.convert_data()
        self.assertEqual(ret, 0)

    def test_convert_data3(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-target', 'numpy', '-o',
                '/home', '-type', 'tf']
        with mock.patch('sys.argv', args):
            with mock.patch("os.path.isfile", return_value=False):
                main = dump_data_conversion.DumpDataConversion()
                main.check_arguments_valid = mock.Mock(return_value=CompareError.MSACCUCMP_NONE_ERROR)
                main.multi_process = mock.Mock()
                main.multi_process.process = mock.Mock(return_value=[0, 'ok'])
                main.convert_data()

    def test_convert_file1(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-target', 'dump', '-o',
                '/home', '-type', 'tf']
        dump_data = DD.DumpData()
        dump_data.version = '2.0'
        dump_data.dump_time = int(round(time.time() * 1000))
        buffer = dump_data.buffer.add()
        buffer.buffer_type = DD.L1
        buffer.size = 8
        dump_data_ser = dump_data.SerializeToString()
        struct_format = 'Q' + str(len(dump_data_ser)) + 'sQ'
        data = struct.pack(
            struct_format, len(dump_data_ser), dump_data_ser, 88)
        with mock.patch('sys.argv', args):
            with mock.patch("src.compare.cmp_utils.utils.read_numpy_file", return_value=np.array([1, 2, 3, 4])):
                with mock.patch('os.open') as open_file, \
                        mock.patch('os.fdopen'):
                    with mock.patch('builtins.open',
                                    mock.mock_open(read_data=data)):
                        open_file.write = None
                        main = dump_data_conversion.DumpDataConversion()
                        main._get_op_name_from_path = mock.Mock(return_value="demo")
                        return_code, input_path = main._convert_file("/home/left.bin")
        self.assertEqual(return_code, 0)

    def test_convert_file2(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-target', 'numpy', '-o',
                '/home', '-type', 'tf']
        dump_data = mock.Mock
        dump_data.input_data = "input_test"
        dump_data.output_data = 'output_test'
        dump_data.buffer = "buffer_test"
        with mock.patch('sys.argv', args):
            with mock.patch("src.compare.cmp_utils.utils.parse_dump_file", return_value=dump_data):
                main = dump_data_conversion.DumpDataConversion()
                main._save_tensor_to_file = mock.Mock()
                main._save_buffer_to_file = mock.Mock()
                main._get_op_name_from_path = mock.Mock(return_value="demo")
                return_code, input_path = main._convert_file("/home/left.bin")
        self.assertEqual(return_code, 0)

    def test_convert_file3(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-target', 'numpy', '-o',
                '/home', '-type', 'tf']
        dump_data = mock.Mock
        dump_data.input_data = "input_test"
        dump_data.output_data = 'output_test'
        dump_data.buffer = "buffer_test"
        with mock.patch('sys.argv', args):
            with mock.patch("src.compare.cmp_utils.utils.parse_dump_file", return_value=dump_data):
                main = dump_data_conversion.DumpDataConversion()
                main._save_tensor_to_file = mock.Mock()
                main._save_buffer_to_file = mock.Mock(
                    side_effect=CompareError(CompareError.MSACCUCMP_INVALID_PATH_ERROR))
                main._get_op_name_from_path = mock.Mock(return_value="demo")
                return_code, input_path = main._convert_file("/home/left.bin")
        self.assertEqual(return_code, CompareError.MSACCUCMP_INVALID_PATH_ERROR)

    def test_convert_file4(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-target', 'numpy', '-o',
                '/home', '-type', 'tf']
        dump_data = mock.Mock
        dump_data.input_data = "input_test"
        dump_data.output_data = 'output_test'
        dump_data.buffer = "buffer_test"
        with mock.patch('sys.argv', args):
            with mock.patch("src.compare.cmp_utils.utils.parse_dump_file", return_value=dump_data):
                main = dump_data_conversion.DumpDataConversion()
                main._save_tensor_to_file = mock.Mock()
                main._save_buffer_to_file = mock.Mock(side_effect=MemoryError)
                main._get_op_name_from_path = mock.Mock(return_value="demo")
                return_code, input_path = main._convert_file("/home/left.bin")
        self.assertEqual(return_code, 1)

    def test_save_buffer_to_file1(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-target', 'numpy', '-o',
                '/home', '-type', 'tf']
        with mock.patch('sys.argv', args):
            main = dump_data_conversion.DumpDataConversion()
            tensor_list = []
            name = "demo"
            input_path = "/home/left/bin"
            main._save_buffer_to_file(tensor_list, name, input_path)

    def test_save_buffer_to_file2(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-target', 'numpy', '-o',
                '/home', '-type', 'tf']
        with mock.patch('sys.argv', args):
            main = dump_data_conversion.DumpDataConversion()
            tensor = mock.Mock()
            tensor.buffer_type = "DD.L1"
            tensor.data = b'\x01\x02'
            tensor_list = [tensor]
            name = "demo"
            input_path = "/home/left/bin"
            main._save_buffer_to_file(tensor_list, name, input_path)

    def test_save_buffer_to_file3(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-target', 'numpy', '-o',
                '/home', '-type', 'tf']
        with mock.patch('sys.argv', args):
            main = dump_data_conversion.DumpDataConversion()
            tensor = mock.Mock()
            tensor.buffer_type = "DD.L1"
            tensor.data = b'\x01\x02'
            tensor_list = [tensor]
            name = "demo"
            input_path = "/home/left/bin"
            with mock.patch("os.open", side_effect=IOError):
                main._save_buffer_to_file(tensor_list, name, input_path)

    def test_save_tensor_to_file1(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-target', 'numpy', '-o',
                '/home', '-type', 'tf']
        with mock.patch('sys.argv', args):
            main = dump_data_conversion.DumpDataConversion()
            input_path = "/home/demo"
            tensor_list = []
            name = "demo"
            tensor_type = "input"
            main._save_tensor_to_file(tensor_list, name, input_path, tensor_type)

    def test_save_tensor_to_file2(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-target', 'numpy', '-o',
                '/home', '-type', 'tf']
        with mock.patch('sys.argv', args):
            main = dump_data_conversion.DumpDataConversion()
            tensor = mock.Mock()
            tensor.shape = [1, 2, 3, 4]
            input_path = "/home/demo"
            tensor_list = [tensor]
            name = "demo"
            tensor_type = "input"
            tensor.data = np.zeros((2, 3, 4))
            with mock.patch("numpy.save"):
                main._save_tensor_to_file(tensor_list, name, input_path, tensor_type)

    def test_save_tensor_to_file3(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-target', 'numpy', '-o',
                '/home', '-type', 'tf']
        with mock.patch('sys.argv', args):
            main = dump_data_conversion.DumpDataConversion()
            tensor = mock.Mock()
            tensor.shape = [1, 2, 3, 4]
            input_path = "/home/demo"
            tensor_list = [tensor]
            name = "demo"
            tensor_type = "input"
            tensor.data = np.array([1, 2, 3, 4])
            main._save_tensor_to_file(tensor_list, name, input_path, tensor_type)

    def test_get_offline_layer_name1(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-target', 'numpy', '-o',
                '/home', '-type', 'tf']
        with mock.patch('sys.argv', args):
            main = dump_data_conversion.DumpDataConversion()
            layer_name = "convert_2d"
            with pytest.raises(CompareError) as error:
                main._get_offline_layer_name(layer_name)
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_DUMP_FILE_ERROR)

    def test_get_offline_layer_name2(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-target', 'dump', '-o',
                '/home', '-type', 'tf']
        with mock.patch('sys.argv', args):
            main = dump_data_conversion.DumpDataConversion()
            layer_name = "convert_2d"
            with pytest.raises(CompareError) as error:
                main._get_offline_layer_name(layer_name)
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_DUMP_FILE_ERROR)

    def test_get_quant_layer_name1(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-target', 'numpy', '-o',
                '/home', '-type', 'tf']
        with mock.patch('sys.argv', args):
            main = dump_data_conversion.DumpDataConversion()
            layer_name = "convert_2d"
            with pytest.raises(CompareError) as error:
                main._get_quant_layer_name(layer_name)
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_DUMP_FILE_ERROR)

    def test_get_quant_layer_name2(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-target', 'dump', '-o',
                '/home', '-type', 'tf']
        with mock.patch('sys.argv', args):
            main = dump_data_conversion.DumpDataConversion()
            layer_name = "convert_2d"
            with pytest.raises(CompareError) as error:
                main._get_quant_layer_name(layer_name)
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_DUMP_FILE_ERROR)

    def test_get_standard_layer_name1(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-target', 'numpy', '-o',
                '/home', '-type', 'tf']
        with mock.patch('sys.argv', args):
            main = dump_data_conversion.DumpDataConversion()
            layer_name = "convert_2d"
            with pytest.raises(CompareError) as error:
                main._get_standard_layer_name(layer_name)
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_DUMP_FILE_ERROR)

    def test_get_standard_layer_name2(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-target', 'dump', '-o',
                '/home', '-type', 'tf']
        with mock.patch('sys.argv', args):
            main = dump_data_conversion.DumpDataConversion()
            layer_name = "convert_2d"
            with pytest.raises(CompareError) as error:
                main._get_standard_layer_name(layer_name)
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_DUMP_FILE_ERROR)

    @staticmethod
    def _make_op_output(dd_format, shape):
        op_output = DD.OpOutput()
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



if __name__ == '__main__':
    unittest.main()
