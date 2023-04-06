import unittest

import struct
import numpy as np
import time
from src.compare.dump_parse import dump_data_conversion
from src.compare.cmp_utils import utils
from src.compare.cmp_utils import common
import dump_data_pb2 as DD
from src.compare.cmp_utils.constant.compare_error import CompareError
from unittest import mock


class TestUtilsMethods(unittest.TestCase):

    def test_convert_data1(self):
        args = ['aaa.py', '-i', '/home/pron.pb', '-target', 'numpy', '-o',
                '/home', '-type', 'tf']
        with mock.patch('sys.argv', args):
            with mock.patch('os.path.exists', return_value=False):
                main = dump_data_conversion.DumpDataConversion()
                ret = main.convert_data()
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_PATH_ERROR)

    def test_convert_data2(self):
        args = ['aaa.py', '-i', '/home/pron.xxx.0.txt', '-target', 'dump', '-o',
                '/home', '-type', 'tf']
        with mock.patch('os.path.exists', return_value=True), \
             mock.patch('os.access', side_effect=[True, False]):
            with mock.patch('sys.argv', args):
                main = dump_data_conversion.DumpDataConversion()
                ret = main.convert_data()
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_PATH_ERROR)

    def test_convert_data3(self):
        args = ['aaa.py', '-i', '/home/pron.xxx.0.txt', '-target', 'xxx', '-o',
                '/home', '-type', 'tf']
        with mock.patch('os.path.exists', return_value=True), \
             mock.patch('os.access', return_value=True):
            with mock.patch('sys.argv', args):
                main = dump_data_conversion.DumpDataConversion()
                ret = main.convert_data()
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_TARGET_ERROR)

    def test_convert_data4(self):
        args = ['aaa.py', '-i', '/home/pron.xxx.0.txt', '-target', 'dump', '-o',
                '/home', '-type', 'xx']
        with mock.patch('os.path.exists', return_value=True), \
             mock.patch('os.access', return_value=True):
            with mock.patch('sys.argv', args):
                main = dump_data_conversion.DumpDataConversion()
                ret = main.convert_data()
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_TYPE_ERROR)

    def test_convert_data5(self):
        args = ['aaa.py', '-i', '/home/pron.pb', '-target', 'numpy', '-o',
                '/home', '-type', 'sim']
        with mock.patch('os.path.exists', return_value=True), \
             mock.patch('os.remove'), \
             mock.patch('os.path.isfile', return_value=True), \
             mock.patch('os.access', return_value=True):
            with mock.patch('sys.argv', args):
                main = dump_data_conversion.DumpDataConversion()
                ret = main.convert_data()
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_TYPE_ERROR)

    def test_convert_data6(self):
        args = ['aaa.py', '-i', '/home/pron.xxx.0.txt', '-target', 'dump', '-o',
                '/home', '-type', 'sim']
        with mock.patch('os.path.exists', return_value=True), \
             mock.patch('os.remove'), \
             mock.patch('os.path.isfile', return_value=True), \
             mock.patch('os.access', return_value=True):
            with mock.patch('sys.argv', args):
                main = dump_data_conversion.DumpDataConversion()
                ret = main.convert_data()
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_TYPE_ERROR)

    def test_convert_data7(self):
        args = ['aaa.py', '-i', '/home/pron.pb', '-target', 'numpy', '-o',
                '/home', '-type', 'offline']
        with mock.patch('os.path.exists', return_value=True), \
             mock.patch('os.remove'), \
             mock.patch('os.path.isfile', return_value=True), \
             mock.patch('os.access', return_value=True):
            with mock.patch('sys.argv', args):
                main = dump_data_conversion.DumpDataConversion()
                ret = main.convert_data()
        self.assertEqual(ret, CompareError.MSACCUCMP_DUMP_FILE_ERROR)

    def test_convert_data8(self):
        args = ['aaa.py', '-i', '/home/pron.xxx.0.txt', '-target', 'dump', '-o',
                '/home', '-type', 'offline']
        with mock.patch('os.path.exists', return_value=True), \
             mock.patch('os.remove'), \
             mock.patch('os.path.isfile', return_value=True), \
             mock.patch('os.access', return_value=True):
            with mock.patch('sys.argv', args):
                main = dump_data_conversion.DumpDataConversion()
                ret = main.convert_data()
        self.assertEqual(ret, CompareError.MSACCUCMP_DUMP_FILE_ERROR)

    def test_convert_data9(self):
        args = ['aaa.py', '-i', '/home/pron.pb', '-target', 'numpy', '-o',
                '/home', '-type', 'quant']
        with mock.patch('os.path.exists', return_value=True), \
             mock.patch('os.remove'), \
             mock.patch('os.path.isfile', return_value=True), \
             mock.patch('os.access', return_value=True):
            with mock.patch('sys.argv', args):
                main = dump_data_conversion.DumpDataConversion()
                ret = main.convert_data()
        self.assertEqual(ret, CompareError.MSACCUCMP_DUMP_FILE_ERROR)

    def test_convert_data10(self):
        args = ['aaa.py', '-i', '/home/pron.xxx.0.txt', '-target', 'dump', '-o',
                '/home', '-type', 'quant']
        with mock.patch('os.path.exists', return_value=True), \
             mock.patch('os.remove'), \
             mock.patch('os.path.isfile', return_value=True), \
             mock.patch('os.access', return_value=True):
            with mock.patch('sys.argv', args):
                main = dump_data_conversion.DumpDataConversion()
                ret = main.convert_data()
        self.assertEqual(ret, CompareError.MSACCUCMP_DUMP_FILE_ERROR)

    def test_convert_data11(self):
        args = ['aaa.py', '-i', '/home/pron.0.1234567891234567.quant',
                '-target', 'numpy', '-o', '/home', '-type', 'quant']
        with mock.patch('os.path.exists', return_value=True), \
             mock.patch('os.remove'), \
             mock.patch('os.path.isfile', return_value=True), \
             mock.patch('os.access', return_value=True), \
             mock.patch('os.path.getsize', return_value=100):
            with mock.patch('sys.argv', args):
                with mock.patch('builtins.open',
                                side_effect=IOError('not found')):
                    main = dump_data_conversion.DumpDataConversion()
                    ret = main.convert_data()
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR)

    def test_convert_data12(self):
        args = ['aaa.py', '-i', '/home/pron.0.1234567891234567.pb',
                '-target', 'numpy', '-o', '/home', '-type', 'caffe']
        data_str = self._make_dump_data_ser(DD.FORMAT_NCHW, [1, 100],
                                            DD.DT_FLOAT16)
        with mock.patch('os.path.exists', return_value=True), \
             mock.patch('os.remove'), \
             mock.patch('os.path.isfile', return_value=True), \
             mock.patch('os.path.getsize', return_value=len(data_str)), \
             mock.patch('os.access', return_value=True):
            with mock.patch('sys.argv', args):
                with mock.patch('builtins.open',
                                mock.mock_open(read_data=data_str)):
                    main = dump_data_conversion.DumpDataConversion()
                    ret = main.convert_data()
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_convert_data13(self):
        args = ['aaa.py', '-i', '/home/pron.0.1234567891234567.dump',
                '-target', 'numpy', '-o', '/home', '-type', 'sim']
        data_str = self._make_dump_data_ser(DD.FORMAT_NCHW, [1, 100],
                                            DD.DT_FLOAT)
        with mock.patch('os.path.exists', return_value=True), \
             mock.patch('os.remove'), \
             mock.patch('os.path.isfile', return_value=True), \
             mock.patch('os.path.getsize', return_value=len(data_str)), \
             mock.patch('os.access', return_value=True):
            with mock.patch('sys.argv', args):
                with mock.patch('builtins.open',
                                mock.mock_open(read_data=data_str)):
                    main = dump_data_conversion.DumpDataConversion()
                    ret = main.convert_data()
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_TYPE_ERROR)

    def test_convert_data14(self):
        args = ['aaa.py', '-i', '/home/CONV2.pron.1.1234567891234567',
                '-target', 'numpy', '-o', '/home', '-type', 'offline']
        data_str = self._make_dump_data_ser(DD.FORMAT_NCHW, [1, 100],
                                            DD.DT_DOUBLE)
        with mock.patch('os.path.exists', return_value=True), \
             mock.patch('os.remove'), \
             mock.patch('os.path.isfile', return_value=True), \
             mock.patch('os.path.getsize', return_value=len(data_str)), \
             mock.patch('os.access', return_value=True):
            with mock.patch('sys.argv', args):
                with mock.patch('builtins.open',
                                mock.mock_open(read_data=data_str)):
                    main = dump_data_conversion.DumpDataConversion()
                    ret = main.convert_data()
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_convert_data15(self):
        args = ['aaa.py', '-i', '/home/pron.pb', '-target', 'numpy', '-o',
                '/home', '-type', 'tf']
        with mock.patch('os.path.exists', return_value=True), \
             mock.patch('os.remove'), \
             mock.patch('os.path.isfile', return_value=True), \
             mock.patch('os.access', return_value=True):
            with mock.patch('sys.argv', args):
                main = dump_data_conversion.DumpDataConversion()
                ret = main.convert_data()
        self.assertEqual(ret, CompareError.MSACCUCMP_DUMP_FILE_ERROR)

    def test_convert_data16(self):
        args = ['aaa.py', '-i', '/home/pron.1.pb', '-target', 'dump', '-o',
                '/home', '-type', 'tf']
        with mock.patch('os.path.exists', return_value=True), \
             mock.patch('os.remove'), \
             mock.patch('os.path.isfile', return_value=True), \
             mock.patch('os.access', return_value=True):
            with mock.patch('sys.argv', args):
                main = dump_data_conversion.DumpDataConversion()
                ret = main.convert_data()
        self.assertEqual(ret, CompareError.MSACCUCMP_DUMP_FILE_ERROR)

    def test_convert_data17(self):
        args = ['aaa.py', '-i', '/home/pron.0.1234567891234567.npy',
                '-target', 'dump', '-o', '/home', '-type', 'quant']
        with mock.patch('os.path.exists', return_value=True), \
             mock.patch('os.remove'), \
             mock.patch('os.path.isfile', return_value=True), \
             mock.patch('os.access', return_value=True), \
             mock.patch('os.path.getsize', return_value=100):
            with mock.patch('sys.argv', args):
                with mock.patch('os.open') as open_file, \
                        mock.patch('os.fdopen'):
                    with mock.patch('numpy.load', return_value=np.arange(100,
                                                                         dtype=np.float)):
                        open_file.write = None
                        main = dump_data_conversion.DumpDataConversion()
                        ret = main.convert_data()
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_convert_data18(self):
        args = ['aaa.py', '-i', '/home/CONV2.pron.1.1234567891234567.0.npy',
                '-target', 'dump', '-o', '/home', '-type', 'offline']
        with mock.patch('os.path.exists', return_value=True), \
             mock.patch('os.remove'), \
             mock.patch('os.path.isfile', return_value=True), \
             mock.patch('os.access', return_value=True), \
             mock.patch('os.path.getsize', return_value=100):
            with mock.patch('sys.argv', args):
                with mock.patch('os.open') as open_file, \
                        mock.patch('os.fdopen'):
                    with mock.patch('numpy.load',
                                    return_value=np.arange(100,
                                                           dtype=np.float16)):
                        open_file.write = None
                        main = dump_data_conversion.DumpDataConversion()
                        ret = main.convert_data()
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_convert_data19(self):
        args = ['aaa.py', '-i', '/home/pron.0.1234567891234567.npy',
                '-target', 'dump', '-o', '/home', '-type', 'tf']
        with mock.patch('os.path.exists', return_value=True), \
             mock.patch('os.remove'), \
             mock.patch('os.path.isfile', return_value=True), \
             mock.patch('os.access', return_value=True), \
             mock.patch('os.path.getsize', return_value=100):
            with mock.patch('sys.argv', args):
                with mock.patch('os.open') as open_file, \
                        mock.patch('os.fdopen'):
                    with mock.patch('numpy.load',
                                    return_value=np.arange(100,
                                                           dtype=np.int32)):
                        open_file.write = None
                        main = dump_data_conversion.DumpDataConversion()
                        ret = main.convert_data()
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_convert_data20(self):
        args = ['aaa.py', '-i', '/home/pron.0.1234567891234567.npy',
                '-target', 'dump', '-o', '/home', '-type', 'sim']
        with mock.patch('os.path.exists', return_value=True), \
             mock.patch('os.remove'), \
             mock.patch('os.path.isfile', return_value=True), \
             mock.patch('os.access', return_value=True), \
             mock.patch('os.path.getsize', return_value=100):
            with mock.patch('sys.argv', args):
                with mock.patch('os.open') as open_file, \
                        mock.patch('os.fdopen'):
                    with mock.patch('numpy.load',
                                    return_value=np.arange(100,
                                                           dtype=np.int8)):
                        open_file.write = None
                        main = dump_data_conversion.DumpDataConversion()
                        ret = main.convert_data()
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_TYPE_ERROR)

    def test_convert_data21(self):
        args = ['aaa.py', '-i', '/home/pron.0.1234567891234567.npy',
                '-target', 'dump', '-o', '/home', '-type', 'caffe']
        with mock.patch('os.path.exists', return_value=True), \
             mock.patch('os.remove'), \
             mock.patch('os.path.isfile', return_value=True), \
             mock.patch('os.access', return_value=True), \
             mock.patch('os.path.getsize', return_value=100):
            with mock.patch('sys.argv', args):
                with mock.patch('os.open') as open_file, \
                        mock.patch('os.fdopen'):
                    with mock.patch('numpy.load',
                                    return_value=np.arange(100,
                                                           dtype=np.uint8)):
                        open_file.write = None
                        main = dump_data_conversion.DumpDataConversion()
                        ret = main.convert_data()
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_convert_data_buffer_data1(self):
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
        args = ['aaa.py', '-i', '/home/CONV2.pron.1.1234567891234567',
                '-target', 'numpy', '-o',
                '/home', '-type', 'offline']
        with mock.patch('os.path.exists', return_value=True), \
             mock.patch('os.remove'), \
             mock.patch('os.path.isfile', return_value=True), \
             mock.patch('os.path.getsize', return_value=len(data)), \
             mock.patch('os.access', return_value=True):
            with mock.patch('sys.argv', args):
                with mock.patch('os.open') as open_file, \
                        mock.patch('os.fdopen'):
                    with mock.patch('builtins.open',
                                    mock.mock_open(read_data=data)):
                        open_file.write = None
                        main = dump_data_conversion.DumpDataConversion()
                        ret = main.convert_data()
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_convert_data23(self):
        args = ['aaa.py', '-i', '/home',
                '-target', 'numpy', '-o', '/home', '-type', 'sim']
        with mock.patch('os.path.exists', return_value=True), \
             mock.patch('os.remove'), \
             mock.patch('os.listdir',
                        return_value=[
                            'a.0.1111111111111111.dump',
                            'prob.0.1111111111111111.dump']), \
             mock.patch('os.path.getsize', return_value=10), \
             mock.patch('os.access', return_value=True):
            with mock.patch('sys.argv', args):
                with mock.patch('os.open') as open_file, \
                        mock.patch('os.fdopen'):
                    with mock.patch('numpy.load',
                                    return_value=np.arange(100,
                                                           dtype=np.uint8)):
                        with mock.patch('numpy.save'):
                            with mock.patch('os.path.isfile',
                                            side_effect=[False, True,
                                                         True]):
                                open_file.write = None
                                main = dump_data_conversion.DumpDataConversion()
                                ret = main.convert_data()
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_TYPE_ERROR)

    @staticmethod
    def _make_dump_data_ser(dd_format, shape, data_type):
        dump_data = DD.DumpData()
        op_output = DD.OpOutput()
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

        op_input = DD.OpInput()
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
        return data_str


if __name__ == '__main__':
    unittest.main()
