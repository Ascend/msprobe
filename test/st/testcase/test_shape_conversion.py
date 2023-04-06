import unittest

import struct
import pytest
import numpy as np
from src.compare.cmp_utils import utils
from src.compare.conversion import shape_conversion
import dump_data_pb2 as DD
from unittest import mock
from src.compare.cmp_utils.constant.compare_error import CompareError
from cmp_utils import common


class TestUtilsMethods(unittest.TestCase):

    def test_process1(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-format', 'HWXS', '-o',
                '/home']
        with pytest.raises(CompareError) as error:
            with mock.patch('sys.argv', args):
                main = shape_conversion.ShapeConversionMain()
                main.process()
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_process2(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-format', 'NCHW', '-o',
                '/home']
        with pytest.raises(CompareError) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=False):
                    main = shape_conversion.ShapeConversionMain()
                    main.process()
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_CONVERT_FUNC_ERROR)

    def test_process3(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-format', 'NHWC', '-o',
                '/home']
        with mock.patch('sys.argv', args):
            with mock.patch('os.path.exists', return_value=True):
                with mock.patch('os.access', return_value=True):
                    with mock.patch('os.path.isfile', return_value=True):
                        with mock.patch('os.path.isdir', return_value=False):
                            main = shape_conversion.ShapeConversionMain()
                            ret = main.process()
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_PATH_ERROR)

    def test_process4(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-format', 'NCHW', '-o',
                '/home', '-shape', '4,5,x']
        with pytest.raises(CompareError) as error:
            with mock.patch('sys.argv', args):
                main = shape_conversion.ShapeConversionMain()
                main.process()
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_SHAPE_ERROR)

    def test_process5(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-format', 'NCHW', '-o',
                '/home', '-shape', '4,5,0']
        with pytest.raises(CompareError) as error:
            with mock.patch('sys.argv', args):
                main = shape_conversion.ShapeConversionMain()
                ret = main.process()
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_SHAPE_ERROR)

    def test_process6(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-format', 'NCHW', '-o',
                '/home', '-index', 'xx']
        with pytest.raises(CompareError) as error:
            with mock.patch('sys.argv', args):
                shape_conversion.ShapeConversionMain()
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_process7(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-format', 'NCHW', '-o',
                '/home', '-tensor', 'xxxx']
        with mock.patch('sys.argv', args):
            main = shape_conversion.ShapeConversionMain()
            ret = main.process()
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_process8(self):
        args = ['aaa.py', '-i', '/home/lef?XX&*t.bin', '-format', 'NCHW', '-o',
                '/home']
        with mock.patch('sys.argv', args):
            main = shape_conversion.ShapeConversionMain()
            ret = main.process()
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_process9(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-format', 'NCHW', '-o',
                '/home']
        data_str = self._make_dump_data_ser(DD.FORMAT_NC1HWC0, [1, 2, 4, 4, 2],
                                            DD.DT_INT64)
        with mock.patch('sys.argv', args):
            with mock.patch('src.compare.cmp_utils.utils.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR), \
                 mock.patch('os.path.getsize', return_value=len(data_str)), \
                 mock.patch('os.path.isdir', return_value=False):
                with mock.patch('numpy.save'):
                    with mock.patch('builtins.open',
                                    mock.mock_open(read_data=data_str)):
                        main = shape_conversion.ShapeConversionMain()
                        ret = main.process()
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_process10(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-format', 'NCHW', '-o',
                '/home', '-index', '10', '-tensor', 'input']
        data_str = self._make_dump_data_ser(DD.FORMAT_NC1HWC0, [1, 2, 4, 4, 2],
                                            DD.DT_INT64)
        with mock.patch('sys.argv', args):
            with mock.patch('src.compare.cmp_utils.utils.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR), \
                 mock.patch('os.path.getsize', return_value=len(data_str)), \
                 mock.patch('os.path.isdir', return_value=False):
                with mock.patch('numpy.save'):
                    with mock.patch('builtins.open',
                                    mock.mock_open(read_data=data_str)):
                        main = shape_conversion.ShapeConversionMain()
                        ret = main.process()
        self.assertEqual(ret,
                         CompareError.MSACCUCMP_INDEX_OUT_OF_BOUNDS_ERROR)

    def test_process11(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-format', 'NCHW', '-o',
                '/home']
        with mock.patch('sys.argv', args):
            with mock.patch('src.compare.cmp_utils.utils.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR), \
                 mock.patch('os.path.getsize', return_value=100), \
                 mock.patch('os.path.isdir', return_value=False):
                with mock.patch('numpy.save'):
                    with mock.patch('builtins.open',
                                    side_effect=IOError('not found')):
                        main = shape_conversion.ShapeConversionMain()
                        ret = main.process()
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR)

    def test_process15(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-format', 'NHWC', '-o',
                '/home', '-tensor', 'input']
        data_str = self._make_dump_data_ser(DD.FORMAT_NC1HWC0, [1, 3, 4, 4, 2],
                                            DD.DT_UINT8)
        with mock.patch('sys.argv', args):
            with mock.patch('src.compare.cmp_utils.utils.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR), \
                 mock.patch('os.path.getsize', return_value=len(data_str)), \
                 mock.patch('os.path.isdir', return_value=False):
                with mock.patch('numpy.save'):
                    with mock.patch('builtins.open',
                                    mock.mock_open(read_data=data_str)):
                        main = shape_conversion.ShapeConversionMain()
                        ret = main.process()
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_process16(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-format', 'HWCN', '-o',
                '/home']
        data_str = self._make_dump_data_ser(DD.FORMAT_NC1HWC0, [1, 2, 4, 4, 3],
                                            DD.DT_INT16)
        with mock.patch('sys.argv', args):
            with mock.patch('src.compare.cmp_utils.utils.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR), \
                 mock.patch('os.path.getsize', return_value=len(data_str)), \
                 mock.patch('os.path.isdir', return_value=False):
                with mock.patch('numpy.save'):
                    with mock.patch('builtins.open',
                                    mock.mock_open(read_data=data_str)):
                        main = shape_conversion.ShapeConversionMain()
                        ret = main.process()
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_process17(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-format', 'HWCN', '-o',
                '/home']
        data_str = self._make_dump_data_ser(DD.FORMAT_NCHW, [1, 2, 4, 4],
                                            DD.DT_UINT16)
        with mock.patch('sys.argv', args):
            with mock.patch('src.compare.cmp_utils.utils.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR), \
                 mock.patch('os.path.getsize', return_value=len(data_str)), \
                 mock.patch('os.path.isdir', return_value=False):
                with mock.patch('numpy.save'):
                    with mock.patch('builtins.open',
                                    mock.mock_open(read_data=data_str)):
                        main = shape_conversion.ShapeConversionMain()
                        ret = main.process()
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_FORMAT_ERROR)

    def test_process18(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-format', 'HWCN', '-o',
                '/home']
        data_str = self._make_dump_data_ser(DD.FORMAT_NHWC, [1, 4, 4, 2],
                                            DD.DT_INT64)
        with mock.patch('sys.argv', args):
            with mock.patch('src.compare.cmp_utils.utils.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR), \
                 mock.patch('os.path.getsize', return_value=len(data_str)), \
                 mock.patch('os.path.isdir', return_value=False):
                with mock.patch('numpy.save'):
                    with mock.patch('builtins.open',
                                    mock.mock_open(read_data=data_str)):
                        main = shape_conversion.ShapeConversionMain()
                        ret = main.process()
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_process19(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-format', 'NCHW', '-o',
                '/home']
        data_str = self._make_dump_data_ser(DD.FORMAT_NHWC, [1, 4, 4, 2],
                                            DD.DT_UINT32)
        with mock.patch('sys.argv', args):
            with mock.patch('src.compare.cmp_utils.utils.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR), \
                 mock.patch('os.path.getsize', return_value=len(data_str)), \
                 mock.patch('os.path.isdir', return_value=False):
                with mock.patch('numpy.save'):
                    with mock.patch('builtins.open',
                                    mock.mock_open(read_data=data_str)):
                        main = shape_conversion.ShapeConversionMain()
                        ret = main.process()
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_process20(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-format', 'NCHW', '-o',
                '/home']
        data_str = self._make_dump_data_ser(DD.FORMAT_HWCN, [4, 4, 2, 1],
                                            DD.DT_UINT64)
        with mock.patch('sys.argv', args):
            with mock.patch('src.compare.cmp_utils.utils.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR), \
                 mock.patch('os.path.getsize', return_value=len(data_str)), \
                 mock.patch('os.path.isdir', return_value=False):
                with mock.patch('numpy.save'):
                    with mock.patch('builtins.open',
                                    mock.mock_open(read_data=data_str)):
                        main = shape_conversion.ShapeConversionMain()
                        ret = main.process()
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_process21(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-format', 'NHWC', '-o',
                '/home']
        data_str = self._make_dump_data_ser(DD.FORMAT_HWCN, [4, 4, 2, 1],
                                            DD.DT_DOUBLE)
        with mock.patch('sys.argv', args):
            with mock.patch('src.compare.cmp_utils.utils.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR), \
                 mock.patch('os.path.getsize', return_value=len(data_str)), \
                 mock.patch('os.path.isdir', return_value=False):
                with mock.patch('numpy.save'):
                    with mock.patch('builtins.open',
                                    mock.mock_open(read_data=data_str)):
                        main = shape_conversion.ShapeConversionMain()
                        ret = main.process()
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_process22(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-format', 'NHWC', '-o',
                '/home']
        data_str = self._make_dump_data_ser(DD.FORMAT_NCHW, [1, 2, 4, 4],
                                            DD.DT_FLOAT)
        with mock.patch('sys.argv', args):
            with mock.patch('src.compare.cmp_utils.utils.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR), \
                 mock.patch('os.path.getsize', return_value=len(data_str)), \
                 mock.patch('os.path.isdir', return_value=False):
                with mock.patch('numpy.save'):
                    with mock.patch('builtins.open',
                                    mock.mock_open(read_data=data_str)):
                        main = shape_conversion.ShapeConversionMain()
                        ret = main.process()
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_process23(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-format', 'FRACTAL_Z', '-o',
                '/home']
        data_str = self._make_dump_data_ser(DD.FORMAT_NHWC, [1, 32, 64, 4],
                                            DD.DT_FLOAT16)
        with mock.patch('sys.argv', args):
            with mock.patch('src.compare.cmp_utils.utils.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR), \
                 mock.patch('os.path.getsize', return_value=len(data_str)), \
                 mock.patch('os.path.isdir', return_value=False):
                with mock.patch('numpy.save'):
                    with mock.patch('builtins.open',
                                    mock.mock_open(read_data=data_str)):
                        main = shape_conversion.ShapeConversionMain()
                        ret = main.process()
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_process24(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-format', 'FRACTAL_Z', '-o',
                '/home']
        data_str = self._make_dump_data_ser(DD.FORMAT_NCHW, [1, 20, 32, 64],
                                            DD.DT_FLOAT16)
        with mock.patch('sys.argv', args):
            with mock.patch('src.compare.cmp_utils.utils.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR), \
                 mock.patch('os.path.getsize', return_value=len(data_str)), \
                 mock.patch('os.path.isdir', return_value=False):
                with mock.patch('numpy.save'):
                    with mock.patch('builtins.open',
                                    mock.mock_open(read_data=data_str)):
                        main = shape_conversion.ShapeConversionMain()
                        ret = main.process()
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_process25(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-format', 'FRACTAL_Z', '-o',
                '/home']
        data_str = self._make_dump_data_ser(DD.FORMAT_HWCN, [32, 64, 3, 1],
                                            DD.DT_FLOAT16)
        with mock.patch('sys.argv', args):
            with mock.patch('src.compare.cmp_utils.utils.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR), \
                 mock.patch('os.path.getsize', return_value=len(data_str)), \
                 mock.patch('os.path.isdir', return_value=False):
                with mock.patch('numpy.save'):
                    with mock.patch('builtins.open',
                                    mock.mock_open(read_data=data_str)):
                        main = shape_conversion.ShapeConversionMain()
                        ret = main.process()
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_process26(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-format', 'ND', '-o',
                '/home', '-shape', '1600,768']
        data_str = self._make_dump_data_ser(DD.FORMAT_FRACTAL_NZ,
                                            [48, 100, 16, 16],
                                            DD.DT_FLOAT16)
        with mock.patch('sys.argv', args):
            with mock.patch('src.compare.cmp_utils.utils.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR), \
                 mock.patch('os.path.getsize', return_value=len(data_str)), \
                 mock.patch('os.path.isdir', return_value=False):
                with mock.patch('numpy.save'):
                    with mock.patch('builtins.open',
                                    mock.mock_open(read_data=data_str)):
                        main = shape_conversion.ShapeConversionMain()
                        ret = main.process()
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_process27(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-format', 'NCHW', '-o',
                '/home', '-shape', '12,4']
        data_str = self._make_dump_data_ser(DD.FORMAT_FRACTAL_NZ, [1, 3, 4, 4],
                                            DD.DT_FLOAT16)
        with mock.patch('sys.argv', args):
            with mock.patch('src.compare.cmp_utils.utils.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR), \
                 mock.patch('os.path.getsize', return_value=len(data_str)), \
                 mock.patch('os.path.isdir', return_value=False):
                with mock.patch('numpy.save'):
                    with mock.patch('builtins.open',
                                    mock.mock_open(read_data=data_str)):
                        main = shape_conversion.ShapeConversionMain()
                        ret = main.process()
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_process28(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-format', 'NCHW', '-o',
                '/home', '-shape', '1600,768']
        data_str = self._make_dump_data_ser(DD.FORMAT_FRACTAL_NZ,
                                            [48, 100, 16, 16],
                                            DD.DT_FLOAT16)
        with mock.patch('sys.argv', args):
            with mock.patch('src.compare.cmp_utils.utils.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR), \
                 mock.patch('os.path.getsize', return_value=len(data_str)), \
                 mock.patch('os.path.isdir', return_value=False):
                with mock.patch('numpy.save'):
                    with mock.patch('builtins.open',
                                    mock.mock_open(read_data=data_str)):
                        main = shape_conversion.ShapeConversionMain()
                        ret = main.process()
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_process29(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-format', 'NCHW', '-o',
                '/home']
        data_str = self._make_dump_data_ser(DD.FORMAT_NCHW,
                                            [1, 100, 16, 16],
                                            DD.DT_FLOAT16)
        with mock.patch('sys.argv', args):
            with mock.patch('src.compare.cmp_utils.utils.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR), \
                 mock.patch('os.path.getsize', return_value=len(data_str)), \
                 mock.patch('os.path.isdir', return_value=False):
                with mock.patch('numpy.save'):
                    with mock.patch('builtins.open',
                                    mock.mock_open(read_data=data_str)):
                        main = shape_conversion.ShapeConversionMain()
                        ret = main.process()
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_process30(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-format', 'NCHW', '-o',
                '/home']
        data_str = self._make_dump_data_ser(DD.FORMAT_FRACTAL_NZ,
                                            [48, 100, 16, 16],
                                            DD.DT_FLOAT16)
        with mock.patch('sys.argv', args):
            with mock.patch('src.compare.cmp_utils.utils.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR), \
                 mock.patch('os.path.getsize', return_value=len(data_str)), \
                 mock.patch('os.path.isdir', return_value=False):
                with mock.patch('numpy.save'):
                    with mock.patch('builtins.open',
                                    mock.mock_open(read_data=data_str)):
                        main = shape_conversion.ShapeConversionMain()
                        ret = main.process()
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_process31(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-format', 'NCHW', '-o',
                '/home', '-shape', '1600,768,1']
        data_str = self._make_dump_data_ser(DD.FORMAT_FRACTAL_NZ,
                                            [48, 100, 16, 16],
                                            DD.DT_FLOAT16)
        with mock.patch('sys.argv', args):
            with mock.patch('src.compare.cmp_utils.utils.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR), \
                 mock.patch('os.path.getsize', return_value=len(data_str)), \
                 mock.patch('os.path.isdir', return_value=False):
                with mock.patch('numpy.save'):
                    with mock.patch('builtins.open',
                                    mock.mock_open(read_data=data_str)):
                        main = shape_conversion.ShapeConversionMain()
                        ret = main.process()
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_process32(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-format', 'NCHW', '-o',
                '/home', '-shape', '2,1600,768']
        data_str = self._make_dump_data_ser(DD.FORMAT_FRACTAL_NZ,
                                            [1, 48, 100, 16, 16],
                                            DD.DT_FLOAT16)
        with mock.patch('sys.argv', args):
            with mock.patch('src.compare.cmp_utils.utils.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR), \
                 mock.patch('os.path.getsize', return_value=len(data_str)), \
                 mock.patch('os.path.isdir', return_value=False):
                with mock.patch('numpy.save'):
                    with mock.patch('builtins.open',
                                    mock.mock_open(read_data=data_str)):
                        main = shape_conversion.ShapeConversionMain()
                        ret = main.process()
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_process33(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-format', 'HWCN', '-o',
                '/home', '-shape', '2,1600,768']
        data_str = self._make_dump_data_ser(DD.FORMAT_NCHW,
                                            [1, 48],
                                            DD.DT_FLOAT16)
        with mock.patch('sys.argv', args):
            with mock.patch('src.compare.cmp_utils.utils.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR), \
                 mock.patch('os.path.getsize', return_value=len(data_str)), \
                 mock.patch('os.path.isdir', return_value=False):
                with mock.patch('numpy.save'):
                    with mock.patch('builtins.open',
                                    mock.mock_open(read_data=data_str)):
                        main = shape_conversion.ShapeConversionMain()
                        ret = main.process()
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_FORMAT_ERROR)

    def test_process_for_big_dump_data1(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-format', 'HWCN', '-o',
                '/home', '-shape', '2,1600,768']
        data_str, file_size = self._make_big_dump_data_ser(
            DD.FORMAT_NCHW, [1, 48], DD.DT_FLOAT16)
        with mock.patch('sys.argv', args):
            with mock.patch('src.compare.cmp_utils.utils.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR), \
                 mock.patch('os.path.getsize', return_value=100), \
                 mock.patch('os.path.isdir', return_value=False):
                with mock.patch('numpy.save'):
                    with mock.patch('builtins.open',
                                    mock.mock_open(read_data=data_str)):
                        main = shape_conversion.ShapeConversionMain()
                        ret = main.process()
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR)

    def test_process_for_big_dump_data2(self):
        args = ['aaa.py', '-i', '/home/left.bin', '-format', 'NHWC', '-o',
                '/home', '-shape', '2,1600,768']
        data_str, file_size = self._make_big_dump_data_ser(
            DD.FORMAT_NCHW, [1, 48], DD.DT_FLOAT16)
        with mock.patch('sys.argv', args):
            with mock.patch('src.compare.cmp_utils.utils.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR), \
                 mock.patch('os.path.getsize', return_value=file_size), \
                 mock.patch('os.path.isdir', return_value=False):
                with mock.patch('numpy.save'):
                    with mock.patch('builtins.open',
                                    mock.mock_open(read_data=data_str)):
                        main = shape_conversion.ShapeConversionMain()
                        ret = main.process()
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_FORMAT_ERROR)

    def test_process_for_NDC1HWC0_to_NCDHW(self):
        arguments = mock.Mock()
        arguments.dump_path = "/home/left.bin"
        arguments.output_path = "/home"
        arguments.dump_version = 2.0
        arguments.output_file_type = "npy"
        arguments.output = '0'
        arguments.format = "NCDHW"
        arguments.custom_script_path = ''
        arguments.shape = ""
        dump_data = DD.DumpData()
        dump_data.output.append(
            self._make_op_output(DD.FORMAT_NDC1HWC0, [1, 4, 16, 2, 2, 16], [1, 256, 4, 2, 2]))
        with mock.patch('src.compare.cmp_utils.utils.check_path_valid',
                        return_value=CompareError.MSACCUCMP_NONE_ERROR):
            with mock.patch('src.compare.cmp_utils.utils.parse_dump_file', return_value=dump_data), \
                 mock.patch('os.path.isfile', return_value=True), \
                 mock.patch('os.chmod'):
                with mock.patch('numpy.save'):
                    main = shape_conversion.FormatConversionMain(arguments)
                    ret = main.convert_format()
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    @staticmethod
    def _make_dump_data_ser(dd_format, shape, data_type):
        dump_data = DD.DumpData()
        dump_data.version = '1.0'
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

    @staticmethod
    def _make_big_dump_data_ser(dd_format, shape, data_type):
        dump_data = DD.DumpData()
        op_output = DD.OpOutput()
        op_output.data_type = data_type
        op_output.format = dd_format
        output_length = 1
        for dim in shape:
            op_output.shape.dim.append(dim)
            output_length *= dim
        data_list = np.arange(output_length)
        output_numpy = np.array(data_list,
                                common.get_dtype_by_data_type(data_type))
        data_format = common.get_struct_format_by_data_type(data_type)
        op_output.size = struct.calcsize(data_format * output_length)
        dump_data.output.append(op_output)
        op_input = DD.OpInput()
        op_input.data_type = data_type
        op_input.format = dd_format
        input_length = 1
        for dim in shape:
            op_input.shape.dim.append(dim)
            input_length *= dim
        data_list = np.arange(input_length)
        input_numpy = np.array(data_list,
                               common.get_dtype_by_data_type(data_type))
        op_input.size = struct.calcsize(data_format * input_length)
        dump_data.input.append(op_input)
        data_str = dump_data.SerializeToString()
        struct_format = 'Q' + str(len(data_str)) + 's' + str(output_length) \
                        + data_format + str(output_length) + data_format
        return struct.pack(struct_format, len(data_str), data_str, *input_numpy,
                           *output_numpy), struct.calcsize(struct_format)

    @staticmethod
    def _make_op_output(dd_format, shape, dim_list=None):
        op_output = DD.OpOutput()
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


if __name__ == '__main__':
    unittest.main()
