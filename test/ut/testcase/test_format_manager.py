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
from unittest import mock
import struct
import pytest

import numpy as np

from format_manager.format_manager import FormatManager
from format_manager.format_manager import ShapeConversion
from format_manager.format_manager import SrcToDest
from dump_parse.dump_data_object import DumpTensor
from cmp_utils.constant.compare_error import CompareError
from cmp_utils.constant.const_manager import DD


class TestUtilsMethods(unittest.TestCase):
    @staticmethod
    def _make_numpy_array(shape_from):
        count = 1
        for dim in shape_from:
            count *= dim
        return np.arange(count).flatten()

    @staticmethod
    def _make_shape(dim_list):
        shape = DumpTensor(shape=dim_list).shape
        return shape

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

    def test_check_arguments_valid1(self):
        manager = FormatManager("")
        manager.check_arguments_valid()
        self.assertEqual(len(manager.custom_support_format), 0)
        self.assertEqual(len(manager.built_in_support_format), 18)

    def test_check_arguments_valid2(self):
        with pytest.raises(CompareError) as error:
            with mock.patch('cmp_utils.path_check.check_path_valid', return_value=2):
                manager = FormatManager("/home/gzj")
                manager.check_arguments_valid()
        self.assertEqual(error.value.args[0], 2)

    def test_check_arguments_valid3(self):
        with pytest.raises(CompareError) as error:
            with mock.patch('cmp_utils.path_check.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR):
                with mock.patch('os.path.exists', return_value=False):
                    manager = FormatManager("/home/gzj")
                    manager.check_arguments_valid()
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_CONVERT_FUNC_ERROR)

    def test_check_arguments_valid4(self):
        with pytest.raises(CompareError) as error:
            with mock.patch('cmp_utils.path_check.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR):
                with mock.patch('os.listdir',
                                return_value=['xxx', 'ccc', 'ddd', 'eee']), \
                     mock.patch('os.path.exists', return_value=True), \
                     mock.patch('os.path.isdir', return_value=True):
                    manager = FormatManager("/home/gzj")
                    manager.check_arguments_valid()
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_CONVERT_FUNC_ERROR)

    def test_check_arguments_valid5(self):
        with pytest.raises(CompareError) as error:
            with mock.patch('cmp_utils.path_check.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR):
                with mock.patch('os.listdir',
                                return_value=['xxx', 'ccc', 'ddd', 'eee']), \
                     mock.patch('os.path.exists', return_value=True), \
                     mock.patch('os.path.isdir', return_value=False):
                    manager = FormatManager("/home/gzj")
                    manager.check_arguments_valid()
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_CONVERT_FUNC_ERROR)

    def test_check_arguments_valid6(self):
        with pytest.raises(CompareError) as error:
            with mock.patch('cmp_utils.path_check.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR):
                with mock.patch('os.listdir',
                                return_value=['convert_xxx_to_yyx.py',
                                              'convert_xxx_to1_yyx.py']), \
                     mock.patch('os.path.exists', return_value=True), \
                     mock.patch('os.path.isdir', return_value=False):
                    manager = FormatManager("/home/gzj")
                    manager.check_arguments_valid()
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_CONVERT_FUNC_ERROR)

    def test_execute_format_convert1(self):
        format_from = DD.FORMAT_NC1HWC0
        format_to = DD.FORMAT_MD
        shape_from = self._make_shape([1, 2, 2, 2, 2])
        shape_to = self._make_shape([1, 3, 2, 2])
        array = self._make_numpy_array(shape_from)
        with pytest.raises(CompareError) as error:
            manager = FormatManager("")
            manager.check_arguments_valid()
            manager.execute_format_convert(SrcToDest(format_from, format_to, shape_from, shape_to), array,
                                           {'group': 1})
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_FORMAT_ERROR)

    def test_convert_shape1(self):
        format_from = DD.FORMAT_ND
        format_to = DD.FORMAT_NCHW
        shape_from = self._make_shape([1, 2, 2, 2, 2])
        group = 1
        shape_to = self._make_shape([1, 3, 2, 2])
        array = np.arange(5)
        manager = FormatManager("")
        manager.check_arguments_valid()
        with pytest.raises(CompareError) as error:
            ShapeConversion(manager).convert_shape(
                SrcToDest(format_from, format_to, shape_from, shape_to), array, {'group': group})
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR)

    def test_convert_shape2(self):
        format_from = DD.FORMAT_NC1HWC0
        format_to = DD.FORMAT_MD
        shape_from = self._make_shape([1, 2, 2, 2, 2])
        group = 1
        shape_to = self._make_shape([1, 3, 2, 2])
        array = self._make_numpy_array(shape_from)
        manager = FormatManager("")
        manager.check_arguments_valid()
        with pytest.raises(CompareError) as error:
            ShapeConversion(manager).convert_shape(
                SrcToDest(format_from, format_to, shape_from, shape_to), array, {'group': group})
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_FORMAT_ERROR)

    def test_convert_shape3(self):
        format_from = DD.FORMAT_FRACTAL_NZ
        format_to = DD.FORMAT_NCHW
        shape_from = self._make_shape([1, 2, 2, 2, 2])
        group = 1
        shape_to = self._make_shape([1, 3, 2, 2])
        array = self._make_numpy_array(shape_from)
        manager = FormatManager("")
        manager.check_arguments_valid()

        with pytest.raises(CompareError) as error:
            ShapeConversion(manager).convert_shape(
                SrcToDest(format_from, format_to, shape_from, shape_to), array, {'group': group})
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_FRACTAL_NZ_DUMP_DATA_ERROR)

    def test_convert_shape4(self):
        format_from = DD.FORMAT_NC1HWC0
        format_to = DD.FORMAT_NHWC
        shape_from = self._make_shape([1, 2, 2, 2, 2])
        group = 1
        shape_to = self._make_shape([1, 3, 2, 2])
        array = self._make_numpy_array(shape_from)
        manager = FormatManager("")
        manager.check_arguments_valid()
        data = ShapeConversion(manager).convert_shape(
            SrcToDest(format_from, format_to, shape_from, shape_to), array, {'group': group})
        self.assertEqual(len(data.shape), 4)
        self.assertEqual(data.size, 16)

    def test_convert_shape5(self):
        format_from = DD.FORMAT_NC1HWC0
        format_to = DD.FORMAT_HWCN
        shape_from = self._make_shape([1, 2, 2, 2, 2])
        group = 1
        shape_to = self._make_shape([1, 3, 2, 2])
        array = self._make_numpy_array(shape_from)
        manager = FormatManager("")
        manager.check_arguments_valid()
        data = ShapeConversion(manager).convert_shape(
            SrcToDest(format_from, format_to, shape_from, shape_to), array, {'group': group})
        self.assertEqual(len(data.shape), 4)
        self.assertEqual(data.size, 16)

    def test_convert_shape6(self):
        format_from = DD.FORMAT_NC1HWC0
        format_to = DD.FORMAT_NCHW
        shape_from = self._make_shape([1, 2, 2, 2, 2])
        group = 1
        shape_to = self._make_shape([1, 3, 2, 2])
        array = self._make_numpy_array(shape_from)
        manager = FormatManager("")
        manager.check_arguments_valid()
        data = ShapeConversion(manager).convert_shape(
            SrcToDest(format_from, format_to, shape_from, shape_to), array, {'group': group})
        self.assertEqual(len(data.shape), 4)
        self.assertEqual(data.size, 16)

    def test_convert_shape7(self):
        format_from = DD.FORMAT_HWCN
        format_to = DD.FORMAT_NCHW
        shape_from = self._make_shape([1, 4, 2, 2])
        group = 1
        shape_to = self._make_shape([1, 3, 2, 2])
        array = self._make_numpy_array(shape_from)
        manager = FormatManager("")
        manager.check_arguments_valid()
        data = ShapeConversion(manager).convert_shape(
            SrcToDest(format_from, format_to, shape_from, shape_to), array, {'group': group})
        self.assertEqual(len(data.shape), 4)
        self.assertEqual(data.size, 16)

    def test_convert_shape8(self):
        format_from = DD.FORMAT_HWCN
        format_to = DD.FORMAT_NHWC
        shape_from = self._make_shape([1, 4, 2, 2])
        shape_to = self._make_shape([1, 3, 2, 2])
        group = 1
        array = self._make_numpy_array(shape_from)
        manager = FormatManager("")
        manager.check_arguments_valid()
        data = ShapeConversion(manager).convert_shape(
            SrcToDest(format_from, format_to, shape_from, shape_to), array, {'group': group})
        self.assertEqual(len(data.shape), 4)
        self.assertEqual(data.size, 16)

    def test_convert_shape9(self):
        format_from = DD.FORMAT_HWCN
        format_to = DD.FORMAT_FRACTAL_Z
        shape_from = self._make_shape([1, 4, 2, 2])
        group = 1
        shape_to = self._make_shape([1, 3, 2, 2])
        array = self._make_numpy_array(shape_from)
        manager = FormatManager("")
        manager.check_arguments_valid()
        data = ShapeConversion(manager).convert_shape(
            SrcToDest(format_from, format_to, shape_from, shape_to), array, {'group': group})
        self.assertEqual(len(data.shape), 4)
        self.assertEqual(data.size, 1024)

    def test_convert_shape10(self):
        format_from = DD.FORMAT_NHWC
        format_to = DD.FORMAT_NCHW
        shape_from = self._make_shape([1, 4, 2, 2])
        group = 1
        shape_to = self._make_shape([1, 3, 2, 2])
        array = self._make_numpy_array(shape_from)
        manager = FormatManager("")
        manager.check_arguments_valid()
        data = ShapeConversion(manager).convert_shape(
            SrcToDest(format_from, format_to, shape_from, shape_to), array, {'group': group})
        self.assertEqual(len(data.shape), 4)
        self.assertEqual(data.size, 16)

    def test_convert_shape11(self):
        format_from = DD.FORMAT_NHWC
        format_to = DD.FORMAT_HWCN
        shape_from = self._make_shape([1, 4, 2, 2])
        group = 1
        shape_to = self._make_shape([1, 3, 2, 2])
        array = self._make_numpy_array(shape_from)
        manager = FormatManager("")
        manager.check_arguments_valid()
        data = ShapeConversion(manager).convert_shape(
            SrcToDest(format_from, format_to, shape_from, shape_to), array, {'group': group})
        self.assertEqual(len(data.shape), 4)
        self.assertEqual(data.size, 16)

    def test_convert_shape12(self):
        format_from = DD.FORMAT_NHWC
        format_to = DD.FORMAT_FRACTAL_Z
        shape_from = self._make_shape([1, 4, 2, 2])
        group = 1
        shape_to = self._make_shape([1, 3, 2, 2])
        array = self._make_numpy_array(shape_from)
        manager = FormatManager("")
        manager.check_arguments_valid()
        data = ShapeConversion(manager).convert_shape(
            SrcToDest(format_from, format_to, shape_from, shape_to), array, {'group': group})
        self.assertEqual(len(data.shape), 4)
        self.assertEqual(data.size, 2048)

    def test_convert_shape13(self):
        format_from = DD.FORMAT_FRACTAL_NZ
        format_to = DD.FORMAT_NCHW
        shape_from = self._make_shape([2, 16, 16, 16])
        group = 1
        shape_to = self._make_shape([256, 32])
        array = self._make_numpy_array(shape_from)
        manager = FormatManager("")
        manager.check_arguments_valid()
        data = ShapeConversion(manager).convert_shape(
            SrcToDest(format_from, format_to, shape_from, shape_to), array, {'group': group})
        self.assertEqual(len(data.shape), 2)
        self.assertEqual(data.size, 256 * 32)

    def test_convert_shape14(self):
        format_from = DD.FORMAT_FRACTAL_NZ
        format_to = DD.FORMAT_ND
        shape_from = self._make_shape([2, 16, 16, 16])
        group = 1
        shape_to = self._make_shape([256, 32])
        array = self._make_numpy_array(shape_from)
        manager = FormatManager("")
        manager.check_arguments_valid()
        data = ShapeConversion(manager).convert_shape(
            SrcToDest(format_from, format_to, shape_from, shape_to), array, {'group': group})
        self.assertEqual(len(data.shape), 2)
        self.assertEqual(data.size, 256 * 32)

    def test_convert_shape15(self):
        format_from = DD.FORMAT_NCHW
        format_to = DD.FORMAT_FRACTAL_Z
        shape_from = self._make_shape([1, 3, 2, 2])
        group = 1
        shape_to = self._make_shape([1, 2, 2, 2])
        array = self._make_numpy_array(shape_from)
        manager = FormatManager("")
        manager.check_arguments_valid()
        data = ShapeConversion(manager).convert_shape(
            SrcToDest(format_from, format_to, shape_from, shape_to), array, {'group': group})
        self.assertEqual(len(data.shape), 4)
        self.assertEqual(data.size, 1024)

    def test_convert_shape16(self):
        format_from = DD.FORMAT_NCHW
        format_to = DD.FORMAT_NHWC
        shape_from = self._make_shape([1, 3, 2, 2])
        group = 1
        shape_to = self._make_shape([1, 2, 2, 2])
        array = self._make_numpy_array(shape_from)
        manager = FormatManager("")
        manager.check_arguments_valid()
        data = ShapeConversion(manager).convert_shape(
            SrcToDest(format_from, format_to, shape_from, shape_to), array, {'group': group})
        self.assertEqual(len(data.shape), 4)
        self.assertEqual(data.size, 12)

    def test_convert_shape17(self):
        format_from = DD.FORMAT_FRACTAL_NZ
        format_to = DD.FORMAT_NHWC
        shape_from = self._make_shape([4, 2, 16, 16, 16])
        group = 1
        shape_to = self._make_shape([4, 256, 32])
        array = self._make_numpy_array(shape_from)
        manager = FormatManager("")
        manager.check_arguments_valid()
        data = ShapeConversion(manager).convert_shape(
            SrcToDest(format_from, format_to, shape_from, shape_to), array, {'group': group})
        self.assertEqual(len(data.shape), 3)
        self.assertEqual(data.size, 256 * 32 * 4)

    def test_convert_shape18(self):
        format_from = DD.FORMAT_NDC1HWC0
        format_to = DD.FORMAT_ND
        shape_from = self._make_shape([1, 8, 1, 224, 224, 16])
        group = 1
        shape_to = self._make_shape([1, 3, 8, 224, 224])
        array = self._make_numpy_array(shape_from)
        manager = FormatManager("")
        manager.check_arguments_valid()
        data = ShapeConversion(manager).convert_shape(
            SrcToDest(format_from, format_to, shape_from, shape_to), array, {'group': group})
        self.assertEqual(len(data.shape), 5)
        self.assertEqual(data.size, 24 * 224 * 224)

    def test_convert_shape_fractal_nz_to_nd_array_not_equal_shape(self):
        format_from = DD.FORMAT_FRACTAL_NZ
        format_to = DD.FORMAT_ND
        shape_from = self._make_shape([20, 2, 16, 16])
        group = 1
        shape_to = self._make_shape([1, 32, 12, 26])
        array = self._make_numpy_array(shape_from)
        manager = FormatManager("")
        manager.check_arguments_valid()
        data = ShapeConversion(manager).convert_shape(
            SrcToDest(format_from, format_to, shape_from, shape_to), array, {'group': group})
        self.assertEqual(data.size, 20 * 2 * 16 * 16)


if __name__ == '__main__':
    unittest.main()
