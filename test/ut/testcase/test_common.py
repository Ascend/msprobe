import unittest

import pytest
import numpy as np
import dump_data_pb2 as DD

from cmp_utils import common
from cmp_utils.constant.compare_error import CompareError


class TestUtilsMethods(unittest.TestCase):

    def test_get_format_string1(self):
        with pytest.raises(CompareError) as error:
            common.get_format_string('XXXXX')
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_FORMAT_ERROR)

    def test_get_format_string2(self):
        ret = common.get_format_string(DD.FORMAT_CHWN)
        self.assertEqual(ret, 'CHWN')

    def test_get_format_string3(self):
        ret = common.get_format_string(DD.FORMAT_RESERVED)
        self.assertEqual(ret, 'RESERVED')

    def test_get_data_type_by_dtype1(self):
        ret = common.get_data_type_by_dtype(np.double)
        self.assertEqual(ret, DD.DT_DOUBLE)

    def test_get_data_type_by_dtype2(self):
        ret = common.get_data_type_by_dtype(np.float64)
        self.assertEqual(ret, DD.DT_DOUBLE)

    def test_get_data_type_by_dtype3(self):
        ret = common.get_data_type_by_dtype(np.int8)
        self.assertEqual(ret, DD.DT_INT8)

    def test_get_data_type_by_dtype4(self):
        ret = common.get_data_type_by_dtype(np.complex128)
        self.assertEqual(ret, DD.DT_COMPLEX128)

    def test_get_dtype_by_data_type1(self):
        with pytest.raises(CompareError) as error:
            common.get_dtype_by_data_type(DD.DT_QINT8)
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_DATA_TYPE_ERROR)

    def test_get_dtype_by_data_type2(self):
        ret = common.get_dtype_by_data_type(DD.DT_DOUBLE)
        self.assertEqual(ret, np.float64)

    def test_get_dtype_by_data_type3(self):
        ret = common.get_dtype_by_data_type(DD.DT_FLOAT)
        self.assertEqual(ret, np.float32)

    def test_get_dtype_by_data_type4(self):
        ret = common.get_dtype_by_data_type(DD.DT_UINT32)
        self.assertEqual(ret, np.uint32)

    def test_get_dtype_by_data_type5(self):
        ret = common.get_dtype_by_data_type(DD.DT_INT32)
        self.assertEqual(ret, np.int32)

    def test_get_dtype_by_data_type6(self):
        ret = common.get_dtype_by_data_type(DD.DT_UINT16)
        self.assertEqual(ret, np.uint16)

    def test_get_dtype_by_data_type7(self):
        ret = common.get_dtype_by_data_type(DD.DT_INT16)
        self.assertEqual(ret, np.int16)

    def test_get_dtype_by_data_type8(self):
        ret = common.get_dtype_by_data_type(DD.DT_UINT64)
        self.assertEqual(ret, np.uint64)

    def test_get_dtype_by_data_type9(self):
        ret = common.get_dtype_by_data_type(DD.DT_INT64)
        self.assertEqual(ret, np.int64)

    def test_get_dtype_by_data_type10(self):
        ret = common.get_dtype_by_data_type(DD.DT_FLOAT16)
        self.assertEqual(ret, np.float16)

    def test_get_struct_format_by_data_type1(self):
        with pytest.raises(CompareError) as error:
            common.get_struct_format_by_data_type(DD.DT_QINT32)
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_DATA_TYPE_ERROR)

    def test_get_struct_format_by_data_type2(self):
        ret = common.get_struct_format_by_data_type(DD.DT_FLOAT)
        self.assertEqual(ret, 'f')

    def test_get_struct_format_by_data_type3(self):
        ret = common.get_struct_format_by_data_type(DD.DT_FLOAT16)
        self.assertEqual(ret, 'e')

    def test_get_struct_format_by_data_type4(self):
        ret = common.get_struct_format_by_data_type(DD.DT_INT8)
        self.assertEqual(ret, 'b')

    def test_get_struct_format_by_data_type5(self):
        ret = common.get_struct_format_by_data_type(DD.DT_UINT8)
        self.assertEqual(ret, 'B')

    def test_get_struct_format_by_data_type6(self):
        ret = common.get_struct_format_by_data_type(DD.DT_INT16)
        self.assertEqual(ret, 'h')

    def test_get_struct_format_by_data_type7(self):
        ret = common.get_struct_format_by_data_type(DD.DT_UINT16)
        self.assertEqual(ret, 'H')

    def test_get_struct_format_by_data_type8(self):
        ret = common.get_struct_format_by_data_type(DD.DT_INT32)
        self.assertEqual(ret, 'i')

    def test_get_struct_format_by_data_type9(self):
        ret = common.get_struct_format_by_data_type(DD.DT_UINT32)
        self.assertEqual(ret, 'I')

    def test_get_struct_format_by_data_type10(self):
        ret = common.get_struct_format_by_data_type(DD.DT_INT64)
        self.assertEqual(ret, 'q')

    def test_get_struct_format_by_data_type11(self):
        ret = common.get_struct_format_by_data_type(DD.DT_UINT64)
        self.assertEqual(ret, 'Q')

    def test_get_struct_format_by_data_type12(self):
        ret = common.get_struct_format_by_data_type(DD.DT_BOOL)
        self.assertEqual(ret, '?')

    def test_get_struct_format_by_data_type13(self):
        ret = common.get_struct_format_by_data_type(DD.DT_DOUBLE)
        self.assertEqual(ret, 'd')

if __name__ == '__main__':
    unittest.main()
