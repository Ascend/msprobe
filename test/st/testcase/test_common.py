import unittest

import pytest
import numpy as np
from src.compare.cmp_utils import common
from src.compare.cmp_utils import utils
import dump_data_pb2 as DD
from src.compare.vector_cmp.compare_detail
from src.compare.cmp_utils.constant.compare_error import CompareError


class TestUtilsMethods(unittest.TestCase):

    def test_get_format_string1(self):
        with pytest.raises(CompareError) as error:
            common.get_format_string('XXXXX')
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_FORMAT_ERROR)

    def test_get_data_type_by_dtype4(self):
        with pytest.raises(CompareError) as error:
            common.get_data_type_by_dtype(np.complex)
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_DATA_TYPE_ERROR)

    def test_get_dtype_by_data_type1(self):
        with pytest.raises(CompareError) as error:
            common.get_dtype_by_data_type(DD.DT_QINT8)
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_DATA_TYPE_ERROR)

    def test_get_struct_format_by_data_type1(self):
        with pytest.raises(CompareError) as error:
            common.get_struct_format_by_data_type(DD.DT_QINT32)
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_DATA_TYPE_ERROR)

    def test_detail_info_check_arguments_valid(self):
        tensor_id = detail.TensorId('xx', 'input', '1')
        detail_info = detail.DetailInfo(tensor_id, 10200, True, 1000000)
        with pytest.raises(CompareError) as error:
            detail_info.check_arguments_valid()
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_INDEX_OUT_OF_BOUNDS_ERROR)


if __name__ == '__main__':
    unittest.main()
