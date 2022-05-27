import unittest

import numpy as np
from format_manager import FormatManager
from format_manager import ShapeConversion
from format_manager import SrcToDest
import dump_data_pb2 as DD


class TestUtilsMethods(unittest.TestCase):
    def test_convert_shape_fractal_nz_to_nd_array_not_eaual_shape(self):
        format_from = DD.FORMAT_FRACTAL_NZ
        format_to = DD.FORMAT_ND
        shape_from = self._make_shape([20, 2, 16, 16])
        group = 1
        shape_to = self._make_shape([1, 32, 12, 26])
        array = self._make_numpy_array(shape_from.dim)
        manager = FormatManager("")
        manager.check_arguments_valid()
        data = ShapeConversion(manager).convert_shape(
            SrcToDest(format_from, format_to, shape_from, shape_to), array, {'group': group})
        self.assertEqual(data.size, 20 * 2 * 16 * 16)

    @staticmethod
    def _make_numpy_array(shape_from):
        count = 1
        for dim in shape_from:
            count *= dim
        return np.arange(count).flatten()

    @staticmethod
    def _make_shape(dim_list):
        shape = DD.Shape()
        for dim in dim_list:
            shape.dim.append(dim)
        return shape


if __name__ == '__main__':
    unittest.main()
