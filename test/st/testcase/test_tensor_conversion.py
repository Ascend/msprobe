import unittest

from unittest import mock
import numpy as np
from src.compare.vector_cmp.fusion_manager import fusion_op
from src.compare.format_convert.format_manager import FormatManager
from src.compare.conversion.tensor_conversion import TensorConversion


class TestUtilsMethods(unittest.TestCase):

    def test_slice_data(self):
        shape = (1, 3, 2, 2)
        origin_numpy = np.ones(shape, np.float16)
        manager = FormatManager("")
        manager.check_arguments_valid()
        tensor_conversion = TensorConversion(self._make_fusion_op(), manager, False)
        compare_data = mock.Mock()
        compare_data.is_standard_quant_vs_origin = mock.Mock(return_value=True)
        left_np = tensor_conversion.slice_data(origin_numpy, [1, 3, 2, 2, 1, 1])
        self.assertEqual(left_np.shape, shape)

    @staticmethod
    def _make_fusion_op():
        attr = fusion_op.OpAttr(['conv1', 'conv1_relu'], '', False, 12)
        output_desc_list = []
        output_desc = fusion_op.OutputDesc('conv1_relu', 0, 'NCHW',
                                           [1, 3, 224, 224])
        output_desc_list.append(output_desc)
        return fusion_op.FusionOp(12, 'conv1conv1_relu', ['a:0,b:0'], 'Relu', output_desc_list, attr)


if __name__ == '__main__':
    unittest.main()
