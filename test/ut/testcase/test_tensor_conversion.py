import unittest
from unittest import mock
import pytest
import numpy as np
import dump_data_pb2 as DD

from vector_cmp.fusion_manager import fusion_op
from format_manager.format_manager import FormatManager
from vector_cmp.fusion_manager.fusion_op import Tensor
from cmp_utils.constant.compare_error import CompareError
from conversion.tensor_conversion import TensorConversion
from dump_parse.dump_data_object import DumpTensor


class TestUtilsMethods(unittest.TestCase):

    def test_get_my_output_and_ground_truth_data1(self):
        attr = fusion_op.OpAttr(['conv1', 'conv1_relu'], '', False, 12)
        fusion_op_info = fusion_op.FusionOp(12, 'conv1conv1_relu', ['a:0,b:0'], 'Relu', None, attr)
        op_output = DumpTensor()
        op_output.tensor_format = DD.FORMAT_NC1HWC0
        manager = FormatManager("")
        manager.check_arguments_valid()
        tensor_conversion = TensorConversion(fusion_op_info, manager, False)
        compare_data = mock.Mock()
        compare_data.is_standard_quant_vs_origin = mock.Mock(return_value=False)
        ground_truth_tensor = Tensor('conv1_relu', 0, 'XFGG', [1, 3])
        ground_truth_tensor.set_data(op_output)
        with pytest.raises(CompareError) as error:
            tensor_conversion.get_my_output_and_ground_truth_data(compare_data, op_output, ground_truth_tensor)
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_FORMAT_ERROR)

    def test_get_my_output_and_ground_truth_data2(self):
        op_output = DumpTensor()
        op_output.data_type = DD.DT_FLOAT16
        op_output.tensor_format = DD.FORMAT_RESERVED
        op_output.shape.append(1)
        op_output.shape.append(4)
        data_list = [1.0, 4.5, 2.0, 3.5]
        op_output.data = np.array(data_list, np.float16)
        manager = FormatManager("")
        manager.check_arguments_valid()
        tensor_conversion = TensorConversion(self._make_fusion_op(), manager, False)
        compare_data = mock.Mock()
        compare_data.is_standard_quant_vs_origin = mock.Mock(return_value=True)
        ground_truth_tensor = Tensor('conv1_relu', 0, 'NCHW', [1, 4])
        ground_truth_tensor.set_data(op_output)
        left, right, shape = tensor_conversion.get_my_output_and_ground_truth_data(compare_data, op_output,
                                                                                   ground_truth_tensor)
        self.assertEqual(len(left), len(data_list))
        self.assertEqual(len(right), len(data_list))
        self.assertEqual(len(shape), 2)
        self.assertEqual(shape[0], 1)
        self.assertEqual(shape[1], 4)

    def test_get_my_output_and_ground_truth_data3(self):
        op_output = DumpTensor()
        op_output.data_type = DD.DT_FLOAT16
        op_output.tensor_format = DD.FORMAT_ND
        data_list = [1.0, 4.5, 2.0, 3.5]
        op_output.data = np.asarray(data_list, np.float16)
        manager = FormatManager("")
        manager.check_arguments_valid()
        tensor_conversion = TensorConversion(self._make_fusion_op(), manager, False)
        compare_data = mock.Mock()
        compare_data.is_standard_quant_vs_origin = mock.Mock(return_value=False)
        ground_truth_tensor = Tensor('conv1_relu', 0, 'NCHW', [1, 4])
        ground_truth_tensor.set_data(op_output)
        left, right, shape = tensor_conversion.get_my_output_and_ground_truth_data(compare_data, op_output,
                                                                                   ground_truth_tensor)
        self.assertEqual(len(left), len(data_list))
        self.assertEqual(len(right), len(data_list))
        self.assertEqual(len(shape), 1)
        self.assertEqual(shape[0], 4)

    def test_get_my_output_and_ground_truth_data4(self):
        left_op_output = self._make_op_output(
            DD.FORMAT_NC1HWC0, [1, 3, 2, 2, 2])
        right_op_output = self._make_op_output(DD.FORMAT_RESERVED, [2, 2, 4, 1])
        manager = FormatManager("")
        manager.check_arguments_valid()
        tensor_conversion = TensorConversion(self._make_fusion_op(), manager, False)
        compare_data = mock.Mock()
        compare_data.is_standard_quant_vs_origin = mock.Mock(return_value=False)
        ground_truth_tensor = Tensor('conv1_relu', 0, 'HWCN', [1, 2, 2, 6])
        ground_truth_tensor.set_data(right_op_output)
        left, right, shape = tensor_conversion.get_my_output_and_ground_truth_data(compare_data, left_op_output,
                                                                                   ground_truth_tensor)
        self.assertEqual(len(left), 16)
        self.assertEqual(len(right), 16)
        self.assertEqual(len(shape), 4)
        self.assertEqual(shape[1], 2)

    def test_get_my_output_and_ground_truth_data5(self):
        left_op_output = self._make_op_output(DD.FORMAT_ND, [1, 10])
        right_op_output = self._make_op_output(DD.FORMAT_RESERVED, [1, 8])
        manager = FormatManager("")
        manager.check_arguments_valid()
        tensor_conversion = TensorConversion(self._make_fusion_op(), manager, True)
        compare_data = mock.Mock()
        compare_data.is_standard_quant_vs_origin = mock.Mock(return_value=False)
        ground_truth_tensor = Tensor('conv1_relu', 0, 'NCHW', [1, 8])
        ground_truth_tensor.set_data(right_op_output)
        left, right, shape = tensor_conversion.get_my_output_and_ground_truth_data(compare_data, left_op_output,
                                                                                   ground_truth_tensor)
        self.assertEqual(len(left), 8)
        self.assertEqual(len(right), 8)
        self.assertEqual(len(shape), 4)
        self.assertEqual(shape[1], 8)

    def test_get_my_output_and_ground_truth_data6(self):
        left_op_output = self._make_op_output(DD.FORMAT_ND, [1, 8])
        right_op_output = self._make_op_output(DD.FORMAT_RESERVED, [1, 8])
        manager = FormatManager("")
        manager.check_arguments_valid()
        tensor_conversion = TensorConversion(self._make_fusion_op(), manager, True)
        compare_data = mock.Mock()
        compare_data.is_standard_quant_vs_origin = mock.Mock(return_value=True)
        ground_truth_tensor = Tensor('conv1_relu', 0, 'NCHW', [1, 8])
        ground_truth_tensor.set_data(right_op_output)
        left, right, shape = tensor_conversion.get_my_output_and_ground_truth_data(compare_data, left_op_output,
                                                                                   ground_truth_tensor)
        self.assertEqual(len(left), 8)
        self.assertEqual(len(right), 8)
        self.assertEqual(len(shape), 2)
        self.assertEqual(shape[1], 8)

    def test_get_my_output_and_ground_truth_data7(self):
        left_op_output = self._make_op_output(
            DD.FORMAT_NC1HWC0, [1, 3, 2, 2, 2])
        right_op_output = self._make_op_output(
            DD.FORMAT_NC1HWC0, [1, 2, 2, 2, 2])
        manager = FormatManager("")
        manager.check_arguments_valid()
        tensor_conversion = TensorConversion(self._make_fusion_op(), manager, False)
        compare_data = mock.Mock()
        compare_data.is_standard_quant_vs_origin = mock.Mock(return_value=False)
        ground_truth_tensor = Tensor('conv1_relu', 0, 'NC1HWC0', [1, 2, 2, 2, 2])
        ground_truth_tensor.set_data(right_op_output)
        left, right, shape = tensor_conversion.get_my_output_and_ground_truth_data(compare_data, left_op_output,
                                                                                   ground_truth_tensor)
        self.assertEqual(len(left), 16)
        self.assertEqual(len(right), 16)
        self.assertEqual(len(shape), 5)
        self.assertEqual(shape[1], 2)

    def test_get_my_output_and_ground_truth_data8(self):
        left_op_output = self._make_op_output(DD.FORMAT_NC1HWC0,
                                              [1, 1, 2, 2, 2])
        right_op_output = self._make_op_output(DD.FORMAT_NC1HWC0,
                                               [1, 2, 2, 2, 2])
        manager = FormatManager("")
        manager.check_arguments_valid()
        with pytest.raises(CompareError) as error:
            tensor_conversion = TensorConversion(self._make_fusion_op(), manager, False)
            compare_data = mock.Mock()
            compare_data.is_standard_quant_vs_origin = mock.Mock(return_value=False)
            ground_truth_tensor = Tensor('conv1_relu', 0, 'NC1HWC0', [1, 2, 2, 2, 2])
            ground_truth_tensor.set_data(right_op_output)
            tensor_conversion.get_my_output_and_ground_truth_data(compare_data, left_op_output,
                                                                  ground_truth_tensor)
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_SHAPE_ERROR)

    def test_get_my_output_and_ground_truth_data9(self):
        left_op_output = self._make_op_output(DD.FORMAT_FRACTAL_Z, [1, 3, 2, 2])
        right_op_output = self._make_op_output(DD.FORMAT_NC1HWC0, [1, 2, 2, 2, 2])
        manager = FormatManager("")
        manager.check_arguments_valid()
        with pytest.raises(CompareError) as error:
            tensor_conversion = TensorConversion(self._make_fusion_op(), manager, False)
            compare_data = mock.Mock()
            compare_data.is_standard_quant_vs_origin = mock.Mock(return_value=False)
            ground_truth_tensor = Tensor('conv1_relu', 0, 'NC1HWC0', [1, 2, 2, 2, 2])
            ground_truth_tensor.set_data(right_op_output)
            tensor_conversion.get_my_output_and_ground_truth_data(compare_data, left_op_output,
                                                                  ground_truth_tensor)
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_FORMAT_ERROR)

    def test_make_detail_dest_format1(self):
        dest1, dest2 = TensorConversion._make_detail_dest_format(
            self._make_op_output(DD.FORMAT_NDHWC, [1, 2, 3, 4, 5, 6]),
            DD.FORMAT_NCDHW)
        self.assertEqual(dest1, DD.FORMAT_NCDHW)
        self.assertEqual(dest1, DD.FORMAT_NCDHW)

    def test_make_detail_dest_format2(self):
        dest1, dest2 = TensorConversion._make_detail_dest_format(
            self._make_op_output(DD.FORMAT_FRACTAL_Z, [1, 2, 3, 4, 5, 6]),
            DD.FORMAT_HWCN)
        self.assertEqual(dest1, DD.FORMAT_HWCN)
        self.assertEqual(dest1, DD.FORMAT_HWCN)

    def test_make_detail_dest_format3(self):
        dest1, dest2 = TensorConversion._make_detail_dest_format(
            self._make_op_output(DD.FORMAT_NC1HWC0, [1, 2, 3, 4, 5, 6]),
            DD.FORMAT_HWCN)
        self.assertEqual(dest1, DD.FORMAT_NCHW)
        self.assertEqual(dest1, DD.FORMAT_NCHW)

    @staticmethod
    def _make_fusion_op():
        attr = fusion_op.OpAttr(['conv1', 'conv1_relu'], '', False, 12)
        output_desc_list = []
        output_desc = fusion_op.OutputDesc('conv1_relu', 0, 'NCHW',
                                           [1, 3, 224, 224])
        output_desc_list.append(output_desc)
        return fusion_op.FusionOp(12, 'conv1conv1_relu', ['a:0,b:0'], 'Relu', output_desc_list, attr)

    @staticmethod
    def _make_op_output(dd_format, shape):
        op_output = DumpTensor()
        op_output.data_type = DD.DT_FLOAT16
        op_output.tensor_format = dd_format
        op_output.shape = shape
        length = np.prod(shape)
        data_list = np.arange(length)
        op_output.data = np.array(data_list, np.float16)
        return op_output


if __name__ == '__main__':
    unittest.main()
