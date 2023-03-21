import unittest
import numpy as np
import pytest

from dump_data_object import DumpDataObj, DumpTensor
from compare_error import CompareError
from ffts_parser import FFTSParser


class TestUtilsMethods(unittest.TestCase):

    def test_manual_cut_axis1(self):
        dump_data = DumpDataObj()
        dump_data.attr = {"outputCutList": [[1, 2, 1, 1], [1, 1, 1, 1]]}
        cut_axis = dump_data.get_cut_axis_manual
        self.assertEqual([[1], []], cut_axis)

    def test_manual_cut_axis2(self):
        dump_data = DumpDataObj()
        dump_data.attr = {"outputCutList": [[1, 2, 1, 1], [1, 2, 1, 1], [1, 2, 1, 1]]}
        cut_axis = dump_data.get_cut_axis_manual
        self.assertEqual([[1], [1], [1]], cut_axis)

    def test_manual_cut_axis3(self):
        dump_data = DumpDataObj()
        dump_data.attr = {"outputCutList": [[1, 1, 1, 1], [1, 1, 1, 1]]}
        cut_axis = dump_data.get_cut_axis_manual
        self.assertEqual([[], []], cut_axis)

    def test_get_output_data(self):
        dump_data = DumpDataObj()
        data1 = DumpTensor(data=np.random.rand(4, 5, 3, 1).reshape(-1), shape=[4, 5, 3, 1])
        dump_data.output_data.append(data1)
        data2 = DumpTensor(data=np.random.rand(2, 3, 5, 1, 1).reshape(-1), shape=[2, 3, 5, 1, 1])
        dump_data.output_data.append(data2)
        output = dump_data.get_output_data
        self.assertEqual(output[0].all(), data1.data.all())
        self.assertEqual(output[1].all(), data2.data.all())

    def test_check_shape_match1(self):
        data = np.zeros((256,))
        shape = [4, 4, 4, 4]
        ret = DumpDataObj.check_shape_match(data, shape)
        self.assertTrue(ret)

    def test_check_shape_match2(self):
        data = np.zeros((258,))
        shape = [4, 4, 4, 4]
        with pytest.raises(CompareError) as error:
            DumpDataObj.check_shape_match(data, shape)
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_UNMATCH_DATA_SHAPE_ERROR)

    def test_parse_ffts1(self):
        dump_data1 = DumpDataObj()
        dump_data2 = DumpDataObj()
        dump_data1.attr = {"slice_instance_num": 2, "outputCutList": [[2, 1, 1, 1], [1, 1, 1, 1]],
                           "threadMode": False}
        dump_data2.attr = {"slice_instance_num": 2, "outputCutList": [[2, 1, 1, 1], [1, 1, 1, 1]],
                           "threadMode": False}
        dump_data_list = [dump_data1, dump_data2]
        dump_file_list = ["Conv2D.Conv2D_lxslice0.2.9.1670205069987341.4.330.0.0",
                          "Conv2D.Conv2D_lxslice1.2.9.1670205069987340.4.330.0.0"]
        output_tensor1 = DumpTensor(data=np.ones((72,)), shape=[12, 2, 1, 3])
        output_tensor2 = DumpTensor(data=np.ones((6,)), shape=[1, 1, 2, 3])
        for dump_data in dump_data_list:
            dump_data.output_data.append(output_tensor1)
            dump_data.output_data.append(output_tensor2)
        ffts_parser = FFTSParser(dump_file_list, dump_data_list)
        dump_file_path, dump_data = ffts_parser.parse_ffts
        std_file_path = "Conv2D.Conv2D_lxslice0.2.9.*"
        std_dump_data = [DumpTensor(data=np.ones((144,)), shape=[24, 2, 1, 3]),
                         DumpTensor(data=np.ones((6,)), shape=[1, 1, 2, 3])]
        self.assertEqual(std_file_path, dump_file_path)
        self.assertEqual(std_dump_data[0].data.all(), dump_data.output_data[0].data.all())
        self.assertEqual(std_dump_data[1].data.all(), dump_data.output_data[1].data.all())
        self.assertEqual(std_dump_data[0].shape, dump_data.output_data[0].shape)
        self.assertEqual(std_dump_data[1].shape, dump_data.output_data[1].shape)

    def test_parse_ffts2(self):
        dump_data1 = DumpDataObj()
        dump_data2 = DumpDataObj()
        dump_data1.attr = {"slice_instance_num": 2, "outputCutList": [[1, 1, 1, 1], [1, 1, 1, 1]],
                           "threadMode": False}
        dump_data2.attr = {"slice_instance_num": 2, "outputCutList": [[1, 1, 1, 1], [1, 1, 1, 1]],
                           "threadMode": False}
        dump_data_list = [dump_data1, dump_data2]
        dump_file_list = ["Conv2D.Conv2D_lxslice0.2.9.1670205069981234.4.330.0.0",
                          "Conv2D.Conv2D_lxslice1.2.9.1670205069980123.4.330.0.0"]
        output_tensor1 = DumpTensor(data=np.ones((24,)), shape=[12, 2, 1, 1])
        output_tensor2 = DumpTensor(data=np.ones((4,)), shape=[1, 1, 2, 2])
        for dump_data in dump_data_list:
            dump_data.output_data.append(output_tensor1)
            dump_data.output_data.append(output_tensor2)
        ffts_parser = FFTSParser(dump_file_list, dump_data_list)
        dump_file_path, dump_data = ffts_parser.parse_ffts
        std_file_path = "Conv2D.Conv2D_lxslice1.2.9.1670205069980123.4.330.0.0"
        std_dump_data = dump_data2
        self.assertEqual(std_file_path, dump_file_path)
        self.assertEqual(std_dump_data.output_data, dump_data.output_data)

    def test_check_file_missing(self):
        dump_file_list = ["Conv2D.Conv2D_lxslice0.2.9.1670205069987341.4.330.0.0"]
        dump_data = DumpDataObj()
        dump_data.attr = {"slice_instance_num": 2, "outputCutList": [[1, 1, 1, 1], [1, 1, 1, 1]],
                           "threadMode": False}
        dump_data_list = [dump_data]
        ffts_parser = FFTSParser(dump_file_list, dump_data_list)
        thread_num = len(dump_file_list)
        ret = ffts_parser.check_file_missing(thread_num)
        self.assertEqual(ret, False)
