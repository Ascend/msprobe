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
import pytest
import numpy as np

from cmp_utils import utils_type
from cmp_utils.constant.compare_error import CompareError
from pytorch_cmp import pytorch_dump_data


class TestUtilsMethods(unittest.TestCase):

    def test_get_original_opname(self):
        compare_data = pytorch_dump_data.CompareData("/home/my_dump.h5", "/home/golden_dump.h5")
        op_name = compare_data.get_original_opname("Admm1:2")
        self.assertEqual(op_name, "Admm1")

    def test_get_all_orders(self):
        compare_data = pytorch_dump_data.CompareData("/home/my_dump.h5", "/home/golden_dump.h5")
        compare_data.my_dump.file_handle = {
            "Admm1": {'3': "input/input0",
                      '4': "input/input1"},
            "Abxx1": {'3': "input/input2",
                      '5': "input/input3"},
        }
        compare_data.my_dump._generate_order_ext_opname_map()
        orders = compare_data.get_all_orders()
        self.assertEqual(list(orders), [3, 4, 5])

    def test_get_ext_opname_by_order(self):
        compare_data = pytorch_dump_data.CompareData("/home/my_dump.h5", "/home/golden_dump.h5")
        compare_data.my_dump.file_handle = {
            "Admm1": {'3': "input/input0",
                      '4': "input/input1"},
            "Abxx1": {'3': "input/input2",
                      '5': "input/input3"},
        }
        compare_data.my_dump._generate_order_ext_opname_map()
        ext_opaname = compare_data.get_ext_opname_by_order(3)
        self.assertEqual(ext_opaname, ["Admm1:0", "Abxx1:0"])

    def test_get_my_dump_datasets(self):
        compare_data = pytorch_dump_data.CompareData("/home/my_dump.h5", "/home/golden_dump.h5")
        compare_data.my_dump.file_handle = {
            "Admm1": {'3': "input/input0",
                      '4': "input/input1"},
            "Abxx1": {'3': "input/input2",
                      '5': "input/input3"},
        }
        compare_data.my_dump._generate_order_ext_opname_map()
        compare_data.my_dump.ext_opname_dataset_map = {
            'Admm1:0': ['/Admm1/3/input0'],
            'Abxx1:0': ['/Abxx1/3/input2'],
            'Admm1:1': ['/Admm1/4/input1'],
            'Abxx1:1': ['/Abxx1/5/input3'],
        }
        my_dump_datasets = compare_data.get_my_dump_datasets("Admm1:0")
        self.assertEqual(my_dump_datasets, ['/Admm1/3/input0'])

    def test_construct_dataset_path1(self):
        compare_data = pytorch_dump_data.CompareData("/home/my_dump.h5", "/home/golden_dump.h5")
        compare_data.my_dump.need_compare_input = True
        compare_data.golden_dump.need_compare_input = True
        compare_data.my_dump.file_handle = {
            "Admm1": {'3': "input/input0",
                      '5': "output/input1"},
            "Abxx1": {'3': "input/input2",
                      '5': "input/input3"},
        }
        compare_data.golden_dump.file_handle = {
            "Admm1": {'4': "input/input0",
                      '5': "output/input1"},
            "Abxx1": {'7': "input/input2",
                      '8': "input/input3"},
        }
        compare_data.my_dump._generate_order_ext_opname_map()
        compare_data.golden_dump._generate_order_ext_opname_map()
        compare_data.my_dump.ext_opname_dataset_map = {
            'Admm1:0': ['/Admm1/3/input/input0'],
            'Abxx1:0': ['/Abxx1/3/input/input2'],
            'Admm1:1': ['/Admm1/5/output/input1'],
            'Abxx1:1': ['/Abxx1/5/input/input3'],
        }
        compare_data.golden_dump.ext_opname_dataset_map = {
            'Admm1:0': ['/Admm1/4/input/input0'],
            'Abxx1:0': ['/Abxx1/7/input/input2'],
            'Admm1:1': ['/Admm1/5/output/input1'],
            'Abxx1:1': ['/Abxx1/8/input/input3'],
        }
        expect_dataset = compare_data._construct_dataset_path("Admm1:0", '/Admm1/3/input/input0', compare_data.golden_dump, "Admm1:0")
        self.assertEqual(expect_dataset, '/Admm1/4/input/input0')
        compare_data.my_dump.need_compare_input = False
        compare_data.golden_dump.need_compare_input = False
        expect_dataset = compare_data._construct_dataset_path("Admm1:1", '/Admm1/5/output/input1', compare_data.golden_dump, "Admm1:1")
        self.assertEqual(expect_dataset, '/Admm1/5/output/input1')

    def test_construct_dataset_path2(self):
        compare_data = pytorch_dump_data.CompareData("/home/my_dump.h5", "/home/golden_dump.h5")
        compare_data.my_dump.need_compare_input = True
        compare_data.my_dump.file_handle = {
            "Admm1": {'3': "input/input0",
                      '4': "input/input1"},
        }
        compare_data.golden_dump.file_handle = {
            "Admm1": {'5': "input/input0",
                      '6': "input/input1"},
        }
        compare_data.my_dump._generate_order_ext_opname_map()
        compare_data.golden_dump._generate_order_ext_opname_map()
        compare_data.my_dump.ext_opname_dataset_map = {
            'Admm1:0': ['/Admm1/3/input/input0'],
            'Admm1:1': ['/Admm1/4/input/input1'],
        }
        compare_data.golden_dump.ext_opname_dataset_map = {
            'Admm1:0': ['/Admm1/5/input/input0'],
            'Admm1:1': ['/Admm1/6/input/input1'],
        }
        expect_dataset = compare_data._construct_dataset_path("Admm1:1", '/Admm1/6/input/input1', compare_data.my_dump, "Admm1:1")
        self.assertEqual(expect_dataset, '/Admm1/4/input/input1')

    def test_opname_map(self):
        compare_data = pytorch_dump_data.CompareData("/home/my_dump.h5", "/home/golden_dump.h5")
        compare_data.my_dump.need_compare_input = True
        map_opname = compare_data._opname_map("NpuAdmm:1", utils_type.DeviceType.CPU.value)
        self.assertEqual(map_opname, "ThnnAdmm:1")
        map_opname = compare_data._opname_map("NpuAdmm:1", utils_type.DeviceType.GPU.value)
        self.assertEqual(map_opname, "CudnnAdmm:1")
        map_opname = compare_data._opname_map("NpuAdmm:1", utils_type.DeviceType.NPU.value)
        self.assertEqual(map_opname, "NpuAdmm:1")
        map_opname = compare_data._opname_map("CudnnAdmm:1")
        self.assertEqual(map_opname, "NpuAdmm:1")
        map_opname = compare_data._opname_map("ThnnAdmm:1")
        self.assertEqual(map_opname, "NpuAdmm:1")

    def test_get_golden_dataset(self):
        compare_data = pytorch_dump_data.CompareData("/home/my_dump.h5", "/home/golden_dump.h5")
        compare_data.my_dump.need_compare_input = True
        compare_data.golden_dump.need_compare_input = True
        compare_data.my_dump.file_handle = {
            "Admm1": {'3': "input/input0",
                      '4': "input/input1"},
        }
        compare_data.golden_dump.file_handle = {
            "Admm1": {'5': "input/input0",
                      '6': "input/input1"},
        }
        compare_data.my_dump._generate_order_ext_opname_map()
        compare_data.golden_dump._generate_order_ext_opname_map()
        compare_data.my_dump.ext_opname_dataset_map = {
            'Admm1:0': ['/Admm1/3/input/input0'],
            'Admm1:1': ['/Admm1/4/input/input1'],
        }
        compare_data.golden_dump.ext_opname_dataset_map = {
            'Admm1:0': ['/Admm1/5/input/input0'],
            'Admm1:1': ['/Admm1/6/input/input1'],
        }
        with mock.patch('pytorch_cmp.hdf5_parser.Hdf5Parser.get_dump_data_attr',
                        side_effect=[1
                                     ]):
            _, golden_dataset, _ = compare_data.get_golden_dataset("Admm1:1", '/Admm1/4/input/input1')
            self.assertEqual(golden_dataset, '/Admm1/6/input/input1')

    def test_get_dump_data(self):
        compare_data = pytorch_dump_data.CompareData("/home/my_dump.h5", "/home/golden_dump.h5")
        compare_data.my_dump.need_compare_input = True
        compare_data.my_dump.file_handle = {
            "Admm1": {'3': "input/input0",
                      '4': "input/input1"},
        }
        compare_data.golden_dump.file_handle = {
            "Admm1": {'5': "input/input0",
                      '6': "input/input1"},
        }
        compare_data.my_dump._generate_order_ext_opname_map()
        compare_data.golden_dump._generate_order_ext_opname_map()
        compare_data.my_dump.ext_opname_dataset_map = {
            'Admm1:0': ['/Admm1/3/input0'],
            'Admm1:1': ['/Admm1/4/input1'],
        }
        compare_data.golden_dump.ext_opname_dataset_map = {
            'Admm1:0': ['/Admm1/5/input0'],
            'Admm1:1': ['/Admm1/6/input1'],
        }
        with mock.patch('pytorch_cmp.hdf5_parser.Hdf5Parser.get_dump_data_attr',
                        side_effect=[(True, 1), (True, 1), (True, 1), (True, (1, 3))]):
            with mock.patch('pytorch_cmp.hdf5_parser.Hdf5Parser.get_dump_data',
                            side_effect=[np.array(np.arange(9)).reshape(3, 3),
                                         np.array(np.arange(9)).reshape(3, 3).T]):
                mydump_data, golden_dump_data, _ = compare_data.get_dump_data(
                    '/Admm1/3/input0', '/Admm1/5/input0')
        self.assertEqual((mydump_data == golden_dump_data).all(), True)

    def test_get_dump_data_case1(self):
        compare_data = pytorch_dump_data.CompareData("/home/my_dump_c1.h5", "/home/golden_dump_c1.h5")
        compare_data.my_dump.need_compare_input = True

        with mock.patch('pytorch_cmp.pytorch_dump_data.CompareData._check_data_type',
                        return_value=[True, "success"]):
            with mock.patch('pytorch_cmp.hdf5_parser.Hdf5Parser.get_dump_data',
                            side_effect=[np.array(np.arange(9)).reshape(3, 3),
                                         np.array(np.arange(9)).reshape(3, 3).T]):
                with mock.patch('pytorch_cmp.hdf5_parser.Hdf5Parser.get_dump_data_attr',
                                side_effect=[(True, 1), (True, (3, 3))]):
                    mydump_data, golden_dump_data, _ = compare_data.get_dump_data(
                '/Admm1/3/input0', '/Admm1/5/input0')
        self.assertEqual((golden_dump_data.T == mydump_data).all(), True)

    def test_get_not_matched_golden_datasets(self):
        compare_data = pytorch_dump_data.CompareData("/home/my_dump.h5", "/home/golden_dump.h5")
        compare_data.my_dump.need_compare_input = True
        compare_data.golden_dump.need_compare_input = True
        compare_data.my_dump.file_handle = {
            "Admm1": {'3': "input/input0",
                      '4': "input/input1"},
        }
        compare_data.golden_dump.file_handle = {
            "Admm1": {'5': "input/input0",
                      '6': "input/input1",
                      '7': "input/input2"},
            "Adxx1": {'8': "input/input0",
                      '9': "input/input1"},
        }
        compare_data.my_dump._generate_order_ext_opname_map()
        compare_data.golden_dump._generate_order_ext_opname_map()
        compare_data.my_dump.ext_opname_dataset_map = {
            'Admm1:0': ['/Admm1/3/input/input0'],
            'Admm1:1': ['/Admm1/4/input/input1'],
        }
        compare_data.golden_dump.ext_opname_dataset_map = {
            'Admm1:0': ['/Admm1/5/input/input0'],
            'Admm1:1': ['/Admm1/6/input/input1', '/Admm1/6/input/input2'],
            'Admm1:2': ['/Admm1/7/input/input2'],
            'Adxx1:0': ['/Adxx1/8/input/input0'],
            'Adxx1:1': ['/Adxx1/9/input/input1'],
        }

        not_matched_in_golden = compare_data.get_not_matched_golden_datasets()
        result = []
        expect_result = ['/Admm1/6/input/input2',
                         '/Admm1/7',
                         '/Adxx1/8',
                         '/Adxx1/9'
                        ]
        for item in not_matched_in_golden:
            result.append(item[2])
        self.assertEqual(result == expect_result, True)

    def test_parse_dump_file(self):
        compare_data = pytorch_dump_data.CompareData("/home/my_dump.h5", "/home/golden_dump.h5")
        compare_data.my_dump.need_compare_input = True
        compare_data.my_dump.file_handle = {
            "Admm1": {'3': "input/input0",
                      '4': "input/input1"},
        }
        compare_data.golden_dump.file_handle = {
            "Admm1": {'5': "input/input0",
                      '6': "input/input1"},
        }
        compare_data.my_dump._generate_order_ext_opname_map()
        compare_data.golden_dump._generate_order_ext_opname_map()
        compare_data.my_dump.ext_opname_dataset_map = {
            'Admm1:0': ['/Admm1/3/input/input0'],
            'Admm1:1': ['/Admm1/4/input/input1'],
        }
        compare_data.golden_dump.ext_opname_dataset_map = {
            'Admm1:0': ['/Admm1/5/input/input0'],
            'Admm1:1': ['/Admm1/6/input/input1'],
        }
        compare_data.my_dump.file_handle = None
        compare_data.my_dump.device_type = 10
        compare_data.golden_dump.file_handle = None
        with mock.patch('pytorch_cmp.hdf5_parser.Hdf5Parser.parse_dump_file',
                        side_effect=[CompareError.MSACCUCMP_NONE_ERROR,
                                     CompareError.MSACCUCMP_NONE_ERROR
                                     ]):
            with mock.patch('pytorch_cmp.hdf5_parser.Hdf5Parser.get_dump_data_attr',
                            side_effect=[10]):
                parse_result = compare_data.parse_dump_file()
        self.assertEqual(parse_result, CompareError.MSACCUCMP_NONE_ERROR)

        compare_data.my_dump.device_type = 1
        with pytest.raises(CompareError) as err:
            with mock.patch('pytorch_cmp.hdf5_parser.Hdf5Parser.parse_dump_file',
                            side_effect=[CompareError.MSACCUCMP_NONE_ERROR,
                                         CompareError.MSACCUCMP_NONE_ERROR
                                        ]):
                with mock.patch('pytorch_cmp.hdf5_parser.Hdf5Parser.get_dump_data_attr',
                            side_effect=[1]):
                    parse_result = compare_data.parse_dump_file()
        self.assertEqual(err.value.code, CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR)

    def test_check_my_dump_file_valid(self):
        compare_data = pytorch_dump_data.CompareData("/home/my_dump.h5", "/home/golden_dump.h5")
        compare_data.my_dump.need_compare_input = True
        compare_data.my_dump.file_handle = {
            "Admm1": {'3': "input/input0",
                      '4': "input/input1"},
        }
        compare_data.golden_dump.file_handle = {
            "Admm1": {'5': "input/input0",
                      '6': "input/input1"},
        }
        compare_data.my_dump._generate_order_ext_opname_map()
        compare_data.golden_dump._generate_order_ext_opname_map()
        compare_data.my_dump.ext_opname_dataset_map = {
            'Admm1:0': ['/Admm1/3/input0'],
            'Admm1:1': ['/Admm1/4/input1'],
        }
        compare_data.golden_dump.ext_opname_dataset_map = {
            'Admm1:0': ['/Admm1/5/input0'],
            'Admm1:1': ['/Admm1/6/input1'],
        }
        compare_data.my_dump.file_handle = None
        compare_data.golden_dump.file_handle = None
        with pytest.raises(CompareError) as err:
            with mock.patch('pytorch_cmp.hdf5_parser.Hdf5Parser.get_dump_data_attr',
                            side_effect=[1]):
                compare_data.check_my_dump_file_valid()
        self.assertEqual(err.value.code, CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR)

        compare_data.my_dump.ext_opname_dataset_map = {}
        self.assertEqual(err.value.code, CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR)

    def test_check_data_type_case1(self):
        compare_data = pytorch_dump_data.CompareData("/home/my_dump.h5", "/home/golden_dump.h5")
        compare_data.my_dump.need_compare_input = True
        result = []
        with mock.patch('pytorch_cmp.hdf5_parser.Hdf5Parser.get_dump_data_attr',
                        side_effect=[(False, ""), (False, "")]):
            result = compare_data._check_data_type('dataset1', 'dataset2')
        self.assertEqual(result[0], False)

    def test_check_data_type_case2(self):
        compare_data = pytorch_dump_data.CompareData("/home/my_dump.h5", "/home/golden_dump.h5")
        compare_data.my_dump.need_compare_input = True
        result = []
        with mock.patch('pytorch_cmp.hdf5_parser.Hdf5Parser.get_dump_data_attr',
                        side_effect=[(True, 2), (True, 2)]):
            result = compare_data._check_data_type('dataset1', 'dataset2')
        self.assertEqual(result[0], True)

    def test_check_data_type_case3(self):
        compare_data = pytorch_dump_data.CompareData("/home/my_dump.h5", "/home/golden_dump.h5")
        compare_data.my_dump.need_compare_input = True
        result = []
        with mock.patch('pytorch_cmp.hdf5_parser.Hdf5Parser.get_dump_data_attr',
                        side_effect=[(True, 6), (True, 7)]):
            result = compare_data._check_data_type('dataset1', 'dataset2')
        self.assertEqual(result[0], True)

    def test_check_data_type_case4(self):
        compare_data = pytorch_dump_data.CompareData("/home/my_dump.h5", "/home/golden_dump.h5")
        compare_data.my_dump.need_compare_input = True
        result = []
        with mock.patch('pytorch_cmp.hdf5_parser.Hdf5Parser.get_dump_data_attr',
                        side_effect=[(True, 3), (True, 7)]):
            result = compare_data._check_data_type('dataset1', 'dataset2')
        self.assertEqual(result[0], False)

    def test_check__check_stride_case1(self):
        compare_data = pytorch_dump_data.CompareData("/home/my_dump1.h5", "/home/golden_dump1.h5")
        compare_data.my_dump.need_compare_input = True
        result = compare_data._check_stride('/cat/9/output/result', [218, 3, 32, 32], [3072, 1024, 1, 32])
        self.assertEqual(result, True)

    def test_check__check_stride_case2(self):
        compare_data = pytorch_dump_data.CompareData("/home/my_dump1.h5", "/home/golden_dump1.h5")
        compare_data.my_dump.need_compare_input = True
        result = compare_data._check_stride('/cat/9/output/result', [218, 3, 32, 32], [3074, 1024, 1, 32])
        self.assertEqual(result, False)

    def test_converted_stride_case1(self):
        compare_data = pytorch_dump_data.CompareData("/home/my_dump2.h5", "/home/golden_dump2.h5")
        compare_data.my_dump.need_compare_input = True

        with mock.patch('pytorch_cmp.hdf5_parser.Hdf5Parser.get_dump_data_attr',
                        side_effect=[(True, 1), (True, (1, 3))]):
            dump_data = np.array(np.arange(9)).reshape(3, 3)
            converted_dump_data = compare_data._converted_stride(
                np.array(np.arange(9)).reshape(3, 3), '/cat/9/output/result')
        self.assertEqual((dump_data.T == converted_dump_data).all(), True)

    def test_converted_stride_case2(self):
        compare_data = pytorch_dump_data.CompareData("/home/my_dump3.h5", "/home/golden_dump3.h5")

        with mock.patch('pytorch_cmp.hdf5_parser.Hdf5Parser.get_dump_data_attr',
                        side_effect=[(True, 1), (True, (3, 3))]):
            dump_data = np.array(np.arange(9)).reshape(3, 3)
            converted_dump_data = compare_data._converted_stride(
                np.array(np.arange(9)).reshape(3, 3), '/cat/9/output/result')
        self.assertEqual((dump_data == converted_dump_data).all(), True)

    def test_open_file(self):
        compare_data = pytorch_dump_data.CompareData("/home/my_dump.h5", "/home/golden_dump.h5")
        compare_data.open_file('r')

    def test_close_file(self):
        compare_data = pytorch_dump_data.CompareData("/home/my_dump.h5", "/home/golden_dump.h5")
        compare_data.close_file()

    def test_data_type_get_name(self):
        self.assertEqual(pytorch_dump_data.DataType.get_name(1), "Float")
        self.assertEqual(pytorch_dump_data.DataType.get_name(6), "Int")
        self.assertEqual(pytorch_dump_data.DataType.get_name(7), "Long")
        self.assertEqual(pytorch_dump_data.DataType.get_name(10), "Half")

    def test_data_type_get_value(self):
        self.assertEqual(pytorch_dump_data.DataType.get_value("Float"), 1)
        self.assertEqual(pytorch_dump_data.DataType.get_value("Int"), 6)
        self.assertEqual(pytorch_dump_data.DataType.get_value("Long"), 7)
        self.assertEqual(pytorch_dump_data.DataType.get_value("Half"), 10)
        self.assertIsNone(pytorch_dump_data.DataType.get_value("Unknown"))

    def test_compare_map_get_mapping_opname(self):
        cmp_map = pytorch_dump_data.CompareMap()
        result = cmp_map.get_mapping_opname("CudnnConvolutionBackward")
        self.assertEqual(result, ["NpuConvolutionBackward"])
        result = cmp_map.get_mapping_opname("NpuConvolutionBackward")
        self.assertEqual(result, ["CudnnConvolutionBackward", "ThnnConvDepthwise2DBackward"])
        result = cmp_map.get_mapping_opname("UnknownOp")
        self.assertEqual(result, [])

    def test_compare_map_get_mapping_param(self):
        cmp_map = pytorch_dump_data.CompareMap()
        result = cmp_map.get_mapping_param("CudnnBatchNormBackward", "epsilon")
        self.assertEqual(result, "eps")
        result = cmp_map.get_mapping_param("NativeBatchNormBackward", "eps")
        self.assertEqual(result, "epsilon")
        result = cmp_map.get_mapping_param("CudnnConvolutionBackward", "self")
        self.assertEqual(result, "input")
        result = cmp_map.get_mapping_param("UnknownOp", "param")
        self.assertEqual(result, "")
        result = cmp_map.get_mapping_param("CudnnBatchNormBackward", "unknown_param")
        self.assertEqual(result, "")

    def test_compare_map_get_mapping_opname_all(self):
        cmp_map = pytorch_dump_data.CompareMap()
        result = cmp_map.get_mapping_opname_all()
        self.assertIn(["NativeBatchNormBackward"], result)
        self.assertIn(["CudnnBatchNormBackward"], result)
        self.assertIn(["NpuConvolutionBackward"], result)

    def test_is_equivalent_type(self):
        compare_data = pytorch_dump_data.CompareData("/home/my_dump.h5", "/home/golden_dump.h5")
        result = compare_data._is_equivalent_type(6, 7, "test_info")
        self.assertEqual(result, True)
        result = compare_data._is_equivalent_type(1, 10, "test_info")
        self.assertEqual(result, True)
        result = compare_data._is_equivalent_type(1, 6, "test_info")
        self.assertEqual(result, False)
        result = compare_data._is_equivalent_type(2, 3, "test_info")
        self.assertEqual(result, False)

    def test_opname_map_by_map_table(self):
        compare_data = pytorch_dump_data.CompareData("/home/my_dump.h5", "/home/golden_dump.h5")
        compare_data.my_dump.ext_opname_dataset_map = {
            "NpuConvolutionBackward:0": ["/NpuConvolutionBackward/3/input/input0"],
        }
        result = compare_data._opname_map_by_map_table(
            "CudnnConvolutionBackward:0", utils_type.DeviceType.NPU.value)
        self.assertEqual(result, "NpuConvolutionBackward:0")
        result = compare_data._opname_map_by_map_table(
            "UnknownOp:0", utils_type.DeviceType.NPU.value)
        self.assertEqual(result, "")

    def test_opname_map_by_map_table_golden(self):
        compare_data = pytorch_dump_data.CompareData("/home/my_dump.h5", "/home/golden_dump.h5")
        compare_data.golden_dump.ext_opname_dataset_map = {
            "CudnnConvolutionBackward:0": ["/CudnnConvolutionBackward/3/input/input0"],
        }
        result = compare_data._opname_map_by_map_table(
            "NpuConvolutionBackward:0", utils_type.DeviceType.GPU.value)
        self.assertEqual(result, "CudnnConvolutionBackward:0")

    def test_set_compare_input_flag(self):
        compare_data = pytorch_dump_data.CompareData("/home/my_dump.h5", "/home/golden_dump.h5")
        compare_data.my_dump.need_compare_input = True
        compare_data.golden_dump.need_compare_input = False
        compare_data.set_compare_input_flag()
        self.assertEqual(compare_data.golden_dump.need_compare_input, True)

    def test_set_compare_input_flag_false(self):
        compare_data = pytorch_dump_data.CompareData("/home/my_dump.h5", "/home/golden_dump.h5")
        compare_data.my_dump.need_compare_input = False
        compare_data.golden_dump.need_compare_input = False
        compare_data.set_compare_input_flag()
        self.assertEqual(compare_data.golden_dump.need_compare_input, False)

    def test_get_original_opname_error(self):
        compare_data = pytorch_dump_data.CompareData("/home/my_dump.h5", "/home/golden_dump.h5")
        with pytest.raises(CompareError) as err:
            compare_data.get_original_opname("NoDelimiterOpName")
        self.assertEqual(err.value.code, CompareError.MSACCUCMP_NAME_ERROR)

    def test_construct_dataset_path_with_param_mapping(self):
        compare_data = pytorch_dump_data.CompareData("/home/my_dump.h5", "/home/golden_dump.h5")
        compare_data.my_dump.need_compare_input = True
        compare_data.golden_dump.need_compare_input = True
        compare_data.my_dump.file_handle = {
            "CudnnBatchNormBackward": {'3': "input/epsilon"},
        }
        compare_data.golden_dump.file_handle = {
            "NativeBatchNormBackward": {'5': "input/eps"},
        }
        compare_data.my_dump._generate_order_ext_opname_map()
        compare_data.golden_dump._generate_order_ext_opname_map()
        compare_data.my_dump.ext_opname_dataset_map = {
            'CudnnBatchNormBackward:0': ['/CudnnBatchNormBackward/3/input/epsilon'],
        }
        compare_data.golden_dump.ext_opname_dataset_map = {
            'NativeBatchNormBackward:0': ['/NativeBatchNormBackward/5/input/eps'],
        }
        expect_dataset = compare_data._construct_dataset_path(
            "CudnnBatchNormBackward:0", '/CudnnBatchNormBackward/3/input/epsilon',
            compare_data.golden_dump, "NativeBatchNormBackward:0")
        self.assertEqual(expect_dataset, '/NativeBatchNormBackward/5/input/eps')

    def test_get_golden_dataset_not_found(self):
        compare_data = pytorch_dump_data.CompareData("/home/my_dump.h5", "/home/golden_dump.h5")
        compare_data.my_dump.need_compare_input = True
        compare_data.golden_dump.need_compare_input = True
        compare_data.my_dump.file_handle = {
            "Admm1": {'3': "input/input0"},
        }
        compare_data.golden_dump.file_handle = {
            "Admm1": {'5': "input/input0"},
        }
        compare_data.my_dump._generate_order_ext_opname_map()
        compare_data.golden_dump._generate_order_ext_opname_map()
        compare_data.my_dump.ext_opname_dataset_map = {
            'Admm1:0': ['/Admm1/3/input/input0'],
        }
        compare_data.golden_dump.ext_opname_dataset_map = {
            'Admm1:0': ['/Admm1/5/input/input0'],
        }
        compare_data.golden_dump.device_type = utils_type.DeviceType.NPU.value
        found, _, message = compare_data.get_golden_dataset(
            "NonExistent:0", '/Admm1/3/input/input0')
        self.assertEqual(found, False)
        self.assertIn("No data match", message)

    def test_converted_stride_npu_device(self):
        compare_data = pytorch_dump_data.CompareData("/home/my_dump.h5", "/home/golden_dump.h5")
        with mock.patch('pytorch_cmp.hdf5_parser.Hdf5Parser.get_dump_data_attr',
                        side_effect=[(True, utils_type.DeviceType.NPU.value)]):
            dump_data = np.array(np.arange(9)).reshape(3, 3)
            converted_dump_data = compare_data._converted_stride(
                dump_data, '/cat/9/output/result')
        self.assertEqual((dump_data == converted_dump_data).all(), True)

    def test_reverse_match_non_load_mode(self):
        compare_data = pytorch_dump_data.CompareData("/home/my_dump.h5", "/home/golden_dump.h5")
        compare_data.my_dump.need_compare_input = True
        compare_data.my_dump.file_handle = {
            "Admm1": {'3': "input/input0"},
        }
        compare_data.my_dump._generate_order_ext_opname_map()
        compare_data.my_dump.ext_opname_dataset_map = {
            'Admm1:0': ['/Admm1/3/input/input0'],
        }
        failed_info = []
        result = compare_data._reverse_match_non_load_mode(
            "NonExistent:0", "/NonExistent/3/input/input0", failed_info)
        self.assertEqual(result, False)
        self.assertTrue(len(failed_info) > 0)

    def test_reverse_match_non_load_mode_load(self):
        compare_data = pytorch_dump_data.CompareData("/home/my_dump.h5", "/home/golden_dump.h5")
        compare_data.my_dump.need_compare_input = False
        failed_info = []
        result = compare_data._reverse_match_non_load_mode(
            "NonExistent:0", "/NonExistent/3/input/input0", failed_info)
        self.assertEqual(result, True)

    def test_reverse_match_process(self):
        compare_data = pytorch_dump_data.CompareData("/home/my_dump.h5", "/home/golden_dump.h5")
        compare_data.my_dump.need_compare_input = True
        compare_data.my_dump.file_handle = {
            "Admm1": {'3': "input/input0"},
        }
        compare_data.golden_dump.file_handle = {
            "Admm1": {'5': "input/input0"},
        }
        compare_data.my_dump._generate_order_ext_opname_map()
        compare_data.golden_dump._generate_order_ext_opname_map()
        compare_data.my_dump.ext_opname_dataset_map = {
            'Admm1:0': ['/Admm1/3/input/input0'],
        }
        compare_data.golden_dump.ext_opname_dataset_map = {
            'Admm1:0': ['/Admm1/5/input/input0'],
        }
        failed_info = []
        result = compare_data._reverse_match_process(
            "Admm1:0", "Admm1:0", "/Admm1/5/input/input0", failed_info)
        self.assertEqual(result, True)

    def test_reverse_match_process_not_found(self):
        compare_data = pytorch_dump_data.CompareData("/home/my_dump.h5", "/home/golden_dump.h5")
        compare_data.my_dump.need_compare_input = True
        compare_data.my_dump.file_handle = {
            "Admm1": {'3': "input/input0"},
        }
        compare_data.golden_dump.file_handle = {
            "Admm1": {'5': "input/input0"},
        }
        compare_data.my_dump._generate_order_ext_opname_map()
        compare_data.golden_dump._generate_order_ext_opname_map()
        compare_data.my_dump.ext_opname_dataset_map = {
            'Admm1:0': ['/Admm1/3/input/input0'],
        }
        compare_data.golden_dump.ext_opname_dataset_map = {
            'Admm1:0': ['/Admm1/5/input/input0'],
        }
        failed_info = []
        result = compare_data._reverse_match_process(
            "Admm1:0", "Admm1:0", "/Admm1/5/input/nonexistent", failed_info)
        self.assertEqual(result, False)
        self.assertTrue(len(failed_info) > 0)

    def test_get_dump_data_type_mismatch(self):
        compare_data = pytorch_dump_data.CompareData("/home/my_dump.h5", "/home/golden_dump.h5")
        compare_data.my_dump.need_compare_input = True
        with mock.patch('pytorch_cmp.pytorch_dump_data.CompareData._check_data_type',
                        return_value=[False, "type mismatch"]):
            with pytest.raises(CompareError) as err:
                compare_data.get_dump_data('/Admm1/3/input0', '/Admm1/5/input0')
        self.assertEqual(err.value.code, CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR)

    def test_check_stride_different_length(self):
        compare_data = pytorch_dump_data.CompareData("/home/my_dump.h5", "/home/golden_dump.h5")
        result = compare_data._check_stride('/cat/9/output/result', [218, 3, 32, 32], [3072, 1024, 1])
        self.assertEqual(result, False)

    def test_check_stride_exceed_data_num(self):
        compare_data = pytorch_dump_data.CompareData("/home/my_dump.h5", "/home/golden_dump.h5")
        result = compare_data._check_stride('/cat/9/output/result', [2, 3], [10, 1])
        self.assertEqual(result, False)

    def test_get_my_dump_datasets_not_found(self):
        compare_data = pytorch_dump_data.CompareData("/home/my_dump.h5", "/home/golden_dump.h5")
        compare_data.my_dump.ext_opname_dataset_map = {
            'Admm1:0': ['/Admm1/3/input0'],
        }
        result = compare_data.get_my_dump_datasets("NonExistent:0")
        self.assertEqual(result, [])

    def test_parse_dump_file_my_dump_error(self):
        compare_data = pytorch_dump_data.CompareData("/home/my_dump.h5", "/home/golden_dump.h5")
        compare_data.my_dump.file_handle = None
        with mock.patch('pytorch_cmp.hdf5_parser.Hdf5Parser.parse_dump_file',
                        return_value=CompareError.MSACCUCMP_PARSE_DUMP_FILE_ERROR):
            parse_result = compare_data.parse_dump_file()
        self.assertEqual(parse_result, CompareError.MSACCUCMP_PARSE_DUMP_FILE_ERROR)
