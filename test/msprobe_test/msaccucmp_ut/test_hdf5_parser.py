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

from cmp_utils.constant.compare_error import CompareError
from pytorch_cmp import hdf5_parser


class TestUtilsMethods(unittest.TestCase):
    mapping_list = [['NativeBatchNormBackward'],
                    ['CudnnBatchNormBackward'],
                    ['NpuConvolutionBackward'],
                    ['NpuConvolutionBackward'],
                    ['CudnnConvolutionBackward', 'ThnnConvDepthwise2DBackward']]

    def test_Hdf5Parser_open_file(self):
        parser = hdf5_parser.Hdf5Parser("/home/test.h5", hdf5_parser.Hdf5Parser.GOLDEN_DUMP_FILE, self.mapping_list)
        ret = parser.open_file('r')
        self.assertEqual(ret, CompareError.MSACCUCMP_OPEN_FILE_ERROR)

    def test_Hdf5Parser_close_file(self):
        parser = hdf5_parser.Hdf5Parser("/home/test.h5", hdf5_parser.Hdf5Parser.GOLDEN_DUMP_FILE, self.mapping_list)
        parser.file_handle = "testhandle"
        with pytest.raises(AttributeError):
            parser.close_file()

    def test_get_dump_data_attr1(self):
        parser = hdf5_parser.Hdf5Parser("/home/test.h5", hdf5_parser.Hdf5Parser.GOLDEN_DUMP_FILE, self.mapping_list)
        parser.file_handle = None
        with pytest.raises(CompareError) as error:
            parser.get_dump_data_attr("/Admm1/6/input1", "DataType")
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR)

    def test_get_dump_data_attr2(self):
        parser = hdf5_parser.Hdf5Parser("/home/test.h5", hdf5_parser.Hdf5Parser.GOLDEN_DUMP_FILE, self.mapping_list)
        parser.file_handle = "testhandle"
        with pytest.raises(CompareError) as error:
            parser.get_dump_data_attr("/Admm1/6/input1", "DataType")
            self.assertEqual(error.value.args[0],
                             CompareError.MSACCUCMP_PARSE_DUMP_FILE_ERROR)

    def get_dump_data1(self):
        parser = hdf5_parser.Hdf5Parser("/home/test.h5", hdf5_parser.Hdf5Parser.GOLDEN_DUMP_FILE, self.mapping_list)
        parser.file_handle = None
        with pytest.raises(CompareError) as error:
            parser.get_dump_data("/Admm1/6/input1")
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR)

    def test_get_dump_data2(self):
        parser = hdf5_parser.Hdf5Parser("/home/test.h5", hdf5_parser.Hdf5Parser.GOLDEN_DUMP_FILE, self.mapping_list)
        parser.file_handle = "testhandle"
        with pytest.raises(CompareError) as error:
            parser.get_dump_data("/Admm1/6/input1")
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_PARSE_DUMP_FILE_ERROR)

    def test_gen_single_order_ext_opname_map(self):
        parser = hdf5_parser.Hdf5Parser("/home/test.h5", hdf5_parser.Hdf5Parser.GOLDEN_DUMP_FILE, self.mapping_list)
        parser.file_handle = {
            "Admm1": {'3': "input/input0",
                      '4': "input/input1"},
            "Abxx1": {'3': "input/input2",
                      '5': "input/input3"},
        }
        order_ext_opname_map = parser._gen_single_order_ext_opname_map("Admm1")
        self.assertEqual(order_ext_opname_map[3], ["Admm1:0"])
        self.assertEqual(order_ext_opname_map[4], ["Admm1:1"])

    def test_parse_all_dataset(self):
        parser = hdf5_parser.Hdf5Parser("/home/test1.h5", hdf5_parser.Hdf5Parser.GOLDEN_DUMP_FILE, self.mapping_list)
        parser.file_handle = {
            "Admm1": {'3': "output0",
                      '4': "output1"},
            "Abxx1": {'3': "output2",
                      '5': "output3"},
            "/Admm1/3": {'output0': 1},
            "/Admm1/4": {'output1': 2},
            "/Abxx1/3": {'output2': 3},
            "/Abxx1/5": {'output3': 4},
            "/BatchNorm/6": {'output0: 6'}
        }
        parser.order_ext_opname_map = {
            3: ["Admm1:0", "Abxx1:0"],
            4: ["Admm1:1"],
            5: ["Abxx1:1"]}
        with mock.patch('pytorch_cmp.hdf5_parser.Hdf5Parser.open_file',
                        return_value=CompareError.MSACCUCMP_NONE_ERROR):
            with mock.patch('pytorch_cmp.hdf5_parser.Hdf5Parser.get_dump_data_attr',
                            side_effect=[(False, ''),
                                         (True, 0),
                                         (True, 1),
                                         (False, ''),
                                         (True, 0),
                                         (False, ''),
                                         (True, 0),
                                         (False, ''),
                                         (True, 0),
                                         (False, ''),
                                         (True, 0)]):
                parser._parse_all_dataset()
        self.assertEqual(parser.ext_opname_dataset_map['Admm1:0'], ['/Admm1/3/output0'])
        self.assertEqual(parser.ext_opname_dataset_map['Abxx1:0'], ['/Abxx1/3/output2'])
        self.assertEqual(parser.ext_opname_dataset_map['Admm1:1'], ['/Admm1/4/output1'])
        self.assertEqual(parser.ext_opname_dataset_map['Abxx1:1'], ['/Abxx1/5/output3'])

    def test_generate_order_ext_opname_map(self):
        parser = hdf5_parser.Hdf5Parser("/home/test.h5", hdf5_parser.Hdf5Parser.GOLDEN_DUMP_FILE, self.mapping_list)
        parser.file_handle = {
            "Admm1": {'3': "input/input0",
                      '4': "input/input1"},
            "Abxx1": {'3': "input/input2",
                      '5': "input/input3"},
        }
        parser._generate_order_ext_opname_map()
        self.assertEqual(parser.order_ext_opname_map[3], ["Admm1:0", "Abxx1:0"])
        self.assertEqual(parser.order_ext_opname_map[4], ["Admm1:1"])
        self.assertEqual(parser.order_ext_opname_map[5], ["Abxx1:1"])

    def test_get_all_orders(self):
        parser1 = hdf5_parser.Hdf5Parser("/home/test1.h5", hdf5_parser.Hdf5Parser.GOLDEN_DUMP_FILE, self.mapping_list)
        parser1.file_handle = {
            "Admm1": {'3': "input/input0",
                      '4': "input/input1"},
            "Abxx1": {'3': "input/input2",
                      '5': "input/input3"},
        }
        parser1._generate_order_ext_opname_map()
        orders = parser1.get_all_orders()
        self.assertEqual(list(orders), [3, 4, 5])

    def test_get_order_by_ext_opname(self):
        parser1 = hdf5_parser.Hdf5Parser("/home/test1.h5", hdf5_parser.Hdf5Parser.GOLDEN_DUMP_FILE, self.mapping_list)
        parser1.file_handle = {
            "Admm1": {'3': "input/input0",
                      '4': "input/input1"},
            "Abxx1": {'3': "input/input2",
                      '5': "input/input3"},
        }
        parser1._generate_order_ext_opname_map()
        order = parser1.get_order_by_ext_opname('Admm1:0')
        self.assertEqual(order, 3)
        order = parser1.get_order_by_ext_opname('Admm1:5')
        self.assertEqual(order, 6)

    def test_get_ext_opname_group_by_order(self):
        parser1 = hdf5_parser.Hdf5Parser("/home/test1.h5", hdf5_parser.Hdf5Parser.GOLDEN_DUMP_FILE, self.mapping_list)
        parser1.file_handle = {
            "Admm1": {'3': "input/input0",
                      '4': "input/input1"},
            "Abxx1": {'3': "input/input2",
                      '5': "input/input3"},
        }
        parser1._generate_order_ext_opname_map()
        ext_opaname = parser1.get_ext_opname_group_by_order(3)
        self.assertEqual(ext_opaname, ["Admm1:0", "Abxx1:0"])

    def test_have_dataset_case1(self):
        parser1 = hdf5_parser.Hdf5Parser("/home/test1.h5", hdf5_parser.Hdf5Parser.GOLDEN_DUMP_FILE, self.mapping_list)

        parser1.ext_opname_dataset_map = {'Admm1:1': ['/Admm1/3/input/input0']}
        parser1.order_ext_opname_map = {3: ['Admm1:1']}

        ret = parser1.have_dataset('Admm1:1', '/Admm1/3/input/input0')
        self.assertEqual(ret, True)

    def test_have_dataset_case2(self):
        parser1 = hdf5_parser.Hdf5Parser("/home/test1.h5", hdf5_parser.Hdf5Parser.GOLDEN_DUMP_FILE, self.mapping_list)

        parser1.ext_opname_dataset_map = {'Admm1:2': ['/Admm1/3/input/input0']}
        parser1.order_ext_opname_map = {3: ['Admm1:2']}

        ret = parser1.have_dataset('Admm1:1', '/Admm1/3/input/input0')
        self.assertEqual(ret, True)

    def test_have_dataset_case3(self):
        parser1 = hdf5_parser.Hdf5Parser("/home/test1.h5", hdf5_parser.Hdf5Parser.GOLDEN_DUMP_FILE, self.mapping_list)

        parser1.ext_opname_dataset_map = {'Admm1:2': ['/Admm1/3/input/input0']}
        parser1.order_ext_opname_map = {3: ['Admm1:2']}

        ret = parser1.have_dataset('Admm1:1', '/Admm1/4/input/input0')
        self.assertEqual(ret, False)

    def test_file_is_empty(self):
        parser = hdf5_parser.Hdf5Parser("/home/test.h5", hdf5_parser.Hdf5Parser.GOLDEN_DUMP_FILE, self.mapping_list)
        ret = parser.file_is_empty()
        self.assertEqual(ret, True)

    def test_is_load_mode(self):
        parser = hdf5_parser.Hdf5Parser("/home/test.h5", hdf5_parser.Hdf5Parser.GOLDEN_DUMP_FILE, self.mapping_list)
        parser.need_compare_input = False
        ret = parser.is_load_mode()
        self.assertEqual(ret, True)

    def test_check_value(self):
        parser = hdf5_parser.Hdf5Parser("/home/test.h5", hdf5_parser.Hdf5Parser.GOLDEN_DUMP_FILE, self.mapping_list)
        with pytest.raises(CompareError) as error:
            tmp = [None] * 1000001
            parser._check_value(tmp)
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INDEX_OUT_OF_BOUNDS_ERROR)

    def test_is_parsed(self):
            parser_map = hdf5_parser.Hdf5Parser("/home/test1.h5", hdf5_parser.Hdf5Parser.GOLDEN_DUMP_FILE,
                                                self.mapping_list)
            parser_map.ext_opname_dataset_map = {'Admm1:2': ['/Admm1/3/input/input0']}
            parser_map.order_ext_opname_map = {3: ['Admm1:2']}

            result = parser_map._is_parsed('Admm1')
            self.assertEqual(result, True)
            result = parser_map._is_parsed('Admm2')
            self.assertEqual(result, False)

    def test_gen_ext_opname_map_special(self):
        parser_map = hdf5_parser.Hdf5Parser("/home/test1.h5", hdf5_parser.Hdf5Parser.GOLDEN_DUMP_FILE,
                                            self.mapping_list)
        parser_map.file_handle = {
            "CudnnAdmm1": {'3': "input/input0",
                           '4': "input/input1"},
            "ThnnAdmm1": {'5': "input/input2",
                          '6': "input/input3"},
        }
        multimap_set = ['CudnnAdmm1', 'ThnnAdmm1']
        opname = 'CudnnAdmm1'

        result = parser_map._gen_ext_opname_map_special(opname, multimap_set)
        expect_result = {3: ['CudnnAdmm1:0'], 4: ['CudnnAdmm1:1'], 5: ['ThnnAdmm1:2'], 6: ['ThnnAdmm1:3']}
        self.assertEqual(result, expect_result)

    def test_dataset_path_valid_true(self):
        parser = hdf5_parser.Hdf5Parser("/home/test.h5", hdf5_parser.Hdf5Parser.GOLDEN_DUMP_FILE, self.mapping_list)
        self.assertTrue(parser._dataset_path_valid("/Admm1/6/input"))

    def test_dataset_path_valid_false(self):
        parser = hdf5_parser.Hdf5Parser("/home/test.h5", hdf5_parser.Hdf5Parser.GOLDEN_DUMP_FILE, self.mapping_list)
        self.assertFalse(parser._dataset_path_valid("invalid_path"))

    def test_data_path_is_input_true(self):
        parser = hdf5_parser.Hdf5Parser("/home/test.h5", hdf5_parser.Hdf5Parser.GOLDEN_DUMP_FILE, self.mapping_list)
        self.assertTrue(parser._data_path_is_input("/Admm1/6/input"))

    def test_data_path_is_input_false(self):
        parser = hdf5_parser.Hdf5Parser("/home/test.h5", hdf5_parser.Hdf5Parser.GOLDEN_DUMP_FILE, self.mapping_list)
        self.assertFalse(parser._data_path_is_input("/Admm1/6/output"))

    def test_data_path_is_input_invalid(self):
        parser = hdf5_parser.Hdf5Parser("/home/test.h5", hdf5_parser.Hdf5Parser.GOLDEN_DUMP_FILE, self.mapping_list)
        self.assertFalse(parser._data_path_is_input("invalid"))

    def test_get_mapping_set_found(self):
        parser = hdf5_parser.Hdf5Parser("/home/test.h5", hdf5_parser.Hdf5Parser.GOLDEN_DUMP_FILE, self.mapping_list)
        result = parser._get_mapping_set("CudnnConvolutionBackward")
        self.assertEqual(result, ['CudnnConvolutionBackward', 'ThnnConvDepthwise2DBackward'])

    def test_get_mapping_set_not_found(self):
        parser = hdf5_parser.Hdf5Parser("/home/test.h5", hdf5_parser.Hdf5Parser.GOLDEN_DUMP_FILE, self.mapping_list)
        result = parser._get_mapping_set("UnknownOp")
        self.assertEqual(result, [])

    def test_get_mapping_set_single_element(self):
        parser = hdf5_parser.Hdf5Parser("/home/test.h5", hdf5_parser.Hdf5Parser.GOLDEN_DUMP_FILE, self.mapping_list)
        result = parser._get_mapping_set("NativeBatchNormBackward")
        self.assertEqual(result, [])

    def test_is_multimap_true(self):
        self.assertTrue(hdf5_parser.Hdf5Parser._is_multimap("CudnnConvolutionBackward",
                                                            ['CudnnConvolutionBackward', 'ThnnConvDepthwise2DBackward']))

    def test_is_multimap_false(self):
        self.assertFalse(hdf5_parser.Hdf5Parser._is_multimap("UnknownOp",
                                                              ['CudnnConvolutionBackward', 'ThnnConvDepthwise2DBackward']))

    def test_need_compare_output_path(self):
        parser = hdf5_parser.Hdf5Parser("/home/test.h5", hdf5_parser.Hdf5Parser.GOLDEN_DUMP_FILE, self.mapping_list)
        self.assertTrue(parser._need_compare("/Admm1/6/output"))

    def test_need_compare_my_dump_input(self):
        parser = hdf5_parser.Hdf5Parser("/home/test.h5", hdf5_parser.Hdf5Parser.MY_DUMP_FILE, self.mapping_list)
        self.assertTrue(parser._need_compare("/Admm1/6/input"))
        self.assertTrue(parser.need_compare_input)

    def test_need_compare_golden_dump_input_first_time(self):
        parser = hdf5_parser.Hdf5Parser("/home/test.h5", hdf5_parser.Hdf5Parser.GOLDEN_DUMP_FILE, self.mapping_list)
        parser.need_compare_input = False
        self.assertFalse(parser._need_compare("/Admm1/6/input"))

    def test_need_compare_golden_dump_input_second_time(self):
        parser = hdf5_parser.Hdf5Parser("/home/test.h5", hdf5_parser.Hdf5Parser.GOLDEN_DUMP_FILE, self.mapping_list)
        parser.need_compare_input = True
        self.assertTrue(parser._need_compare("/Admm1/6/input"))

    def test_parse_one_dataset_tensor(self):
        parser = hdf5_parser.Hdf5Parser("/home/test.h5", hdf5_parser.Hdf5Parser.GOLDEN_DUMP_FILE, self.mapping_list)
        parser.file_handle = {"dummy": "handle"}
        with mock.patch.object(parser, 'get_dump_data_attr', return_value=(False, None)):
            parser._parse_one_dataset(hdf5_parser.DataSetType.TENSOR.value, "Admm1:0", "/Admm1/3/output0")
        self.assertEqual(parser.ext_opname_dataset_map["Admm1:0"], ["/Admm1/3/output0"])

    def test_parse_one_dataset_non_tensor(self):
        parser = hdf5_parser.Hdf5Parser("/home/test.h5", hdf5_parser.Hdf5Parser.GOLDEN_DUMP_FILE, self.mapping_list)
        parser._parse_one_dataset(hdf5_parser.DataSetType.VEC_TENSOR.value, "Admm1:0", "/Admm1/3/output0")
        self.assertEqual(parser.ext_opname_dataset_map.get("Admm1:0", []), [])

    def test_parse_one_dataset_with_device_type(self):
        parser = hdf5_parser.Hdf5Parser("/home/test.h5", hdf5_parser.Hdf5Parser.GOLDEN_DUMP_FILE, self.mapping_list)
        with mock.patch.object(parser, 'get_dump_data_attr', return_value=(True, 10)):
            parser._parse_one_dataset(hdf5_parser.DataSetType.TENSOR.value, "Admm1:0", "/Admm1/3/output0")
        self.assertEqual(parser.device_type, 10)

    def test_parse_dataset_recursively_with_attr(self):
        parser = hdf5_parser.Hdf5Parser("/home/test.h5", hdf5_parser.Hdf5Parser.GOLDEN_DUMP_FILE, self.mapping_list)
        with mock.patch.object(parser, 'get_dump_data_attr', return_value=(True, hdf5_parser.DataSetType.TENSOR.value)):
            with mock.patch.object(parser, '_parse_one_dataset') as mock_parse:
                parser._parse_dataset_recursively("Admm1:0", "/Admm1/3/output0")
                mock_parse.assert_called_once_with(hdf5_parser.DataSetType.TENSOR.value, "Admm1:0", "/Admm1/3/output0")

    def test_parse_dataset_recursively_without_attr(self):
        parser = hdf5_parser.Hdf5Parser("/home/test.h5", hdf5_parser.Hdf5Parser.GOLDEN_DUMP_FILE, self.mapping_list)
        parser.file_handle = {
            "/Admm1/3": {"output0": 1}
        }
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] > 1:
                return None
            return parser._parse_dataset_recursively(*args, **kwargs)

        with mock.patch.object(parser, 'get_dump_data_attr', return_value=(False, None)):
            with mock.patch.object(parser, '_parse_dataset_recursively', side_effect=side_effect):
                parser._parse_dataset_recursively("Admm1:0", "/Admm1/3")

    def test_parse_dataset_recursively_file_handle_none(self):
        parser = hdf5_parser.Hdf5Parser("/home/test.h5", hdf5_parser.Hdf5Parser.GOLDEN_DUMP_FILE, self.mapping_list)
        parser.file_handle = None
        parser.ext_opname_dataset_map = {"Admm1:0": ["/Admm1/3/output0"]}
        with pytest.raises(CompareError) as error:
            parser._parse_dataset_recursively("Admm1:0", "/Admm1/3")
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR)

    def test_get_dump_data_valid_path(self):
        parser = hdf5_parser.Hdf5Parser("/home/test.h5", hdf5_parser.Hdf5Parser.GOLDEN_DUMP_FILE, self.mapping_list)
        parser.file_handle = {"Admm1/6/input1": mock.MagicMock()}
        parser.file_handle["Admm1/6/input1"].__getitem__ = mock.MagicMock(return_value=[1, 2, 3])
        with mock.patch.object(parser, '_dataset_path_valid', return_value=True):
            result = parser.get_dump_data("Admm1/6/input1")
            self.assertEqual(result, [1, 2, 3])

    def test_get_dump_data_invalid_path(self):
        parser = hdf5_parser.Hdf5Parser("/home/test.h5", hdf5_parser.Hdf5Parser.GOLDEN_DUMP_FILE, self.mapping_list)
        parser.file_handle = "testhandle"
        with mock.patch.object(parser, '_dataset_path_valid', return_value=False):
            result = parser.get_dump_data("invalid_path")
            self.assertEqual(result, [])

    def test_get_order_by_ext_opname_empty(self):
        parser = hdf5_parser.Hdf5Parser("/home/test.h5", hdf5_parser.Hdf5Parser.GOLDEN_DUMP_FILE, self.mapping_list)
        order = parser.get_order_by_ext_opname("Admm1:0")
        self.assertEqual(order, 0)

    def test_have_dataset_invalid_ext_opname(self):
        parser = hdf5_parser.Hdf5Parser("/home/test.h5", hdf5_parser.Hdf5Parser.GOLDEN_DUMP_FILE, self.mapping_list)
        with pytest.raises(CompareError) as error:
            parser.have_dataset("Admm1", "/Admm1/3/output0")
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_NAME_ERROR)

    def test_parse_dump_file_success(self):
        parser = hdf5_parser.Hdf5Parser("/home/test.h5", hdf5_parser.Hdf5Parser.GOLDEN_DUMP_FILE, self.mapping_list)
        with mock.patch.object(parser, 'open_file', return_value=CompareError.MSACCUCMP_NONE_ERROR):
            with mock.patch.object(parser, '_generate_order_ext_opname_map'):
                with mock.patch.object(parser, '_parse_all_dataset'):
                    ret = parser.parse_dump_file()
                    self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_parse_dump_file_fail(self):
        parser = hdf5_parser.Hdf5Parser("/home/test.h5", hdf5_parser.Hdf5Parser.GOLDEN_DUMP_FILE, self.mapping_list)
        with mock.patch.object(parser, 'open_file', return_value=CompareError.MSACCUCMP_OPEN_FILE_ERROR):
            ret = parser.parse_dump_file()
            self.assertEqual(ret, CompareError.MSACCUCMP_OPEN_FILE_ERROR)

    def test_close_file_success(self):
        parser = hdf5_parser.Hdf5Parser("/home/test.h5", hdf5_parser.Hdf5Parser.GOLDEN_DUMP_FILE, self.mapping_list)
        mock_handle = mock.MagicMock()
        parser.file_handle = mock_handle
        parser.close_file()
        mock_handle.close.assert_called_once()
        self.assertIsNone(parser.file_handle)

    def test_file_is_empty_with_handle(self):
        parser = hdf5_parser.Hdf5Parser("/home/test.h5", hdf5_parser.Hdf5Parser.GOLDEN_DUMP_FILE, self.mapping_list)
        parser.file_handle = {"Admm1": {}}
        ret = parser.file_is_empty()
        self.assertTrue(ret)

    def test_is_load_mode_false(self):
        parser = hdf5_parser.Hdf5Parser("/home/test.h5", hdf5_parser.Hdf5Parser.GOLDEN_DUMP_FILE, self.mapping_list)
        parser.need_compare_input = True
        ret = parser.is_load_mode()
        self.assertFalse(ret)

    def test_gen_single_order_ext_opname_map_file_none(self):
        parser = hdf5_parser.Hdf5Parser("/home/test.h5", hdf5_parser.Hdf5Parser.GOLDEN_DUMP_FILE, self.mapping_list)
        parser.file_handle = None
        result = parser._gen_single_order_ext_opname_map("Admm1")
        self.assertEqual(result, {})

    def test_gen_ext_opname_map_special_already_parsed(self):
        parser = hdf5_parser.Hdf5Parser("/home/test.h5", hdf5_parser.Hdf5Parser.GOLDEN_DUMP_FILE, self.mapping_list)
        parser.order_ext_opname_map = {3: ["Admm1:0"]}
        result = parser._gen_ext_opname_map_special("Admm1", ["Admm1"])
        self.assertEqual(result, {})

    def test_generate_order_ext_opname_map_file_none(self):
        parser = hdf5_parser.Hdf5Parser("/home/test.h5", hdf5_parser.Hdf5Parser.GOLDEN_DUMP_FILE, self.mapping_list)
        parser.file_handle = None
        parser._generate_order_ext_opname_map()
        self.assertEqual(parser.order_ext_opname_map, {})


if __name__ == '__main__':
    unittest.main()

