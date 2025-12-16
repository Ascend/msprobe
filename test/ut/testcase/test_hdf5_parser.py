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


if __name__ == '__main__':
    unittest.main()

