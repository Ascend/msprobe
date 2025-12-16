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

from dump_parse import dump, mapping
from cmp_utils.constant.compare_error import CompareError


class TestUtilsMethods(unittest.TestCase):
    def test_check_arguments_valid1(self):
        with pytest.raises(CompareError) as error:
            with mock.patch('cmp_utils.path_check.check_path_valid', return_value=1):
                dump_info = dump.DumpInfo('/home', 1)
                dump_info.check_arguments_valid()
        self.assertEqual(error.value.code, 1)

    def test_check_arguments_valid2(self):
        with mock.patch('cmp_utils.path_check.check_path_valid', return_value=0):
            dump_info = dump.DumpInfo('/home', 1)
            dump_info._make_op_name_to_file_map = mock.Mock()
            dump_info.check_arguments_valid()

    def test_match_dump_pattern1(self):
        pattern = r"^([A-Za-z0-9_-]+\.[0-9]+)\.[0-9]{1,255}\.npy$"
        item = "demo.npy"
        info = 'op_name.output_index.timestamp.npy'
        expect = 3
        dump_info = dump.DumpInfo('/home', 1)
        with pytest.raises(CompareError) as error:
            dump_info._match_dump_pattern(pattern, item, info, expect)
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_DUMP_FILE_ERROR)

    def test_match_dump_pattern2(self):
        pattern = r"^([A-Za-z0-9_-]+\.[0-9]+)\.[0-9]{1,255}\.npy$"
        item = "demo.npy"
        info = 'op_name.output_index.timestamp.npy'
        expect = 3
        with pytest.raises(CompareError) as error:
            dump_info = dump.DumpInfo('/home', 1)
            dump_info.hash_to_file_name_map = {"1": item}
            dump_info._match_dump_pattern(pattern, item, info, expect)
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_DUMP_FILE_ERROR)

    def test_match_dump_pattern3(self):
        pattern = r"^([A-Za-z0-9_-]+\.[0-9]+)\.[0-9]{1,255}\.npy$"
        item = "Add.0.1223242.npy"
        info = 'op_name.output_index.timestamp.npy'
        expect = 3
        dump_info = dump.DumpInfo('/home', 1)
        dump_info._match_dump_pattern(pattern, item, info, expect)

    def test_check_dump_file_is_quant1(self):
        dump_type = dump.DumpType(0)
        op_name = "xxxz_dequant_layer"
        dump_info = dump.DumpInfo('/home', 1)
        dump_info._check_dump_file_is_quant(dump_type, op_name)

    def test_check_dump_file_is_quant2(self):
        dump_type = dump.DumpType(3)
        op_name = "xxxz_dequant_layer"
        dump_info = dump.DumpInfo('/home', 1)
        dump_info._check_dump_file_is_quant(dump_type, op_name)

    def test_check_file_match_pattern1(self):
        dump_info = dump.DumpInfo('/home', 1)
        item = "Add.0.1223242.npy"
        dump_info._match_dump_pattern = mock.Mock
        dump_info._check_file_match_pattern(item)

    def test_check_file_match_pattern2(self):
        dump_info = dump.DumpInfo('/home', 1)
        item = "Add.0.1223242.pb"
        dump_info._match_dump_pattern = mock.Mock
        dump_info._check_file_match_pattern(item)

    def test_check_file_match_pattern3(self):
        dump_info = dump.DumpInfo('/home', 1)
        item = "Add.0.1223242.quant"
        dump_info._match_dump_pattern = mock.Mock
        dump_info._check_file_match_pattern(item)

    def test_check_file_match_pattern4(self):
        dump_info = dump.DumpInfo('/home', 1)
        item = "12345232833"
        dump_info._match_dump_pattern = mock.Mock
        dump_info._check_file_match_pattern(item)

    def test_make_op_name_to_file_map(self):
        dump_info = dump.DumpInfo('/home', 1)
        dump_info._read_mapping_file = mock.Mock()
        dump_info._check_file_match_pattern = mock.Mock(return_value=["Add", 0])
        dump_info._check_dump_file_is_quant = mock.Mock()
        dump_info._judge_dump_type = mock.Mock()
        with mock.patch("os.path.getsize", return_value=1024):
            with mock.patch("os.listdir", return_value=["mapping.csv", '23423125315', "Add.0.1223242.npy"]), \
                 mock.patch("os.path.isfile", return_value=True), \
                 mock.patch("os.path.exists", return_value=False):
                dump_info._make_op_name_to_file_map()

    def test_get_op_dump_file1(self):
        op_name = "Add_dmeo"
        dump_info = dump.DumpInfo('/home', 1)
        with pytest.raises(CompareError) as error:
            dump_info.get_op_dump_file(op_name)
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR)

    def test_get_op_dump_file2(self):
        op_name = "Add_dmeo"
        dump_info = dump.DumpInfo('/home', 1)
        dump_info.op_name_to_file_map = {op_name: []}
        with pytest.raises(CompareError) as error:
            dump_info.get_op_dump_file(op_name)
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR)

    def test_get_op_dump_file3(self):
        op_name = "Add_dmeo"
        dump_info = dump.DumpInfo('/home', 1)
        dump_info.type = mock.Mock()
        dump_info.type.name = "input"
        dump_info.op_name_to_file_map = {op_name: ["/home/demo"]}
        dump_info.get_op_dump_file(op_name)

    def test_get_op_dump_file4(self):
        op_name = "Add_dmeo"
        dump_info = dump.DumpInfo('/home', 1)
        dump_info.type = mock.Mock()
        dump_info.type.name = "input"
        dump_info.op_name_to_file_map = {op_name: ["/home/demo", "/home/test"]}
        dump_path, _ = dump_info.get_op_dump_file(op_name)
        self.assertEqual(dump_path[-1], "/home/test")

    def test_get_op_dump_data(self):
        op_name = "Add_dmeo"
        dump_info = dump.DumpInfo('/home', 1)
        dump_info.type = dump.DumpType.Quant
        dump_info.get_op_dump_file = mock.Mock(return_value=(["/home/demo"], 0))
        with mock.patch("dump_parse.dump_utils.parse_dump_file", return_value=1):
            dump_file_path, dump_data = dump_info.get_op_dump_data(op_name)
        self.assertEqual(dump_file_path, "/home/demo")
        self.assertEqual(dump_data, 1)

    def test_get_data_info1(self):
        dump_info = dump.DumpInfo('/home', 1)
        dump_info.type = dump.DumpType.Offline
        dump_info.get_data_info()
        self.assertEqual(dump_info.data_info, 'side is dump data of the unquantized model executed on the AI processor')

    def test_get_data_info2(self):
        dump_info = dump.DumpInfo('/home', 1)
        dump_info.type = dump.DumpType.Standard
        dump_info.get_data_info()
        self.assertEqual(dump_info.data_info, 'side is dump data of the unquantized original model')

    def test_get_data_info3(self):
        dump_info = dump.DumpInfo('/home', 1)
        dump_info.type = dump.DumpType.Quant
        dump_info.get_data_info()
        self.assertEqual(dump_info.data_info, 'side is dump data of the quantized original model')

    def test_judge_dump_type1(self):
        dump_info = dump.DumpInfo('/home', 1)
        with pytest.raises(CompareError) as error:
            dump_info._judge_dump_type()
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_DUMP_FILE_ERROR)

    def test_judge_dump_type2(self):
        dump_info = dump.DumpInfo('/home', 1)
        dump_info.type = dump.DumpType.Numpy
        dump_info._judge_dump_type()

    def test_judge_dump_type3(self):
        dump_info = dump.DumpInfo('/home', 1)
        dump_info.type = dump.DumpType.Numpy
        dump_info.quant = mock.Mock()
        dump_info._judge_dump_type()

    def test_check_offline_standard_valid1(self):
        info = "demo"
        fusion_json_file_path = ""
        quant_fusion_rule_file_path = ""
        close_fusion_rule_file_path = ""
        dump_compare_data = dump.CompareData('/home/left', '/home/right', 2)
        with pytest.raises(CompareError) as error:
            dump_compare_data._check_offline_standard_valid(info, fusion_json_file_path, quant_fusion_rule_file_path,
                                                            close_fusion_rule_file_path)
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_check_offline_standard_valid2(self):
        info = "demo"
        fusion_json_file_path = "/home/demo/1.json"
        quant_fusion_rule_file_path = ""
        close_fusion_rule_file_path = ""
        dump_compare_data = dump.CompareData('/home/left', '/home/right', 2)
        dump_compare_data.left_dump_info.quant = dump.DumpType.Quant
        with pytest.raises(CompareError) as error:
            dump_compare_data._check_offline_standard_valid(info, fusion_json_file_path, quant_fusion_rule_file_path,
                                                            close_fusion_rule_file_path)
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_check_offline_standard_valid3(self):
        info = "demo"
        fusion_json_file_path = "/home/demo/1.json"
        quant_fusion_rule_file_path = "/home/demo/2.json"
        close_fusion_rule_file_path = ""
        dump_compare_data = dump.CompareData('/home/left', '/home/right', 2)
        with pytest.raises(CompareError) as error:
            dump_compare_data._check_offline_standard_valid(info, fusion_json_file_path, quant_fusion_rule_file_path,
                                                            close_fusion_rule_file_path)
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_check_offline_standard_valid4(self):
        info = "demo"
        fusion_json_file_path = "/home/demo/1.json"
        quant_fusion_rule_file_path = "/home/demo/2.json"
        close_fusion_rule_file_path = "/home/demo/3.json"
        dump_compare_data = dump.CompareData('/home/left', '/home/right', 2)
        with pytest.raises(CompareError) as error:
            dump_compare_data._check_offline_standard_valid(info, fusion_json_file_path, quant_fusion_rule_file_path,
                                                            close_fusion_rule_file_path)
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_check_offline_quant_valid1(self):
        info = "demo"
        fusion_json_file_path = ""
        quant_fusion_rule_file_path = ""
        close_fusion_rule_file_path = ""
        dump_compare_data = dump.CompareData('/home/left', '/home/right', 2)
        with pytest.raises(CompareError) as error:
            dump_compare_data._check_offline_quant_valid(info, fusion_json_file_path, quant_fusion_rule_file_path,
                                                         close_fusion_rule_file_path)
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_DUMP_TYPE_ERROR)

    def test_check_offline_quant_valid2(self):
        info = "demo"
        fusion_json_file_path = ""
        quant_fusion_rule_file_path = ""
        close_fusion_rule_file_path = ""
        dump_compare_data = dump.CompareData('/home/left', '/home/right', 2)
        dump_compare_data.left_dump_info.quant = dump.DumpType.Quant
        with pytest.raises(CompareError) as error:
            dump_compare_data._check_offline_quant_valid(info, fusion_json_file_path, quant_fusion_rule_file_path,
                                                         close_fusion_rule_file_path)
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_check_offline_quant_valid3(self):
        info = "demo"
        fusion_json_file_path = "/home/demo/1.json"
        quant_fusion_rule_file_path = "/home/demo/2.json"
        close_fusion_rule_file_path = ""
        dump_compare_data = dump.CompareData('/home/left', '/home/right', 2)
        dump_compare_data.left_dump_info.quant = dump.DumpType.Quant
        with pytest.raises(CompareError) as error:
            dump_compare_data._check_offline_quant_valid(info, fusion_json_file_path, quant_fusion_rule_file_path,
                                                         close_fusion_rule_file_path)
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_check_offline_quant_valid4(self):
        info = "demo"
        fusion_json_file_path = "/home/demo/1.json"
        quant_fusion_rule_file_path = ""
        close_fusion_rule_file_path = "/home/demo/2.json"
        dump_compare_data = dump.CompareData('/home/left', '/home/right', 2)
        dump_compare_data.left_dump_info.quant = dump.DumpType.Quant
        with pytest.raises(CompareError) as error:
            dump_compare_data._check_offline_quant_valid(info, fusion_json_file_path, quant_fusion_rule_file_path,
                                                         close_fusion_rule_file_path)
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_check_left_type_offline_valid1(self):
        info = "demo"
        fusion_json_file_path = "/home/demo/1.json"
        quant_fusion_rule_file_path = "/home/demo/2.json"
        close_fusion_rule_file_path = ""
        dump_compare_data = dump.CompareData('/home/left', '/home/right', 2)
        dump_compare_data.right_dump_info.type = dump.DumpType.Offline
        dump_compare_data.left_dump_info.quant = True
        with pytest.raises(CompareError) as error:
            dump_compare_data._check_left_type_offline_valid(info, fusion_json_file_path, quant_fusion_rule_file_path,
                                                             close_fusion_rule_file_path)
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_DUMP_TYPE_ERROR)

    def test_check_left_type_offline_valid2(self):
        info = "demo"
        fusion_json_file_path = "/home/demo/1.json"
        quant_fusion_rule_file_path = "/home/demo/2.json"
        close_fusion_rule_file_path = ""
        dump_compare_data = dump.CompareData('/home/left', '/home/right', 2)
        dump_compare_data.right_dump_info.type = dump.DumpType.Offline
        dump_compare_data.left_dump_info.quant = False
        with pytest.raises(CompareError) as error:
            dump_compare_data._check_left_type_offline_valid(info, fusion_json_file_path, quant_fusion_rule_file_path,
                                                             close_fusion_rule_file_path)
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_check_left_type_offline_valid3(self):
        info = "demo"
        fusion_json_file_path = "/home/demo/1.json"
        quant_fusion_rule_file_path = "/home/demo/2.json"
        close_fusion_rule_file_path = ""
        dump_compare_data = dump.CompareData('/home/left', '/home/right', 2)
        dump_compare_data.right_dump_info.type = dump.DumpType.Standard
        dump_compare_data.left_dump_info.quant = False
        dump_compare_data._check_offline_standard_valid = mock.Mock()
        dump_compare_data._check_left_type_offline_valid(info, fusion_json_file_path, quant_fusion_rule_file_path,
                                                         close_fusion_rule_file_path)

    def test_check_left_type_offline_valid4(self):
        info = "demo"
        fusion_json_file_path = "/home/demo/1.json"
        quant_fusion_rule_file_path = "/home/demo/2.json"
        close_fusion_rule_file_path = ""
        dump_compare_data = dump.CompareData('/home/left', '/home/right', 2)
        dump_compare_data.right_dump_info.type = dump.DumpType.Quant
        dump_compare_data.left_dump_info.quant = False
        dump_compare_data._check_offline_quant_valid = mock.Mock()
        dump_compare_data._check_left_type_offline_valid(info, fusion_json_file_path, quant_fusion_rule_file_path,
                                                         close_fusion_rule_file_path)

    def test_check_left_type_offline_valid5(self):
        info = "demo"
        fusion_json_file_path = "/home/demo/1.json"
        quant_fusion_rule_file_path = ""
        close_fusion_rule_file_path = ""
        dump_compare_data = dump.CompareData('/home/left', '/home/right', 2)
        dump_compare_data.right_dump_info.type = dump.DumpType.Offline
        dump_compare_data.left_dump_info.quant = False
        with pytest.raises(CompareError) as error:
            dump_compare_data._check_left_type_offline_valid(info, fusion_json_file_path, quant_fusion_rule_file_path,
                                                             close_fusion_rule_file_path)
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_check_left_type_offline_valid6(self):
        info = "demo"
        fusion_json_file_path = ""
        quant_fusion_rule_file_path = ""
        close_fusion_rule_file_path = "/home/demo/1.json"
        dump_compare_data = dump.CompareData('/home/left', '/home/right', 2)
        dump_compare_data.right_dump_info.type = dump.DumpType.Offline
        dump_compare_data.left_dump_info.quant = False
        with pytest.raises(CompareError) as error:
            dump_compare_data._check_left_type_offline_valid(info, fusion_json_file_path, quant_fusion_rule_file_path,
                                                             close_fusion_rule_file_path)
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_check_left_type_quant_valid1(self):
        info = "demo"
        fusion_json_file_path = "/home/demo/1.json"
        quant_fusion_rule_file_path = "/home/demo/2.json"
        close_fusion_rule_file_path = ""
        dump_compare_data = dump.CompareData('/home/left', '/home/right', 2)
        dump_compare_data.right_dump_info.type = dump.DumpType.Offline
        with pytest.raises(CompareError) as error:
            dump_compare_data._check_left_type_quant_valid(info, fusion_json_file_path, quant_fusion_rule_file_path,
                                                           close_fusion_rule_file_path)
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_DUMP_TYPE_ERROR)

    def test_check_left_type_quant_valid2(self):
        info = "demo"
        fusion_json_file_path = "/home/demo/1.json"
        quant_fusion_rule_file_path = ""
        close_fusion_rule_file_path = ""
        dump_compare_data = dump.CompareData('/home/left', '/home/right', 2)
        dump_compare_data.right_dump_info.type = dump.DumpType.Standard
        with pytest.raises(CompareError) as error:
            dump_compare_data._check_left_type_quant_valid(info, fusion_json_file_path, quant_fusion_rule_file_path,
                                                           close_fusion_rule_file_path)
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_DUMP_TYPE_ERROR)

    def test_check_left_type_quant_valid3(self):
        info = "demo"
        fusion_json_file_path = "/home/demo/1.json"
        quant_fusion_rule_file_path = "/home/demo/2.json"
        close_fusion_rule_file_path = ""
        dump_compare_data = dump.CompareData('/home/left', '/home/right', 2)
        dump_compare_data.right_dump_info.type = dump.DumpType.Standard
        with pytest.raises(CompareError) as error:
            dump_compare_data._check_left_type_quant_valid(info, fusion_json_file_path, quant_fusion_rule_file_path,
                                                           close_fusion_rule_file_path)
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_DUMP_TYPE_ERROR)

    def test_check_left_type_quant_valid4(self):
        info = "demo"
        fusion_json_file_path = "/home/demo/1.json"
        quant_fusion_rule_file_path = "/home/demo/2.json"
        close_fusion_rule_file_path = "/home/demo/3.json"
        dump_compare_data = dump.CompareData('/home/left', '/home/right', 2)
        dump_compare_data.right_dump_info.type = dump.DumpType.Standard
        with pytest.raises(CompareError) as error:
            dump_compare_data._check_left_type_quant_valid(info, fusion_json_file_path, quant_fusion_rule_file_path,
                                                           close_fusion_rule_file_path)
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_DUMP_TYPE_ERROR)

    def test_compare_data_check_arguments_valid1(self):
        fusion_json_file_path = ""
        quant_fusion_rule_file_path = ""
        close_fusion_rule_file_path = ""
        dump_compare_data = dump.CompareData('/home/left', '/home/right', 2)
        dump_compare_data.left_dump_info.type = dump.DumpType.Offline
        dump_compare_data.left_dump_info.check_arguments_valid = mock.Mock()
        dump_compare_data.right_dump_info.check_arguments_valid = mock.Mock()
        dump_compare_data.left_dump_info.get_data_info = mock.Mock(return_value="left_demo")
        dump_compare_data.right_dump_info.get_data_info = mock.Mock(return_value="right_demo")
        dump_compare_data._check_left_type_offline_valid = mock.Mock()
        dump_compare_data.check_arguments_valid(fusion_json_file_path, quant_fusion_rule_file_path,
                                                close_fusion_rule_file_path)

    def test_compare_data_check_arguments_valid2(self):
        fusion_json_file_path = ""
        quant_fusion_rule_file_path = ""
        close_fusion_rule_file_path = ""
        dump_compare_data = dump.CompareData('/home/left', '/home/right', 2)
        dump_compare_data.left_dump_info.type = dump.DumpType.Quant
        dump_compare_data.left_dump_info.check_arguments_valid = mock.Mock()
        dump_compare_data.right_dump_info.check_arguments_valid = mock.Mock()
        dump_compare_data.left_dump_info.get_data_info = mock.Mock(return_value="left_demo")
        dump_compare_data.right_dump_info.get_data_info = mock.Mock(return_value="right_demo")
        dump_compare_data._check_left_type_quant_valid = mock.Mock()
        dump_compare_data.check_arguments_valid(fusion_json_file_path, quant_fusion_rule_file_path,
                                                close_fusion_rule_file_path)

    def test_compare_data_check_arguments_valid3(self):
        fusion_json_file_path = ""
        quant_fusion_rule_file_path = ""
        close_fusion_rule_file_path = ""
        dump_compare_data = dump.CompareData('/home/left', '/home/right', 2)
        dump_compare_data.left_dump_info.type = dump.DumpType.Standard
        dump_compare_data.left_dump_info.check_arguments_valid = mock.Mock()
        dump_compare_data.right_dump_info.check_arguments_valid = mock.Mock()
        dump_compare_data.left_dump_info.get_data_info = mock.Mock(return_value="left_demo")
        dump_compare_data.right_dump_info.get_data_info = mock.Mock(return_value="right_demo")
        dump_compare_data._check_left_type_quant_valid = mock.Mock()
        with pytest.raises(CompareError) as error:
            dump_compare_data.check_arguments_valid(fusion_json_file_path, quant_fusion_rule_file_path,
                                                    close_fusion_rule_file_path)
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_DUMP_TYPE_ERROR)


if __name__ == '__main__':
    unittest.main()
