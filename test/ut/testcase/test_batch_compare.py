#!/usr/bin/env python
# coding=utf-8
"""
Function:
This file mainly involves xxxx function.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2021
"""
import unittest
import pytest

from src.compare.vector_cmp.batch_compare import BatchCompare
import utils
from unittest import mock
from cmp_utils.constant.compare_error import CompareError


class TestUtilsMethods(unittest.TestCase):
    def test_init(self):
        test = BatchCompare()
        self.assertNotEqual(test, None)

    def test_make_model_name_to_json_map(self):
        json_dir_path = '/home/more_json'
        json_file_array = ['bert_qa_layernorm_71.json']
        with mock.patch("os.listdir", return_value=json_file_array):
            with mock.patch("os.path.getsize", return_value=100):
                with mock.patch("builtins.open", mock.mock_open(read_data=None)):
                    with mock.patch("json.load", return_value=self._make_json_object()):
                        batch_compare_test = BatchCompare()
                        batch_compare_test._make_model_name_to_json_map(json_dir_path)
                        test_map = batch_compare_test.model_name_to_json_map
        self.assertEqual(len(test_map), 1)
        self.assertEqual(test_map.get("ge_default_20210420113943_71"), "/home/more_json/bert_qa_layernorm_71.json")

    def test_make_json_path_to_dump_path_map1(self):
        batch_compare_test = BatchCompare()
        batch_compare_test.model_name_to_json_map = {'0': '/home/202134565663/0/0/0/0'}
        npu_dump_dir = '/home/202134565663'
        with mock.patch("os.listdir", return_value=["0"]):
            with mock.patch("os.path.isdir", return_value=True):
                with mock.patch("utils.check_path_valid", return_value=CompareError.MSACCUCMP_NONE_ERROR):
                    batch_compare_test._make_json_path_to_dump_path_map(npu_dump_dir)
        self.assertEqual(batch_compare_test.json_path_to_dump_path_map,
                         {'/home/202134565663/0/0/0/0': ['/home/202134565663/0/0/0/0']})

    def test_make_json_path_to_dump_path_map2(self):
        batch_compare_test = BatchCompare()
        batch_compare_test.model_name_to_json_map = {'1_0': '/home/202134565663/0/0/0/0'}
        npu_dump_dir = '/home/202134565663'
        with mock.patch("os.listdir", return_value=["0_0"]):
            with mock.patch("os.path.isdir", return_value=True):
                with mock.patch("utils.check_path_valid", return_value=CompareError.MSACCUCMP_NONE_ERROR):
                    batch_compare_test._make_json_path_to_dump_path_map(npu_dump_dir)
        self.assertEqual(batch_compare_test.json_path_to_dump_path_map,
                         {'/home/202134565663/0/0/0/0': ['/home/202134565663/0_0/0_0/0_0/0_0']})

    def test_make_json_path_to_dump_path_map3(self):
        batch_compare_test = BatchCompare()
        batch_compare_test.model_name_to_json_map = {'1_0': '/home/202134565663/0/0/0/0'}
        npu_dump_dir = '/home/202134565663'
        with pytest.raises(utils.CompareError) as error:
            with mock.patch("os.listdir", return_value=["0"]):
                with mock.patch("os.path.isdir", return_value=True):
                    with mock.patch("utils.check_path_valid", return_value=CompareError.MSACCUCMP_NONE_ERROR):
                        batch_compare_test._make_json_path_to_dump_path_map(npu_dump_dir)
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PATH_ERROR)

    def test_execute_batch_compare(self):
        batch_compare_test = BatchCompare()
        arguments = mock.Mock()
        arguments.output_path = '/home/result'
        arguments.my_dump_path = '/home/202134565663'
        arguments.fusion_rule_file = "/home/b.json"
        arguments.quant_fusion_rule_file = ""
        arguments.close_fusion_rule_file = ""
        arguments.golden_dump_path = "/home/dt"
        arguments.dump_version = 1
        arguments.op_name = ""
        arguments.custom_script_path = "result"
        batch_compare_test.json_path_to_dump_path_map = {
            '/home/more_json/bert_qa_layernorm_71.json': '/home/202134565663/0/ge_default_20210420113943_71/10/1'}
        arguments.fusion_rule_file = ""
        with pytest.raises(CompareError) as error:
            batch_compare_test._execute_batch_compare(arguments)
        self.assertEqual(error.value.args[0], 3)

    def test_compare(self):
        arguments = mock.Mock()
        arguments.output_path = '/home/result'
        arguments.my_dump_path = '/home/202134565663'
        arguments.fusion_rule_file = '/home/more_json'
        arguments.quant_fusion_rule_file = ""
        arguments.close_fusion_rule_file = ""
        arguments.golden_dump_path = "/home/dt"
        arguments.dump_version = 1
        arguments.op_name = ""
        arguments.custom_script_path = ""
        arguments.algorithm = 'all'
        arguments.algorithm_options = ''
        json_file_array = ['0_71.json']
        with pytest.raises(utils.CompareError) as error:
            with mock.patch("os.listdir", side_effect=[['alg_MaxAbsoluteError.py'], json_file_array]):
                with mock.patch("builtins.open", mock.mock_open(read_data=None)):
                    with mock.patch("json.load", return_value=self._make_json_object()):
                        with mock.patch("utils.check_path_valid",
                                        return_value=CompareError.MSACCUCMP_NONE_ERROR):
                            with mock.patch("os.path.exists", return_value=True):
                                with mock.patch("os.access", return_value=False):
                                    batch_compare_test = BatchCompare()
                                    batch_compare_test.compare(arguments)
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PATH_ERROR)

    def test_check_argument_valid1(self):
        batch_compare_test = BatchCompare()
        arguments = mock.Mock()
        arguments.op_name = "xx"
        with pytest.raises(utils.CompareError) as error:
            batch_compare_test.check_argument_valid(arguments)
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_check_argument_valid2(self):
        batch_compare_test = BatchCompare()
        arguments = mock.Mock()
        arguments.op_name = ""
        arguments.my_dump_path = '/home/gzj/xxx'
        with pytest.raises(utils.CompareError) as error:
            with mock.patch("utils.check_path_valid",
                            return_value=CompareError.MSACCUCMP_NONE_ERROR):
                batch_compare_test.check_argument_valid(arguments)
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PATH_ERROR)

    @staticmethod
    def _make_json_object():
        return {'graph': [
            {'name': 'ge_default_20210420113943_71',
             'op': [
                 {
                     "attr": [],
                     "dst_index": [],
                     "dst_name": [],
                     "has_out_attr": True,
                     "input": [],
                     "input_desc": [],
                     "name": "input_ids",
                     "output_desc": [],
                     "output_i": [],
                     "type": "Data"
                 }
             ]}]}
