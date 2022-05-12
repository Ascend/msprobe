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

from batch_compare import BatchCompare
import utils
from compare_error import CompareError
from unittest import mock


class TestUtilsMethods(unittest.TestCase):

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
        arguments.range = None
        json_file_array = ['0_71.json']
        with pytest.raises(CompareError) as error:
            with mock.patch("os.listdir", return_value=json_file_array):
                with mock.patch("os.path.getsize", return_value=100):
                    with mock.patch("builtins.open", mock.mock_open(read_data=None)):
                        with mock.patch("json.load", return_value=self._make_json_object()):
                            with mock.patch("utils.check_path_valid",
                                            return_value=CompareError.MSACCUCMP_NONE_ERROR):
                                with mock.patch("os.path.exists", return_value=False):
                                    batch_compare_test = BatchCompare()
                                    batch_compare_test.compare(arguments)
        self.assertEqual(error.value.args[0], 31)

    def test_compare2(self):
        arguments = mock.Mock()
        arguments.output_path = '/home/result'
        arguments.my_dump_path = '/home/202134565663'
        arguments.fusion_rule_file = '/home/more_json'
        arguments.quant_fusion_rule_file = ""
        arguments.close_fusion_rule_file = ""
        arguments.golden_dump_path = "/home/dt"
        arguments.dump_version = 1
        arguments.algorithm = 'all'
        arguments.op_name = ""
        arguments.custom_script_path = ""
        arguments.algorithm_options = ''
        arguments.range = None
        json_file_array = ['0.json']
        with pytest.raises(utils.CompareError) as error:
            with mock.patch("os.listdir", return_value=json_file_array):
                with mock.patch("os.path.getsize", return_value=100):
                    with mock.patch("builtins.open", mock.mock_open(read_data=None)):
                        with mock.patch("json.load", return_value=self._make_json_object()):
                            with mock.patch("utils.check_path_valid",
                                            return_value=CompareError.MSACCUCMP_NONE_ERROR):
                                with mock.patch("os.path.exists", return_value=False):
                                    batch_compare_test = BatchCompare()
                                    batch_compare_test.compare(arguments)
        self.assertEqual(error.value.args[0], 3)

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
