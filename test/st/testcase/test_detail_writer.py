#!/usr/bin/env python
# coding=utf-8
"""
Function:
This file mainly involves st test for detail_write module.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2021
"""
import unittest
import pytest
import numpy as np

from detail_writer import DetailWriter
from detail_writer import TopN
from unittest import mock


class TestUtilsMethods(unittest.TestCase):

    def test_write(self):
        op = 'test'
        detail_info = mock.Mock()
        detail_info.top_n = 10
        detail_info.max_line = 20
        detail_info.my_output_ops = op
        detail_info.ground_truth_ops = op
        detail_info.detail_format = 'N C H W'
        detail_info.ignore_result = False
        detail_info.tensor_id.get_file_prefix = mock.Mock(return_value='test_input_1')
        detail_info.tensor_id.get_tensor_id = mock.Mock(return_value='test:input:1')
        detail_info.make_detail_header = mock.Mock(
            return_value='Index,N C H W,LeftOp,RightOp,AbsoluteError,RelativeError')
        detail_writer = DetailWriter('/home/demo/', detail_info)
        dim = (1, 3, 4, 5)
        my_out_put_data = np.array([np.random.random() for _ in range(3 * 4 * 5)])
        ground_truth_data = np.array([np.random.random() for _ in range(3 * 4 * 5)])

        with mock.patch('cmp_utils.file_utils.FileUtils.save_array_to_file'), \
                mock.patch('os.open') as open_file, mock.patch('os.fdopen'):
            open_file.write = None
            open_file.close = None
            detail_writer.write(dim, my_out_put_data, ground_truth_data)

    def test_new_top_n_obj1(self):
        top_n = TopN(False)
        self.assertEqual(top_n.is_bool, False)

    def test_new_top_n_obj2(self):
        top_n = TopN(True)
        self.assertEqual(top_n.is_bool, True)

    def test_bool_write1(self):
        with mock.patch('os.open') as open_file, mock.patch('os.fdopen'):
            DetailWriter._write_one_detail(0, open_file, ['1', '2'], 1, 2, 1, 2, True)

    def test_bool_write2(self):
        with mock.patch('os.open') as open_file, mock.patch('os.fdopen'):
            DetailWriter._write_one_detail(0, open_file, ['1', '2', '2'], 1, 2, 1, 2, False)

    def test_get_top_n_res1(self):
        top_n = TopN(True)
        top_n._get_top_n_result([[0, 'name', True, False, 1, 2]], 0, 1)

    def test_get_top_n_res2(self):
        top_n = TopN(False)
        top_n._get_top_n_result([[0, 'name', 1, 2, 1, 2]], 0, 1)

    def test_cal_err1(self):
        my_output_data = np.array([1.1, 2, 3])
        ground_truth_data = np.array([11, 22, 33])
        detail_info = mock.Mock()
        detail_writer = DetailWriter('/home/demo/', detail_info)
        absolute_error, relative_error = detail_writer._cal_err(my_output_data, ground_truth_data, False)
        self.assertEqual(9.9, absolute_error[0])
        self.assertEqual(20, absolute_error[1])
        self.assertEqual(30, absolute_error[2])
        self.assertEqual(0.9, relative_error[0])
        self.assertEqual(round(0.90909091, 3), round(relative_error[1], 3))
        self.assertEqual(round(0.90909091, 3), round(relative_error[2], 3))

    def test_cal_err2(self):
        my_output_data = np.array([True, False])
        ground_truth_data = np.array([False, True])
        detail_info = mock.Mock()
        detail_writer = DetailWriter('/home/demo/', detail_info)
        detail_writer._total_num = 2
        absolute_error, relative_error = detail_writer._cal_err(my_output_data, ground_truth_data, True)
        self.assertTrue(np.isnan(absolute_error[0]))
        self.assertTrue(np.isnan(absolute_error[1]))
        self.assertTrue(np.isnan(relative_error[0]))
        self.assertTrue(np.isnan(relative_error[1]))

    def test_get_err_top_n_index1(self):
        my_output_data = np.array([1.1, 2, 3])
        ground_truth_data = np.array([11, 22, 33])
        detail_info = mock.Mock()
        detail_writer = DetailWriter('/home/demo/', detail_info)
        absolute_error, relative_error = detail_writer._cal_err(my_output_data, ground_truth_data, False)
        top_n = 2
        absolute_top_n_index, relative_top_n_index = detail_writer._get_error_top_n_index(absolute_error,
                                                                                          relative_error, top_n,
                                                                                          False)
        self.assertEqual([1, 2], absolute_top_n_index)
        self.assertEqual([1, 2], relative_top_n_index)

    def test_get_err_top_n_index2(self):
        my_output_data = np.array([True, False, True])
        ground_truth_data = np.array([False, True, True])
        detail_info = mock.Mock()
        detail_writer = DetailWriter('/home/demo/', detail_info)
        detail_writer._total_num = 2
        absolute_error, relative_error = detail_writer._cal_err(my_output_data, ground_truth_data, True)
        top_n = 2
        absolute_top_n_index, relative_top_n_index = detail_writer._get_error_top_n_index(absolute_error,
                                                                                          relative_error, top_n,
                                                                                          True)
        self.assertEqual([0, 1], absolute_top_n_index)
        self.assertEqual([0, 1], relative_top_n_index)

