# coding=utf-8
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
import numpy as np
from unittest.mock import patch

from msprobe.core.common.const import CompareConst
from msprobe.core.compare.npy_compare import handle_inf_nan, reshape_value, \
    npy_data_check, statistics_data_check, get_relative_err, GetCosineSimilarity, GetMaxAbsErr, GetMaxRelativeErr, \
    GetErrRatio, error_value_process, compare_ops_apply, GetEuclideanDistance


op_name = 'Functional.conv2d.0.backward.input.0'


class TestUtilsMethods(unittest.TestCase):
    def test_handle_inf_nan_normal(self):
        n_value = np.array([1, 2, 3, 4])
        b_value = np.array([1, 2, 3, 4])

        a, b = handle_inf_nan(n_value, b_value)

        self.assertTrue(np.array_equal(a, n_value) and np.array_equal(b, b_value))

    def test_handle_inf_nan_with_inf(self):
        n_value = np.array([1, 2, np.inf, 4])
        b_value = np.array([1, 2, 3, 4])

        a, b = handle_inf_nan(n_value, b_value)

        self.assertTrue(a == CompareConst.NAN and b == CompareConst.NAN)

    def test_handle_inf_nan_with_nan(self):
        n_value = np.array([1, 2, 3, 4])
        b_value = np.array([1, 2, np.nan, 4])

        a, b = handle_inf_nan(n_value, b_value)

        self.assertTrue(a == CompareConst.NAN and b == CompareConst.NAN)

    def test_handle_inf_nan_both_nan(self):
        n_value = np.array([1, 2, np.nan, 4])
        b_value = np.array([1, 2, np.nan, 4])

        a, b = handle_inf_nan(n_value, b_value)

        self.assertTrue(np.array_equal(a, np.array([1, 2, 0, 4])))
        self.assertTrue(np.array_equal(b, np.array([1, 2, 0, 4])))

    def test_handle_inf_nan_both_inf(self):
        n_value = np.array([1, 2, np.inf, 4])
        b_value = np.array([1, 2, np.inf, 4])

        a, b = handle_inf_nan(n_value, b_value)

        self.assertTrue(np.array_equal(a, np.array([1, 2, 0, 4])))
        self.assertTrue(np.array_equal(b, np.array([1, 2, 0, 4])))

    def test_reshape_value_normal(self):
        n_value = np.array([[1, 2], [3, 4]])
        b_value = np.array([[1, 2, 3], [3, 4, 5]])
        a, b = reshape_value(n_value, b_value)
        self.assertTrue(np.array_equal(a, np.array([1., 2., 3., 4.])) and np.array_equal(b, np.array([1., 2., 3., 3., 4., 5.])))

    def test_reshape_value_not_shape(self):
        n_value = np.array([])
        b_value = np.array([])
        a, b = reshape_value(n_value, b_value)
        self.assertTrue(np.array_equal(a, n_value) and np.array_equal(b, b_value))

    def test_reshape_value_bool(self):
        n_value = np.array(True)
        b_value = np.array(True)
        a, b = reshape_value(n_value, b_value)
        self.assertTrue(np.array_equal(a, np.array(1.)) and np.array_equal(b, np.array(1.)))

    def test_data_check(self):
        n_value_1 = None
        b_value_1 = None
        error_flag_1, error_message_1 = npy_data_check(n_value_1, b_value_1)
        self.assertEqual(error_message_1, 'Dump file is not ndarray.\n')

        n_value_2 = ''
        b_value_2 = ''
        error_flag_2, error_message_2 = npy_data_check(n_value_2, b_value_2)
        self.assertEqual(error_message_2, 'Dump file is not ndarray.\n')

        n_value_3 = np.array([])
        b_value_3 = np.array([])
        error_flag_3, error_message_3 = npy_data_check(n_value_3, b_value_3)
        self.assertEqual(error_message_3, 'This is empty data, can not compare.\n')

        n_value_4 = np.array(1)
        b_value_4 = np.array(2)
        error_flag_4, error_message_4 = npy_data_check(n_value_4, b_value_4)
        self.assertEqual(error_message_4, 'This is type of scalar data, can not compare.\n')

        n_value_5 = np.array([1])
        b_value_5 = np.array([1, 2])
        error_flag_5, error_message_5 = npy_data_check(n_value_5, b_value_5)
        self.assertEqual(error_message_5, 'Shape of NPU and bench Tensor do not match.\n')

        n_value_6 = np.array([1, 2], dtype=float)
        b_value_6 = np.array([1, 2], dtype=int)
        error_flag_6, error_message_6 = npy_data_check(n_value_6, b_value_6)
        self.assertEqual(error_message_6, 'Dtype of NPU and bench Tensor do not match. Skipped.\n')

        n_value_7 = np.array([1, np.nan], dtype=float)
        b_value_7 = np.array([1, 2], dtype=float)
        error_flag_7, error_message_7 = npy_data_check(n_value_7, b_value_7)
        self.assertEqual(error_message_7, 'The position of inf or nan in NPU and bench Tensor do not match.\n')

        n_value_8 = np.array([1, 2], dtype=float)
        b_value_8 = np.array([1, 2], dtype=float)
        error_flag_8, error_message_8 = npy_data_check(n_value_8, b_value_8)
        self.assertFalse(error_flag_8)

    def test_statistics_data_check(self):
        result_dict_1 = {'NPU Name': None}
        error_flag_1, error_message_1 = statistics_data_check(result_dict_1)
        self.assertEqual(error_message_1, 'Dump file not found.\nThis is type of scalar data, can not compare.\n')
        self.assertTrue(error_flag_1)

        result_dict_2 = {'NPU Tensor Shape': [1], 'Bench Tensor Shape': [2]}
        error_flag_2, error_message_2 = statistics_data_check(result_dict_2)
        self.assertEqual(error_message_2, 'Dump file not found.\nTensor shapes do not match.\n')
        self.assertTrue(error_flag_2)

        result_dict_3 = {'NPU Dtype': 'torch.float32', 'Bench Dtype': 'torch.float16'}
        error_flag_3, error_message_3 = statistics_data_check(result_dict_3)
        self.assertEqual(error_message_3, 'Dump file not found.\nThis is type of scalar data, can not compare.\n''Dtype of NPU and bench Tensor do not match. Skipped.\n')
        self.assertTrue(error_flag_3)

    def test_get_relative_err(self):
        n_value = np.array([1, 2])
        b_value = np.array([1, 1])
        result = get_relative_err(n_value, b_value)

        self.assertTrue(np.array_equal(result, [0.0, 1.0]))

    def test_GetCosineSimilarity_normal(self):
        op = GetCosineSimilarity()

        n_value_1 = np.array(1)
        b_value_1 = np.array(1)
        relative_err = get_relative_err(n_value_1, b_value_1)
        n_value_1, b_value_1 = reshape_value(n_value_1, b_value_1)
        err_msg = "This is type of 0-d tensor, can not calculate 'Cosine', 'EucDist', 'One Thousandth Err Ratio' and 'Five Thousandths Err Ratio'. "
        result, err_msg = op.apply(n_value_1, b_value_1, relative_err, err_msg)
        self.assertEqual(result, CompareConst.UNSUPPORTED)
        self.assertEqual(err_msg, "This is type of 0-d tensor, can not calculate 'Cosine', 'EucDist', 'One Thousandth Err Ratio' and 'Five Thousandths Err Ratio'. ")

        n_value_2 = np.array([1, 2])
        b_value_2 = np.array([1, 2])
        relative_err = get_relative_err(n_value_2, b_value_2)
        n_value_2, b_value_2 = reshape_value(n_value_2, b_value_2)
        err_msg = ""
        result, err_msg = op.apply(n_value_2, b_value_2, relative_err, err_msg)
        self.assertEqual(result, 1.0)
        self.assertEqual(err_msg, "")

        n_value_3 = np.array([0, 0])
        b_value_3 = np.array([0, 0])
        relative_err = get_relative_err(n_value_3, b_value_3)
        n_value_3, b_value_3 = reshape_value(n_value_3, b_value_3)
        err_msg = ""
        result, err_msg = op.apply(n_value_3, b_value_3, relative_err, err_msg)
        self.assertEqual(result, 1.0)
        self.assertEqual(err_msg, "")

        n_value_4 = np.array([0, 0])
        b_value_4 = np.array([1, 2])
        relative_err = get_relative_err(n_value_4, b_value_4)
        n_value_4, b_value_4 = reshape_value(n_value_4, b_value_4)
        err_msg = ""
        result, err_msg = op.apply(n_value_4, b_value_4, relative_err, err_msg)
        self.assertEqual(result, CompareConst.NAN)
        self.assertEqual(err_msg, 'Cannot compare by Cosine Similarity, All the data is Zero in npu dump data.')

        n_value_5 = np.array([1, 2])
        b_value_5 = np.array([0, 0])
        relative_err = get_relative_err(n_value_5, b_value_5)
        n_value_5, b_value_5 = reshape_value(n_value_5, b_value_5)
        err_msg = ""
        result, err_msg = op.apply(n_value_5, b_value_5, relative_err, err_msg)
        self.assertEqual(result, CompareConst.NAN)
        self.assertEqual(err_msg, 'Cannot compare by Cosine Similarity, All the data is Zero in Bench dump data.')

    def test_GetCosineSimilarity_not_shape(self):
        op = GetCosineSimilarity()

        n_value_1 = np.array([1])
        b_value_1 = np.array([1])
        relative_err = get_relative_err(n_value_1, b_value_1)
        n_value_1, b_value_1 = reshape_value(n_value_1, b_value_1)
        err_msg = ""

        result, err_msg = op.apply(n_value_1, b_value_1, relative_err, err_msg)
        self.assertEqual(result, CompareConst.UNSUPPORTED)
        self.assertEqual(err_msg, "This is a 1-d tensor of length 1.")

    @patch("numpy.isnan", return_value=True)
    def test_GetCosineSimilarity_isnan(self, mock_isnan):
        op = GetCosineSimilarity()

        n_value = np.array([1, 2])
        b_value = np.array([1, 1])
        relative_err = get_relative_err(n_value, b_value)
        n_value, b_value = reshape_value(n_value, b_value)
        err_msg = ""

        result, err_msg = op.apply(n_value, b_value, relative_err, err_msg)

        self.assertEqual(result, CompareConst.NAN)
        self.assertEqual(err_msg, "Cannot compare by Cosine Similarity, the dump data has NaN.")
        mock_isnan.assert_called_once()

    def test_GetCosineSimilarity_correct_data(self):
        op = GetCosineSimilarity()

        result_origin = CompareConst.NAN
        result = op.correct_data(result_origin)
        self.assertEqual(result, CompareConst.NAN)

        result_origin = 1
        result = op.correct_data(result_origin)
        self.assertEqual(result, float(result_origin))

    def test_GetMaxAbsErr_normal(self):
        op = GetMaxAbsErr()

        n_value = np.array([1, 2])
        b_value = np.array([0, 0])
        relative_err = get_relative_err(n_value, b_value)
        n_value, b_value = reshape_value(n_value, b_value)
        err_msg = ""

        result, err_msg = op.apply(n_value, b_value, relative_err, err_msg)

        self.assertEqual(result, 2.0)
        self.assertEqual(err_msg, "")

    @patch("numpy.isnan", return_value=True)
    def test_GetMaxAbsErr_isnan(self, mock_isnan):
        op = GetMaxAbsErr()

        n_value = np.array([1, 2])
        b_value = np.array([1, 1])
        relative_err = get_relative_err(n_value, b_value)
        n_value, b_value = reshape_value(n_value, b_value)
        err_msg = ""

        result, err_msg = op.apply(n_value, b_value, relative_err, err_msg)

        self.assertEqual(result, CompareConst.NAN)
        self.assertEqual(err_msg, "Cannot compare by MaxAbsError, the data contains nan/inf/-inf in dump data.")
        mock_isnan.assert_called_once()

    def test_GetMaxRelativeErr_normal(self):
        op = GetMaxRelativeErr()

        n_value = np.array([1, 2])
        b_value = np.array([1, 1])
        relative_err = get_relative_err(n_value, b_value)
        n_value, b_value = reshape_value(n_value, b_value)
        err_msg = ""

        result, err_msg = op.apply(n_value, b_value, relative_err, err_msg)

        self.assertEqual(result, 1.0)
        self.assertEqual(err_msg, "")

    @patch("numpy.isnan", return_value=True)
    def test_GetMaxRelativeErr_isnan(self, mock_isnan):
        op = GetMaxRelativeErr()

        n_value = np.array([1, 2])
        b_value = np.array([1, 1])
        relative_err = get_relative_err(n_value, b_value)
        n_value, b_value = reshape_value(n_value, b_value)
        err_msg = ""

        result, err_msg = op.apply(n_value, b_value, relative_err, err_msg)

        self.assertEqual(result, CompareConst.NAN)
        self.assertEqual(err_msg, "Cannot compare by MaxRelativeError, the data contains nan/inf/-inf in dump data.")
        mock_isnan.assert_called_once()

    def test_GetThousandErrRatio_normal(self):
        op = GetErrRatio(CompareConst.THOUSAND_RATIO_THRESHOLD)

        n_value = np.array([1, 2])
        b_value = np.array([1, 1])
        relative_err = get_relative_err(n_value, b_value)
        n_value, b_value = reshape_value(n_value, b_value)
        err_msg = ""

        result, err_msg = op.apply(n_value, b_value, relative_err, err_msg)

        self.assertEqual(result, 0.5)
        self.assertEqual(err_msg, "")

    def test_GetThousandErrRatio_not_shape(self):
        op = GetErrRatio(CompareConst.THOUSAND_RATIO_THRESHOLD)

        n_value = np.array(1)   # 标量
        b_value = np.array(1)
        relative_err = np.array(0)
        err_msg = "This is type of 0-d tensor, can not calculate 'Cosine', 'EucDist', 'One Thousandth Err Ratio' and 'Five Thousandths Err Ratio'. "

        result, err_msg = op.apply(n_value, b_value, relative_err, err_msg)

        self.assertEqual(result, CompareConst.UNSUPPORTED)
        self.assertEqual(err_msg, "This is type of 0-d tensor, can not calculate 'Cosine', 'EucDist', 'One Thousandth Err Ratio' and 'Five Thousandths Err Ratio'. ")

    def test_GetThousandErrRatio_not_size(self):
        op = GetErrRatio(CompareConst.THOUSAND_RATIO_THRESHOLD)

        n_value = np.array([1, 2])
        b_value = np.array([1, 2])
        relative_err = np.array([])     # 空数组
        err_msg = ""

        result, err_msg = op.apply(n_value, b_value, relative_err, err_msg)

        self.assertEqual(result, CompareConst.NAN)
        self.assertEqual(err_msg, "")

    def test_GetFiveThousandErrRatio_normal(self):
        op = GetErrRatio(CompareConst.FIVE_THOUSAND_RATIO_THRESHOLD)

        n_value = np.array([1, 2])
        b_value = np.array([1, 1])
        relative_err = get_relative_err(n_value, b_value)
        n_value, b_value = reshape_value(n_value, b_value)
        err_msg = ""

        result, err_msg = op.apply(n_value, b_value, relative_err, err_msg)

        self.assertEqual(result, 0.5)
        self.assertEqual(err_msg, "")

    def test_error_value_process_read_none(self):
        n_value = CompareConst.READ_NONE
        result, err_msg = error_value_process(n_value)

        self.assertEqual(result, CompareConst.UNSUPPORTED)
        self.assertEqual(err_msg, "")

    def test_error_value_process_unreadable(self):
        n_value = CompareConst.UNREADABLE

        result, err_msg = error_value_process(n_value)

        self.assertEqual(result, CompareConst.UNSUPPORTED)
        self.assertEqual(err_msg, "")

    def test_error_value_process_none(self):
        n_value = CompareConst.NONE

        result, err_msg = error_value_process(n_value)

        self.assertEqual(result, CompareConst.UNSUPPORTED)
        self.assertEqual(err_msg, "")

    def test_error_value_process_shape_unmatch(self):
        n_value = CompareConst.SHAPE_UNMATCH

        result, err_msg = error_value_process(n_value)

        self.assertEqual(result, CompareConst.SHAPE_UNMATCH)
        self.assertEqual(err_msg, "")

    def test_error_value_process_nan(self):
        n_value = CompareConst.NAN

        result, err_msg = error_value_process(n_value)

        self.assertEqual(result, CompareConst.N_A)
        self.assertEqual(err_msg, "")

    def test_error_value_process_other(self):
        n_value = "abc"

        result, err_msg = error_value_process(n_value)

        self.assertEqual(result, CompareConst.N_A)
        self.assertEqual(err_msg, "")

    def test_compare_ops_apply(self):
        n_value = np.array([1, 1])
        b_value = np.array([1, 1])
        error_flag = False
        err_msg = ''
        a, b = compare_ops_apply(n_value, b_value, error_flag, err_msg)
        self.assertEqual(a, [1.0, 0.0, 0.0, 0.0, 1.0, 1.0])
        self.assertEqual(b, '')


class TestGetEuclideanDistance(unittest.TestCase):

    def setUp(self):
        self.euc_distance = GetEuclideanDistance()

    def test_euclidean_distance_normal(self):
        # 测试计算两个张量之间的欧氏距离
        n_value = np.array([1, 2, 3])
        b_value = np.array([4, 5, 6])
        relative_err = None
        err_msg = ""

        result, msg = self.euc_distance.apply(n_value, b_value, relative_err, err_msg)
        expected_distance = np.linalg.norm(n_value - b_value)
        self.assertEqual(result, expected_distance)
        self.assertEqual(msg, '')

    def test_euclidean_distance_0d_tensor(self):
        # 测试计算两个张量之间的欧氏距离
        n_value = np.array(1)
        b_value = np.array(1)
        relative_err = None
        err_msg = "This is type of 0-d tensor, can not calculate 'Cosine', 'EucDist', 'One Thousandth Err Ratio' and 'Five Thousandths Err Ratio'. "

        result, msg = self.euc_distance.apply(n_value, b_value, relative_err, err_msg)
        self.assertEqual(result, CompareConst.UNSUPPORTED)
        self.assertEqual(msg, "This is type of 0-d tensor, can not calculate 'Cosine', 'EucDist', 'One Thousandth Err Ratio' and 'Five Thousandths Err Ratio'. ")
