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
from msprobe.core.common.const import CompareConst
from msprobe.core.compare.algorithm.builtin_algorithm.alg_cosine_similarity import compare as cosine_compare
from msprobe.core.compare.algorithm.builtin_algorithm.alg_euclidean_distance import compare as euc_compare
from msprobe.core.compare.algorithm.builtin_algorithm.alg_five_thousand_error import compare as five_thousand_compare
from msprobe.core.compare.algorithm.builtin_algorithm.alg_max_absolute_error import compare as max_abs_compare
from msprobe.core.compare.algorithm.builtin_algorithm.alg_max_relative_error import compare as max_rel_compare
from msprobe.core.compare.algorithm.builtin_algorithm.alg_one_thousand_error import compare as one_thousand_compare
from msprobe.core.compare.npy_compare import (
    compare_ops_apply,
    error_value_process,
    npy_data_check,
    statistics_data_check,
)
from msprobe.core.compare.utils import get_relative_err


class TestNpyDataCheck(unittest.TestCase):
    def test_data_check_none(self):
        error_flag, error_message = npy_data_check(None, None)
        self.assertTrue(error_flag)
        self.assertEqual(error_message, "Dump file is not ndarray.\n")

    def test_data_check_str(self):
        error_flag, error_message = npy_data_check("", "")
        self.assertTrue(error_flag)
        self.assertEqual(error_message, "Dump file is not ndarray.\n")

    def test_data_check_empty(self):
        error_flag, error_message = npy_data_check(np.array([]), np.array([]))
        self.assertTrue(error_flag)
        self.assertEqual(error_message, "This is empty data, can not compare.\n")

    def test_data_check_scalar(self):
        error_flag, error_message = npy_data_check(np.array(1), np.array(2))
        self.assertTrue(error_flag)
        self.assertEqual(error_message, "This is type of scalar data, can not compare.\n")

    def test_data_check_shape_unmatch(self):
        error_flag, error_message = npy_data_check(np.array([1]), np.array([1, 2]))
        self.assertTrue(error_flag)
        self.assertEqual(error_message, "Shape of NPU and bench Tensor do not match.\n")

    def test_data_check_dtype_unmatch(self):
        error_flag, error_message = npy_data_check(
            np.array([1, 2], dtype=float), np.array([1, 2], dtype=int)
        )
        self.assertTrue(error_flag)
        self.assertEqual(error_message, "Dtype of NPU and bench Tensor do not match. Skipped.\n")

    def test_data_check_nan_inf_mismatch(self):
        error_flag, error_message = npy_data_check(
            np.array([1, np.nan], dtype=float), np.array([1, 2], dtype=float)
        )
        self.assertTrue(error_flag)
        self.assertEqual(error_message, "The position of inf or nan in NPU and bench Tensor do not match.\n")

    def test_data_check_normal_no_false_positive(self):
        error_flag, error_message = npy_data_check(
            np.array([1, 2], dtype=float), np.array([1, 2], dtype=float)
        )
        self.assertFalse(error_flag)
        self.assertEqual(error_message, "")

    def test_data_check_both_nan_aligned_no_false_positive(self):
        error_flag, error_message = npy_data_check(
            np.array([1, np.nan], dtype=float), np.array([1, np.nan], dtype=float)
        )
        self.assertFalse(error_flag)
        self.assertEqual(error_message, "")


class TestStatisticsDataCheck(unittest.TestCase):
    def test_statistics_data_check_not_found(self):
        error_flag, error_message = statistics_data_check({"NPU Name": None})
        self.assertTrue(error_flag)
        self.assertEqual(
            error_message, "Dump file not found.\nThis is type of scalar data, can not compare.\n"
        )

    def test_statistics_data_check_shape_mismatch(self):
        error_flag, error_message = statistics_data_check(
            {"NPU Tensor Shape": [1], "Bench Tensor Shape": [2]}
        )
        self.assertTrue(error_flag)
        self.assertEqual(error_message, "Dump file not found.\nTensor shapes do not match.\n")

    def test_statistics_data_check_dtype_mismatch(self):
        error_flag, error_message = statistics_data_check(
            {"NPU Dtype": "torch.float32", "Bench Dtype": "torch.float16"}
        )
        self.assertTrue(error_flag)


class TestGetRelativeErr(unittest.TestCase):
    def test_get_relative_err_numpy_input(self):
        result = get_relative_err(np.array([1.0, 2.0]), np.array([1.0, 1.0]))
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_allclose(result, np.array([0.0, 1.0]))

    def test_get_relative_err_with_zero_denominator(self):
        result = get_relative_err(np.array([0.0, 2.0]), np.array([0.0, 1.0]))
        self.assertFalse(np.isnan(result).any())


class TestErrorValueProcess(unittest.TestCase):
    def test_error_value_process_read_none(self):
        result, err_msg = error_value_process(CompareConst.READ_NONE)
        self.assertEqual(result, CompareConst.UNSUPPORTED)
        self.assertEqual(err_msg, "")

    def test_error_value_process_unreadable(self):
        result, err_msg = error_value_process(CompareConst.UNREADABLE)
        self.assertEqual(result, CompareConst.UNSUPPORTED)
        self.assertEqual(err_msg, "")

    def test_error_value_process_none(self):
        result, err_msg = error_value_process(CompareConst.NONE)
        self.assertEqual(result, CompareConst.UNSUPPORTED)
        self.assertEqual(err_msg, "")

    def test_error_value_process_shape_unmatch(self):
        result, err_msg = error_value_process(CompareConst.SHAPE_UNMATCH)
        self.assertEqual(result, CompareConst.SHAPE_UNMATCH)
        self.assertEqual(err_msg, "")

    def test_error_value_process_nan(self):
        result, err_msg = error_value_process(CompareConst.NAN)
        self.assertEqual(result, CompareConst.N_A)
        self.assertEqual(err_msg, "")

    def test_error_value_process_other(self):
        result, err_msg = error_value_process("abc")
        self.assertEqual(result, CompareConst.N_A)
        self.assertEqual(err_msg, "")


class TestCompareOpsApply(unittest.TestCase):
    def test_compare_ops_apply_normal(self):
        n_value = np.array([1.0, 1.0])
        b_value = np.array([1.0, 1.0])
        results, err_msg = compare_ops_apply(n_value, b_value, False, "")

        self.assertEqual(len(results), len(CompareConst.BUILTIN_COMPARE_COLUMNS))
        self.assertEqual(err_msg, "")
        self.assertEqual(results[0], 1.0)
        self.assertEqual(results[1], 0.0)
        self.assertEqual(results[2], 0.0)
        self.assertEqual(results[3], 0.0)
        self.assertEqual(results[4], 1.0)
        self.assertEqual(results[5], 1.0)

    def test_compare_ops_apply_numpy_input(self):
        n_value = np.array([1.0, 1.0])
        b_value = np.array([1.0, 1.0])
        results, err_msg = compare_ops_apply(n_value, b_value, False, "")

        self.assertEqual(len(results), len(CompareConst.BUILTIN_COMPARE_COLUMNS))
        self.assertEqual(results[0], 1.0)

    def test_compare_ops_apply_with_error_flag(self):
        results, err_msg = compare_ops_apply(
            CompareConst.SHAPE_UNMATCH, CompareConst.SHAPE_UNMATCH, True, ""
        )
        self.assertEqual(len(results), len(CompareConst.BUILTIN_COMPARE_COLUMNS))
        for r in results:
            self.assertEqual(r, CompareConst.SHAPE_UNMATCH)


class TestCosineAlgorithm(unittest.TestCase):
    def test_cosine_normal(self):
        n_value = np.array([1.0, 2.0])
        b_value = np.array([1.0, 2.0])
        result, err_msg = cosine_compare(n_value, b_value)
        self.assertEqual(result, 1.0)
        self.assertEqual(err_msg, "")

    def test_cosine_0d_tensor(self):
        n_value = np.array(1.0)
        b_value = np.array(1.0)
        result, err_msg = cosine_compare(n_value, b_value)
        self.assertEqual(result, CompareConst.UNSUPPORTED)
        self.assertIn("0-d tensor", err_msg)

    def test_cosine_length_1(self):
        n_value = np.array([1.0])
        b_value = np.array([1.0])
        result, err_msg = cosine_compare(n_value, b_value)
        self.assertEqual(result, CompareConst.UNSUPPORTED)
        self.assertIn("length 1", err_msg)

    def test_cosine_all_zero_npu(self):
        n_value = np.array([0.0, 0.0])
        b_value = np.array([1.0, 2.0])
        result, err_msg = cosine_compare(n_value, b_value)
        self.assertEqual(result, CompareConst.NAN)
        self.assertIn("Zero in npu", err_msg)

    def test_cosine_all_zero_bench(self):
        n_value = np.array([1.0, 2.0])
        b_value = np.array([0.0, 0.0])
        result, err_msg = cosine_compare(n_value, b_value)
        self.assertEqual(result, CompareConst.NAN)
        self.assertIn("Zero in Bench", err_msg)

    def test_cosine_both_zero(self):
        n_value = np.array([0.0, 0.0])
        b_value = np.array([0.0, 0.0])
        result, err_msg = cosine_compare(n_value, b_value)
        self.assertEqual(result, 1.0)
        self.assertEqual(err_msg, "")

    def test_cosine_float16_distinct_vectors(self):
        n_value = np.array([1.0, 0.0], dtype=np.float16)
        b_value = np.array([0.0, 1.0], dtype=np.float16)
        result, err_msg = cosine_compare(n_value, b_value)
        self.assertEqual(result, 0.0)
        self.assertEqual(err_msg, "")


class TestEuclideanDistanceAlgorithm(unittest.TestCase):
    def test_euclidean_normal(self):
        n_value = np.array([1.0, 2.0, 3.0])
        b_value = np.array([4.0, 5.0, 6.0])
        result, err_msg = euc_compare(n_value, b_value)
        expected = float(np.linalg.norm(np.array([1.0, 2.0, 3.0]) - np.array([4.0, 5.0, 6.0])))
        self.assertAlmostEqual(float(result), expected, places=5)
        self.assertEqual(err_msg, "")

    def test_euclidean_0d_tensor(self):
        n_value = np.array(1.0)
        b_value = np.array(1.0)
        result, err_msg = euc_compare(n_value, b_value)
        self.assertEqual(result, CompareConst.UNSUPPORTED)
        self.assertIn("0-d tensor", err_msg)

    def test_euclidean_length_1(self):
        n_value = np.array([1.0])
        b_value = np.array([1.0])
        result, err_msg = euc_compare(n_value, b_value)
        self.assertEqual(result, CompareConst.UNSUPPORTED)
        self.assertIn("length 1", err_msg)

    def test_euclidean_identical(self):
        n_value = np.array([1.0, 2.0, 3.0])
        b_value = np.array([1.0, 2.0, 3.0])
        result, err_msg = euc_compare(n_value, b_value)
        self.assertEqual(result, 0.0)
        self.assertEqual(err_msg, "")


class TestMaxAbsErrAlgorithm(unittest.TestCase):
    def test_max_abs_err_normal(self):
        n_value = np.array([1.0, 2.0])
        b_value = np.array([0.0, 0.0])
        result, err_msg = max_abs_compare(n_value, b_value)
        self.assertEqual(result, 2.0)
        self.assertEqual(err_msg, "")

    def test_max_abs_err_identical(self):
        n_value = np.array([1.0, 2.0])
        b_value = np.array([1.0, 2.0])
        result, err_msg = max_abs_compare(n_value, b_value)
        self.assertEqual(result, 0.0)
        self.assertEqual(err_msg, "")

    def test_max_abs_err_0d_tensor(self):
        n_value = np.array(5.0)
        b_value = np.array(3.0)
        result, err_msg = max_abs_compare(n_value, b_value)
        self.assertEqual(result, 2.0)
        self.assertEqual(err_msg, "")


class TestMaxRelativeErrAlgorithm(unittest.TestCase):
    def test_max_relative_err_normal(self):
        n_value = np.array([1.0, 2.0])
        b_value = np.array([1.0, 1.0])
        result, err_msg = max_rel_compare(n_value, b_value)
        self.assertEqual(result, 1.0)
        self.assertEqual(err_msg, "")

    def test_max_relative_err_identical(self):
        n_value = np.array([1.0, 2.0])
        b_value = np.array([1.0, 2.0])
        result, err_msg = max_rel_compare(n_value, b_value)
        self.assertEqual(result, 0.0)
        self.assertEqual(err_msg, "")


class TestOneThousandErrAlgorithm(unittest.TestCase):
    def test_one_thousand_normal(self):
        n_value = np.array([1.0, 2.0])
        b_value = np.array([1.0, 1.0])
        result, err_msg = one_thousand_compare(n_value, b_value)
        self.assertEqual(result, 0.5)
        self.assertEqual(err_msg, "")

    def test_one_thousand_all_below_threshold(self):
        n_value = np.array([1.0, 1.0])
        b_value = np.array([1.0, 1.0])
        result, err_msg = one_thousand_compare(n_value, b_value)
        self.assertEqual(result, 1.0)
        self.assertEqual(err_msg, "")

    def test_one_thousand_0d_tensor(self):
        n_value = np.array(1.0)
        b_value = np.array(1.0)
        result, err_msg = one_thousand_compare(n_value, b_value)
        self.assertEqual(result, CompareConst.UNSUPPORTED)
        self.assertIn("0-d tensor", err_msg)


class TestFiveThousandErrAlgorithm(unittest.TestCase):
    def test_five_thousand_normal(self):
        n_value = np.array([1.0, 2.0])
        b_value = np.array([1.0, 1.0])
        result, err_msg = five_thousand_compare(n_value, b_value)
        self.assertEqual(result, 0.5)
        self.assertEqual(err_msg, "")

    def test_five_thousand_all_below_threshold(self):
        n_value = np.array([1.0, 1.0])
        b_value = np.array([1.0, 1.0])
        result, err_msg = five_thousand_compare(n_value, b_value)
        self.assertEqual(result, 1.0)
        self.assertEqual(err_msg, "")


if __name__ == "__main__":
    unittest.main()
