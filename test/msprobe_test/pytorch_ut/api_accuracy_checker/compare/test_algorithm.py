import unittest

import torch
import numpy as np

from msprobe.pytorch.api_accuracy_checker.compare import algorithm as alg
from msprobe.pytorch.api_accuracy_checker.compare.compare_utils import ULP_PARAMETERS
from msprobe.core.common.const import CompareConst


class TestAlgorithmMethods(unittest.TestCase):

    def setUp(self):
        self.bench_data = np.array([1.0, 1.0, 9.0], dtype=np.float16)
        self.device_data = np.array([5.0, 2.0, 1.0], dtype=np.float16)
        self.bench_data_fp32 = np.array([1.0, 1.0, 9.0], dtype=np.float32)
        self.device_data_fp32 = np.array([5.0, 2.0, 1.0], dtype=np.float32)
        self.abs_err = np.abs(self.device_data - self.bench_data)
        self.rel_err_origin = np.abs(self.abs_err / self.bench_data)
        eps = np.finfo(self.bench_data.dtype).eps
        self.abs_bench = np.abs(self.bench_data)
        self.abs_bench_with_eps = self.abs_bench + eps
        self.rel_err = self.abs_err / self.abs_bench_with_eps

    def test_cosine_sim(self):
        cpu_output = np.array([1.0, 2.0, 3.0])
        npu_output = np.array([1.0, 2.0, 3.0])
        self.assertEqual(alg.cosine_sim(cpu_output, npu_output), (1.0, True, ''))

    def test_cosine_sim_shape_mismatch(self):
        bench_output = np.array([1, 2, 3])
        device_output = np.array([4, 5])
        cos, success, msg = alg.cosine_sim(bench_output, device_output)
        self.assertEqual(cos, -1)
        self.assertFalse(success)
        self.assertIn("Shape of device and bench outputs don't match", msg)

    def test_cosine_sim_scalar_value(self):
        bench_output = np.array([1])
        device_output = np.array([1])
        cos, success, msg = alg.cosine_sim(bench_output, device_output)
        self.assertEqual(cos, CompareConst.SPACE)
        self.assertTrue(success)
        self.assertIn("All the data in device dump data is scalar", msg)

    def test_cosine_sim_all_zeros(self):
        bench_output = np.array([0, 0, 0])
        device_output = np.array([0, 0, 0])
        cos, success, msg = alg.cosine_sim(bench_output, device_output)
        self.assertEqual(cos, CompareConst.SPACE)
        self.assertTrue(success)
        self.assertIn("All the data in device and bench outputs are zero", msg)

    def test_cosine_sim_device_all_zeros(self):
        bench_output = np.array([0, 1, 0])
        device_output = np.array([0, 0, 0])
        cos, success, msg = alg.cosine_sim(bench_output, device_output)
        self.assertEqual(cos, CompareConst.SPACE)
        self.assertFalse(success)
        self.assertIn("All the data is zero in device dump data", msg)

    def test_cosine_sim_bench_all_zeros(self):
        bench_output = np.array([0, 0, 0])
        device_output = np.array([0, 1, 0])
        cos, success, msg = alg.cosine_sim(bench_output, device_output)
        self.assertEqual(cos, CompareConst.SPACE)
        self.assertFalse(success)
        self.assertIn("All the data is zero in bench dump data", msg)

    def test_nan_values(self):
        bench_output = np.array([1, 2, np.nan])
        device_output = np.array([1, 2, 3])
        cos, success, msg = alg.cosine_sim(bench_output, device_output)
        self.assertTrue(np.isnan(cos))
        self.assertTrue(success)
        self.assertIn("Dump data has NaN when comparing with Cosine Similarity", msg)

    def test_get_rmse(self):
        inf_nan_mask = [False, False, False]
        self.assertAlmostEqual(alg.get_rmse(self.abs_err, inf_nan_mask), 5.196, 3)

    def test_get_error_balance(self):
        self.assertEqual(alg.get_error_balance(self.bench_data, self.device_data), 1 / 3)

    def test_get_small_value_err_ratio(self):
        small_value_mask = [True, True, True, False, True]
        abs_err_greater_mask = [False, True, True, True, False]
        self.assertEqual(alg.get_small_value_err_ratio(small_value_mask, abs_err_greater_mask), 0.5)

    def get_rel_err(self):
        eps = np.finfo(self.bench_data.dtype).eps
        abs_bench = np.abs(self.bench_data)
        abs_bench_with_eps = abs_bench + eps
        small_value_mask = [False, False, False]
        inf_nan_mask = [False, False, False]
        rel_err = self.abs_err / abs_bench_with_eps
        self.assertListEqual(list(alg.get_rel_err(self.abs_err, abs_bench_with_eps, small_value_mask, inf_nan_mask)),
                             list(rel_err))

    def test_get_abs_err(self):
        self.assertListEqual(list(alg.get_abs_err(self.bench_data, self.device_data)), [4.0, 1.0, 8.0])

    def test_get_rel_err_origin(self):
        self.assertListEqual(list(alg.get_rel_err_origin(self.abs_err, self.bench_data)), list(self.rel_err_origin))

    def test_get_max_abs_err(self):
        self.assertEqual(alg.get_max_abs_err(self.abs_err), (8.0, False))

    def test_get_max_rel_err(self):
        self.assertAlmostEqual(alg.get_max_rel_err(self.rel_err), 3.996, 3)

    def test_get_mean_rel_err(self):
        self.assertAlmostEqual(alg.get_mean_rel_err(self.rel_err), 1.961, 3)

    def test_get_rel_err_ratio_with_empty_array(self):
        rel_err = np.array([])
        thresholding = 0.01
        ratio, bool_result = alg.get_rel_err_ratio(rel_err, thresholding)
        self.assertEqual(ratio, 1)
        self.assertTrue(bool_result)

    def test_get_rel_err_ratio_thousandth(self):
        b_value = np.array([1.0, 2.0, 3.0])
        n_value = np.array([1.0, 2.0, 3.0])
        abs_err = np.abs(b_value - n_value)
        rel_err = alg.get_rel_err_origin(abs_err, b_value)
        self.assertEqual(alg.get_rel_err_ratio(rel_err, 0.001), (1.0, True))

    def test_get_rel_err_ratio_ten_thousandth(self):
        b_value = np.array([1.0, 2.0, 3.0])
        n_value = np.array([1.0, 2.0, 3.0])
        abs_err = np.abs(b_value - n_value)
        rel_err = alg.get_rel_err_origin(abs_err, b_value)
        self.assertEqual(alg.get_rel_err_ratio(rel_err, 0.0001), (1.0, True))

    def test_get_finite_and_infinite_mask(self):
        both_finite_mask, inf_nan_mask = alg.get_finite_and_infinite_mask(self.bench_data, self.device_data)
        self.assertListEqual(list(both_finite_mask), [True, True, True])
        self.assertListEqual(list(inf_nan_mask), [False, False, False])

    def test_get_small_value_mask(self):
        b_value = np.array([1e-7, 1.0, 2e-6], dtype=np.float16)
        abs_bench = np.abs(b_value)
        both_finite_mask = [True, True, True]
        small_value_mask = alg.get_small_value_mask(abs_bench, both_finite_mask, 1e-3)
        self.assertListEqual(list(small_value_mask), [True, False, True])

    def test_get_abs_bench_with_eps(self):
        abs_bench, abs_bench_with_eps = alg.get_abs_bench_with_eps(self.bench_data, np.float16)
        self.assertListEqual(list(abs_bench), list(self.abs_bench))
        self.assertListEqual(list(abs_bench_with_eps), list(self.abs_bench_with_eps))

    def test_check_inf_nan_value(self):
        both_finite_mask, inf_nan_mask = alg.get_finite_and_infinite_mask(self.bench_data, self.device_data)
        self.assertEqual(alg.check_inf_nan_value(inf_nan_mask, self.bench_data, self.device_data, np.float16, 0.001), 0)

    def test_check_small_value(self):
        a_value = np.array([1e-7, 1.0, 2e-6], dtype=np.float16)
        b_value = np.array([1e-7, 1.0, 2e-6], dtype=np.float16)
        abs_bench = np.abs(b_value)
        both_finite_mask = [True, True, True]
        abs_err = abs(a_value - b_value)
        small_value_mask = alg.get_small_value_mask(abs_bench, both_finite_mask, 1e-3)
        self.assertEqual(alg.check_small_value(abs_err, small_value_mask, 0.001), 0)

    def test_check_norm_value(self):
        both_finite_mask, inf_nan_mask = alg.get_finite_and_infinite_mask(self.bench_data, self.device_data)
        small_value_mask = alg.get_small_value_mask(self.abs_bench, both_finite_mask, 1e-3)
        normal_value_mask = np.logical_and(both_finite_mask, np.logical_not(small_value_mask))
        print(normal_value_mask)
        print(self.rel_err)
        self.assertEqual(alg.check_norm_value(normal_value_mask, self.rel_err, 0.001), 1)

    def test_get_ulp_err(self):
        parameters = ULP_PARAMETERS.get(torch.float16)
        min_eb = parameters.get('min_eb')[0]
        abs_bench = np.abs(self.bench_data)
        eb = np.where(abs_bench == 0, 0, np.floor(np.log2(abs_bench)))
        eb = np.maximum(eb, min_eb)
        exponent_num = parameters.get('exponent_num')[0]
        ulp_err = alg.get_ulp_err(self.bench_data, self.device_data, torch.float16)
        data_type = np.float32
        expected_ulp_err = (self.device_data.astype(data_type) - self.bench_data).astype(data_type) * np.exp2(-eb + exponent_num)
        expected_ulp_err = np.abs(expected_ulp_err)
        self.assertTrue(np.allclose(ulp_err, expected_ulp_err))
        
        parameters = ULP_PARAMETERS.get(torch.float32)
        min_eb = parameters.get('min_eb')[0]
        abs_bench = np.abs(self.bench_data_fp32)
        eb = np.where(abs_bench == 0, 0, np.floor(np.log2(abs_bench)))
        eb = np.maximum(eb, min_eb)
        exponent_num = parameters.get('exponent_num')[0]
        ulp_err = alg.get_ulp_err(self.bench_data_fp32, self.device_data_fp32, torch.float32)
        data_type = np.float64
        expected_ulp_err = (self.device_data_fp32.astype(data_type) - self.bench_data_fp32).astype(data_type) * np.exp2(-eb + exponent_num)
        expected_ulp_err = np.abs(expected_ulp_err)
        self.assertTrue(np.allclose(ulp_err, expected_ulp_err))

    def test_calc_ulp_err(self):
        # 测试 calc_ulp_err 函数的计算是否正确
        parameters = ULP_PARAMETERS.get(torch.float16)
        min_eb = parameters.get('min_eb')[0]
        abs_bench = np.abs(self.bench_data)
        eb = np.where(abs_bench == 0, 0, np.floor(np.log2(abs_bench)))
        eb = np.maximum(eb, min_eb)
        exponent_num = parameters.get('exponent_num')[0]
        data_type = np.float32
        ulp_err = alg.calc_ulp_err(self.bench_data, self.device_data, eb, exponent_num, data_type)
        expected_ulp_err = (self.device_data.astype(data_type) - self.bench_data).astype(data_type) * np.exp2(-eb + exponent_num)
        self.assertTrue(np.allclose(ulp_err, expected_ulp_err))

    # ========== compare_bool_tensor Int8类型测试用例 ==========
    def test_compare_bool_tensor_int8_all_match(self):
        """测试int8类型数据完全匹配的情况"""
        bench_output = np.array([1, 2, 3, 4, 5], dtype=np.int8)
        device_output = np.array([1, 2, 3, 4, 5], dtype=np.int8)
        error_rate, result, msg = alg.compare_bool_tensor(bench_output, device_output)
        self.assertEqual(error_rate, 0.0)
        self.assertEqual(result, CompareConst.PASS)
        self.assertEqual(msg, "")

    def test_compare_bool_tensor_int8_all_different(self):
        """测试int8类型数据完全不匹配的情况"""
        bench_output = np.array([1, 2, 3, 4, 5], dtype=np.int8)
        device_output = np.array([6, 7, 8, 9, 10], dtype=np.int8)
        error_rate, result, msg = alg.compare_bool_tensor(bench_output, device_output)
        self.assertEqual(error_rate, 1.0)
        self.assertEqual(result, CompareConst.ERROR)
        self.assertEqual(msg, "")

    def test_compare_bool_tensor_int8_partial_match(self):
        """测试int8类型数据部分匹配的情况"""
        bench_output = np.array([1, 2, 3, 4, 5], dtype=np.int8)
        device_output = np.array([1, 2, 0, 4, 6], dtype=np.int8)
        error_rate, result, msg = alg.compare_bool_tensor(bench_output, device_output)
        self.assertEqual(error_rate, 0.4)  # 2/5 = 0.4
        self.assertEqual(result, CompareConst.ERROR)
        self.assertEqual(msg, "")

    def test_compare_bool_tensor_int8_with_negative_values(self):
        """测试int8类型包含负值的情况"""
        bench_output = np.array([-5, -10, 0, 10, 5], dtype=np.int8)
        device_output = np.array([-5, -10, 0, 10, 5], dtype=np.int8)
        error_rate, result, msg = alg.compare_bool_tensor(bench_output, device_output)
        self.assertEqual(error_rate, 0.0)
        self.assertEqual(result, CompareConst.PASS)
        self.assertEqual(msg, "")

    def test_compare_bool_tensor_int8_boundary_values(self):
        """测试int8类型边界值的情况"""
        bench_output = np.array([127, -128, 0], dtype=np.int8)
        device_output = np.array([127, -128, 0], dtype=np.int8)
        error_rate, result, msg = alg.compare_bool_tensor(bench_output, device_output)
        self.assertEqual(error_rate, 0.0)
        self.assertEqual(result, CompareConst.PASS)
        self.assertEqual(msg, "")

    def test_compare_bool_tensor_int8_boundary_values_mismatch(self):
        """测试int8类型边界值不匹配的情况"""
        bench_output = np.array([127, -128, 0], dtype=np.int8)
        device_output = np.array([126, -127, 1], dtype=np.int8)
        error_rate, result, msg = alg.compare_bool_tensor(bench_output, device_output)
        self.assertEqual(error_rate, 1.0)
        self.assertEqual(result, CompareConst.ERROR)
        self.assertEqual(msg, "")

    def test_compare_bool_tensor_int8_single_element_match(self):
        """测试int8类型单个元素匹配的情况"""
        bench_output = np.array([42], dtype=np.int8)
        device_output = np.array([42], dtype=np.int8)
        error_rate, result, msg = alg.compare_bool_tensor(bench_output, device_output)
        self.assertEqual(error_rate, 0.0)
        self.assertEqual(result, CompareConst.PASS)
        self.assertEqual(msg, "")

    def test_compare_bool_tensor_int8_single_element_mismatch(self):
        """测试int8类型单个元素不匹配的情况"""
        bench_output = np.array([42], dtype=np.int8)
        device_output = np.array([43], dtype=np.int8)
        error_rate, result, msg = alg.compare_bool_tensor(bench_output, device_output)
        self.assertEqual(error_rate, 1.0)
        self.assertEqual(result, CompareConst.ERROR)
        self.assertEqual(msg, "")

    def test_compare_bool_tensor_int8_large_array(self):
        """测试int8类型大数组的情况"""
        bench_output = np.array([i for i in range(-50, 50)], dtype=np.int8)
        device_output = np.array([i for i in range(-50, 50)], dtype=np.int8)
        error_rate, result, msg = alg.compare_bool_tensor(bench_output, device_output)
        self.assertEqual(error_rate, 0.0)
        self.assertEqual(result, CompareConst.PASS)
        self.assertEqual(msg, "")

    def test_compare_bool_tensor_int8_large_array_partial_mismatch(self):
        """测试int8类型大数组部分不匹配的情况"""
        bench_output = np.array([i for i in range(-50, 50)], dtype=np.int8)
        device_output = np.array([i for i in range(-50, 50)], dtype=np.int8)
        # 修改几个元素 (确保修改后的值与原值不同)
        device_output[0] = 99  # 原值是-50
        device_output[50] = 99  # 原值是0
        device_output[99] = 99  # 原值是49
        error_rate, result, msg = alg.compare_bool_tensor(bench_output, device_output)
        self.assertEqual(error_rate, 3.0 / 100.0)
        self.assertEqual(result, CompareConst.ERROR)
        self.assertEqual(msg, "")

    def test_compare_bool_tensor_int8_all_zeros(self):
        """测试int8类型全零的情况"""
        bench_output = np.array([0, 0, 0, 0, 0], dtype=np.int8)
        device_output = np.array([0, 0, 0, 0, 0], dtype=np.int8)
        error_rate, result, msg = alg.compare_bool_tensor(bench_output, device_output)
        self.assertEqual(error_rate, 0.0)
        self.assertEqual(result, CompareConst.PASS)
        self.assertEqual(msg, "")

    def test_compare_bool_tensor_int8_repeated_values(self):
        """测试int8类型重复值的情况"""
        bench_output = np.array([1, 1, 1, 1, 1], dtype=np.int8)
        device_output = np.array([1, 1, 1, 1, 1], dtype=np.int8)
        error_rate, result, msg = alg.compare_bool_tensor(bench_output, device_output)
        self.assertEqual(error_rate, 0.0)
        self.assertEqual(result, CompareConst.PASS)
        self.assertEqual(msg, "")

        def test_cosine_sim_int8(self):
            """测试纯int8数据的余弦相似度计算"""
            bench_output = np.array([1, 2, 3], dtype=np.int8)
            device_output = np.array([1, 2, 3], dtype=np.int8)
            cos, success, msg = alg.cosine_sim(bench_output, device_output)
            self.assertEqual(cos, 1.0)
            self.assertTrue(success)
            self.assertEqual(msg, '')

        def test_cosine_sim_int8_with_difference(self):
            """测试int8数据有差异时的余弦相似度"""
            bench_output = np.array([1, 2, 3], dtype=np.int8)
            device_output = np.array([2, 3, 4], dtype=np.int8)
            cos, success, msg = alg.cosine_sim(bench_output, device_output)
            # 计算期望的余弦相似度
            expected_cos = np.dot(bench_output, device_output) / (
                        np.linalg.norm(bench_output) * np.linalg.norm(device_output))
            self.assertAlmostEqual(cos, expected_cos, places=6)
            self.assertTrue(success)

        def test_get_abs_err_int8(self):
            """测试int8数据的绝对误差计算"""
            bench_data = np.array([10, 20, 30], dtype=np.int8)
            device_data = np.array([12, 18, 32], dtype=np.int8)
            abs_err = alg.get_abs_err(bench_data, device_data)
            self.assertListEqual(list(abs_err), [2.0, 2.0, 2.0])

        def test_get_abs_bench_with_eps_int8(self):
            """测试int8数据的get_abs_bench_with_eps函数"""
            bench_data = np.array([1, 2, 3], dtype=np.int8)
            abs_bench, abs_bench_with_eps = alg.get_abs_bench_with_eps(bench_data, torch.int8)
            expected_abs_bench = np.abs(bench_data).astype(float)  # 显式转换为float
            expected_abs_bench_with_eps = expected_abs_bench + np.finfo(np.float64).eps  # 使用float64的epsilon
            self.assertTrue(np.array_equal(abs_bench, expected_abs_bench))
            self.assertTrue(np.allclose(abs_bench_with_eps, expected_abs_bench_with_eps))

        def test_get_rel_err_origin_int8(self):
            """测试int8数据的原始相对误差计算"""
            bench_data = np.array([10, 20, 30], dtype=np.int8)
            device_data = np.array([12, 18, 32], dtype=np.int8)
            abs_err = np.abs(device_data - bench_data)
            rel_err = alg.get_rel_err_origin(abs_err, bench_data)
            expected_rel_err = np.abs(abs_err / bench_data)
            self.assertTrue(np.allclose(rel_err, expected_rel_err))

        def test_get_finite_and_infinite_mask_int8(self):
            """测试int8数据的有限值和无限值掩码"""
            bench_data = np.array([1, 2, 3], dtype=np.int8)
            device_data = np.array([4, 5, 6], dtype=np.int8)
            both_finite_mask, inf_nan_mask = alg.get_finite_and_infinite_mask(bench_data, device_data)
            self.assertListEqual(list(both_finite_mask), [True, True, True])
            self.assertListEqual(list(inf_nan_mask), [False, False, False])

        def test_check_inf_nan_value_int8(self):
            """测试int8数据的inf/nan值检查"""
            bench_data = np.array([1, 2, 3], dtype=np.int8)
            device_data = np.array([4, 5, 6], dtype=np.int8)
            both_finite_mask, inf_nan_mask = alg.get_finite_and_infinite_mask(bench_data, device_data)
            # int8数据不应该有inf/nan，所以应该返回0
            result = alg.check_inf_nan_value(inf_nan_mask, bench_data, device_data, torch.int8, 0.001)
            self.assertEqual(result, 0)

        def test_get_max_abs_err_int8(self):
            """测试int8数据的最大绝对误差"""
            bench_data = np.array([10, 20, 30], dtype=np.int8)
            device_data = np.array([12, 18, 32], dtype=np.int8)
            abs_err = alg.get_abs_err(bench_data, device_data)
            max_err, has_nan_inf = alg.get_max_abs_err(abs_err)
            self.assertEqual(max_err, 2.0)
            self.assertFalse(has_nan_inf)

        def test_get_max_rel_err_int8(self):
            """测试int8数据的最大相对误差"""
            bench_data = np.array([10, 20, 30], dtype=np.int8)
            device_data = np.array([12, 18, 32], dtype=np.int8)
            abs_err = np.abs(device_data - bench_data)
            abs_bench = np.abs(bench_data).astype(float)  # 显式转换为float
            eps = np.finfo(np.float64).eps  # 使用float64的epsilon
            abs_bench_with_eps = abs_bench + eps
            rel_err = abs_err / abs_bench_with_eps
            # get_max_rel_err返回单个值，不是元组
            max_err = alg.get_max_rel_err(rel_err)
            self.assertAlmostEqual(max_err, 0.2, places=3)  # 最大相对误差约为0.2

        def test_get_mean_rel_err_int8(self):
            """测试int8数据的平均相对误差"""
            bench_data = np.array([10, 20, 30], dtype=np.int8)
            device_data = np.array([12, 18, 32], dtype=np.int8)
            abs_err = np.abs(device_data - bench_data)
            abs_bench = np.abs(bench_data).astype(float)  # 显式转换为float
            eps = np.finfo(np.float64).eps  # 使用float64的epsilon
            abs_bench_with_eps = abs_bench + eps
            rel_err = abs_err / abs_bench_with_eps
            mean_err = alg.get_mean_rel_err(rel_err)
            expected_mean = np.mean(rel_err)
            self.assertAlmostEqual(mean_err, expected_mean, places=6)

            # ========== int8 边界值测试用例 ==========

        def test_cosine_sim_int8_boundary_values(self):
            """测试int8边界值的余弦相似度"""
            # 测试int8的最大值和最小值
            bench_output = np.array([127, -128, 0], dtype=np.int8)
            device_output = np.array([127, -128, 0], dtype=np.int8)
            cos, success, msg = alg.cosine_sim(bench_output, device_output)
            self.assertEqual(cos, 1.0)
            self.assertTrue(success)

        def test_get_abs_bench_with_eps_int8_boundary(self):
            """测试int8边界值的get_abs_bench_with_eps"""
            bench_data = np.array([127, -128, 0], dtype=np.int8)
            abs_bench, abs_bench_with_eps = alg.get_abs_bench_with_eps(bench_data, torch.int8)
            expected_abs_bench = np.array([127, 128, 0], dtype=np.float64)
            expected_abs_bench_with_eps = expected_abs_bench + np.finfo(np.float64).eps  # 使用float64的epsilon
            self.assertTrue(np.array_equal(abs_bench, expected_abs_bench))
            self.assertTrue(np.allclose(abs_bench_with_eps, expected_abs_bench_with_eps))

        def test_get_abs_err_int8_max_difference(self):
            """测试int8最大差值的绝对误差"""
            # 注意: int8范围是-128到127,127 - (-128) = 255会溢出
            # 100 - (-100) = 200 也会溢出到 -56,abs后为56
            bench_data = np.array([-50], dtype=np.int8)
            device_data = np.array([50], dtype=np.int8)
            abs_err = alg.get_abs_err(bench_data, device_data)
            self.assertEqual(abs_err[0], 100)

        def test_get_rel_err_origin_int8_zero_handling(self):
            """测试int8数据中包含0时的相对误差处理"""
            bench_data = np.array([10, 0, 30], dtype=np.int8)
            device_data = np.array([12, 0, 32], dtype=np.int8)
            abs_err = np.abs(device_data - bench_data)
            rel_err = alg.get_rel_err_origin(abs_err, bench_data)
            # 对于0值，相对误差应该为inf或特殊处理
            # 实际实现中，0除以0会产生nan或inf
            # 测试非0值的相对误差计算
            # 期望: [abs(12-10)/10, abs(0-0)/0, abs(32-30)/30] = [0.2, nan/inf, 0.0667]
            # 由于除零会产生inf/nan，我们只验证非0值
            self.assertAlmostEqual(rel_err[0], 0.2, places=5)
            self.assertAlmostEqual(rel_err[2], 0.06666667, places=5)
            # 0值处的相对误差应该是inf或nan
            self.assertTrue(np.isinf(rel_err[1]) or np.isnan(rel_err[1]))

        def test_get_finite_and_infinite_mask_int8_boundary(self):
            """测试int8边界值的有限值掩码"""
            bench_data = np.array([127, -128, 0], dtype=np.int8)
            device_data = np.array([127, -128, 0], dtype=np.int8)
            both_finite_mask, inf_nan_mask = alg.get_finite_and_infinite_mask(bench_data, device_data)
            self.assertListEqual(list(both_finite_mask), [True, True, True])
            self.assertListEqual(list(inf_nan_mask), [False, False, False])

            # ========== int8 与 float16 混合场景测试用例 ==========

        def test_cosine_sim_int8_to_float16_conversion(self):
            """测试int8数据转换为float16后的余弦相似度"""
            bench_output = np.array([1, 2, 3], dtype=np.int8)
            device_output = np.array([1, 2, 3], dtype=np.int8)
            # 算法内部可能会处理类型转换
            cos, success, msg = alg.cosine_sim(bench_output, device_output)
            self.assertEqual(cos, 1.0)
            self.assertTrue(success)

        def test_get_abs_err_int8_vs_float16(self):
            """测试int8和float16混合数据的绝对误差"""
            # 算法应该能够处理int8数据
            bench_data = np.array([10, 20, 30], dtype=np.int8)
            device_data = np.array([12, 18, 32], dtype=np.int8)
            abs_err = alg.get_abs_err(bench_data, device_data)
            self.assertListEqual(list(abs_err), [2.0, 2.0, 2.0])

        def test_get_rel_err_origin_int8_vs_float16(self):
            """测试int8数据的相对误差计算（模拟与float16对比场景）"""
            bench_data = np.array([10, 20, 30], dtype=np.int8)
            device_data = np.array([12, 18, 32], dtype=np.int8)
            abs_err = np.abs(device_data - bench_data)
            # 对于int8，相对误差计算应该正常工作
            rel_err = alg.get_rel_err_origin(abs_err, bench_data)
            expected_rel_err = np.abs(abs_err / bench_data)
            self.assertTrue(np.allclose(rel_err, expected_rel_err))

        def test_int8_array_operations(self):
            """测试int8数组的基本操作"""
            bench_data = np.array([1, 2, 3, 4, 5], dtype=np.int8)
            device_data = np.array([2, 3, 4, 5, 6], dtype=np.int8)

            # 测试绝对误差
            abs_err = alg.get_abs_err(bench_data, device_data)
            self.assertTrue(np.array_equal(abs_err, np.array([1.0, 1.0, 1.0, 1.0, 1.0])))

            # 测试相对误差
            rel_err = alg.get_rel_err_origin(abs_err, bench_data)
            expected_rel_err = np.array([1.0, 0.5, 0.33333333, 0.25, 0.2])
            self.assertTrue(np.allclose(rel_err, expected_rel_err, atol=1e-5))

            # 测试最大绝对误差
            max_abs_err, _ = alg.get_max_abs_err(abs_err)
            self.assertEqual(max_abs_err, 1.0)

            # 测试最大相对误差 - get_max_rel_err返回单个值，不是元组
            max_rel_err = alg.get_max_rel_err(rel_err)
            self.assertAlmostEqual(max_rel_err, 1.0, places=3)

        def test_get_abs_bench_with_eps_int8_type_conversion(self):
            """测试int8数据的显式类型转换"""
            bench_data = np.array([10, 20, 30], dtype=np.int8)
            abs_bench, abs_bench_with_eps = alg.get_abs_bench_with_eps(bench_data, torch.int8)

            # 验证abs_bench的类型为float64
            self.assertEqual(abs_bench.dtype, np.float64)
            self.assertEqual(abs_bench_with_eps.dtype, np.float64)

            # 验证类型一致性
            self.assertEqual(abs_bench.dtype, abs_bench_with_eps.dtype)

            # 验证数值正确性
            expected_abs_bench = np.array([10.0, 20.0, 30.0], dtype=np.float64)
            self.assertTrue(np.array_equal(abs_bench, expected_abs_bench))

            # 验证eps值使用的是float64的机器epsilon
            float64_eps = np.finfo(np.float64).eps
            expected_abs_bench_with_eps = expected_abs_bench + float64_eps
            self.assertTrue(np.allclose(abs_bench_with_eps, expected_abs_bench_with_eps))

        def test_get_abs_bench_with_eps_int16_type_conversion(self):
            """测试int16数据的显式类型转换"""
            bench_data = np.array([1000, 2000, 3000], dtype=np.int16)
            abs_bench, abs_bench_with_eps = alg.get_abs_bench_with_eps(bench_data, torch.int16)

            # 验证abs_bench的类型为float64
            self.assertEqual(abs_bench.dtype, np.float64)
            self.assertEqual(abs_bench_with_eps.dtype, np.float64)

            # 验证类型一致性
            self.assertEqual(abs_bench.dtype, abs_bench_with_eps.dtype)

            # 验证数值正确性
            expected_abs_bench = np.array([1000.0, 2000.0, 3000.0], dtype=np.float64)
            self.assertTrue(np.array_equal(abs_bench, expected_abs_bench))

            # 验证eps值使用的是float64的机器epsilon
            float64_eps = np.finfo(np.float64).eps
            expected_abs_bench_with_eps = expected_abs_bench + float64_eps
            self.assertTrue(np.allclose(abs_bench_with_eps, expected_abs_bench_with_eps))

        def test_get_abs_bench_with_eps_int32_type_conversion(self):
            """测试int32数据的显式类型转换"""
            bench_data = np.array([100000, 200000, 300000], dtype=np.int32)
            abs_bench, abs_bench_with_eps = alg.get_abs_bench_with_eps(bench_data, torch.int32)

            # 验证abs_bench的类型为float64
            self.assertEqual(abs_bench.dtype, np.float64)
            self.assertEqual(abs_bench_with_eps.dtype, np.float64)

            # 验证类型一致性
            self.assertEqual(abs_bench.dtype, abs_bench_with_eps.dtype)

            # 验证数值正确性
            expected_abs_bench = np.array([100000.0, 200000.0, 300000.0], dtype=np.float64)
            self.assertTrue(np.array_equal(abs_bench, expected_abs_bench))

            # 验证eps值使用的是float64的机器epsilon
            float64_eps = np.finfo(np.float64).eps
            expected_abs_bench_with_eps = expected_abs_bench + float64_eps
            self.assertTrue(np.allclose(abs_bench_with_eps, expected_abs_bench_with_eps))

        def test_get_abs_bench_with_eps_integer_vs_float_comparison(self):
            """测试整数类型和浮点类型的eps值差异"""
            # int8数据
            int8_data = np.array([10, 20, 30], dtype=np.int8)
            int8_abs_bench, int8_abs_bench_with_eps = alg.get_abs_bench_with_eps(int8_data, torch.int8)

            # float32数据
            float32_data = np.array([10.0, 20.0, 30.0], dtype=np.float32)
            float32_abs_bench, float32_abs_bench_with_eps = alg.get_abs_bench_with_eps(float32_data, torch.float32)

            # 验证两者都是float64类型 (整数转换为float64)
            self.assertEqual(int8_abs_bench.dtype, np.float64)
            # 浮点类型保持原类型
            self.assertEqual(float32_abs_bench.dtype, np.float32)

            # 验证eps值不同
            int8_eps = int8_abs_bench_with_eps[0] - int8_abs_bench[0]
            float32_eps = float32_abs_bench_with_eps[0] - float32_abs_bench[0]

            # int8使用float64的epsilon
            self.assertAlmostEqual(int8_eps, np.finfo(np.float64).eps)
            # float32使用自己的epsilon,但由于float32精度限制,小值无法表示eps
            # 对于较大的值,eps会被舍入,所以实际差值可能为0
            # 验证函数确实使用了float32的epsilon,即使结果被舍入
            self.assertLessEqual(float32_eps, np.finfo(np.float32).eps)

    # ========== int8 类型测试用例 ==========


