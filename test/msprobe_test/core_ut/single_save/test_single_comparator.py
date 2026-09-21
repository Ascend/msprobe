# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

import os
import tempfile
import shutil
import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import sys

# 添加被测试模块的路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from msprobe.core.single_save.single_comparator import SingleComparator, CompareResult


class TestSingleComparator(unittest.TestCase):
    """SingleComparator单元测试类"""

    def setUp(self):
        """测试前准备临时目录"""
        self.temp_dir = tempfile.mkdtemp()
        self.dir1 = os.path.join(self.temp_dir, "dir1")
        self.dir2 = os.path.join(self.temp_dir, "dir2")
        self.output_dir = os.path.join(self.temp_dir, "output")

        # 创建测试目录结构
        os.makedirs(self.dir1, exist_ok=True)
        os.makedirs(self.dir2, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # 创建测试数据目录结构
        self.data_dir1 = os.path.join(self.dir1, "data")
        self.data_dir2 = os.path.join(self.dir2, "data")
        os.makedirs(self.data_dir1, exist_ok=True)
        os.makedirs(self.data_dir2, exist_ok=True)

    def tearDown(self):
        """测试后清理临时目录"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_compare_result_dataclass(self):
        """测试CompareResult数据类"""
        result = CompareResult(
            max_abs_error=0.1,
            max_relative_error=0.01,
            same_percentage=99.5,
            first_mismatch_index=5,
            percentage_within_thousandth=99.9,
            percentage_within_hundredth=99.99
        )

        self.assertEqual(result.max_abs_error, 0.1)
        self.assertEqual(result.max_relative_error, 0.01)
        self.assertEqual(result.same_percentage, 99.5)
        self.assertEqual(result.first_mismatch_index, 5)
        self.assertEqual(result.percentage_within_thousandth, 99.9)
        self.assertEqual(result.percentage_within_hundredth, 99.99)

    def test_compare_arrays_identical(self):
        """测试比较完全相同的数组"""
        array1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        array2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = SingleComparator.compare_arrays(array1, array2)

        self.assertEqual(result.max_abs_error, 0.0)
        self.assertEqual(result.max_relative_error, 0.0)
        self.assertEqual(result.same_percentage, 100.0)
        self.assertIsNone(result.first_mismatch_index)
        self.assertEqual(result.percentage_within_thousandth, 100.0)
        self.assertEqual(result.percentage_within_hundredth, 100.0)

    def test_compare_arrays_different(self):
        """测试比较不同的数组"""
        array1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        array2 = np.array([1.0, 2.0, 3.5, 4.0, 5.0])

        result = SingleComparator.compare_arrays(array1, array2)

        self.assertEqual(result.max_abs_error, 0.5)
        self.assertAlmostEqual(result.max_relative_error, 0.142857, places=5)  # 0.5/3.5
        self.assertEqual(result.same_percentage, 80.0)  # 4/5相同
        self.assertEqual(result.first_mismatch_index, 2)

    def test_compare_arrays_different_shapes(self):
        """测试比较不同形状的数组"""
        array1 = np.array([[1.0, 2.0], [3.0, 4.0]])  # 2x2
        array2 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # 2x3

        result = SingleComparator.compare_arrays(array1, array2)

        # 应比较相同部分(2x2)
        self.assertEqual(result.max_abs_error, 1.0)

    def test_compare_arrays_with_zeros(self):
        """测试包含0的数组（测试相对误差的分母为0的情况）"""
        array1 = np.array([0.0, 0.0, 1.0])
        array2 = np.array([0.0, 0.1, 1.0])

        result = SingleComparator.compare_arrays(array1, array2)

        # 应该能正确处理分母为0的情况
        self.assertEqual(result.max_abs_error, 0.1)
        # 第一个元素0-0/0，会被np.nan_to_num处理为0

    def test_compare_arrays_with_nan_inf(self):
        """测试包含特殊值的数组"""
        array1 = np.array([1.0, np.nan, np.inf])
        array2 = np.array([1.0, 2.0, np.inf])

        result = SingleComparator.compare_arrays(array1, array2)
        # 主要测试是否能正常处理，不抛出异常

    def test_get_steps(self):
        """测试获取step目录"""
        # 创建测试目录结构
        tag_path = os.path.join(self.temp_dir, "test_tag")
        os.makedirs(tag_path, exist_ok=True)

        # 创建step目录
        os.makedirs(os.path.join(tag_path, "step0"), exist_ok=True)
        os.makedirs(os.path.join(tag_path, "step1"), exist_ok=True)
        os.makedirs(os.path.join(tag_path, "step10"), exist_ok=True)
        os.makedirs(os.path.join(tag_path, "other_dir"), exist_ok=True)  # 不应被识别

        steps = list(SingleComparator.get_steps(tag_path))

        self.assertEqual(len(steps), 3)
        step_numbers = [step for step, _ in steps]
        self.assertIn(0, step_numbers)
        self.assertIn(1, step_numbers)
        self.assertIn(10, step_numbers)

    def test_get_steps_invalid_name(self):
        """测试无效的step目录名"""
        tag_path = os.path.join(self.temp_dir, "test_tag_invalid")
        os.makedirs(tag_path, exist_ok=True)
        os.makedirs(os.path.join(tag_path, "step_abc"), exist_ok=True)  # 无效名称

        with self.assertRaises(RuntimeError):
            list(SingleComparator.get_steps(tag_path))

    def test_get_ranks(self):
        """测试获取rank目录"""
        step_path = os.path.join(self.temp_dir, "test_step")
        os.makedirs(step_path, exist_ok=True)

        os.makedirs(os.path.join(step_path, "rank0"), exist_ok=True)
        os.makedirs(os.path.join(step_path, "rank1"), exist_ok=True)

        ranks = list(SingleComparator.get_ranks(step_path))

        self.assertEqual(len(ranks), 2)
        rank_numbers = [rank for rank, _ in ranks]
        self.assertIn(0, rank_numbers)
        self.assertIn(1, rank_numbers)

    def test_get_micro_steps(self):
        """测试获取micro_step目录"""
        rank_path = os.path.join(self.temp_dir, "test_rank")
        os.makedirs(rank_path, exist_ok=True)

        os.makedirs(os.path.join(rank_path, "micro_step0"), exist_ok=True)
        os.makedirs(os.path.join(rank_path, "micro_step1"), exist_ok=True)
        os.makedirs(os.path.join(rank_path, "other_dir"), exist_ok=True)  # 应返回0

        micro_steps = list(SingleComparator.get_micro_steps(rank_path))

        # 应该包含3个条目
        self.assertEqual(len(micro_steps), 3)

        # 检查micro_step编号
        micro_step_numbers = []
        for micro_step, path in micro_steps:
            micro_step_numbers.append(micro_step)
            if "other_dir" in path:
                self.assertEqual(micro_step, 0)

        self.assertIn(0, micro_step_numbers)
        self.assertIn(1, micro_step_numbers)

    def test_get_arrays(self):
        """测试获取npy文件"""
        micro_step_path = os.path.join(self.temp_dir, "test_micro_step")
        os.makedirs(micro_step_path, exist_ok=True)

        # 创建npy文件
        np.save(os.path.join(micro_step_path, "0.npy"), np.array([1, 2, 3]))
        np.save(os.path.join(micro_step_path, "1.npy"), np.array([4, 5, 6]))
        np.save(os.path.join(micro_step_path, "10.npy"), np.array([7, 8, 9]))

        arrays = list(SingleComparator.get_arrays(micro_step_path))

        self.assertEqual(len(arrays), 3)

        # 检查array_id
        array_ids = [array_id for array_id, _ in arrays]
        self.assertIn(0, array_ids)
        self.assertIn(1, array_ids)
        self.assertIn(10, array_ids)

    def test_get_array_paths_simple_structure(self):
        """测试简单目录结构"""
        # 创建测试目录结构: tag/step0/rank0/array_0.npy
        tag_dir = os.path.join(self.data_dir1, "tag1")
        step_dir = os.path.join(tag_dir, "step0")
        rank_dir = os.path.join(step_dir, "rank0")
        os.makedirs(rank_dir, exist_ok=True)

        # 保存npy文件
        array_path = os.path.join(rank_dir, "array_0.npy")
        np.save(array_path, np.array([1, 2, 3]))

        array_paths = SingleComparator.get_array_paths(self.data_dir1)

        self.assertIn("tag1", array_paths)
        self.assertEqual(len(array_paths["tag1"]), 1)

        step, rank, micro_step, array_id, path = array_paths["tag1"][0]
        self.assertEqual(step, 0)
        self.assertEqual(rank, 0)
        self.assertEqual(micro_step, 0)
        self.assertEqual(array_id, 0)
        self.assertEqual(path, array_path)

    def test_get_array_paths_with_micro_steps(self):
        """测试包含micro_step的目录结构"""
        # 创建测试目录结构: tag/step0/rank0/micro_step0/array_0.npy
        tag_dir = os.path.join(self.data_dir1, "tag2")
        step_dir = os.path.join(tag_dir, "step1")
        rank_dir = os.path.join(step_dir, "rank1")
        micro_step_dir = os.path.join(rank_dir, "micro_step0")
        os.makedirs(micro_step_dir, exist_ok=True)

        # 保存npy文件
        array_path = os.path.join(micro_step_dir, "array_0.npy")
        np.save(array_path, np.array([1, 2, 3, 4, 5]))

        array_paths = SingleComparator.get_array_paths(self.data_dir1)

        self.assertIn("tag2", array_paths)
        self.assertEqual(len(array_paths["tag2"]), 1)

        step, rank, micro_step, array_id, path = array_paths["tag2"][0]
        self.assertEqual(step, 1)
        self.assertEqual(rank, 1)
        self.assertEqual(micro_step, 0)
        self.assertEqual(array_id, 0)
        self.assertEqual(path, array_path)

    def test_get_array_paths_empty_directory(self):
        """测试空目录"""
        array_paths = SingleComparator.get_array_paths(self.temp_dir)
        self.assertEqual(len(array_paths), 0)

    def test_get_array_paths_nonexistent_directory(self):
        """测试不存在的目录"""
        array_paths = SingleComparator.get_array_paths("/nonexistent/path")
        self.assertEqual(len(array_paths), 0)

    def test_compare_single_tag_no_common(self):
        """测试没有共同tag的情况"""
        # 创建不同的tag
        tag_dir1 = os.path.join(self.data_dir1, "tag1")
        tag_dir2 = os.path.join(self.data_dir2, "tag2")
        os.makedirs(tag_dir1, exist_ok=True)
        os.makedirs(tag_dir2, exist_ok=True)

        array_paths1 = {"tag1": [(0, 0, 0, 0, "dummy_path")]}
        array_paths2 = {"tag2": [(0, 0, 0, 0, "dummy_path")]}

        # 应该不抛出异常
        SingleComparator.compare_single_tag("tag1", array_paths1, array_paths2, self.output_dir)

        # 检查输出文件是否存在
        output_file = os.path.join(self.output_dir, "tag1.xlsx")
        self.assertTrue(os.path.exists(output_file))

    def test_compare_single_tag_with_real_data(self):
        """测试真实数据比较"""
        # 创建测试数据
        tag_dir1 = os.path.join(self.data_dir1, "tag1")
        tag_dir2 = os.path.join(self.data_dir2, "tag1")

        # 相同结构
        step_dir1 = os.path.join(tag_dir1, "step0")
        step_dir2 = os.path.join(tag_dir2, "step0")
        rank_dir1 = os.path.join(step_dir1, "rank0")
        rank_dir2 = os.path.join(step_dir2, "rank0")
        os.makedirs(rank_dir1, exist_ok=True)
        os.makedirs(rank_dir2, exist_ok=True)

        # 保存测试数组
        array1 = np.array([1.0, 2.0, 3.0])
        array2 = np.array([1.0, 2.1, 3.0])  # 第二个元素不同

        array_path1 = os.path.join(rank_dir1, "array_0.npy")
        array_path2 = os.path.join(rank_dir2, "array_0.npy")
        np.save(array_path1, array1)
        np.save(array_path2, array2)

        # 获取路径
        array_paths1 = SingleComparator.get_array_paths(self.data_dir1)
        array_paths2 = SingleComparator.get_array_paths(self.data_dir2)

        # 执行比较
        SingleComparator.compare_single_tag("tag1", array_paths1, array_paths2, self.output_dir)

        # 检查输出文件
        output_file = os.path.join(self.output_dir, "tag1.xlsx")
        self.assertTrue(os.path.exists(output_file))

        # 读取并验证结果
        df = pd.read_excel(output_file)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['step'], 0)
        self.assertEqual(df.iloc[0]['rank'], 0)
        self.assertEqual(df.iloc[0]['micro_step'], 0)
        self.assertEqual(df.iloc[0]['id'], 0)
        self.assertAlmostEqual(df.iloc[0]['相同元素百分比(%)'], 66.666, delta=0.1)
        self.assertEqual(df.iloc[0]['首个不匹配元素索引'], 1)
        self.assertAlmostEqual(df.iloc[0]['最大绝对误差'], 0.1, places=5)

    @patch('multiprocessing.Pool')
    def test_compare_data(self, mock_pool_class):
        """测试compare_data方法"""
        # 模拟multiprocessing.Pool
        mock_pool = MagicMock()
        mock_pool_class.return_value.__enter__.return_value = mock_pool
        mock_pool.starmap_async.return_value = MagicMock()
        mock_pool.starmap_async.return_value.ready.return_value = True
        mock_pool.starmap_async.return_value._number_left = 0

        # 创建测试数据
        tag_dir1 = os.path.join(self.data_dir1, "tag1")
        tag_dir2 = os.path.join(self.data_dir2, "tag1")
        os.makedirs(tag_dir1, exist_ok=True)
        os.makedirs(tag_dir2, exist_ok=True)

        SingleComparator.compare_data(self.data_dir1, self.data_dir2, self.output_dir, num_processes=2)

        # 验证pool被正确调用
        mock_pool_class.assert_called_once_with(processes=2)

    @patch('multiprocessing.Pool')
    def test_compare_data_exception(self, mock_pool_class):
        """测试compare_data方法中的异常处理"""
        mock_pool = MagicMock()
        mock_pool_class.return_value.__enter__.return_value = mock_pool

        # 模拟异常
        mock_pool.starmap_async.side_effect = Exception("Test exception")

        # 应该不抛出异常，而是记录错误
        SingleComparator.compare_data(self.data_dir1, self.data_dir2, self.output_dir)

    def test_compare_class_method(self):
        """测试主compare方法"""
        # 创建完整的目录结构
        data_dir1 = os.path.join(self.dir1, "data")
        data_dir2 = os.path.join(self.dir2, "data")
        os.makedirs(data_dir1, exist_ok=True)
        os.makedirs(data_dir2, exist_ok=True)

        # 创建测试数据
        tag_dir1 = os.path.join(data_dir1, "tag1", "step0", "rank0")
        tag_dir2 = os.path.join(data_dir2, "tag1", "step0", "rank0")
        os.makedirs(tag_dir1, exist_ok=True)
        os.makedirs(tag_dir2, exist_ok=True)

        # 保存相同数组
        array = np.array([1.0, 2.0, 3.0])
        np.save(os.path.join(tag_dir1, "array_0.npy"), array)
        np.save(os.path.join(tag_dir2, "array_0.npy"), array)

        # 执行比较
        with patch('multiprocessing.Pool') as mock_pool_class:
            mock_pool = MagicMock()
            mock_pool_class.return_value.__enter__.return_value = mock_pool
            mock_pool.starmap_async.return_value = MagicMock()
            mock_pool.starmap_async.return_value.ready.return_value = True
            mock_pool.starmap_async.return_value._number_left = 0

            SingleComparator.compare(self.dir1, self.dir2, self.output_dir, num_processes=2)

        # 验证输出目录被创建
        self.assertTrue(os.path.exists(self.output_dir))

    def test_compare_class_method_invalid_path(self):
        """测试compare方法无效路径"""
        with self.assertRaises(Exception):  # 具体异常类型依赖于check_file_or_directory_path的实现
            SingleComparator.compare("/invalid/path1", "/invalid/path2", self.output_dir)

    def test_compare_class_method_invalid_process_num(self):
        """测试无效的进程数"""
        with patch('msprobe.core.common.utils.check_process_num') as mock_check:
            mock_check.side_effect = ValueError("Invalid process number")

            with self.assertRaises(ValueError):
                SingleComparator.compare(self.dir1, self.dir2, self.output_dir, num_processes=0)

    def test_result_header(self):
        """测试结果表头"""
        expected_header = [
            'step', 'rank', 'micro_step', 'id', 'shape1', 'shape2',
            '相同元素百分比(%)', '首个不匹配元素索引', '最大绝对误差',
            '最大相对误差', '误差在千分之一内元素占比(%)', '误差在百分之一内元素占比(%)'
        ]

        self.assertEqual(SingleComparator.result_header, expected_header)
        self.assertEqual(len(SingleComparator.result_header), 12)

    def test_multidimensional_array_comparison(self):
        """测试多维数组比较"""
        # 测试3维数组
        array1 = np.random.rand(3, 4, 5)
        array2 = array1.copy()
        array2[0, 0, 0] += 0.001  # 微小差异

        result = SingleComparator.compare_arrays(array1, array2)

        self.assertAlmostEqual(result.max_abs_error, 0.001, places=5)
        self.assertLess(result.same_percentage, 100.0)

    def test_large_array_comparison(self):
        """测试大数组比较（性能测试）"""
        array1 = np.ones((100, 100))
        array2 = np.ones((100, 100))
        array2[50, 50] = 2.0

        result = SingleComparator.compare_arrays(array1, array2)

        self.assertEqual(result.max_abs_error, 1.0)
        self.assertEqual(result.same_percentage, 99.99)  # 9999/10000

    def test_complex_directory_structure(self):
        """测试复杂目录结构"""
        # 创建复杂目录结构
        base_dir = os.path.join(self.temp_dir, "complex")
        os.makedirs(base_dir, exist_ok=True)

        # 创建多种结构的目录
        structures = [
            ("tag1/step0/rank0/0.npy", 0, 0, 0, 0),
            ("tag1/step1/rank0/micro_step0/1.npy", 1, 0, 0, 1),
            ("tag1/step2/rank1/micro_step1/2.npy", 2, 1, 1, 2),
        ]

        for path_template, step, rank, micro_step, array_id in structures:
            full_path = os.path.join(base_dir, path_template)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            np.save(full_path, np.array([1, 2, 3]))

        array_paths = SingleComparator.get_array_paths(base_dir)

        self.assertIn("tag1", array_paths)
        self.assertEqual(len(array_paths["tag1"]), 3)

        # 验证所有结构都被正确解析
        found_structures = set()
        for step, rank, micro_step, array_id, _ in array_paths["tag1"]:
            found_structures.add((step, rank, micro_step, array_id))

        for _, step, rank, micro_step, array_id in structures:
            self.assertIn((step, rank, micro_step, array_id), found_structures)


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)
