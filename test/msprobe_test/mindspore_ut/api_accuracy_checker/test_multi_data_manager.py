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

# Python 标准库
import multiprocessing
import os
import threading
from collections import namedtuple
from multiprocessing import Manager

# 第三方库
import unittest
from unittest.mock import MagicMock, patch

# 自定义模块
from msprobe.mindspore.api_accuracy_checker.data_manager import (
    get_detail_csv_header,
    get_result_csv_header,
    write_csv_header
)
from msprobe.mindspore.api_accuracy_checker.multi_data_manager import MultiDataManager

class TestMultiDataManager(unittest.TestCase):

    def setUp(self):
        # 创建测试目录和文件路径
        self.csv_dir = "./test_data_csv_dir"
        self.result_csv_path = os.path.join(self.csv_dir, "result.csv")
        self.detail_csv_path = os.path.join(self.csv_dir, "details.csv")

        # 创建测试目录
        os.makedirs(self.csv_dir, exist_ok=True)

        # **添加以下代码，创建结果文件并写入表头**
        # 创建 result.csv 文件并写入表头
        with open(self.result_csv_path, 'w') as f:
            f.write("API Name,Forward Test Success,Backward Test Success,Message\n")  # 必需的表头字段

        # 创建 detail.csv 文件并写入表头
        with open(self.detail_csv_path, 'w') as f:
            f.write("api_name,details\n")  # 必需的表头字段

        # 初始化共享变量
        self.manager = Manager()
        self.shared_is_first_write = self.manager.Value('b', True)

        # 创建 MultiDataManager 实例
        self.data_manager = MultiDataManager(self.csv_dir, self.result_csv_path, self.shared_is_first_write)

    def test_save_results_first_write(self):
        # 测试初次写入表头的情况
        self.data_manager.is_first_write = True
        self.shared_is_first_write.value = True
        api_name = "TestAPI"

        with patch("msprobe.mindspore.api_accuracy_checker.multi_data_manager.write_csv_header") as mock_write_header:
            with patch.object(self.data_manager, 'to_detail_csv') as mock_to_detail_csv:
                with patch.object(self.data_manager, 'to_result_csv') as mock_to_result_csv:
                    # 调用 save_results
                    self.data_manager.save_results(api_name)

                    # 验证表头写入方法被调用两次（detail 和 result）
                    mock_write_header.assert_any_call(self.data_manager.detail_out_path, get_detail_csv_header)
                    mock_write_header.assert_any_call(self.data_manager.result_out_path, get_result_csv_header)
                    self.assertEqual(mock_write_header.call_count, 2)

                    # 验证 is_first_write 和 shared_is_first_write 的值已更新
                    self.assertFalse(self.data_manager.is_first_write)
                    self.assertFalse(self.shared_is_first_write.value)

    def test_save_results_multiple_calls(self):
        # 测试连续多次调用 save_results
        api_name = "TestAPI"
        self.data_manager.is_first_write = True

        with patch("msprobe.mindspore.api_accuracy_checker.multi_data_manager.write_csv_header") as mock_write_header:
            with patch.object(self.data_manager, 'to_detail_csv') as mock_to_detail_csv:
                with patch.object(self.data_manager, 'to_result_csv') as mock_to_result_csv:
                    # 连续调用 save_results
                    for _ in range(3):
                        self.data_manager.save_results(api_name)

                    # 验证表头写入方法只被调用一次
                    self.assertEqual(mock_write_header.call_count, 2)
                    mock_write_header.assert_any_call(self.data_manager.detail_out_path, get_detail_csv_header)
                    mock_write_header.assert_any_call(self.data_manager.result_out_path, get_result_csv_header)

                    # 验证详细输出和结果摘要写入次数
                    self.assertEqual(mock_to_detail_csv.call_count, 3)
                    self.assertEqual(mock_to_result_csv.call_count, 3)

    def test_save_results_with_shared_is_first_write_false(self):
        # 测试 shared_is_first_write 已经为 False 的情况
        self.data_manager.is_first_write = True
        self.shared_is_first_write.value = False
        api_name = "TestAPI"

        with patch("msprobe.mindspore.api_accuracy_checker.multi_data_manager.write_csv_header") as mock_write_header:
            self.data_manager.save_results(api_name)

            # 验证表头写入方法未被调用
            mock_write_header.assert_not_called()

    def test_save_results_exception_handling(self):
        # 测试 save_results 方法在出现异常时的处理
        api_name = "TestAPI"

        with patch.object(self.data_manager, 'to_detail_csv', side_effect=Exception("Test Exception")):
            with patch.object(self.data_manager, 'to_result_csv') as mock_to_result_csv:
                # 调用 save_results，应该抛出异常
                with self.assertRaises(Exception) as context:
                    self.data_manager.save_results(api_name)

                # 验证异常信息
                self.assertEqual(str(context.exception), "Test Exception")

                # 验证 to_result_csv 未被调用
                mock_to_result_csv.assert_not_called()

    def test_clear_results_after_save(self):
        # 测试在调用 save_results 后，results 是否被清空
        self.data_manager.results = {'some_key': 'some_value'}
        api_name = "TestAPI"

        with patch.object(self.data_manager, 'to_detail_csv'):
            with patch.object(self.data_manager, 'to_result_csv'):
                self.data_manager.save_results(api_name)

                # 验证 results 已被清空
                self.assertEqual(self.data_manager.results, {})

    def test_thread_safety_with_threads(self):
        # 使用多线程测试线程安全性
        self.data_manager.is_first_write = True
        self.shared_is_first_write.value = True
        api_name = "TestAPI"
        call_counts = {'write_header': 0}

        original_write_csv_header = write_csv_header

        def write_csv_header_counter(*args, **kwargs):
            call_counts['write_header'] += 1
            return original_write_csv_header(*args, **kwargs)

        with patch("msprobe.mindspore.api_accuracy_checker.multi_data_manager.write_csv_header",
                   side_effect=write_csv_header_counter):
            def run_save_results():
                self.data_manager.save_results(api_name)

            threads = []
            for _ in range(5):
                t = threading.Thread(target=run_save_results)
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            # 验证表头写入方法只被调用一次
            if 'write_header' in call_counts:  # 确保 key 在有效范围内
                self.assertEqual(call_counts['write_header'], 2)  # detail 和 result 各一次

    def test_save_results_with_existing_api_names(self):
        # 测试当 api_names_set 已包含某个 API 名称时的行为
        api_name = "TestAPI"
        self.data_manager.api_names_set.add(api_name)

        with patch.object(self.data_manager, 'to_detail_csv') as mock_to_detail_csv:
            with patch.object(self.data_manager, 'to_result_csv') as mock_to_result_csv:
                self.data_manager.save_results(api_name)

                # 验证 to_detail_csv 和 to_result_csv 仍被调用
                mock_to_detail_csv.assert_called_once()
                mock_to_result_csv.assert_called_once()

    def test_save_results_without_results(self):
        # 测试在 results 未设置的情况下调用 save_results
        del self.data_manager.results  # 删除 results 属性
        api_name = "TestAPI"

        with self.assertRaises(AttributeError):
            self.data_manager.save_results(api_name)

    def test_save_results_with_empty_results(self):
        # 测试在 results 为空的情况下调用 save_results
        self.data_manager.results = {}
        api_name = "TestAPI"

        with patch.object(self.data_manager, 'to_detail_csv') as mock_to_detail_csv:
            with patch.object(self.data_manager, 'to_result_csv') as mock_to_result_csv:
                self.data_manager.save_results(api_name)

                # 验证 to_detail_csv 和 to_result_csv 被调用，即使 results 为空
                mock_to_detail_csv.assert_called_once()
                mock_to_result_csv.assert_called_once()

    def test_clear_results_empty(self):
        # 测试 clear_results 在 results 已为空时的行为
        self.data_manager.results = {}
        self.data_manager.clear_results()
        self.assertEqual(self.data_manager.results, {})

    def tearDown(self):
        # 清理测试目录
        if os.path.exists(self.csv_dir):
            for root, dirs, files in os.walk(self.csv_dir, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
            os.rmdir(self.csv_dir)


if __name__ == "__main__":
    unittest.main()
