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

import unittest
from unittest.mock import patch, mock_open, MagicMock
import os
from msprobe.mindspore.api_accuracy_checker.api_accuracy_checker import DataManager
from msprobe.core.common.const import CompareConst
from msprobe.mindspore.common.const import MsCompareConst


class TestDataManager(unittest.TestCase):

    def setUp(self):
        # 设置测试的CSV目录和结果路径
        self.csv_dir = "./test_data_manager_csv_dir"
        self.result_csv_path = os.path.join(self.csv_dir, "result.csv")
        self.details_csv_path = os.path.join(self.csv_dir, "details.csv")  # 新增 details.csv 路径

        # 确保目录存在
        if not os.path.exists(self.csv_dir):
            os.makedirs(self.csv_dir)

        # 确保文件存在，并写入正确的表头
        with open(self.result_csv_path, 'w') as f:
            f.write("API Name,Forward Test Success,Backward Test Success,Message\n")  # 写入必需的表头字段
            # 写入一些示例数据
            f.write("API1,pass,pass,All tests passed\n")
            f.write("API2,pass,pass,Forward test failed\n")
            f.write("api_name,pass,pass,Backward test failed\n")

        # 确保 details.csv 文件存在，并写入一些默认数据
        with open(self.details_csv_path, 'w') as f:
            f.write("api_name,details\n")  # 表头
            f.write("API1,Detail for API1\n")
            f.write("API2,Detail for API2\n")

        # 创建 DataManager 实例
        self.data_manager = DataManager(self.csv_dir, self.result_csv_path)

    def test_is_unique_api(self):
        # 测试唯一 API 名称检测
        self.assertTrue(self.data_manager.is_unique_api("API4"))
        self.assertFalse(self.data_manager.is_unique_api("API4"))

    @patch('os.path.exists', return_value=True)  # Mock路径存在
    @patch('msprobe.core.common.file_utils.check_file_or_directory_path')
    @patch('os.path.isfile', return_value=True)  # Mock文件存在
    @patch('os.access', return_value=True)  # Mock文件可读权限
    def test_resume_from_last_csv(self, mock_access, mock_isfile, mock_check_file, mock_exists):
        # 测试恢复断点功能
        self.data_manager.resume_from_last_csv(self.result_csv_path)

        # 验证路径和输出是否正确
        self.assertEqual(self.data_manager.csv_dir, os.path.dirname(self.result_csv_path))
        self.assertIsNotNone(self.data_manager.detail_out_path)
        self.assertIsNotNone(self.data_manager.result_out_path)

    @patch('msprobe.core.common.file_utils.FileOpen', mock_open())
    def test_clear_results(self):
        # 测试清除结果
        self.data_manager.results[("API1", "FORWARD")] = ["test_data"]
        self.data_manager.clear_results()
        self.assertEqual(len(self.data_manager.results), 0)

    @patch('msprobe.core.common.file_utils.write_csv')
    def test_record(self, mock_write_csv):
        # 测试记录数据
        output_list = [("API1", "FORWARD", MagicMock(api_name="API1"), MagicMock())]
        self.data_manager.record(output_list)
        self.assertIn(("API1", "FORWARD"), self.data_manager.results)
        self.assertEqual(len(self.data_manager.results[("API1", "FORWARD")]), 1)

    def test_record_exception_skip(self):
        self.data_manager.record_exception_skip("API3", "FORWARD", "custom err msg")
        self.assertEqual(self.data_manager.results_exception_skip["API3"]["FORWARD"], "custom err msg")

    def tearDown(self):
        # 清理创建的测试目录和文件
        if os.path.exists(self.csv_dir):
            for root, dirs, files in os.walk(self.csv_dir, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
            os.rmdir(self.csv_dir)

if __name__ == "__main__":
    unittest.main()
