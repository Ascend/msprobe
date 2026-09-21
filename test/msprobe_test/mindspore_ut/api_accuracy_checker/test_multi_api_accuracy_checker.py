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
import os
from multiprocessing import Manager, Queue

# 第三方库
import unittest
from unittest.mock import patch, MagicMock

# 自定义模块
from msprobe.mindspore.api_accuracy_checker.multi_api_accuracy_checker import (
    MultiApiAccuracyChecker
)
from msprobe.mindspore.api_accuracy_checker.multi_data_manager import MultiDataManager
from msprobe.core.common.const import Const


class TestMultiApiAccuracyChecker(unittest.TestCase):
    def setUp(self):
        # 初始化参数
        self.args = MagicMock()
        self.args.out_path = "./test_output"
        self.args.result_csv_path = "./test_output/result.csv"
        self.args.device_id = [0, 1]  # 模拟两个设备ID

        # **创建测试输出目录**
        if not os.path.exists(self.args.out_path):
            os.makedirs(self.args.out_path)

        # **创建空的 result.csv 文件**
        with open(self.args.result_csv_path, 'w') as f:
            f.write("API Name,Forward Test Success,Backward Test Success,Message\n")  # 写入表头

        # 创建 MultiApiAccuracyChecker 实例
        self.checker = MultiApiAccuracyChecker(self.args)

        # 模拟 api_infos 数据
        self.checker.api_infos = {
            'API_1': MagicMock(),
            'API_2': MagicMock(),
            'API_3': MagicMock(),
            'API_4': MagicMock(),
        }

    @patch('msprobe.mindspore.api_accuracy_checker.multi_api_accuracy_checker.logger')
    def test_process_forward_no_forward_info(self, mock_logger):
        """测试当 check_forward_info 返回 False 时，process_forward 返回 None 并记录调试日志。"""
        # 设置 current_device_id
        self.checker.current_device_id = 0

        api_info = MagicMock()
        api_info.check_forward_info.return_value = False

        result = self.checker.process_forward("API_1", api_info)

        self.assertEqual(result, Const.EXCEPTION_NONE)
        mock_logger.debug.assert_called_with(
            "[Device 0] API: API_1 lacks forward information, skipping forward check."
        )

    def tearDown(self):
        # 清理资源
        if hasattr(self.checker, 'manager'):
            self.checker.manager.shutdown()
        # 清理测试输出目录
        if os.path.exists(self.args.out_path):
            for root, dirs, files in os.walk(self.args.out_path, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
            os.rmdir(self.args.out_path)


if __name__ == '__main__':
    unittest.main()
