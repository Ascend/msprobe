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
import os
import sys
import json
import tempfile
from unittest.mock import patch, MagicMock, mock_open

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from msprobe.core.compare.find_first.utils import (
    RankPath, FileCache, is_communication_op, is_ignore_op, 
    DiffAnalyseConst, analyze_diff_in_group
)
from msprobe.core.common.const import Const


class TestRankPath(unittest.TestCase):
    def setUp(self):
        # 创建临时文件用于测试
        self.temp_dir = tempfile.mkdtemp()
        self.dump_path = os.path.join(self.temp_dir, "dump.json")
        # 创建一个空文件
        with open(self.dump_path, "w") as f:
            f.write("{}")
    
    def tearDown(self):
        # 清理临时文件
        if os.path.exists(self.dump_path):
            os.remove(self.dump_path)
        os.rmdir(self.temp_dir)
    
    def test_init(self):
        # 测试正常初始化
        rank_path = RankPath(1, self.dump_path)
        self.assertEqual(rank_path.rank, 1)
        self.assertEqual(rank_path.dump_path, self.dump_path)
    
    @patch('msprobe.core.compare.find_first.utils.check_file_or_directory_path')
    def test_init_with_invalid_path(self, mock_check):
        # 测试无效路径
        mock_check.side_effect = ValueError("Invalid path")
        with self.assertRaises(ValueError):
            RankPath(1, "/invalid/path")


class TestFileCache(unittest.TestCase):
    def setUp(self):
        # 重置单例
        FileCache._instance = None
        self.cache = FileCache()
        # 创建临时文件用于测试
        self.temp_dir = tempfile.mkdtemp()
        self.json_path = os.path.join(self.temp_dir, "test.json")
        self.test_data = {"key": "value"}
        with open(self.json_path, "w") as f:
            json.dump(self.test_data, f)
    
    def tearDown(self):
        # 清理临时文件
        if os.path.exists(self.json_path):
            os.remove(self.json_path)
        os.rmdir(self.temp_dir)
    
    def test_singleton(self):
        # 测试单例模式
        cache2 = FileCache()
        self.assertIs(self.cache, cache2)
    
    def test_load_json(self):
        # 测试加载JSON文件
        result = self.cache.load_json(self.json_path)
        self.assertEqual(result, self.test_data)
        
        # 测试缓存功能
        result2 = self.cache.load_json(self.json_path)
        self.assertEqual(result2, self.test_data)
        self.assertEqual(self.cache._access_cnt[self.json_path], 1)  # 访问计数应该增加
    
    @patch('msprobe.core.compare.find_first.utils.load_json')
    def test_cleanup(self, mock_load_json):
        # 模拟大文件加载
        mock_load_json.return_value = {"large": "x" * 1000000}  # 创建一个大对象
        
        # 修改最大内存使用量为很小的值，强制清理
        original_max = self.cache._max_memory_usage
        self.cache._max_memory_usage = 100  # 设置为很小的值
        
        # 加载多个文件触发清理
        for i in range(5):
            self.cache.load_json(f"{self.json_path}_{i}")
        
        # 恢复原始值
        self.cache._max_memory_usage = original_max


class TestCommunicationFunctions(unittest.TestCase):
    def test_is_communication_op(self):
        # 测试通信算子识别
        self.assertTrue(is_communication_op("Distributed.all_reduce.1"))
        self.assertTrue(is_communication_op("Distributed.send.2"))
        self.assertTrue(is_communication_op("Distributed.broadcast.3"))
        self.assertTrue(is_communication_op(f"{Const.MINT_DIST_API_TYPE_PREFIX}.all_gather.4"))
        self.assertTrue(is_communication_op(f"{Const.MS_API_TYPE_COM}.reduce.5"))
        
        # 测试非通信算子
        self.assertFalse(is_communication_op("Torch.add.1"))
        self.assertFalse(is_communication_op("Torch.matmul.2"))
    
    def test_is_ignore_op(self):
        # 测试忽略算子识别
        self.assertTrue(is_ignore_op("Torch.empty.1"))
        self.assertTrue(is_ignore_op("Torch.fill.2"))
        
        # 测试非忽略算子
        self.assertFalse(is_ignore_op("Torch.add.1"))
        self.assertFalse(is_ignore_op("Torch.matmul.2"))


class TestDiffAnalyseConst(unittest.TestCase):
    def test_constants(self):
        # 测试常量定义
        self.assertIn('send', DiffAnalyseConst.COMMUNICATION_KEYWORDS)
        self.assertIn('recv', DiffAnalyseConst.COMMUNICATION_KEYWORDS)
        self.assertIn('all_reduce', DiffAnalyseConst.COMMUNICATION_KEYWORDS)
        
        # 测试P2P API映射
        self.assertEqual(DiffAnalyseConst.P2P_API_MAPPING['send'], 'recv')
        self.assertEqual(DiffAnalyseConst.P2P_API_MAPPING['recv'], 'send')
        
        # 测试方向常量
        self.assertEqual(DiffAnalyseConst.OPPOSITE_DIR[DiffAnalyseConst.SRC], DiffAnalyseConst.DST)
        self.assertEqual(DiffAnalyseConst.OPPOSITE_DIR[DiffAnalyseConst.DST], DiffAnalyseConst.SRC)


class TestAnalyzeDiffInGroup(unittest.TestCase):
    def test_analyze_diff_in_group_empty(self):
        # 测试空组
        result = analyze_diff_in_group([])
        self.assertEqual(result, [])
    
    def test_analyze_diff_in_group(self):
        # 创建模拟的通信节点
        mock_node1 = MagicMock()
        mock_node1.type = DiffAnalyseConst.SRC
        mock_node1.is_diff = True
        mock_node1.compute_ops = [MagicMock(), MagicMock()]
        
        mock_node2 = MagicMock()
        mock_node2.type = DiffAnalyseConst.DST
        mock_node2.is_diff = False
        mock_node2.data.is_diff = True
        
        # 测试分析函数
        result = analyze_diff_in_group([mock_node1, mock_node2])
        
        # 验证结果包含所有异常节点
        self.assertEqual(len(result), 4)  # 2个计算节点 + 2个通信节点


if __name__ == '__main__':
    unittest.main()