import unittest
from unittest.mock import Mock, patch
import tempfile
import os
from collections import defaultdict

from msprobe.core.dump.dump2db.db_utils import (
    DumpDB,
    parse_full_key
)
from msprobe.core.common.const import Data2DBConst


class TestParseFunctions(unittest.TestCase):
    """测试解析函数"""

    def test_parse_full_key_batch(self):
        """批量测试解析full key"""
        test_cases = [
            # (full_key, expected_patterns)
            ("Module.layer1.conv.input.0",
             [
                 ("input.0", Data2DBConst.TAG_INDEX),
                 ("layer1.conv", Data2DBConst.TAG_MODULE),
                 ("Module", Data2DBConst.TAG_DEFAULT)
             ]),
            ("Module.testlayername.layer.1.blocks.1.conv2.Conv2d.input.2",
             [
                 ("layer.1", Data2DBConst.TAG_LAYER),
                 ("blocks.1", Data2DBConst.TAG_LAYER),
                 ("testlayername", Data2DBConst.TAG_MODULE),
                 ("conv2.Conv2d", Data2DBConst.TAG_MODULE),
                 ("Module", Data2DBConst.TAG_DEFAULT),
                 ("input.2", Data2DBConst.TAG_INDEX)
             ]),
            ("short.key",  []),
            ("function_name.input.0", [
             ("function_name", Data2DBConst.TAG_FUNCTION),
             ("input.0", Data2DBConst.TAG_INDEX)
             ]),
            ("Module.Layer0.parameters.weight",
             [
                 ("parameters.weight", Data2DBConst.TAG_INDEX),
                 ("parameters", Data2DBConst.TAG_DEFAULT),
                 ("Module", Data2DBConst.TAG_DEFAULT),
                 ("Layer0", Data2DBConst.TAG_MODULE)
             ]),
            ("Module.layer1.parameters.weight.0",
             [
                 ("weight.0", Data2DBConst.TAG_INDEX),
                 ("parameters", Data2DBConst.TAG_DEFAULT),
                 ("Module", Data2DBConst.TAG_DEFAULT),
                 ("layer1.parameters", Data2DBConst.TAG_MODULE)
             ]),
            ("Module.layer1.parameters.mlp.0.weight",
             [
                 ("0.weight", Data2DBConst.TAG_INDEX),
                 ("parameters", Data2DBConst.TAG_DEFAULT),
                 ("Module", Data2DBConst.TAG_DEFAULT),
                 ("layer1.parameters.mlp", Data2DBConst.TAG_MODULE)
             ]),
        ]

        for i, (full_key, expected_patterns) in enumerate(test_cases):
            with self.subTest(case=i, full_key=full_key):
                tags = parse_full_key(full_key)
                self.assertIsInstance(tags, set)
                for pattern, expected_type in expected_patterns:
                    self.assertTrue(
                        ((pattern, expected_type) in tags),
                        f"Pattern '{pattern}' with type '{expected_type}' not found in {tags}")
                self.assertEqual(len(tags), len(expected_patterns))


class TestDumpDB(unittest.TestCase):
    """测试DumpDB类"""

    def setUp(self):
        """测试前置设置"""
        self.mock_db_manager = Mock()
        self.mock_db_manager.select_data.return_value = {}
        self.mock_db_manager.insert_data.return_value = 1
        self.db_path = tempfile.mkstemp(suffix='.db')

        with patch('msprobe.core.dump.dump2db.db_utils.DBManager') as mock_db_manager_class:
            mock_db_manager_class.return_value = self.mock_db_manager
            self.dump_db = DumpDB(self.db_path)
        self.mock_db_manager.insert_data.reset_mock()
        self.mock_db_manager.select_data.reset_mock()

    def tearDown(self):
        """测试后置清理"""
        if os.path.exists(self.db_path[1]):
            os.unlink(self.db_path[1])

    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.dump_db.db_path, self.db_path)
        self.assertIsInstance(self.dump_db._metric_id_cache, dict)
        self.assertIsInstance(self.dump_db._processed_targets, defaultdict)

    def test_get_metric_id_existing(self):
        """测试获取已存在的metric ID"""
        # 设置缓存
        self.dump_db._metric_id_cache["forward"] = 1
        result = self.dump_db.get_metric_id("forward")
        self.assertEqual(result, 1)

        # 不应该查询数据库
        self.mock_db_manager.select_data.assert_not_called()

    def test_get_metric_id_new(self):
        """测试获取新的metric ID"""
        # 模拟数据库查询返回结果
        self.mock_db_manager.select_data.return_value = [{"metric_id": 2}]
        # 清空初始化时缓存
        self.dump_db._metric_id_cache.clear()
        result = self.dump_db.get_metric_id("backward")
        self.assertEqual(result, 2)

        # 验证缓存被更新
        self.assertEqual(self.dump_db._metric_id_cache["backward"], 2)

    def test_cache_targets_new_target(self):
        """测试缓存新target"""
        target = ("layer1", 0, 0)
        metric_id = 1

        result = self.dump_db.cache_targets(target, metric_id)

        self.assertEqual(result, {'id': 0})
        self.assertIn(target, self.dump_db._processed_targets[metric_id])
        self.assertIn(target, self.dump_db._new_targets[metric_id])

    def test_cache_targets_existing_target(self):
        """测试缓存已存在的target"""
        target = ("layer1", 0, 0)
        metric_id = 1
        self.dump_db._processed_targets[metric_id][target] = {'id': 123}

        result = self.dump_db.cache_targets(target, metric_id)

        self.assertEqual(result, {'id': 123})
        self.assertNotIn(target, self.dump_db._new_targets[metric_id])

    def test_batch_insert_targets(self):
        """测试批量插入targets"""
        # 设置测试数据
        metric_id = 1
        targets = [("layer1", 0, 0), ("layer2", 0, 1)]

        for target in targets:
            self.dump_db.cache_targets(target, metric_id)

        # 模拟数据库查询返回
        self.mock_db_manager.select_data.return_value = [
            {"target_id": 1, "target_name": "layer1",
                "vpp_stage": 0, "micro_step": 0},
            {"target_id": 2, "target_name": "layer2",
                "vpp_stage": 0, "micro_step": 1}
        ]

        self.dump_db.batch_insert_targets()

        # 验证插入被调用
        self.mock_db_manager.insert_data.assert_called_once()

        # 验证缓存被更新
        self.assertEqual(
            self.dump_db._processed_targets[metric_id][targets[0]]['id'], 1)
        self.assertEqual(
            self.dump_db._processed_targets[metric_id][targets[1]]['id'], 2)

        # 验证新targets列表被清空
        self.assertEqual(len(self.dump_db._new_targets[metric_id]), 0)

    def test_batch_insert_data(self):
        """测试批量插入数据"""
        batch_data = [
            [0, 100, {"id": 1}, 1, 1.0, 0.0],
            [0, 101, {"id": 2}, 1, 2.0, 1.0]
        ]

        # 模拟表存在
        self.mock_db_manager.table_exists.return_value = True

        self.dump_db.batch_insert_data(batch_data)

        # 验证插入被调用
        self.mock_db_manager.insert_data.assert_called_once()

        # 验证target id被转换
        call_args = self.mock_db_manager.insert_data.call_args[0]
        rows = call_args[1]
        self.assertEqual(rows[0][2], 1)  # 应该转换为数字ID

    def test_extract_tags_from_processed_targets(self):
        """测试从已处理的targets中提取标签"""
        # 设置测试数据
        metric_id = 1
        target = ("Module.layer1.conv.input.0", 0, 0)
        self.dump_db._processed_targets[metric_id][target] = {'id': 1}

        # 模拟标签查询返回
        self.mock_db_manager.select_data.return_value = [
            {"tag_id": 1, "tag_name": "input.0",
                "category": "index", "metric_id": 1},
            {"tag_id": 2, "tag_name": "layer1",
                "category": "module", "metric_id": 1}
        ]

        self.dump_db.extract_tags_from_processed_targets()

        # 验证标签插入被调用
        self.mock_db_manager.insert_data.assert_called()

        # 验证映射关系插入被调用
        mapping_calls = [call for call in self.mock_db_manager.insert_data.call_args_list
                         if call[0][0] == "tag_target_mapping"]
        self.assertTrue(len(mapping_calls) > 0)

    def test_extract_tags_from_processed_targets_no_targets(self):
        """测试提取标签-无targets情况"""
        self.dump_db.extract_tags_from_processed_targets()

        # 不应该调用插入
        self.mock_db_manager.insert_data.assert_not_called()


class TestDumpDBIntegration(unittest.TestCase):
    """DumpDB集成测试"""

    def setUp(self):
        """测试前置设置"""
        self.db_path = tempfile.mkstemp(suffix='.db')

    def tearDown(self):
        """测试后置清理"""
        if os.path.exists(self.db_path[1]):
            os.unlink(self.db_path[1])

    @patch('msprobe.core.dump.dump2db.db_utils.DBManager')
    def test_complete_workflow(self, mock_db_manager_class):
        """测试完整工作流程"""
        mock_db_manager = Mock()
        mock_db_manager_class.return_value = mock_db_manager

        # 测试metric ID获取
        mock_db_manager.select_data.return_value = [{"metric_id": 1}]
        # 创建DumpDB实例
        dump_db = DumpDB(self.db_path)

        metric_id = dump_db.get_metric_id("forward")
        self.assertEqual(metric_id, 1)

        # 测试target缓存
        target = ("layer1.input.0", 0, 0)
        cache_result = dump_db.cache_targets(target, metric_id)
        self.assertEqual(cache_result, {'id': 0})

        # 测试批量插入targets
        mock_db_manager.select_data.return_value = [
            {"target_id": 1, "target_name": "layer1.input.0",
                "vpp_stage": 0, "micro_step": 0}
        ]
        dump_db.batch_insert_targets()

        # 验证target ID被更新
        self.assertEqual(
            dump_db._processed_targets[metric_id][target]['id'], 1)

        # 测试批量插入数据
        batch_data = {
            "metric_1_step_0_49": [
                [0, 100, {"id": 1}, 1.0, 0.0, 0.5, 0.1, 100]
            ]
        }
        mock_db_manager.table_exists.return_value = True
        dump_db.batch_insert_data(batch_data)

        # 验证数据插入被调用
        mock_db_manager.insert_data.assert_called()
