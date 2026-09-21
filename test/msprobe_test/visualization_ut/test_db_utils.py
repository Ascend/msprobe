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
import sqlite3
import unittest
from unittest.mock import Mock, patch
import tempfile
from msprobe.visualization.db_utils import (  # 请替换为实际模块名
    create_table_sql_from_dict,
    create_insert_sql_from_dict,
    to_db,
    add_table_index,
    post_process_db,
    node_to_db,
    config_to_db,
    get_graph_unique_id,
    get_node_unique_id,
    node_columns,
    indexes
)


class TestDatabaseFunctions(unittest.TestCase):

    def setUp(self):
        # 创建临时文件作为测试数据库
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db').name
        self.addCleanup(os.remove, self.temp_db)

    def test_create_table_sql_from_dict(self):
        """测试表创建SQL语句生成"""
        test_table = "test_table"
        test_columns = {
            "id": "INTEGER PRIMARY KEY",
            "name": "TEXT NOT NULL"
        }

        expected_sql = """CREATE TABLE IF NOT EXISTS test_table (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);"""

        generated_sql = create_table_sql_from_dict(test_table, test_columns)
        self.assertEqual(generated_sql, expected_sql)

    def test_create_insert_sql_from_dict(self):
        """测试插入SQL语句生成"""
        test_table = "test_table"
        test_columns = {"id": "INTEGER", "name": "TEXT"}

        # 测试普通插入
        expected_sql = "INSERT INTO test_table (id, name) VALUES (?, ?)"
        generated_sql = create_insert_sql_from_dict(test_table, test_columns)
        self.assertEqual(generated_sql, expected_sql)

        # 测试忽略插入
        expected_sql = "INSERT OR IGNORE INTO test_table (id, name) VALUES (?, ?)"
        generated_sql = create_insert_sql_from_dict(test_table, test_columns, ignore_insert=True)
        self.assertEqual(generated_sql, expected_sql)

    def test_to_db(self):
        """测试数据库写入功能"""
        # 创建测试表和数据
        create_sql = create_table_sql_from_dict("test_table", {"id": "INTEGER PRIMARY KEY", "name": "TEXT"})
        insert_sql = create_insert_sql_from_dict("test_table", {"id": "INTEGER", "name": "TEXT"})
        test_data = [(1, "test1"), (2, "test2"), (3, "test3")]

        # 执行写入
        to_db(self.temp_db, create_sql, insert_sql, test_data)

        # 验证数据是否正确写入
        conn = sqlite3.connect(self.temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM test_table")
        result = cursor.fetchall()
        conn.close()

        self.assertEqual(result, test_data)

    def test_add_table_index(self):
        """测试索引添加功能"""
        # 先创建测试表
        conn = sqlite3.connect(self.temp_db)
        cursor = conn.cursor()
        cursor.execute(create_table_sql_from_dict('tb_nodes', node_columns))
        conn.commit()
        conn.close()

        # 添加索引
        add_table_index(self.temp_db)

        # 验证索引是否创建
        conn = sqlite3.connect(self.temp_db)
        cursor = conn.cursor()
        cursor.execute("PRAGMA index_list(tb_nodes)")
        indexes_result = cursor.fetchall()
        conn.close()

        # 检查是否存在预期的索引
        index_names = [idx[1] for idx in indexes_result]
        for index_name in indexes.keys():
            self.assertIn(index_name, index_names)

    def test_post_process_db(self):
        """测试数据库后处理功能"""
        # 创建测试表
        conn = sqlite3.connect(self.temp_db)
        cursor = conn.cursor()
        cursor.execute(create_table_sql_from_dict('tb_nodes', node_columns))
        conn.commit()
        conn.close()

        with patch('msprobe.visualization.db_utils.add_table_index') as mock_add_index:
            post_process_db(self.temp_db)
            mock_add_index.assert_called_once_with(self.temp_db)

    def test_get_graph_unique_id(self):
        """测试图形唯一ID生成"""
        # 创建模拟graph对象
        mock_graph = Mock()
        mock_graph.data_source = "test_source"
        mock_graph.step = 5
        mock_graph.rank = 2

        expected_id = "test_source_5_2"
        self.assertEqual(get_graph_unique_id(mock_graph), expected_id)

    def test_get_node_unique_id(self):
        """测试节点唯一ID生成"""
        # 创建模拟对象
        mock_graph = Mock()
        mock_graph.data_source = "test_source"
        mock_graph.step = 5
        mock_graph.rank = 2

        mock_node = Mock()
        mock_node.id = "node_123"

        expected_id = "test_source_5_2_node_123"
        self.assertEqual(get_node_unique_id(mock_graph, mock_node), expected_id)

    @patch('msprobe.visualization.db_utils.to_db')
    @patch('msprobe.visualization.db_utils.get_node_unique_id')
    @patch('msprobe.visualization.db_utils.get_graph_unique_id')
    @patch('msprobe.visualization.builder.msprobe_adapter.format_node_data')
    def test_node_to_db(self, mock_format, mock_graph_id, mock_node_id, mock_to_db):
        """测试节点数据写入数据库"""
        # 配置模拟
        mock_graph_id.return_value = "graph_123"
        mock_node_id.return_value = "node_456"
        mock_format.return_value = {}

        # 创建模拟graph和node
        mock_node = Mock()
        mock_node.id = "node1"
        mock_node.op.value = "OPERATION"
        mock_node.upnode = None
        mock_node.subnodes = []
        mock_node.data = {}
        mock_node.micro_step_id = 1
        mock_node.matched_node_link = {}
        mock_node.stack_info = {}
        mock_node.parallel_merge_info = None
        mock_node.matched_distributed = {}
        mock_node.input_data = {}
        mock_node.output_data = {}

        mock_graph = Mock()
        mock_graph.get_sorted_nodes.return_value = [mock_node]
        mock_graph.data_source = "test_source"
        mock_graph.data_path = "/test/path"
        mock_graph.step = 1
        mock_graph.rank = 0

        # 执行测试
        node_to_db(mock_graph, self.temp_db)

        # 验证to_db被正确调用
        self.assertTrue(mock_to_db.called)

    @patch('msprobe.visualization.db_utils.to_db')
    def test_config_to_db(self, mock_to_db):
        """测试配置数据写入数据库"""
        mock_config = Mock()
        mock_config.graph_b = False
        mock_config.task = "test_task"
        mock_config.tool_tip = "test tooltip"
        mock_config.micro_steps = 10
        mock_config.overflow_check = 1
        mock_config.node_colors = {}
        mock_config.rank_list = [0, 1, 2, 3]
        mock_config.step_list = [0]

        # 执行测试
        config_to_db(mock_config, self.temp_db)

        # 验证to_db被正确调用
        self.assertTrue(mock_to_db.called)


if __name__ == '__main__':
    unittest.main()
