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
import json
import os
import sqlite3
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from msprobe.visualization.db_utils import (
    DBActuator,
    add_table_index,
    create_insert_sql_from_dict,
    create_table_sql_from_dict,
    node_columns,
    to_db,
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



class TestDbUtilsExtra(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp.name, "test.db")
        with open(self.db_path, "wb"):
            pass

    def tearDown(self):
        self.tmp.cleanup()

    def _create_nodes_table(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute(create_table_sql_from_dict("tb_nodes", node_columns))
        conn.commit()
        conn.close()

    def test_invalid_identifiers(self):
        with self.assertRaises(ValueError):
            create_table_sql_from_dict("bad;drop", {"id": "TEXT"})
        with self.assertRaises(ValueError):
            create_table_sql_from_dict("tb", {"bad col": "TEXT"})
        with self.assertRaises(ValueError):
            create_insert_sql_from_dict("bad;drop", {"id": 1})
        with self.assertRaises(ValueError):
            create_insert_sql_from_dict("tb", {"bad col": 1})

    def test_create_insert_sql_ignore(self):
        sql = create_insert_sql_from_dict("tb", {"id": 1, "name": "x"}, ignore_insert=True)
        self.assertIn("INSERT OR IGNORE INTO", sql)
        sql = create_insert_sql_from_dict("tb", {"id": 1})
        self.assertIn("INSERT INTO", sql)

    def test_to_db(self):
        create_table_sql = create_table_sql_from_dict("tb_nodes", {"id": "TEXT PRIMARY KEY", "value": "TEXT"})
        insert_sql = create_insert_sql_from_dict("tb_nodes", {"id": None, "value": None})
        to_db(self.db_path, create_table_sql, insert_sql, [("a", "1"), ("b", "2")])

        conn = sqlite3.connect(self.db_path)
        count = conn.execute("SELECT COUNT(*) FROM tb_nodes").fetchone()[0]
        conn.close()
        self.assertEqual(count, 2)

    def test_to_db_with_pbar(self):
        create_table_sql = create_table_sql_from_dict("tb_nodes", {"id": "TEXT PRIMARY KEY", "value": "TEXT"})
        insert_sql = create_insert_sql_from_dict("tb_nodes", {"id": None, "value": None})
        pbar_info = MagicMock()
        with patch("msprobe.visualization.db_utils.update_shared_dict") as update_shared, \
                patch("msprobe.visualization.db_utils.update_pbar_info") as update_pbar:
            to_db(self.db_path, create_table_sql, insert_sql, [("a", "1")], pbar_info=pbar_info)
        update_shared.assert_called_once()
        update_pbar.assert_called_once()

    def test_to_db_retries_on_lock(self):
        create_table_sql = create_table_sql_from_dict("tb_nodes", {"id": "TEXT PRIMARY KEY"})
        insert_sql = create_insert_sql_from_dict("tb_nodes", {"id": None})
        fake_conn = MagicMock()
        with patch("msprobe.visualization.db_utils.sqlite3.connect",
                   side_effect=[sqlite3.OperationalError("database is locked"), fake_conn]), \
                patch("msprobe.visualization.db_utils.time.sleep") as sleep:
            to_db(self.db_path, create_table_sql, insert_sql, [("a",)])
        sleep.assert_called_once()
        fake_conn.commit.assert_called_once()

    def test_to_db_sqlite_error(self):
        create_table_sql = create_table_sql_from_dict("tb_nodes", {"id": "TEXT PRIMARY KEY"})
        insert_sql = create_insert_sql_from_dict("tb_nodes", {"id": None})
        with patch("msprobe.visualization.db_utils.sqlite3.connect", side_effect=sqlite3.Error("boom")):
            with self.assertRaises(sqlite3.Error):
                to_db(self.db_path, create_table_sql, insert_sql, [("a",)])

    def test_add_table_index(self):
        self._create_nodes_table()
        add_table_index(self.db_path)
        conn = sqlite3.connect(self.db_path)
        indexes = conn.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()
        conn.close()
        self.assertIn(("index1",), indexes)

    def test_query_distributed_nodes_info(self):
        self._create_nodes_table()
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO tb_nodes (id, graph_id, node_order, node_name, node_type, micro_step_id, "
            "modified, step, rank, data_source, is_distributed) VALUES "
            "('1', 'g', 0, 'node1', 'api', 0, 0, 0, 0, 'NPU', 1)"
        )
        conn.commit()
        conn.close()

        with DBActuator(self.db_path) as db:
            rows = db.query_distributed_nodes_info()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], "node1")

    def test_query_distributed_nodes_info_without_connection(self):
        db = object.__new__(DBActuator)
        db.db_path = self.db_path
        db.conn = None
        self.assertEqual(db.query_distributed_nodes_info(), [])

    def test_update_matched_distributed(self):
        self._create_nodes_table()
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO tb_nodes (id, graph_id, node_order, node_name, node_type, micro_step_id, "
            "modified, step, rank, data_source, is_distributed) VALUES "
            "('1', 'g', 0, 'node1', 'api', 0, 0, 0, 0, 'NPU', 1)"
        )
        conn.commit()
        conn.close()

        node = SimpleNamespace(id="node1", matched_distributed=[1, 2])
        skipped = SimpleNamespace(id="node2", matched_distributed=None)
        with DBActuator(self.db_path) as db:
            db.update_matched_distributed({0: {"node1": node, "node2": skipped}})

        conn = sqlite3.connect(self.db_path)
        value = conn.execute("SELECT matched_distributed FROM tb_nodes WHERE node_name='node1'").fetchone()[0]
        conn.close()
        self.assertEqual(json.loads(value), [1, 2])

    def test_update_matched_distributed_without_connection(self):
        db = object.__new__(DBActuator)
        db.db_path = self.db_path
        db.conn = None
        db.update_matched_distributed({})

    def test_update_matched_distributed_serialize_failure(self):
        self._create_nodes_table()
        node = SimpleNamespace(id="node1", matched_distributed=object())
        with DBActuator(self.db_path) as db, \
                patch("msprobe.visualization.db_utils.json.dumps", side_effect=TypeError("bad")):
            db.update_matched_distributed({0: {"node1": node}})

    def test_exit_closes_connection(self):
        db = object.__new__(DBActuator)
        db.db_path = self.db_path
        db.conn = MagicMock()
        self.assertFalse(db.__exit__(None, None, None))
        db.conn.close.assert_called_once()

    def test_init_connection_failure(self):
        db = object.__new__(DBActuator)
        db.db_path = self.db_path
        with patch("msprobe.visualization.db_utils.sqlite3.connect", side_effect=sqlite3.Error("boom")):
            with self.assertRaises(sqlite3.Error):
                db._init_connection()

if __name__ == '__main__':
    unittest.main()
