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
import sqlite3
import os
import re
import tempfile
from typing import Dict, List
from unittest.mock import patch, MagicMock

from msprobe.core.common.log import logger
from msprobe.core.common.db_manager import DBManager, TrendSql


def normalize_spaces(text):
    return re.sub(r'\s+', ' ', text)


class TestDBManager(unittest.TestCase):
    def setUp(self):
        # 创建临时数据库文件
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db_path = self.temp_db.name
        self.db_manager = DBManager(self.db_path)

        # 创建测试表
        self.test_table = "test_table"
        self.create_test_table()

    def tearDown(self):
        # 关闭并删除临时数据库文件
        if hasattr(self, 'temp_db'):
            self.temp_db.close()
            os.unlink(self.db_path)

    def create_test_table(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.test_table} (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    value INTEGER,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def test_get_connection_success(self):
        """测试成功获取数据库连接"""
        conn, curs = self.db_manager._get_connection()
        self.assertIsInstance(conn, sqlite3.Connection)
        self.assertIsInstance(curs, sqlite3.Cursor)
        self.db_manager._release_connection(conn, curs)

    @patch.object(logger, 'error')
    def test_get_connection_success_failed(self, mock_logger):
        """测试错误日志记录"""
        with patch('sqlite3.connect', side_effect=sqlite3.Error("Test error")):
            with self.assertRaises(sqlite3.Error):
                self.db_manager._get_connection()
            mock_logger.assert_called_with(
                "Database connection failed: Test error")

    def test_insert_data_basic(self):
        """测试基本数据插入"""
        test_data = [
            (1, "item1", 100),
            (2, "item2", 200)
        ]
        columns = ["id", "name", "value"]

        inserted = self.db_manager.insert_data(
            table_name=self.test_table,
            data=test_data,
            key_list=columns
        )
        self.assertEqual(inserted, 2)

        # 验证数据是否实际插入
        results = self.db_manager.select_data(
            table_name=self.test_table,
            columns=["id", "name", "value"]
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["name"], "item1")

    def test_insert_data_without_keys(self):
        """测试无列名的数据插入"""
        test_data = [
            (3, "item3", 300, 333),
            (4, "item4", 400, 333)
        ]

        inserted = self.db_manager.insert_data(
            table_name=self.test_table,
            data=test_data
        )
        self.assertEqual(inserted, 2)

    def test_insert_data_empty(self):
        """测试空数据插入"""
        inserted = self.db_manager.insert_data(
            table_name=self.test_table,
            data=[]
        )
        self.assertEqual(inserted, 0)

    def test_insert_data_mismatch_keys(self):
        """测试列名与数据不匹配的情况"""
        test_data = [(5, "item5")]
        with self.assertRaises(ValueError):
            self.db_manager.insert_data(
                table_name=self.test_table,
                data=test_data,
                key_list=["id", "name", "value"]  # 多了一个列
            )

    def test_select_data_basic(self):
        """测试基本数据查询"""
        # 先插入测试数据
        self.db_manager.insert_data(
            table_name=self.test_table,
            data=[(10, "test10", 1000)],
            key_list=["id", "name", "value"]
        )

        results = self.db_manager.select_data(
            table_name=self.test_table,
            columns=["name", "value"],
            where={"id": 10}
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "test10")
        self.assertEqual(results[0]["value"], 1000)

    def test_select_data_no_where(self):
        """测试无条件查询"""
        # 插入多条数据
        test_data = [
            (20, "item20", 2000),
            (21, "item21", 2100)
        ]
        self.db_manager.insert_data(
            table_name=self.test_table,
            data=test_data,
            key_list=["id", "name", "value"]
        )

        results = self.db_manager.select_data(
            table_name=self.test_table,
            columns=["id", "name", "value"]
        )
        self.assertGreaterEqual(len(results), 2)

    def test_update_data_basic(self):
        """测试基本数据更新"""
        # 先插入测试数据
        self.db_manager.insert_data(
            table_name=self.test_table,
            data=[(30, "old_name", 3000)],
            key_list=["id", "name", "value"]
        )

        updated = self.db_manager.update_data(
            table_name=self.test_table,
            updates={"name": "new_name", "value": 3500},
            where={"id": 30}
        )
        self.assertEqual(updated, 1)

        # 验证更新结果
        results = self.db_manager.select_data(
            table_name=self.test_table,
            columns=["id", "name", "value"],
            where={"id": 30}
        )
        self.assertEqual(results[0]["name"], "new_name")
        self.assertEqual(results[0]["value"], 3500)

    def test_execute_sql_select(self):
        """测试执行SELECT SQL语句"""
        self.db_manager.insert_data(
            table_name=self.test_table,
            data=[(50, "sql_item", 5000)],
            key_list=["id", "name", "value"]
        )

        results = self.db_manager.execute_sql(
            sql=f"SELECT name, value FROM {self.test_table} WHERE id = ?",
            params=(50,)
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "sql_item")

    def test_execute_sql_non_select(self):
        """测试执行非SELECT SQL语句"""
        # 先插入数据
        self.db_manager.insert_data(
            table_name=self.test_table,
            data=[(60, "to_delete", 6000)],
            key_list=["id", "name", "value"]
        )

        # 执行DELETE语句
        self.db_manager.execute_sql(
            sql=f"DELETE FROM {self.test_table} WHERE id = 60"
        )

        # 验证数据已被删除
        results = self.db_manager.select_data(
            table_name=self.test_table,
            columns=["id", "name", "value"],
            where={"id": 60}
        )
        self.assertEqual(len(results), 0)

    def test_table_exists_true(self):
        """测试表存在检查(存在的情况)"""
        exists = self.db_manager.table_exists(self.test_table)
        self.assertTrue(exists)

    def test_table_exists_false(self):
        """测试表存在检查(不存在的情况)"""
        exists = self.db_manager.table_exists("non_existent_table")
        self.assertFalse(exists)

    def test_execute_multi_sql(self):
        """测试批量执行多个SQL语句"""
        sql_commands = [
            f"INSERT INTO {self.test_table} (id, name, value) VALUES (70, 'multi1', 7000)",
            f"INSERT INTO {self.test_table} (id, name, value) VALUES (71, 'multi2', 7100)",
            f"SELECT * FROM {self.test_table} WHERE id IN (70, 71)"
        ]

        results = self.db_manager.execute_multi_sql(sql_commands)

        # 应该只有最后一个SELECT语句有结果
        self.assertEqual(len(results), 1)
        self.assertEqual(len(results[0]), 2)


class TestDBOperationDecorator(unittest.TestCase):
    """测试 _db_operation 装饰器的错误处理能力"""

    def setUp(self):
        # 创建临时数据库文件
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db_path = self.temp_db.name
        self.db_manager = DBManager(self.db_path)

        # 创建测试表
        self.test_table = "test_table"
        self.create_test_table()

    def tearDown(self):
        # 关闭并删除临时数据库文件
        if hasattr(self, 'temp_db'):
            self.temp_db.close()
            if os.path.exists(self.db_path):
                os.unlink(self.db_path)

    def create_test_table(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.test_table} (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    value INTEGER
                )
            """)
            conn.commit()

    @patch.object(logger, 'error')
    def test_insert_data_table_not_exists(self, mock_logger):
        """测试 insert_data - 表不存在时装饰器捕获异常"""
        result = self.db_manager.insert_data(
            table_name="nonexistent_table_xyz",
            data=[(1, "test_user", 100)],
            key_list=["id", "name", "value"]
        )

        # 装饰器应捕获异常并返回 None
        self.assertIsNone(result)

        # 应记录错误日志
        mock_logger.assert_called_once()
        log_msg = mock_logger.call_args[0][0]
        self.assertIn("Database operation failed", log_msg)
        self.assertIn("operation=insert_data", log_msg)
        self.assertIn("nonexistent_table_xyz", log_msg)

    @patch.object(logger, 'error')
    def test_insert_data_sql_context_recorded(self, mock_logger):
        """测试 insert_data - SQL 上下文被正确记录"""
        result = self.db_manager.insert_data(
            table_name="nonexistent_table_abc",
            data=[(1, "Alice", 100), (2, "Bob", 200)],
            key_list=["id", "name", "value"]
        )

        self.assertIsNone(result)

        # 验证日志包含 SQL 语句和参数信息
        log_msg = mock_logger.call_args[0][0]
        self.assertIn("INSERT OR IGNORE INTO nonexistent_table_abc", log_msg)
        self.assertIn("batch_count=2", log_msg)
        self.assertIn("first_row_sample=(1", log_msg)

    @patch.object(logger, 'error')
    def test_select_data_table_not_exists(self, mock_logger):
        """测试 select_data - 表不存在时装饰器捕获异常"""
        result = self.db_manager.select_data(
            table_name="nonexistent_select_table",
            columns=["id", "name"]
        )

        self.assertIsNone(result)
        mock_logger.assert_called_once()

        log_msg = mock_logger.call_args[0][0]
        self.assertIn("operation=select_data", log_msg)
        self.assertIn("nonexistent_select_table", log_msg)

    @patch.object(logger, 'error')
    def test_select_data_sql_context_recorded(self, mock_logger):
        """测试 select_data - WHERE 条件被记录"""
        result = self.db_manager.select_data(
            table_name="nonexistent_table",
            columns=["id", "name"],
            where={"id": 1, "name": "test"}
        )

        self.assertIsNone(result)

        log_msg = mock_logger.call_args[0][0]
        self.assertIn("SELECT id, name FROM nonexistent_table", log_msg)
        # WHERE 参数应被记录
        self.assertIn("params=(1, 'test')",
                      log_msg) or self.assertIn("id", log_msg)

    @patch.object(logger, 'error')
    def test_update_data_table_not_exists(self, mock_logger):
        """测试 update_data - 表不存在时装饰器捕获异常"""
        result = self.db_manager.update_data(
            table_name="nonexistent_update_table",
            updates={"name": "new_name", "value": 200},
            where={"id": 1}
        )

        self.assertIsNone(result)
        mock_logger.assert_called_once()

        log_msg = mock_logger.call_args[0][0]
        self.assertIn("operation=update_data", log_msg)
        self.assertIn("UPDATE nonexistent_update_table", log_msg)

    @patch.object(logger, 'error')
    def test_execute_sql_invalid_syntax(self, mock_logger):
        """测试 execute_sql - 无效 SQL 语法"""
        result = self.db_manager.execute_sql("INVALID SQL SYNTAX HERE")

        self.assertIsNone(result)
        mock_logger.assert_called_once()

        log_msg = mock_logger.call_args[0][0]
        self.assertIn("operation=execute_sql", log_msg)
        self.assertIn("INVALID SQL SYNTAX", log_msg)

    @patch.object(logger, 'error')
    def test_execute_multi_sql_with_errors(self, mock_logger):
        """测试 execute_multi_sql - 批量执行包含错误的 SQL"""
        sql_commands = [
            "CREATE TABLE IF NOT EXISTS temp_table (id INTEGER)",
            "INSERT INTO nonexistent_batch_table (id) VALUES (1)",  # 错误
            "SELECT * FROM another_nonexistent",  # 错误
        ]

        result = self.db_manager.execute_multi_sql(sql_commands)

        # 装饰器应捕获异常并返回 None（不是空列表）
        self.assertIsNone(result)
        mock_logger.assert_called_once()

        log_msg = mock_logger.call_args[0][0]
        self.assertIn("operation=execute_multi_sql", log_msg)
        self.assertIn("command_index", log_msg)

    @patch.object(logger, 'error')
    def test_sql_injection_prevention_in_column_names(self, mock_logger):
        """测试 SQL 注入防护 - 恶意列名被拦截"""
        with self.assertRaises(ValueError) as cm:
            self.db_manager.select_data(
                table_name=self.test_table,
                columns=["id", "name; DROP TABLE test_table--"]
            )

        self.assertIn("Invalid SQL identifier", str(cm.exception))

    @patch.object(logger, 'error')
    def test_sql_injection_prevention_in_where_keys(self, mock_logger):
        """测试 SQL 注入防护 - WHERE 子句中的恶意键名被拦截"""
        with self.assertRaises(ValueError) as cm:
            self.db_manager.select_data(
                table_name=self.test_table,
                columns=["id", "name"],
                where={"id; DROP TABLE": 1}  # 恶意键名
            )

        self.assertIn("Invalid SQL identifier", str(cm.exception))

    def test_multiple_concurrent_failures(self):
        """测试连续多次失败操作，验证装饰器稳定性"""
        operations = [
            lambda: self.db_manager.select_data(
                table_name="nonexistent_1", columns=["id"]),
            lambda: self.db_manager.insert_data(
                table_name="nonexistent_2", data=[(1,)], key_list=["id"]),
            lambda: self.db_manager.update_data(
                table_name="nonexistent_3", updates={"x": 1}),
            lambda: self.db_manager.execute_sql("INVALID SQL 1"),
            lambda: self.db_manager.execute_sql("INVALID SQL 2"),
        ]

        with patch.object(logger, 'error') as mock_logger:
            for i, op in enumerate(operations):
                with self.subTest(operation_index=i):
                    result = op()
                    self.assertIsNone(result, f"操作 {i} 应返回 None")

            # 应记录 5 次错误
            self.assertEqual(mock_logger.call_count, 5)


class TestTrendSql(unittest.TestCase):
    """测试TrendSql类"""

    def test_get_table_definition_all_tables(self):
        """测试获取所有表定义"""
        result = TrendSql.get_table_definition()
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 4)
        self.assertTrue(all("CREATE TABLE" in sql for sql in result))

    def test_get_table_definition_single_table(self):
        """测试获取单个表定义"""
        tables = [
            "monitoring_targets",
            "monitoring_metrics",
            "monitoring_tags",
            "tag_target_mapping"
        ]

        for table_name in tables:
            with self.subTest(table=table_name):
                result = TrendSql.get_table_definition(table_name)
                self.assertIn(f"CREATE TABLE", result)
                self.assertIn(table_name, result)

    def test_get_table_definition_invalid_table(self):
        """测试获取不存在的表定义"""
        with self.assertRaises(ValueError):
            TrendSql.get_table_definition("invalid_table")

    def test_create_monitoring_targets_table(self):
        """测试监测目标表创建SQL"""
        result = TrendSql.create_monitoring_targets_table()
        self.assertIn("CREATE TABLE IF NOT EXISTS monitoring_targets", result)
        self.assertIn("target_id INTEGER PRIMARY KEY AUTOINCREMENT", result)
        self.assertIn("UNIQUE(target_name, vpp_stage, micro_step)", result)

    def test_get_metric_mapping_sql(self):
        """测试获取指标映射SQL"""
        result = TrendSql.get_metric_mapping_sql()
        result = normalize_spaces(result)
        self.assertIn("SELECT metric_id, metric_name", result)
