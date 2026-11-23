# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
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

import re
import sqlite3
from typing import List, Tuple, Dict, Any
from functools import wraps

from msprobe.core.common.log import logger
from msprobe.core.common.file_utils import check_path_before_create

SAFE_SQL_PATTERN = re.compile(r'^[a-zA-Z0-9_]+$')


def check_identifier_safety(name):
    """验证标识符是否安全（防止SQL注入）"""
    if not isinstance(name, str) or SAFE_SQL_PATTERN.match(name) is None:
        raise ValueError(f"Invalid SQL identifier: {name}, potential SQL injection risk!")


def _db_operation(operation_name: str = ""):
    """数据库操作装饰器，自动管理连接，并记录 SQL 上下文"""

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            conn, curs = None, None
            # 清除上一次的 SQL 记录
            if hasattr(self, '_last_sql'):
                delattr(self, '_last_sql')
            if hasattr(self, '_last_params'):
                delattr(self, '_last_params')

            try:
                conn, curs = self._get_connection()
                result = func(self, conn, curs, *args, **kwargs)
                return result
            except sqlite3.Error as err:
                op = operation_name or func.__name__
                # 获取 SQL 上下文（如果存在）
                sql_info = getattr(self, '_last_sql', 'N/A')
                params_info = getattr(self, '_last_params', 'N/A')

                logger.error(
                    f"Database operation failed | operation={op} | "
                    f"db_path={getattr(self, 'db_path', 'unknown')} | "
                    f"sql={sql_info} | params={params_info} | error={err}"
                )
                if conn:
                    conn.rollback()
                return None
            finally:
                self._release_connection(conn, curs)

        return wrapper

    return decorator


class DBManager:
    """
    数据库管理类，封装常用数据库操作
    """

    DEFAULT_FETCH_SIZE = 10000
    DEFAULT_INSERT_SIZE = 10000
    MAX_ROW_COUNT = 100000000

    def __init__(self, db_path: str):
        """
        初始化DBManager
        :param db_path: 数据库文件路径
        """
        self.db_path = db_path
        self._last_sql = ""
        self._last_params = ""

    @staticmethod
    def _get_where_sql(where_list):
        where_sql = ""

        if not where_list:
            return where_sql, tuple()

        where_clauses = []
        where_values = []
        if where_list:
            for col, val in where_list.items():
                check_identifier_safety(col)
                where_clauses.append(f"{col} = ?")
                where_values.append(val)
            if where_clauses:
                where_sql = " WHERE " + " AND ".join(where_clauses)
        return where_sql, tuple(where_values)

    @_db_operation("insert_data")
    def insert_data(
        self,
        conn: sqlite3.Connection,
        curs: sqlite3.Cursor,
        table_name: str,
        data: List[Tuple],
        key_list: List[str] = None,
    ) -> int:
        """
        批量插入数据
        :param table_name: 表名
        :param data: 要插入的数据列表
        :param batch_size: 每批插入的大小
        :return: 插入的行数
        """
        check_identifier_safety(table_name)

        if not data:
            return 0
        columns = len(data[0])
        if key_list:
            if not isinstance(key_list, list):
                raise TypeError(f"key_list must be a list, got {type(key_list)}")
            if columns != len(key_list):
                raise ValueError(
                    f"When inserting into table {table_name}, the length of key list ({key_list})"
                    f"does not match the data({columns})."
                )
            for key in key_list:
                check_identifier_safety(key)

        batch_size = self.DEFAULT_INSERT_SIZE
        placeholders = ", ".join(["?"] * columns)
        if key_list:
            keys = ", ".join(key_list)
            sql = f"INSERT OR IGNORE INTO {table_name} ({keys}) VALUES ({placeholders})"
        else:
            sql = f"INSERT OR IGNORE INTO {table_name} VALUES ({placeholders})"
        self._last_sql = sql
        self._last_params = f"batch_count={len(data)}, first_row_sample={data[0] if data else None}"
        inserted_rows = 0
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            curs.executemany(sql, batch)
            inserted_rows += curs.rowcount

        conn.commit()
        return inserted_rows

    @_db_operation("select_data")
    def select_data(
        self,
        conn: sqlite3.Connection,
        curs: sqlite3.Cursor,
        table_name: str,
        columns: List[str] = None,
        where: dict = None,
    ) -> List[Dict]:
        """
        查询数据
        :param table_name: 表名
        :param columns: 要查询的列
        :param where: WHERE条件
        :return: 查询结果列表(字典形式)
        """
        check_identifier_safety(table_name)

        if not columns:
            raise ValueError("columns parameter cannot be empty, specify columns to select (e.g. ['id', 'name'])")
        if not isinstance(columns, list) or not all(isinstance(col, str) for col in columns):
            raise TypeError("columns must be a list of strings (e.g. ['id', 'name'])")

        for col in columns:
            check_identifier_safety(col)

        # 新增：校验 table_name
        check_identifier_safety(table_name)

        cols = ", ".join(columns)

        # 修改：增加 Bandit 抑制说明
        sql = f"SELECT {cols} FROM {table_name}"  # nosec B608

        where_sql, where_params = self._get_where_sql(where)
        # 设置 SQL 上下文
        self._last_sql = sql
        self._last_params = where_params
        curs.execute(sql + where_sql, where_params)

        return [dict(row) for row in curs.fetchall()]

    @_db_operation("update_data")
    def update_data(
        self,
        conn: sqlite3.Connection,
        curs: sqlite3.Cursor,
        table_name: str,
        updates: Dict[str, Any],
        where: dict = None,
    ) -> int:
        """
        更新数据
        :param table_name: 表名
        :param updates: 要更新的字段和值
        :param where: WHERE条件
        :param where_params: WHERE条件参数
        :return: 影响的行数
        """
        check_identifier_safety(table_name)
        if not updates:
            raise ValueError("columns parameter cannot be empty, specify it to update (e.g. {'name': 'xxx'}")
        if not isinstance(updates, dict):
            raise TypeError(f"updates must be a dictionary, got: {type(updates)}")
        for key in updates.keys():
            check_identifier_safety(key)

        set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
        # table_name 和 column 已经过安全校验
        sql = f"UPDATE {table_name} SET {set_clause}"  # nosec B608

        params = tuple(updates.values())

        where_sql, where_params = self._get_where_sql(where)
        self._last_sql = sql
        self._last_params = where_params
        curs.execute(sql + where_sql, params + where_params)
        conn.commit()
        return curs.rowcount

    @_db_operation("execute_sql")
    def execute_sql(self, conn: sqlite3.Connection, curs: sqlite3.Cursor, sql: str, params: Tuple = None) -> List[Dict]:
        """
        执行自定义SQL查询
        :param sql: SQL语句
        :param params: 参数
        :return: 查询结果
        """
        self._last_sql = sql
        curs.execute(sql, params or ())
        if sql.strip().upper().startswith("SELECT"):
            return [dict(row) for row in curs.fetchall()]
        conn.commit()
        return []

    def table_exists(self, table_name: str) -> bool:
        """
        :param table_name: 表名
        :return: 查询结果
        """
        result = self.select_data(  # pylint: disable=no-value-for-parameter
            table_name="sqlite_master", columns=["name"], where={"type": "table", "name": table_name}
        )
        return len(result) > 0

    @_db_operation("execute_multi_sql")
    def execute_multi_sql(
        self,
        conn: sqlite3.Connection,
        curs: sqlite3.Cursor,
        sql_commands: List[str],
    ) -> List[List[Dict]]:
        """
        批量执行多个SQL语句
        :param sql_commands: [sql1, sql2, ...]
        :return: 每个SELECT语句的结果列表
        """
        results = []
        for i, sql in enumerate(sql_commands):
            self._last_sql = sql
            self._last_params = f"command_index={i}"
            curs.execute(sql)
            if sql.strip().upper().startswith("SELECT"):
                results.append([dict(row) for row in curs.fetchall()])
        conn.commit()
        return results

    def _get_connection(self) -> Tuple[sqlite3.Connection, sqlite3.Cursor]:
        """获取数据库连接和游标"""
        check_path_before_create(self.db_path)
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # 使用Row工厂获取字典形式的结果
            curs = conn.cursor()
            return conn, curs
        except sqlite3.Error as err:
            logger.error(f"Database connection failed: {err}")
            raise

    def _release_connection(self, conn: sqlite3.Connection, curs: sqlite3.Cursor) -> None:
        """释放数据库连接"""
        try:
            if curs is not None:
                curs.close()
            if conn is not None:
                conn.close()
        except sqlite3.Error as err:
            logger.error(f"Failed to release database connection: {err}")


class TrendSql:
    """趋势可视化数据库表参数类"""

    @staticmethod
    def create_monitoring_targets_table():
        """监测目标表"""
        return """
        CREATE TABLE IF NOT EXISTS monitoring_targets (
            target_id INTEGER PRIMARY KEY AUTOINCREMENT,
            target_name TEXT NOT NULL,
            vpp_stage INTEGER NOT NULL,
            micro_step INTEGER NOT NULL DEFAULT 0,
            UNIQUE(target_name, vpp_stage, micro_step)
        )"""

    @staticmethod
    def create_monitoring_metrics_table():
        """监测指标表"""
        return """
        CREATE TABLE IF NOT EXISTS monitoring_metrics (
            metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
            metric_name TEXT UNIQUE NOT NULL
        )"""

    @staticmethod
    def get_metric_mapping_sql():
        """从monitoring_metrics表获取所有metric名称和ID"""
        return """
        SELECT metric_id, metric_name
        FROM monitoring_metrics
        """

    @staticmethod
    def get_global_stats_sql():
        """从global_stats表获取最新的统计配置"""
        return """
        SELECT * FROM global_stats
        ORDER BY ROWID DESC
        LIMIT 1
        """

    @staticmethod
    def create_global_stats_table(columns_config):
        """根据字典配置创建全局统计表"""
        # 根据字典的键值类型动态创建列
        column_definitions = []
        for column_name, column_value in columns_config.items():
            if isinstance(column_value, (list, set)):
                column_definitions.append(f"{column_name} TEXT DEFAULT NULL")
            elif isinstance(column_value, (int, float)):
                column_definitions.append(f"{column_name} INTEGER DEFAULT 0")

        create_sql = f"""
        CREATE TABLE IF NOT EXISTS global_stats (
            {', '.join(column_definitions)}
        )"""
        return create_sql

    @staticmethod
    def create_tags_table():
        # 新增：标签表
        return """
        CREATE TABLE IF NOT EXISTS monitoring_tags (
            tag_id INTEGER PRIMARY KEY AUTOINCREMENT,
            tag_name TEXT NOT NULL,
            category TEXT NOT NULL,
            metric_id INTEGER NOT NULL,
            UNIQUE(tag_name, category, metric_id),
            FOREIGN KEY (metric_id) REFERENCES monitoring_metrics(metric_id)
        )"""

    @staticmethod
    def create_tag_mapping_table():
        return """
        CREATE TABLE IF NOT EXISTS tag_target_mapping (
            tag_id INTEGER NOT NULL,
            target_id INTEGER NOT NULL,
            PRIMARY KEY (tag_id, target_id),
            FOREIGN KEY (tag_id) REFERENCES monitoring_tags(tag_id),
            FOREIGN KEY (target_id) REFERENCES monitoring_target(target_id)
        )"""

    @staticmethod
    def create_trend_table(stats):
        stat_columns = [f"{stat} REAL DEFAULT NULL" for stat in stats]
        create_sql = f"""
        CREATE TABLE IF NOT EXISTS trend_data (
            rank INTEGER NOT NULL,
            step INTEGER NOT NULL,
            target_id INTEGER NOT NULL,
            metric_id INTEGER NOT NULL,
            {', '.join(stat_columns)},
            PRIMARY KEY (rank, step, target_id, metric_id),
            FOREIGN KEY (target_id) REFERENCES monitoring_targets(target_id),
            FOREIGN KEY (metric_id) REFERENCES monitoring_metrics(metric_id)
        ) WITHOUT ROWID"""
        return create_sql

    @classmethod
    def get_table_definition(cls, table_name=""):
        """
        获取表定义SQL
        :param table_name: 表名
        :return: 建表SQL语句
        :raises ValueError: 当表名不存在时
        """
        table_creators = {
            "monitoring_targets": cls.create_monitoring_targets_table,
            "monitoring_metrics": cls.create_monitoring_metrics_table,
            "monitoring_tags": cls.create_tags_table,
            "tag_target_mapping": cls.create_tag_mapping_table,
        }
        if not table_name:
            return [table_creators.get(table, lambda x: "")() for table in table_creators]
        if table_name not in table_creators:
            raise ValueError(f"Unsupported table name: {table_name}")
        return table_creators[table_name]()
