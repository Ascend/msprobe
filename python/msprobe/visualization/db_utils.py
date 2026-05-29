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

import os
import sqlite3
import json
import re
import time
from typing import Dict
from msprobe.core.common.log import logger
from msprobe.core.common.file_utils import check_path_before_create, FileChecker
from msprobe.core.common.const import FileCheckConst, Const
from msprobe.visualization.utils import GraphConst, update_shared_dict, update_pbar_info, post_process_db_pbar
from msprobe.visualization.builder.msprobe_adapter import format_node_data

TEXT_PRIMARY_KEY = 'TEXT PRIMARY KEY'
TEXT_NOT_NULL = 'TEXT NOT NULL'
INTEGER_NOT_NULL = 'INTEGER NOT NULL'
TEXT = 'TEXT'
INTEGER = 'INTEGER'
DB_INSERT_SIZE = 1000

node_columns = {
    'id': TEXT_PRIMARY_KEY,
    'graph_id': TEXT_NOT_NULL,
    'node_order': INTEGER_NOT_NULL,
    'node_name': TEXT_NOT_NULL,
    'node_type': TEXT_NOT_NULL,
    'up_node': TEXT,
    'sub_nodes': TEXT,
    'precision_index': INTEGER,
    'overflow_level': TEXT,
    'micro_step_id': INTEGER_NOT_NULL,
    'matched_node_link': TEXT,
    'stack_id': TEXT,
    'parallel_merge_info': TEXT,
    'matched_distributed': TEXT,
    'modified': INTEGER_NOT_NULL,
    'input_data': TEXT,
    'output_data': TEXT,
    'data_source': TEXT,
    'dump_data_dir': TEXT,
    'step': INTEGER_NOT_NULL,
    'rank': INTEGER_NOT_NULL,
    'is_distributed': INTEGER,
}

config_columns = {
    'id': TEXT_PRIMARY_KEY,
    'graph_type': TEXT_NOT_NULL,
    'task': TEXT,
    'tool_tip': TEXT,
    'micro_steps': INTEGER,
    'overflow_check': INTEGER,
    'node_colors': TEXT_NOT_NULL,
    'rank_list': TEXT_NOT_NULL,
    'step_list': TEXT_NOT_NULL,
}

stack_columns = {'id': TEXT_PRIMARY_KEY, 'stack_info': TEXT}

indexes = {
    "index1": ["step", "rank", "data_source", "up_node", "node_order"],
    "index2": ["step", "rank", "data_source", "node_name"],
    "index3": ["step", "rank", "data_source", "node_order"],
    "index4": ["step", "rank", "node_order"],
    "index5": ["step", "rank", "micro_step_id", "node_order"],
    "index6": ["step", "rank", "modified", "matched_node_link"],
    "index7": ["is_distributed"],
}

SAFE_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9_]+$')


def is_safe_identifier(name):
    """验证标识符是否安全（防止SQL注入）"""
    return isinstance(name, str) and SAFE_NAME_PATTERN.match(name) is not None


def create_table_sql_from_dict(table_name, columns_dict):
    """
    根据提供的表名和列定义字典生成CREATE TABLE SQL语句。
    """
    if not is_safe_identifier(table_name):
        raise ValueError(f"Invalid table name: {table_name} - potential SQL injection risk!")

    sql = f"CREATE TABLE IF NOT EXISTS {table_name} (\n"

    column_definitions = []
    for column_name, column_type in columns_dict.items():
        if not is_safe_identifier(column_name):
            raise ValueError(f"Invalid column name: {column_name} - potential SQL injection risk!")

        column_definitions.append(f"    {column_name} {column_type}")

    sql += ",\n".join(column_definitions)
    sql += "\n);"

    return sql


def create_insert_sql_from_dict(table_name, columns_dict, ignore_insert=False):
    """
    根据提供的表名和数据字典生成INSERT INTO SQL语句。
    """
    if not is_safe_identifier(table_name):
        raise ValueError(f"Invalid table name: {table_name} - potential SQL injection risk!")

    columns = list(columns_dict.keys())

    for column_name in columns:
        if not is_safe_identifier(column_name):
            raise ValueError(f"Invalid column name: {column_name} - potential SQL injection risk!")

    placeholders = ["?"] * len(columns)

    columns_string = ", ".join(columns)
    placeholders_string = ", ".join(placeholders)

    sql_prefix = "INSERT OR IGNORE INTO" if ignore_insert else "INSERT INTO"
    sql = f"{sql_prefix} {table_name} ({columns_string}) VALUES ({placeholders_string})"
    return sql


def to_db(db_path, create_table_sql, insert_sql, data, pbar_info=None):
    max_retries = 10
    initial_delay = 0.1
    if not os.path.exists(db_path):
        check_path_before_create(db_path)
    else:
        FileChecker(
            db_path, FileCheckConst.FILE, FileCheckConst.READ_WRITE_ABLE, FileCheckConst.DB_SUFFIX
        ).common_check()

    retry_count = 0
    current_delay = initial_delay
    total_data = len(data)
    commited_data = 0

    if pbar_info:
        update_shared_dict(pbar_info.current_stage_dict, pbar_info.task_id, 1)

    while retry_count <= max_retries:
        conn = None
        try:
            conn = sqlite3.connect(db_path, timeout=30)
            cursor = conn.cursor()
            # 启用WAL模式提升多进程读写并发能力
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute(create_table_sql)
            for i in range(0, total_data, DB_INSERT_SIZE):
                batch = data[i : i + DB_INSERT_SIZE]
                cursor.executemany(insert_sql, batch)

                if pbar_info:
                    commited_data += len(batch)
                    update_pbar_info(pbar_info, commited_data, total_data)
            conn.commit()
            return
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e).lower():
                retry_count += 1
                if retry_count > max_retries:
                    logger.error(f"Database lock conflict retry attempts exhausted ({max_retries}): {e}")
                    raise RuntimeError(f"DB lock retry exhausted: {e}") from e

                logger.warning(f"DB lock conflict (retry {retry_count}/{max_retries}), wait {current_delay:.2f}s : {e}")
                time.sleep(current_delay)
                current_delay *= 2
                continue

            logger.error(f"An sqlite3 error occurred: {e}")
            raise e
        except sqlite3.Error as e:
            logger.error(f"An sqlite3 error occurred: {e}")
            raise e
        except Exception as e:
            logger.error(f"An unknown error occurred: {e}")
            raise e
        finally:
            if conn:
                conn.close()


def add_table_index(db_path):
    FileChecker(db_path, FileCheckConst.FILE, FileCheckConst.READ_WRITE_ABLE, FileCheckConst.DB_SUFFIX).common_check()
    try:
        conn = sqlite3.connect(db_path)
    except sqlite3.Error as e:
        logger.error(f"Unable to create database connection: {e}")
        raise RuntimeError("Unable to create database connection") from e

    try:
        cursor = conn.cursor()
        for index_name, columns in indexes.items():
            if not is_safe_identifier(index_name):
                raise ValueError(f"Invalid index name: {index_name} - potential SQL injection risk!")

            for column in columns:
                if not is_safe_identifier(column):
                    raise ValueError(f"Invalid column name in index: {column} - potential SQL injection risk!")

            columns_str = ', '.join(columns)
            index_sql = f'''
                CREATE INDEX IF NOT EXISTS {index_name} ON tb_nodes ({columns_str});
            '''
            cursor.execute(index_sql)
        conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Failed to add table index: {e}")
        raise RuntimeError("Failed to add table index") from e
    finally:
        conn.close()


def post_process_db(db_path, pbar_info=None, is_parallel_merge=False):
    logger.info('Start adding index to db file, please wait...')
    add_table_index(db_path)
    if pbar_info:
        post_process_db_pbar(pbar_info, is_parallel_merge)
    logger.info('Adding index to db file completed.')


def node_to_db(graph, db_name, pbar_info=None):
    create_table_sql = create_table_sql_from_dict('tb_nodes', node_columns)
    insert_sql = create_insert_sql_from_dict('tb_nodes', node_columns)
    data = []
    stack_dict = {}
    for i, node in enumerate(graph.get_sorted_nodes()):
        stack_info_text = json.dumps(node.stack_info)
        if stack_info_text not in stack_dict:
            stack_dict[stack_info_text] = get_stack_unique_id(graph, stack_dict)
        data.append(
            (
                get_node_unique_id(graph, node),
                get_graph_unique_id(graph),
                i,
                node.id,
                node.op.value,
                node.upnode.id if node.upnode else '',
                json.dumps([node.id for node in node.subnodes]) if node.subnodes else '',
                node.data.get(GraphConst.JSON_INDEX_KEY),
                node.data.get(GraphConst.OVERFLOW_LEVEL),
                node.micro_step_id if node.micro_step_id is not None else 0,
                json.dumps(node.matched_node_link),
                stack_dict.get(stack_info_text),
                json.dumps(node.parallel_merge_info) if node.parallel_merge_info else '',
                json.dumps(node.matched_distributed),
                0,
                json.dumps(format_node_data(node.input_data, node.id, graph.compare_mode)),
                json.dumps(format_node_data(node.output_data, node.id, graph.compare_mode)),
                graph.data_source,
                graph.data_path,
                graph.step,
                graph.rank,
                1 if node.id.startswith(Const.DISTRIBUTED) else 0,
            )
        )
    to_db(db_name, create_table_sql, insert_sql, data, pbar_info=pbar_info)
    stack_to_db(stack_dict, db_name)


def config_to_db(config, db_name):
    create_table_sql = create_table_sql_from_dict('tb_config', config_columns)
    insert_sql = create_insert_sql_from_dict('tb_config', config_columns, ignore_insert=True)
    data = [
        (
            "1",
            "compare" if config.graph_b else "build",
            config.task,
            config.tool_tip,
            config.micro_steps,
            config.overflow_check,
            json.dumps(config.node_colors),
            json.dumps(config.rank_list),
            json.dumps(config.step_list),
        )
    ]
    to_db(db_name, create_table_sql, insert_sql, data)


def stack_to_db(stack_dict, db_name):
    create_table_sql = create_table_sql_from_dict('tb_stack', stack_columns)
    insert_sql = create_insert_sql_from_dict('tb_stack', stack_columns)
    data = []
    for stack_info_text, unique_id in stack_dict.items():
        data.append((unique_id, stack_info_text))
    to_db(db_name, create_table_sql, insert_sql, data)


def get_graph_unique_id(graph):
    return f'{graph.data_source}_{graph.step}_{graph.rank}'


def get_node_unique_id(graph, node):
    return f'{get_graph_unique_id(graph)}_{node.id}'


def get_stack_unique_id(graph, stack_dict):
    return f'{get_graph_unique_id(graph)}_{len(stack_dict)}'


class DBActuator:
    def __init__(self, db_path):
        self.db_path = db_path
        self._init_connection()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            try:
                self.conn.close()
                logger.info(f"Database connection to {self.db_path} closed successfully")
            except sqlite3.Error as e:
                logger.error(f"Failed to close database connection: {e}")
        return False

    def query_distributed_nodes_info(self):
        query = """
 	         SELECT
 	             node_name,
 	             input_data,
 	             rank,
 	             step,
 	             data_source,
 	             precision_index,
 	             overflow_level
 	         FROM
 	             tb_nodes
 	         WHERE
 	             is_distributed = 1
 	         """
        if not self.conn:
            logger.warning("Database connection is not initialized.")
            return []
        try:
            with self.conn as conn:
                cursor = conn.execute(query)
                rows = cursor.fetchall()
                cursor.close()

            return rows
        except Exception as e:
            logger.error(f'Failed to query distributed nodes: {e}')
            return []

    def update_matched_distributed(
        self, distributed_info: Dict[int, dict], step=0, data_source=GraphConst.JSON_NPU_KEY, batch_size=1000
    ):
        update_data_list = []
        update_sql = """
     	         UPDATE tb_nodes
     	         SET matched_distributed = ?
     	         WHERE step = ? AND rank = ? AND data_source = ? AND node_name = ?;

     	         """
        if not self.conn:
            logger.warning("Database connection is not initialized.")
            return
        for rank, nodes_dict in distributed_info.items():
            for node in nodes_dict.values():
                if not node.matched_distributed:
                    continue
                try:
                    matched_distributed = json.dumps(node.matched_distributed)
                except Exception as e:
                    logger.warning(f"Failed to serialize matched_distributed for node {node.id}: {e}")
                    continue
                update_data_list.append((matched_distributed, step, rank, data_source, node.id))

        try:
            with self.conn as conn:
                cursor = conn.cursor()
                for i in range(0, len(update_data_list), batch_size):
                    batch_data = update_data_list[i : i + batch_size]
                    cursor.executemany(update_sql, batch_data)
                cursor.close()

        except Exception as e:
            logger.error(f"Failed to update matched_distributed: {e}")
            raise

    def _init_connection(self):
        FileChecker(
            self.db_path, FileCheckConst.FILE, FileCheckConst.READ_WRITE_ABLE, FileCheckConst.DB_SUFFIX
        ).common_check()
        try:
            self.conn = sqlite3.connect(self.db_path)
        except sqlite3.Error as e:
            logger.error(f"Failed to connect to database: {e}")
            raise e
