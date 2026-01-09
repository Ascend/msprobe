# Copyright (c) 2025, Huawei Technologies.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import re
from typing import Dict, List, Tuple, Optional
from ..database.db_connection import DBConnection


class MonvisRepo:

    def __init__(self, log_dir):
        self.db = DBConnection(log_dir)
        self.conn = self.db.conn
        self.is_db_connected = self.db.is_connected()

    @classmethod
    def get_module_name(cls, row: Dict) -> str:
        """Generate module name from row data."""
        return "_".join((
            str(row["target_id"]),
            str(row["vpp_stage"]),
            row["target_name"],
            str(row["micro_step"])
        ))

    def query_metrics_stat(self) -> List[str]:
        """Get all available metrics from database."""
        query = """
            SELECT m.metric_name, GROUP_CONCAT(ms.stat_name) as stats
            FROM monitoring_metrics m
            LEFT JOIN metric_stats ms ON m.metric_id = ms.metric_id
            GROUP BY m.metric_id
        """
        with self.conn as c:
            cursor = c.execute(query)
            rows = cursor.fetchall()
        return rows

    def query_global_stats(self) -> Dict[str, int]:
        """Get global statistics from database."""
        query = "SELECT stat_name, stat_value FROM global_stats"
        with self.conn as c: 
            cursor = c.execute(query)
            rows = cursor.fetchall()
        return {row['stat_name']: row['stat_value'] for row in rows}
    
    def query_module_names(self) -> Dict[int, str]:
        """Get all module names with their target IDs."""
        query = "SELECT target_id, vpp_stage, target_name, micro_step FROM monitoring_targets"
        with self.conn as c:
            cursor = c.execute(query)
            rows = cursor.fetchall()
        return {row["target_id"]: self.get_module_name(row) for row in rows}
    
    def query_metric_id(self, metric_name: str) -> Optional[int]:
        """Get metric ID for a given metric name."""
        query = "SELECT metric_id FROM monitoring_metrics WHERE metric_name = ?"
        with self.conn as c:
            cursor = c.execute(query, (metric_name,))
            row = cursor.fetchone()
        return row['metric_id'] if row else None

    def query_relevant_tables(self, metric_id: int) -> List[str]:
        """Get all tables relevant to a specific metric ID."""
        query = """
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name LIKE 'metric_%_step_%'
        """
        with self.conn as c:
            cursor = c.execute(query)
            tables = [table['name'] for table in cursor]
        relevant_tables = []
        for table in tables:
            match = re.match(r'metric_(\d+)_step_(\d+)_(\d+)', table)
            if match and int(match.group(1)) == metric_id:
                relevant_tables.append(table)
        return relevant_tables
    
    def query_heatmap_data(self, table: str, stat: str, condition: str, params: tuple) -> List[Dict]:
        """Get data for heatmap visualization."""
        query = f"""
            SELECT t.rank, t.step, t.target_id, t.{stat}, m.target_name, m.vpp_stage, m.micro_step
            FROM {table} t
            JOIN monitoring_targets m ON t.target_id = m.target_id
            WHERE {condition}
        """
        with self.conn as c:
            cursor = c.execute(query, params)
            rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def query_trend_data(self, table: str, stat: str, condition: str, params: tuple) -> List[Tuple]:
        """Get data for trend visualization."""
        query = f"""
            SELECT t.step, t.rank, t.target_id, t.{stat}, m.target_name, m.vpp_stage, m.micro_step
            FROM {table} t
            JOIN monitoring_targets m ON t.target_id = m.target_id
            WHERE {condition}
        """
        with self.conn as c:
            cursor = c.execute(query, params)
        return [dict(row) for row in cursor]
    
