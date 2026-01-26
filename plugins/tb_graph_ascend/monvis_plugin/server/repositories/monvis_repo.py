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


    def _get_module_name(self,row: Dict) -> str:
        """Generate module name from row data."""
        return "_".join((
            str(row["target_id"]),
            str(row["vpp_stage"]),
            row["target_name"],
            str(row["micro_step"])
        ))
    

    def _check_stat(self,stat):
        valid_stats = [
            "norm", "min", "max", "zeros", "nans", "mean",
            "entropy", "softmax_max", "sr", "kernel_norm", "std_x", "jacobian",
            "proxy", "token_similarity"
        ]
        if stat not in valid_stats:
            raise ValueError(f"Invalid stat: {stat}")
        
    def get_metric_id(self, metric_name: str) -> Optional[int]:
        """Get metric ID for a given metric name."""
        query = "SELECT metric_id FROM monitoring_metrics WHERE metric_name = ?"
        cursor = self.conn.execute(query, (metric_name,))
        row = cursor.fetchone()
        return row['metric_id'] if row else None    

    def query_metrics_stat(self) -> List[str]:
        """Get all available metrics from database."""
        query = """
            SELECT * FROM global_stats 
            ORDER BY ROWID DESC 
            LIMIT 1
        """
        with self.conn as c:
            cursor = c.execute(query)
            rows = cursor.fetchone()
        return rows

    def query_global_stats(self) -> Dict[str, int]:
        """Get global statistics from database."""
        query = """
        SELECT max_rank, min_step, max_step FROM global_stats 
        ORDER BY ROWID DESC 
        LIMIT 1
        """
        cursor = self.conn.execute(query)
        row = cursor.fetchone()
        if row:
            return {
                "max_rank": row['max_rank'],
                "min_step": row['min_step'],
                "max_step": row['max_step']
            }
        else:
            return {}
        
    def query_module_names_with_tags(self, tags: List[str] = None, metric_name: str = None) -> Dict[int, str]:
        """Get all module names with their target IDs, filtered by tags and metric."""
        # 检查是否存在tag_target_mapping表

        query = """
            SELECT DISTINCT mt.target_id, mt.vpp_stage, mt.target_name, mt.micro_step
            FROM monitoring_targets mt
        """

        conditions = []
        params = []
        # 添加metric过滤条件
        if metric_name:
            metric_id = self.get_metric_id(metric_name)
            if metric_id:
                # 使用trend_data表进行过滤
                conditions.append("""
                    mt.target_id IN (
                        SELECT DISTINCT target_id 
                        FROM trend_data 
                        WHERE metric_id = ?
                    )
                """)
                params.append(metric_id)

        # 添加tag过滤条件（使用tag-target映射）
        cursor = self.conn.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='tag_target_mapping'
        """)
        has_tag_mapping = cursor.fetchone() is not None
        if has_tag_mapping and tags:
            tag_conditions = []
            for tag in tags:
                tag_conditions.append("""
                    mt.target_id IN (
                        SELECT ttm.target_id 
                        FROM tag_target_mapping ttm
                        JOIN monitoring_tags mtags ON ttm.tag_id = mtags.tag_id
                        WHERE mtags.tag_name = ?
                    )
                """)
                params.extend([f"{tag}",])
            conditions.append(f"({' AND '.join(tag_conditions)})")

        # 构建完整查询
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        cursor = self.conn.execute(query, params)
        return {row["target_id"]: self._get_module_name(row) for row in cursor}
    
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
    
    def query_tags(self, metric_name: str = None) -> List[Dict[str, str]]:
        """Get all available tags from database, with categories if available."""
        # 检查是否存在monitoring_tags表
        cursor = self.conn.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='monitoring_tags'
        """)

        if cursor.fetchone():
            # 使用tags表
            query = "SELECT DISTINCT tag_name, category FROM monitoring_tags"
            params = ()

            if metric_name:
                metric_id = self.get_metric_id(metric_name)
                if metric_id:
                    # 检查是否存在tag_target_mapping表
                    cursor = self.conn.execute("""
                        SELECT name FROM sqlite_master 
                        WHERE type='table' AND name='tag_target_mapping'
                    """)

                    has_tag_mapping = cursor.fetchone() is not None

                    if has_tag_mapping:
                        # 使用tag-target映射表进行高效查询
                        query = """
                            SELECT DISTINCT mt.tag_name, mt.category 
                            FROM monitoring_tags mt
                            JOIN tag_target_mapping ttm ON mt.tag_id = ttm.tag_id
                            JOIN trend_data td ON ttm.target_id = td.target_id
                            WHERE td.metric_id = ?
                        """
                        params = (metric_id,)
                    else:
                        # 回退到原来的查询
                        query = """
                            SELECT DISTINCT mt.tag_name, mt.category 
                            FROM monitoring_tags mt
                            WHERE mt.metric_id = ?
                        """
                        params = (metric_id,)

            cursor = self.conn.execute(query, params)
            tags = [
                {
                    "id": row['tag_name'],
                    "text": row['tag_name'],
                    "category": row['category']
                }
                for row in cursor
            ]

            # 按类别排序
            tags.sort(key=lambda x: x['category'])
            return tags
        return []
    
    def query_heatmap_data_with_tags(self, stat: str, condition: str,
                                   params: tuple, metric_id: int, tags: List[str] = None) -> List[Dict]:
        """Get data for heatmap visualization with tag filtering."""
        # 检查是否存在tag_target_mapping表
        cursor = self.conn.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='tag_target_mapping'
        """)

        has_tag_mapping = cursor.fetchone() is not None

        self._check_stat(stat)

        base_query = f"""
            SELECT t.rank, t.step, t.target_id, t.{stat}, m.target_name, m.vpp_stage, m.micro_step
            FROM trend_data t
            JOIN monitoring_targets m ON t.target_id = m.target_id
            WHERE t.metric_id = ? AND {condition}
        """
        params = (metric_id,) + params

        if has_tag_mapping and tags:
            # 使用tag-target映射表进行高效查询
            tag_conditions = []
            tag_params = []
            for tag in tags:
                tag_conditions.append("""
                    t.target_id IN (
                        SELECT ttm.target_id 
                        FROM tag_target_mapping ttm
                        JOIN monitoring_tags mtags ON ttm.tag_id = mtags.tag_id
                        WHERE mtags.tag_name = ?
                    )
                """)
                tag_params.extend([f"{tag}",])

            base_query += f" AND ({' AND '.join(tag_conditions)})"
            params =params + tuple(tag_params)
                
        cursor = self.conn.execute(base_query, params)
        return [dict(row) for row in cursor]
    
    def query_trend_data_with_tags(self, stat: str, condition: str,
                                 params: tuple, metric_id: int, tags: List[str] = None) -> List[Dict]:
        """Get data for trend visualization with tag filtering."""
        self._check_stat(stat)

        base_query = f"""
            SELECT t.step, t.rank, t.target_id, t.{stat}, m.target_name, m.vpp_stage, m.micro_step
            FROM trend_data t
            JOIN monitoring_targets m ON t.target_id = m.target_id
            WHERE t.metric_id = ? AND {condition}
        """

        # 检查是否存在tag_target_mapping表
        cursor = self.conn.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='tag_target_mapping'
        """)
        has_tag_mapping = cursor.fetchone() is not None
        if has_tag_mapping and tags:
            # 添加tag过滤条件
            tag_conditions = []
            tag_params = []
            for tag in tags:
                tag_conditions.append("""
                    t.target_id IN (
                        SELECT ttm.target_id 
                        FROM tag_target_mapping ttm
                        JOIN monitoring_tags mtags ON ttm.tag_id = mtags.tag_id
                        WHERE mtags.tag_name = ?
                    )
                """)
                tag_params.extend([f"{tag}",])

            base_query += f" AND ({' AND '.join(tag_conditions)})"
            params = params + tuple(tag_params)
        params = (metric_id,) + params

        cursor = self.conn.execute(base_query, params)
        return [dict(row) for row in cursor]
    
