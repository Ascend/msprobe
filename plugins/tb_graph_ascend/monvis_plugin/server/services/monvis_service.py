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

from tensorboard.util import tb_logging
from ..repositories.monvis_repo import MonvisRepo
logger = tb_logging.get_logger()


class MonvisService():

    def __init__(self, db_path):
        self.db_path = db_path
        self.repo = MonvisRepo(db_path)
        self.is_db_connected = self.repo.is_db_connected

    def get_metrics_stat(self):
        metrics = []
        reslut = self.repo.query_metrics_stat()
        for row in reslut:
            metric_name = row['metric_name']
            stats = row['stats'].split(',') if row['stats'] else []
            metrics.append({
                "name": metric_name,
                "stats": stats
        })
        return {
            'success': True,
            'data': metrics
        }

    def get_values(self, metric, stat, dimension):
        if not metric or not stat:
            return {'success': False, 'error': 'metric and stat must not be empty'}

        valid_dimensions = {'step', 'rank', 'module_name'}
        if dimension not in valid_dimensions:
            return {'success': False, 'error': f'invalid dimension: {dimension}'}

        try:
            stats = self.repo.query_global_stats()
            values = {}

            if dimension == 'step':
                if 'min_step' not in stats or 'max_step' not in stats:
                    return {'success': False, 'error': 'Step info not found in global_stats'}
                values = {v: f'Step {v}' for v in range(stats['min_step'], stats['max_step'] + 1)}

            elif dimension == 'rank':
                if 'max_rank' not in stats:
                    return {'success': False, 'error': 'Rank info not found in global_stats'}
                values = {v: f'Rank {v}' for v in range(stats['max_rank'] + 1)}

            else:  # module_name                
                values = self.repo.query_module_names()

            return {'success': True, 'data': values}

        except Exception as e:
            return {'success': False, 'error': f'internal error: {str(e)}'}
        
    def get_heatmap_data(self, metric, stat, dimension, value):
        if not all([metric, stat, dimension in ['step', 'rank', 'module_name'], value]):
            return {
                'success': False,
                'error': 'Invalid parameters'
            }

        try:
            metric_id = self.repo.query_metric_id(metric)
            if not metric_id:
                return {'success': False, 'error': 'metric not found'}

            relevant_tables = self.repo.query_relevant_tables(metric_id)
            if not relevant_tables:
                return {'success': False, 'error': 'no relevant tables found'}
            
            selected_value = int(value)
            heatmap_data = []

            if dimension == "step":
                for table in relevant_tables:
                    rows = self.repo.query_heatmap_data(
                        table, stat, "t.step = ?", (selected_value,)
                    )
                    heatmap_data.extend(
                        [row['rank'], (row['target_id'],
                                        self.repo.get_module_name(row)), row[stat]]
                        for row in rows
                    )

            elif dimension == "rank":
                for table in relevant_tables:
                    rows = self.repo.query_heatmap_data(
                        table, stat, "t.rank = ?", (selected_value,)
                    )
                    heatmap_data.extend(
                        [row['step'], (row['target_id'],
                                        self.repo.get_module_name(row)), row[stat]]
                        for row in rows
                    )

            elif dimension == "module_name":
                for table in relevant_tables:
                    rows = self.repo.query_heatmap_data(
                        table, stat, "m.target_id = ?", (selected_value,)
                    )
                    heatmap_data.extend(
                        [row['step'], (row['rank'], row['rank']), row[stat]]
                        for row in rows
                    )

            return {
                'success': True,
                'data': heatmap_data
            }
        except Exception as e:
            return {'success': False, 'error': f'internal error: {str(e)}'}
        
    def get_trend_data(self, metric, stat, dimension, dim_x, dim_y):
        if not all([metric, stat, dimension in ['step', 'rank', 'module_name'], dim_x, dim_y]):
            return {
                'success': False,
                'error': 'Invalid parameters'
            }

        try:
            metric_id = self.repo.query_metric_id(metric)
            if not metric_id:
                return{
                    'success': False,
                    'error': 'metric not found'
                }

            relevant_tables = self.repo.query_relevant_tables(metric_id)
            if not relevant_tables:
                return {
                    'success': False,
                    'error': 'no relevant tables found'
                }

            dim_x = int(dim_x)
            dim_y = int(dim_y)
            trend_data = []
            if dimension == "step":
                for table in relevant_tables:
                    rows = self.repo.query_trend_data(
                        table, stat, "t.rank = ? AND t.target_id = ?", (
                            dim_x, dim_y)
                    )
                    trend_data.extend((row['step'], row[stat]) for row in rows)
                dimensions, values = zip(
                    *sorted(trend_data, key=lambda x: x[0]))

            elif dimension == "rank":
                for table in relevant_tables:
                    rows = self.repo.query_trend_data(
                        table, stat, "t.step = ? AND t.target_id = ?", (
                            dim_x, dim_y)
                    )
                    trend_data.extend((row['rank'], row[stat]) for row in rows)
                dimensions, values = zip(*sorted(trend_data, key=lambda x: x[0]))

            elif dimension == "module_name":
                for table in relevant_tables:
                    rows = self.repo.query_trend_data(
                        table, stat, "t.step = ? AND t.rank = ?", (
                            dim_x, dim_y)
                    )
                    trend_data.extend(
                            (row['target_id'], self.repo.get_module_name(row), row[stat])
                            for row in rows
                        )
                dimensions, values = list(zip(
                    *sorted(trend_data, key=lambda x: x[0])))[1:]
            return {
                'success': True,
                'data': {
                    'dimensions': dimensions,
                    'values': values
                }
            }
        except Exception as e:
            return {'success': False, 'error': f'internal error: {str(e)}'}

