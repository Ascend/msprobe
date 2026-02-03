# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
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
        metrics_data = []
        result = self.repo.query_metrics_stat()
        if  result:    
            # 解析global_stats表中的metric统计信息
            for column_name in result.keys():
                if column_name not in ['max_rank', 'min_step', 'max_step']:
                    # 这些列包含metric的统计信息，格式如：norm,max,min,mean
                    metric_name = column_name
                    stats_value = result[column_name]
                    if stats_value:
                        # 分割统计项
                        stats_list = stats_value.split(',')
                        metrics_data.append({
                            "name": metric_name,
                            "stats": stats_list
                        })
        return {
            'success': True,
            'data': metrics_data
        }

    def get_values(self, metric, stat, dimension,tags):
        if not metric or not stat:
            return {'success': False, 'error': 'metric and stat must not be empty'}

        if not all([metric, stat, dimension in ['step', 'rank', 'module_name']]):
            return {'success': False, 'error': f'invalid dimension: {dimension}'}

        try:
            if dimension == "step":
                stats = self.repo.query_global_stats()
                if "min_step" not in stats or "max_step" not in stats:
                    return {'success': False, 'error': 'Step info not found in global_stats'}
                steps = range(stats["min_step"], stats["max_step"] + 1)
                values = {v: f"Step{v}" for v in steps}

            elif dimension == "rank":
                stats = self.repo.query_global_stats()
                if "max_rank" not in stats:
                    return {'success': False, 'error': 'Rank info not found in global_stats'}
                ranks = range(stats["max_rank"] + 1)
                values = {v: f"Rank{v}" for v in ranks}

            elif dimension == "module_name":
                values = self.repo.query_module_names_with_tags(
                    tags,
                    metric
                )
            else:
                values = {}

            return {'success': True, 'data': values}

        except Exception as e:
            return {'success': False, 'error': f'internal error: {str(e)}'}
    
    def get_tags(self, metric):
        if not metric:
            return {'success': False, 'error': 'metric must not be empty'}

        try:
            tags = self.repo.query_tags(metric)
            return {'success': True, 'data': tags}

        except Exception as e:
            return {'success': False, 'error': f'internal error: {str(e)}'}
        
    def get_heatmap_data(self, metric, stat, dimension, value,tags):

        if not all([metric, stat, dimension in ['step', 'rank', 'module_name'], value]):
            return {'success': False, 'error': 'Invalid parameters'}

        try:
            metric_id = self.repo.get_metric_id(metric)
            if not metric_id:
                return {
                    'success': False,
                    'error': f'No metric found with name: {metric}'
                }

            selected_value = int(value)
            heatmap_data = []

            if dimension == "step":
                rows = self.repo.query_heatmap_data_with_tags(
                    stat, "t.step = ?", (selected_value,), metric_id, tags
                )
                heatmap_data.extend(
                    [row.get('rank'), (row.get('target_id'),
                                   self.repo._get_module_name(row)), row[stat]]
                    for row in rows
                )

            elif dimension == "rank":
                rows = self.repo.query_heatmap_data_with_tags(
                    stat, "t.rank = ?", (selected_value,), metric_id, tags
                )
                heatmap_data.extend(
                    [row.get('step'), (row.get('target_id'),
                                   self.repo._get_module_name(row)), row[stat]]
                    for row in rows
                )

            elif dimension == "module_name":
                rows = self.repo.query_heatmap_data_with_tags(
                    stat, "m.target_id = ?", (selected_value,
                                              ), metric_id, tags
                )
                heatmap_data.extend(
                    [row.get('step'), (row.get('rank'), row.get('rank')), row.get(stat)]
                    for row in rows
                )

            return {'success': True, 'data': heatmap_data}

        except Exception as e:
            return {'success': False, 'error': f'internal error: {str(e)}'}
        
    def get_trend_data(self, metric, stat, dimension, dim_x, dim_y,tags):
        if not all([metric, stat, dimension in ['step', 'rank', 'module_name']]):
            return {
                'success': False,
                'error': 'Invalid parameters'
            }
            
        try:
            metric_id = self.repo.get_metric_id(metric)
            if not metric_id:
                return {
                    'success': False,
                    'error': f'No metric found with name: {metric}'
                }
            dim_x = int(dim_x)
            dim_y = int(dim_y)
            trend_data = []
            dimensions, values = [], []

            if dimension == "step":
                rows = self.repo.query_trend_data_with_tags(
                    stat, "t.rank = ? AND t.target_id = ?", (
                        dim_x, dim_y), metric_id
                )
                trend_data.extend(
                    (int(row.get('step')), row[stat]) for row in rows)
                if trend_data:
                    dimensions, values = zip(
                        *sorted(trend_data, key=lambda x: x[0]))

            elif dimension == "rank":
                rows = self.repo.query_trend_data_with_tags(
                    stat, "t.step = ? AND t.target_id = ?", (
                        dim_x, dim_y), metric_id
                )
                trend_data.extend(
                    (int(row.get('rank')), row[stat]) for row in rows)
                if trend_data:
                    dimensions, values = zip(
                        *sorted(trend_data, key=lambda x: x[0]))

            elif dimension == "module_name":
                rows = self.repo.query_trend_data_with_tags(
                    stat, "t.step = ? AND t.rank = ?", (
                        dim_x, dim_y), metric_id, tags
                )
                trend_data.extend(
                    (int(row.get('target_id')),
                     self.repo._get_module_name(row), row[stat])
                    for row in rows
                )

                if trend_data:
                    sorted_data = sorted(trend_data, key=lambda x: x[0])
                    dimensions = [item[1] for item in sorted_data]
                    values = [item[2] for item in sorted_data]

            return {'success': True, 'data': {'dimensions': dimensions, 'values': values}}

        except Exception as e:
            return {'success': False, 'error': f'internal error: {str(e)}'}
