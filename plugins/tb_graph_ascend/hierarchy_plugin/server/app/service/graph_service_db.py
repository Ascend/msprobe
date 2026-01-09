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
import os
from tensorboard.util import tb_logging

from .graph_service_base import GraphServiceStrategy
from ..repositories.graph_repo_db import GraphRepoDB
from ..utils.global_state import GraphState
from ..utils.graph_utils import GraphUtils

from ..model.layout_hierarchy_model import LayoutHierarchyModel
from ..model.match_nodes_model import MatchNodesController
from ..utils.constant import NPU, BENCH, SINGLE, UN_MATCHED_VALUE
from ..utils.constant import MAX_RELATIVE_ERR, MIN_RELATIVE_ERR, MEAN_RELATIVE_ERR, NORM_RELATIVE_ERR

logger = tb_logging.get_logger()


class DbGraphService(GraphServiceStrategy):

    def __init__(self, run_path, tag):
        super().__init__(run_path, tag)
        runs = GraphState.get_global_value('runs')
        db_path = os.path.join(runs.get(self.run), f"{tag}.vis.db")
        self.repo = GraphRepoDB(db_path)
        self.conn = self.repo.get_db_connection()
        self.config_info = {}

    def load_graph_data(self):
        if not self.repo:
            return {'success': False, 'error': GraphUtils.t('dbInitError')}
        if not self.conn:
            self.conn = self.repo.get_db_connection()
        GraphState.set_global_value("all_node_info_cache", {})  # 切换文件清缓存
        return {'success': True}

    def load_graph_config_info(self):
        try:
            self.config_info = self.repo.query_config_info()
            GraphState.set_global_value("config_info", self.config_info)
            
            # 读取目录下配置文件列表
            modify_matched_files = GraphUtils.find_config_files(self.run)
            self.config_info['matchedConfigFiles'] = modify_matched_files or []
            return {'success': True, 'data': self.config_info}
        except Exception as e:
            logger.error(f"load graph config info failed, {e}")
            return {'success': False, 'error': GraphUtils.t('loadConfigError')}

    def load_graph_all_node_list(self, meta_data):
        try:
            if not self.conn:
                return {'success': False, 'error': GraphUtils.t('dbInitError')}
            rank = meta_data.get('rank')
            step = meta_data.get('step')
            micro_step = meta_data.get('microStep')
            if rank is None or step is None or micro_step is None:
                return {'success': False, 'error': GraphUtils.t('rankStepNullError')}
            result = {}
            if not self.config_info:
                self.config_info = self.repo.query_config_info() 
            # 单图
            if self.config_info.get('isSingleGraph'):
                # DB：根据graphType rank step micro_step 查询节点列表
                node_name_list = self.repo.query_node_name_list(rank, step, micro_step)
                result['npuNodeList'] = node_name_list
                return {'success': True, 'data': result} 
            # 双图
            else:
                config_data = GraphState.get_global_value("config_data")
                all_node_info = self.repo.query_all_node_info_in_one(rank, step, micro_step)
                config_data['npuMatchNodes'] = all_node_info.get('npu_match_node')
                config_data['benchMatchNodes'] = all_node_info.get('bench_match_node')
                config_data['npuUnMatchNodes'] = all_node_info.get('npu_unmatch_node')
                config_data['benchUnMatchNodes'] = all_node_info.get('bench_unmatch_node')
        
                result['npuMatchNodes'] = all_node_info.get('npu_match_node')
                result['benchMatchNodes'] = all_node_info.get('bench_match_node')
                result['npuUnMatchNodes'] = all_node_info.get('npu_unmatch_node')
                result['benchUnMatchNodes'] = all_node_info.get('bench_unmatch_node')
                result['npuNodeList'] = all_node_info.get('npu_node_list')
                result['benchNodeList'] = all_node_info.get('bench_node_list')
                
                return {'success': True, 'data': result}
        except Exception as e:
            logger.error(f"load graph all node list failed, {e}")
            return {'success': False, 'error': GraphUtils.t('loadNodeError')}

    def change_node_expand_state(self, node_info, meta_data):
        try:
            if not self.conn:
                return {'success': False, 'error': GraphUtils.t('dbInitError')}
            graph_type = node_info.get('nodeType')
            node_name = node_info.get('nodeName')
            rank = meta_data.get('rank')
            step = meta_data.get('step')
            micro_step = meta_data.get('microStep')
            if rank is None or step is None or micro_step is None:
                return {'success': False, 'error': GraphUtils.t('rankStepNullError')}
            # 单图
            rank_step = {
                'rank': rank,
                'step': step
            }
            if self.config_info.get('isSingleGraph'):
                hierarchy = LayoutHierarchyModel.change_expand_state(node_name, SINGLE, self.repo, micro_step,
                                                                     rank_step)
            # NPU
            elif graph_type == NPU:
                hierarchy = LayoutHierarchyModel.change_expand_state(node_name, NPU, self.repo, micro_step, rank_step)
            # 标杆
            elif graph_type == BENCH:
                hierarchy = LayoutHierarchyModel.change_expand_state(node_name, BENCH, self.repo, micro_step, rank_step)
            else:
                return {'success': True, 'data': {}}
            return {'success': True, 'data': hierarchy}
        except Exception as e:
            logger.error('node expand or collapse failed:' + str(e))
            return {'success': False, 'error': GraphUtils.t('expandNodeError'), 'data': None}

    def search_node_by_precision(self, meta_data, values):
        try:
            if not self.conn:
                return {'success': False, 'error': GraphUtils.t('dbInitError')}
            rank = meta_data.get('rank')
            step = meta_data.get('step')
            micro_step = meta_data.get('microStep')
            if rank is None or step is None or micro_step is None:
                return {'success': False, 'error': GraphUtils.t('rankStepNullError')}
            is_filter_unmatch_nodes = True if UN_MATCHED_VALUE in values else False
            if is_filter_unmatch_nodes:
                values.remove(UN_MATCHED_VALUE)
          
            update_precision_cache = GraphState.get_global_value("update_precision_cache", {})
            node_name_list = []
            # 先查全局缓存
            if update_precision_cache:
                for node_name, node_info in update_precision_cache.items():
                    if not node_info.get("is_leaf_nodes"):
                        continue 
                    matched_node_link = node_info.get('matched_node_link', None)
                    if is_filter_unmatch_nodes and (not matched_node_link):
                        node_name_list.append(node_name)
                        
                    if isinstance(node_info.get("precision_index"), (int, float, complex)) and any(low <= node_info.
                                            get("precision_index", -1) <= high for low, high in values):
                        node_name_list.append(node_name)
            # 在查数据库
            else:
                node_name_list = self.repo.query_node_list_by_precision(step, rank, micro_step, values,
                                                                        is_filter_unmatch_nodes)
            return {'success': True, 'data': node_name_list}
        except Exception as e:
            logger.error('node search by precision failed:' + str(e))
            return {'success': False, 'error': GraphUtils.t('searchNodeError'), 'data': None}
    
    def search_node_by_overflow(self, meta_data, values):
        try:
            if not self.conn:
                return {'success': False, 'error': GraphUtils.t('dbInitError')}
            rank = meta_data.get('rank')
            step = meta_data.get('step')
            micro_step = meta_data.get('microStep')
            if rank is None or step is None or micro_step is None:
                return {'success': False, 'error': GraphUtils.t('rankStepNullError')}
            node_name_list = self.repo.query_node_list_by_overflow(step, rank, micro_step, values)
            return {'success': True, 'data': node_name_list}
        except Exception as e:
            logger.error('node search by overflow failed:' + str(e))
            return {'success': False, 'error': GraphUtils.t('searchNodeError'), 'data': None}

    def update_hierarchy_data(self, graph_type):
        if (graph_type == NPU or graph_type == BENCH):
            hierarchy = LayoutHierarchyModel.update_hierarchy_data(graph_type)
            return {'success': True, 'data': hierarchy}
        else:
            return {'success': False, 'error': GraphUtils.t('graphTypeError')}

    def get_node_info(self, node_info, meta_data):
        try:
            if not self.conn:
                return {'success': False, 'error': GraphUtils.t('dbInitError')}
            result = {}
            graph_type = node_info.get('nodeType')
            node_name = node_info.get('nodeName')
            rank = meta_data.get('rank')
            step = meta_data.get('step')
            if rank is None or step is None:
                return {'success': False, 'error': GraphUtils.t('rankStepNullError')}
            if self.config_info.get('isSingleGraph') or graph_type == SINGLE:
                result['npu'] = self.repo.query_node_info(node_name, graph_type, rank, step)
            else:
                matched_node_type = BENCH if graph_type == NPU else NPU
                node = self.repo.query_node_info(node_name, graph_type, rank, step)
                matched_node_link = node.get('matched_node_link', []) 
                if isinstance(node.get('matched_node_link', []), list) and len(matched_node_link) > 0:
                    matched_node = self.repo.query_node_info(matched_node_link[-1], matched_node_type, rank, step)
                else:
                    matched_node = None
                result['npu'] = node if graph_type == NPU else matched_node
                result['bench'] = node if graph_type == BENCH else matched_node
            return {'success': True, 'data': result}
        except Exception as e:
            logger.error('get node info failed:' + str(e))
            return {'success': False, 'error': GraphUtils.t('getNodeInfoError'), 'data': None}

    def add_match_nodes(self, npu_node_name, bench_node_name, meta_data, is_match_children):
        try:
            if not self.conn:
                return {'success': False, 'error': GraphUtils.t('dbInitError')}
            rank = meta_data.get('rank')
            step = meta_data.get('step')
            if rank is None or step is None:
                return {'success': False, 'error': GraphUtils.t('rankStepNullError')}
            task = self.config_info.get('task')
            # 根据任务类型计算误差
            if task == 'md5' or task == 'summary':
                if is_match_children:
                    graph_data = self.repo.query_node_and_sub_nodes(npu_node_name, bench_node_name, rank, step)
                    match_result = MatchNodesController.process_task_add_child_layer(graph_data, npu_node_name,
                                                                               bench_node_name, task)
                else:
                    graph_data = self.repo.query_matched_nodes_info(npu_node_name, bench_node_name, rank, step)
                    match_result = MatchNodesController.process_task_add(graph_data, npu_node_name,
                                                                         bench_node_name, task)
                return self._generate_matched_result(match_result, rank, step)
            else:
                return {'success': False, 'error': GraphUtils.t('taskTypeError')}
        except Exception as e:
            logger.error(str(e))
            return {'success': False, 'error': GraphUtils.t('matchNodeError'), 'data': None}

    def add_match_nodes_by_config(self, config_file_name, meta_data):
        if not self.conn:
            return {'success': False, 'error': GraphUtils.t('dbInitError')}
        result = {}
        rank = meta_data.get('rank')
        step = meta_data.get('step')
        if rank is None or step is None:
            return {'success': False, 'error': GraphUtils.t('rankStepNullError')}
        task = self.config_info.get('task')
        try:
            match_node_links, error = GraphUtils.safe_load_data(meta_data.get('run'), config_file_name)
            graph_data = self.repo.query_matched_nodes_info_by_config(match_node_links, rank, step)
            if error:
                return {'success': False, 'error': GraphUtils.t('loadConfigFileError')}
            if not match_node_links:
                return {'success': False, 'error': GraphUtils.t('matchNodeLinksNullError')}
            # 根据任务类型计算误差
            if task == 'md5' or task == 'summary':
                match_result = MatchNodesController.process_task_add_child_layer_by_config(graph_data,
                                                                                           match_node_links, task)
                result = self._generate_matched_result(match_result, rank, step)
                result.setdefault('data', {})['matchResult'] = [item.get('success', False) for item in match_result]
                return result
            else:
                return {'success': False, 'error': GraphUtils.t('taskTypeError')}
        except Exception as e:
            logger.error(str(e))
            return {'success': False, 'error': GraphUtils.t('matchNodeError'), 'data': None}

    def delete_match_nodes(self, npu_node_name, bench_node_name, meta_data, is_unmatch_children):
        try:
            if not self.conn:
                return {'success': False, 'error': GraphUtils.t('dbInitError')}
            rank = meta_data.get('rank')
            step = meta_data.get('step')
            if rank is None or step is None:
                return {'success': False, 'error': GraphUtils.t('rankStepNullError')}
            task = self.config_info.get('task')
            
            # 根据任务类型计算误差
            if task == 'md5' or task == 'summary':
                if is_unmatch_children:
                    # DB：当前节点及其所有的子节点信息
                    graph_data = self.repo.query_node_and_sub_nodes(npu_node_name, bench_node_name, rank, step)
                    match_result = MatchNodesController.process_task_delete_child_layer(graph_data, npu_node_name,
                                                                                  bench_node_name, task)
                else:
                    graph_data = self.repo.query_matched_nodes_info(npu_node_name, bench_node_name, rank, step)
                    match_result = MatchNodesController.process_task_delete(graph_data, npu_node_name,
                                                                            bench_node_name, task)
                return self._generate_matched_result(match_result, rank, step)
            else:
                return {'success': False, 'error': GraphUtils.t('taskTypeError')}
        except Exception as e:
            logger.error('delete_match_nodes error: {}'.format(e))
            return {'success': False, 'error': GraphUtils.t('deleteMatchNodeFailed'), 'data': None}
    
    def update_precision_error(self, meta_data, filter_value):
        try:
            if not self.conn:
                return {'success': False, 'error': GraphUtils.t('dbConncetionNotInit')}

            rank = meta_data.get('rank')
            step = meta_data.get('step')
            if rank is None or step is None:
                return {'success': False, 'error': GraphUtils.t('rankStepNullError')}
            npu_node_list = self.repo.query_node_info_by_data_source(step, rank, NPU)
            update_data_hierarchy = {}
            update_data_db = []
          
            for _, node_info in npu_node_list.items():
                output_statistical_diff = node_info.get('output_data', None)                
                if not node_info.get('matched_node_link') or not output_statistical_diff:
                    update_data_hierarchy[node_info.get('node_name')] = {
                       "precision_index": node_info.get('data').get('precision_index'),
                       "node_name": node_info.get('node_name'),
                       "matched_node_link": node_info.get('matched_node_link'),
                       'is_leaf_nodes': node_info.get('subnodes', []) == [],
                       'graph_type': NPU
                    }  
                    continue
                max_rel_error = -1
                #  根据filter_value 的选择指标计算新的误差值
                for _, diff_values in output_statistical_diff.items():
                    filter_diff_rel = []
                    if MAX_RELATIVE_ERR in filter_value:
                        filter_diff_rel.append(diff_values.get('MaxRelativeErr'))
                    if MIN_RELATIVE_ERR in filter_value:
                        filter_diff_rel.append(diff_values.get('MinRelativeErr'))
                    if NORM_RELATIVE_ERR in filter_value:
                        filter_diff_rel.append(diff_values.get('NormRelativeErr'))
                    if MEAN_RELATIVE_ERR in filter_value:
                        filter_diff_rel.append(diff_values.get('MeanRelativeErr'))
                    # 过滤掉N/A
                    filter_diff_rel = [x for x in filter_diff_rel if x and x != 'N/A']
                    # 如果output指标中存在 Nan/inf/-inf, 直接标记为最大值
                    if "Nan" in filter_diff_rel or "inf" in filter_diff_rel or "-inf" in filter_diff_rel:
                        max_rel_error = 1
                        break
                    filter_diff_rel = [GraphUtils.convert_to_float(x) for x in filter_diff_rel]
                    max_rel_error_for_key = max(filter_diff_rel) if filter_diff_rel else 0
                    max_rel_error = max(max_rel_error, max_rel_error_for_key)
                if max_rel_error != -1:
                    update_data_db.append((
                        min(max_rel_error, 1),
                        step,
                        rank,
                        node_info.get('node_name')
                    ))
                update_data_hierarchy[node_info.get('node_name')] = {
                       "precision_index": min(max_rel_error, 1),
                       "node_name": node_info.get('node_name'),
                       "matched_node_link": node_info.get('matched_node_link'),
                       'is_leaf_nodes': node_info.get('subnodes', []) == [],
                       'graph_type': NPU
                    }  
            if len(update_data_hierarchy) > 0:
                # 视图：调用更新update_hirarchy方法，同步更新图
                LayoutHierarchyModel.update_current_hierarchy_data(list(update_data_hierarchy.values()))
                # 更新全局缓存，使用update_data_hierarchy覆盖原来的缓存即可，注意，切换视图需要重置缓存信息
                GraphState.set_global_value("update_precision_cache", update_data_hierarchy)
                return {'success': True, 'data': {}}
            else:
                return {'success': False, 'error': GraphUtils.t('noUpdateNodesFound')}
        except Exception as e:
            logger.error('update_precision_error error: {}'.format(e))
            return {'success': False, 'error': GraphUtils.t('updateNodeAccuracyFailed'), 'data': None}   

    def update_colors(self, colors):
        try:
            if not self.conn:
                return {'success': False, 'error': GraphUtils.t('dbConncetionNotInit')}
            # DB：更新颜色
            update_db_res = self.repo.update_config_colors(colors)
            if not update_db_res:
                return {'success': False, 'error': GraphUtils.t('updateDatabaseFailed')}
            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': GraphUtils.t('updateColorFailed'), 'data': None}

    def save_matched_relations(self, meta_data):
        try:
            if not self.conn:
                return {'success': False, 'error': GraphUtils.t('dbConncetionNotInit')}
            run = meta_data.get('run')
            tag = meta_data.get('tag')
            rank = meta_data.get('rank')
            step = meta_data.get('step')
            if rank is None or step is None:
                return {'success': False, 'error': GraphUtils.t('rankStepNullError')}
            # DB：根据step rank modify match_node_link查询已经修改的匹配成功的节点关系
            modify_matched_nodes_list = self.repo.query_modify_matched_nodes_list(rank, step)
            confilg_file_name = f"{tag}_{step}_{rank}.vis.config"
            _, error = GraphUtils.safe_save_data(modify_matched_nodes_list, run, confilg_file_name)
            if error:
                return {'success': False, 'error': error}
            else:
                return {'success': True, 'data': confilg_file_name}
       
        except Exception as e:
            logger.error('save_matched_relations error: {}'.format(e))
            return {'success': False, 'error': GraphUtils.t('saveMatchRelationshipFailed'), 'data': None}

    def _generate_matched_result(self, match_result, rank, step):
        update_data = []
        for item in match_result:
            if item.get('success') is True:
                nodes = item.get('data', [])
                for node in nodes:
                    update_data.append(node)
        if len(update_data) > 0:
            # DB：更新数据库节点信息
            update_db_res = self.repo.update_nodes_info(update_data, rank, step)
            if not update_db_res:
                return {'success': False, 'error': GraphUtils.t('updateDatabaseFailed')}
            # 视图：调用更新update_hirarchy方法，同步更新图
            LayoutHierarchyModel.update_current_hierarchy_data(update_data)
            # 返回：返回更新后的节点信息
            config_data = GraphState.get_global_value("config_data")
            result = {
                'success': True,
                'data': {
                    'npuMatchNodes': config_data.get('npuMatchNodes', {}),
                    'benchMatchNodes': config_data.get('benchMatchNodes', {}),
                    'npuUnMatchNodes': config_data.get('npuUnMatchNodes', []),
                    'benchUnMatchNodes': config_data.get('benchUnMatchNodes', [])
                }
            }
        else:
            result = {'success': False, 'error': GraphUtils.t('selectedNodeCantMatched')}
        return result
                     
