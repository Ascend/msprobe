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
import time
import json
from tensorboard.util import tb_logging

from .graph_service_base import GraphServiceStrategy
from ..repositories.graph_repo_vis import GraphRepoVis
from ..utils.global_state import GraphState
from ..utils.graph_utils import GraphUtils
from ..model.layout_hierarchy_model import LayoutHierarchyModel
from ..model.match_nodes_model import MatchNodesController
from ..utils.constant import NPU_PREFIX, BENCH_PREFIX, NPU, BENCH, SINGLE, UN_MATCHED_VALUE
from ..utils.constant import MAX_RELATIVE_ERR, MIN_RELATIVE_ERR, MEAN_RELATIVE_ERR, NORM_RELATIVE_ERR

logger = tb_logging.get_logger()


class JsonGraphService(GraphServiceStrategy):

    def __init__(self, run_path, tag):
        super().__init__(run_path, tag)
        self.repo = None

    def load_graph_data(self):
        runs = GraphState.get_global_value('runs')
        run_path = runs.get(self.run)
        buffer = ""
        read_bytes = 0
        chunk_size = 1024 * 1024 * 60  # 缓冲区
        json_data = None  # 最终存储的变量
        _, error = GraphUtils.safe_load_data(run_path, f"{self.tag}.vis", True)
        if error:
            return {'success': False, 'error': GraphUtils.t('loadErrorBySecurityIssue')}
        file_path = os.path.join(run_path, f"{self.tag}.vis")
        file_path = os.path.normpath(file_path)  # 标准化路径
        file_size = os.path.getsize(file_path)  
        if file_size == 0:
            return {'success': False, 'error': '文件为空'}
        with open(file_path, 'r', encoding='utf-8') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                read_bytes += len(chunk)
                buffer += chunk
                current_progress = min(95, int((read_bytes / file_size) * 100)) 
                reading_info = {
                    'progress': current_progress,
                    'status': 'reading',
                    'size': file_size,
                    'read': read_bytes
                }
                yield f"data: {json.dumps(reading_info)}\n\n"
                time.sleep(0.01)  # 避免发送过快

        if json_data is None and buffer:  # 最终验证数据
            try:
                yield f"data: {json.dumps({'progress': current_progress, 'status': 'loading'})}\n\n"
                json_data = GraphUtils.safe_json_loads(buffer)
                yield f"data: {json.dumps({'progress': 99, 'status': 'loading'})}\n\n"
            except json.JSONDecodeError as e:
                yield f"data: {json.dumps({'progress': current_progress, 'error': GraphUtils.t('parseJsonFailed')})}\n\n"

        if json_data is not None:  # 验证存储
            GraphState.set_global_value('current_file_data', json_data)
            GraphState.set_global_value('current_tag', self.tag)
            GraphState.set_global_value('current_run', run_path)
            # 初始化GraphJson
            self.repo = GraphRepoVis(json_data)
            yield f"data: {json.dumps({'done': True, 'progress': 100, 'status': 'loading'})}\n\n"
        else:
            yield f"data: {json.dumps({'progress': current_progress, 'error': GraphUtils.t('parseJsonFailed')})}\n\n"

    def load_graph_config_info(self):
        run_name = self.run
        tag = self.tag
        graph_data, error_message = GraphUtils.get_graph_data({'run': run_name, 'tag': tag})
        if error_message or not graph_data:
            return {'success': False, 'error': str(error_message)}
        config = {}
        try:
            # 读取全局信息,tag层面
            if graph_data.get('MicroSteps', {}):
                config['microSteps'] = graph_data.get('MicroSteps')
            if graph_data.get('ToolTip', {}):
                config['tooltips'] = graph_data.get('ToolTip')
            config['overflowCheck'] = bool(graph_data.get('OverflowCheck')) if 'OverflowCheck' in graph_data else True
            config['isSingleGraph'] = False if graph_data.get(NPU) else True
            # 读取配置信息，run层面
            config_data = GraphState.get_global_value("config_data", {})
            config_data_run = config_data.get(run_name, {})
            if not config_data_run:  # 如果没有run的配置信息，则读取第一个文件中的Colors
                first_run_tags = GraphState.get_global_value("first_run_tags")
                first_tag = first_run_tags.get(run_name)
                if not first_tag:
                    return {'success': False, 'error': GraphUtils.t('loadConfigFailed')}
                first_graph_data, error_message = GraphUtils.get_graph_data({'run': run_name, 'tag': first_tag})
                config_data_run['colors'] = first_graph_data.get('Colors')
                config_data[run_name] = config_data_run
                GraphState.set_global_value('config_data', config_data)
                config['colors'] = first_graph_data.get('Colors')
            else:
                config['colors'] = config_data_run.get('colors')
            # 读取目录下配置文件列表
            config_files = GraphUtils.find_config_files(run_name)
            config['matchedConfigFiles'] = config_files or []
            config['task'] = graph_data.get('task')
            GraphState.set_global_value("config_info", config)
            return {'success': True, 'data': config}
        except Exception as e:
            return {'success': False, 'error': GraphUtils.t('loadConfigFailed')}

    def load_graph_all_node_list(self, meta_data):
        micro_step = meta_data.get('microStep')
        graph_data, error_message = GraphUtils.get_graph_data({'run': self.run, 'tag': self.tag})
        if error_message or not graph_data:
            return {'success': False, 'error': str(error_message)}
        result = {}
        try:
            if not graph_data.get(NPU):
                nodes = GraphUtils.split_graph_data_by_microstep(graph_data, micro_step)
                node_name_list = list(nodes.keys())
                result['npuNodeList'] = node_name_list
                return {'success': True, 'data': result}

            else:
                # 双图
                config_data = GraphState.get_global_value("config_data")
                npu_node = GraphUtils.split_graph_data_by_microstep(graph_data.get(NPU), micro_step)
                bench_node = GraphUtils.split_graph_data_by_microstep(graph_data.get(BENCH), micro_step)
                npu_node_name_list = list(npu_node.keys())
                bench_node_name_list = list(bench_node.keys())
                npu_unmatehed_name_list = [
                    key
                    for key, value in npu_node.items()
                    if not value.get("matched_node_link")
                ]
                bench_unmatehed_name_list = [
                    key
                    for key, value in bench_node.items()
                    if not value.get("matched_node_link")
                ]
                # 保存未匹配和已匹配的节点到全局变量
                config_data['npuUnMatchNodes'] = npu_unmatehed_name_list
                config_data['benchUnMatchNodes'] = bench_unmatehed_name_list
                config_data['npuMatchNodes'] = graph_data.setdefault('npu_match_nodes', {})
                config_data['benchMatchNodes'] = graph_data.setdefault('bench_match_nodes', {})
                # 返回结果
                result['npuNodeList'] = npu_node_name_list
                result['benchNodeList'] = bench_node_name_list
                result['npuUnMatchNodes'] = npu_unmatehed_name_list
                result['benchUnMatchNodes'] = bench_unmatehed_name_list
                result['npuMatchNodes'] = graph_data.setdefault('npu_match_nodes', {})
                result['benchMatchNodes'] = graph_data.setdefault('bench_match_nodes', {})
                for npu_node_name, npu_node in graph_data.get(NPU, {}).get('node', {}).items():
                    if npu_node.get('matched_node_link', None):
                        bench_node_name = npu_node.get('matched_node_link', [None])[-1].replace(BENCH_PREFIX, '', 1)
                        result['npuMatchNodes'][npu_node_name] = bench_node_name
                        result['benchMatchNodes'][bench_node_name] = npu_node_name

                GraphState.set_global_value('config_data', config_data)
                return {'success': True, 'data': result}
        except Exception as e:
            logger.error('get node list error:' + str(e))
            return {'success': False, 'error': GraphUtils.t('getNodeListFailed')}

    def change_node_expand_state(self, node_info, meta_data):

        try:
            graph_data, error_message = GraphUtils.get_graph_data(meta_data)
            if error_message:
                return {'success': False, 'error': str(error_message)}
            if self.repo is None:
                return {'success': False, 'error': GraphUtils.t('initGraphJsonFailed')}
            graph_type = node_info.get('nodeType')
            node_name = node_info.get('nodeName')
            micro_step = meta_data.get('microStep')
            # 单图
            if not graph_data.get(NPU):
                hierarchy = LayoutHierarchyModel.change_expand_state(node_name, SINGLE, self.repo, micro_step, {})
            # NPU
            elif graph_type == NPU:
                hierarchy = LayoutHierarchyModel.change_expand_state(node_name, NPU, self.repo, micro_step, {})
            # 标杆
            elif graph_type == BENCH:
                hierarchy = LayoutHierarchyModel.change_expand_state(node_name, BENCH, self.repo, micro_step, {})
            else:
                return {'success': True, 'data': {}}
            return {'success': True, 'data': hierarchy}
        except Exception as e:
            logger.error('node expand or collapse error:' + str(e))
            node_type_name = ""
            if graph_data.get(NPU):
                node_type_name = GraphUtils.t('debug') if graph_type == NPU else GraphUtils.t('bench')
        return {'success': False, 'error': f'{node_type_name}{GraphUtils.t('expandNodeError')}', 'data': None}

    def search_node_by_precision(self, meta_data, values):
        # 遍历所有的NPU节点，如果节点的精度值在values中，则返回该节点
        graph_data, error_message = GraphUtils.get_graph_data(meta_data)
        if error_message:
            return {'success': False, 'error': str(error_message)}
     
        precision = []
        is_filter_unmatch_nodes = True if UN_MATCHED_VALUE in values else False
        try:
            if is_filter_unmatch_nodes:
                values.remove(UN_MATCHED_VALUE)
            # 单图
            if not graph_data.get(NPU):
                node_list = GraphUtils.split_graph_data_by_microstep(graph_data.get('node', {}),
                                                                     meta_data.get("microStep", -1))
            # 多图
            else:
                node_list = GraphUtils.split_graph_data_by_microstep(graph_data.get(NPU),
                                                                     meta_data.get("microStep", -1))
            for node_name, node in node_list.items():
                subnodes = node.get("subnodes", None)
                if subnodes != [] and subnodes is not None:
                    continue 
                matched_node_link = node.get('matched_node_link', None)
                if is_filter_unmatch_nodes and (matched_node_link is None or matched_node_link == []):
                    precision.append(node_name)
                if any(low <= node.get('data', {}).get("precision_index", -1) < high for low, high in values):
                    precision.append(node_name)
            return {'success': True, 'data': precision}
        except Exception as e:
            logger.error('search precision node failed:' + str(e))
            return {'success': False, 'error': GraphUtils.t('obtainNodesAccuracyFailed')}
    
    def search_node_by_overflow(self, meta_data, values):
        # 遍历所有的NPU节点，如果节点的精度值在values中，则返回该节点
        graph_data, error_message = GraphUtils.get_graph_data(meta_data)
        if error_message:
            return {'success': False, 'error': str(error_message)}
        overflow = []   
        try:
            # 单图
            if not graph_data.get(NPU):
                node_list = GraphUtils.split_graph_data_by_microstep(graph_data.get('node', {}),
                                                                     meta_data.get("microStep", -1))
                for node_name, node in node_list.items():
                    subnodes = node.get("subnodes", None)
                    if subnodes != [] and subnodes is not None:
                        continue 
                    if node.get('data', {}).get("overflow_level", -1) in values:
                        overflow.append(node_name)
                return {'success': True, 'data': overflow}
            else:
                return {'success': False, 'error': GraphUtils.t('notSupportOverflow')}         
        except Exception as e:
            logger.error('search overflow node failed:' + str(e))
            return {'success': False, 'error': GraphUtils.t('obtainNodesOverflowFailed')}

    def update_precision_error(self, meta_data, filter_value):
        try:
            graph_data, error_message = GraphUtils.get_graph_data(meta_data)
            if error_message:
                return {'success': False, 'error': str(error_message)}
            npu_node_list = graph_data.get(NPU, {}).get('node', {})
            for _, node_info in npu_node_list.items():
                output_statistical_diff = node_info.get('output_data', None)
                if not node_info.get('matched_node_link') or not output_statistical_diff:
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
                    node_info.setdefault('data', {})['precision_index'] = min(max_rel_error, 1)
            return {'success': True, 'data': {}}
        except Exception as e:
            logger.error('update precision error error:' + str(e))
            return {'success': False, 'error': GraphUtils.t('updatePrecisionFailed')}

    def update_hierarchy_data(self, graph_type):
        if (graph_type == NPU or graph_type == BENCH):
            hierarchy = LayoutHierarchyModel.update_hierarchy_data(graph_type)
            return {'success': True, 'data': hierarchy}
        else:
            return {'success': False, 'error': GraphUtils.t('nodeTypeError')}
 
    def get_node_info(self, node_info, meta_data):
        graph_data, error_message = GraphUtils.get_graph_data(meta_data)
        if error_message:
            return {'success': False, 'error': str(error_message)}
        try:
            graph_type = node_info.get('nodeType')
            node_name = node_info.get('nodeName')
            result = {}
            if not graph_data.get(NPU) or graph_type == SINGLE:
                result['npu'] = graph_data.get('node', {}).get(node_name)
            else:
                matched_node_type = BENCH if graph_type == NPU else NPU
                matched_node_preifx = NPU_PREFIX if matched_node_type == NPU else BENCH_PREFIX
                node = graph_data.get(graph_type, {}).get('node', {}).get(node_name, {})
                matched_node_link = node.get('matched_node_link', []) if isinstance(node.get('matched_node_link', []),
                                                                                    list) else []
                matched_node_name = matched_node_link[-1].replace(matched_node_preifx, '',
                                                                  1) if matched_node_link else None
                matched_node = graph_data.get(matched_node_type, {}).get('node', {}).get(
                    matched_node_name) if matched_node_name else None
                result['npu'] = node if graph_type == NPU else matched_node
                result['bench'] = node if graph_type == BENCH else matched_node
            return {'success': True, 'data': result}
        except Exception as e:
            logger.error('get node info error:' + str(e))
            return {'success': False, 'error': GraphUtils.t('getNodeInfoError'), 'data': None}

    def add_match_nodes(self, npu_node_name, bench_node_name, meta_data, is_match_children):
        graph_data, error_message = GraphUtils.get_graph_data(meta_data)
        if error_message:
            return {'success': False, 'error': str(error_message)}
        task = graph_data.get('task')
        result = {}
        try:
            # 根据任务类型计算误差
            if task == 'md5' or task == 'summary':
                if is_match_children:
                    match_result = MatchNodesController.process_task_add_child_layer(graph_data,
                                                                    npu_node_name, bench_node_name, task)              
                else:
                    match_result = MatchNodesController.process_task_add(graph_data,
                                                                         npu_node_name, bench_node_name, task)
                return self._generate_matched_result(match_result)
            else:
                return {'success': False, 'error': GraphUtils.t('taskTypeError')}
        except Exception as e:
            return {'success': False, 'error': GraphUtils.t('operationFailed'), 'data': None}

    def add_match_nodes_by_config(self, config_file_name, meta_data):
        graph_data, error_message = GraphUtils.get_graph_data(meta_data)
        if error_message:
            return {'success': False, 'error': GraphUtils.t('readFileFailed')}
        match_node_links, error = GraphUtils.safe_load_data(meta_data.get('run'), config_file_name)
        if error:
            return {'success': False, 'error': GraphUtils.t('loadConfigFileError')}
        if not match_node_links:
            return {'success': False, 'error': GraphUtils.t('matchNodeLinksNullError')}
        task = graph_data.get('task')
        try:
            # 根据任务类型计算误差
            if task == 'md5' or task == 'summary':
                match_result = MatchNodesController.process_task_add_child_layer_by_config(graph_data,
                                                                                           match_node_links, task)
                result = self._generate_matched_result(match_result)
                if result.get('data'):
                    result['data']['matchResult'] = [item.get('success', False) for item in match_result]
                return result
            else:
                return {'success': False, 'error': GraphUtils.t('taskTypeError')}
        except Exception as e:
            logger.error(str(e))
            return {'success': False, 'error': GraphUtils.t('operationFailed'), 'data': None}

    def delete_match_nodes(self, npu_node_name, bench_node_name, meta_data, is_unmatch_children):
        graph_data, error_message = GraphUtils.get_graph_data(meta_data)
        if error_message:
            return {'success': False, 'error': str(error_message)}
        task = graph_data.get('task')
        try:
            # 根据任务类型计算误差
            if task == 'md5' or task == 'summary':
                if is_unmatch_children:
                    match_result = MatchNodesController.process_task_delete_child_layer(graph_data, npu_node_name,
                                                                                  bench_node_name, task)
                else:
                    match_result = MatchNodesController.process_task_delete(graph_data, npu_node_name,
                                                                            bench_node_name, task)
                
                return self._generate_matched_result(match_result)
            else:
                return {'success': False, 'error': GraphUtils.t('taskTypeError')}
        except Exception as e:
            return {'success': False, 'error': GraphUtils.t('operationFailed'), 'data': None}

    def save_data(self, meta_data):
        if not meta_data:
            return {'success': False, 'error': GraphUtils.t('parameterEmpty')}
        graph_data, error_message = GraphUtils.get_graph_data(meta_data)
        if error_message:
            return {'success': False, 'error': str(error_message)}

        try:
            _, error = GraphUtils.safe_save_data(graph_data, self.run, f"{self.tag}.vis")
            if error:
                return {'success': False, 'error': error}
        except (ValueError, IOError, PermissionError) as e:
            return {'success': False, 'error': GraphUtils.t('saveFileFailed')}
        return {'success': True}

    def update_colors(self, colors):
        """Set new colors in jsondata."""
        try:
            config_data = GraphState.get_global_value("config_data", {})
            first_run_tags = GraphState.get_global_value("first_run_tags")
            config_data_run = config_data.get(self.run, {})
            first_run_tag = first_run_tags.get(self.run)
            first_file_data, error = GraphUtils.safe_load_data(self.run, f"{first_run_tag}.vis")
            if error:
                logger.error(f"Error loading data: {error}")
                return {'success': False, 'error': GraphUtils.t('loadConfigFailed')}
            first_file_data['Colors'] = colors
            config_data_run['colors'] = colors
            config_data[self.run] = config_data_run
            GraphState.set_global_value("config_data", config_data)
            GraphUtils.safe_save_data(first_file_data, self.run, f"{first_run_tag}.vis")
            return {'success': True, 'error': None, 'data': {}}
        except Exception as e:
            return {'success': False, 'error': GraphUtils.t('updateColorFailed'), 'data': None}

    def save_matched_relations(self, meta_data):
        run = meta_data.get('run')
        tag = meta_data.get('tag')
        config_data = GraphState.get_global_value("config_data")
        # 匹配列表和未匹配列表
        npu_match_nodes_list = config_data.get('manualMatchNodes', {})
        try:
            _, error = GraphUtils.safe_save_data(npu_match_nodes_list, run, f"{tag}.vis.config")
            if error:
                return {'success': False, 'error': error}
            else:
                return {'success': True, 'data': f"{tag}.vis.config"}
        except (ValueError, IOError, PermissionError) as e:
            return {'success': False, 'error': GraphUtils.t('operationFailed')}
  
    def _generate_matched_result(self, match_result):
        update_data = []
        for item in match_result:
            if item.get('success') is True:
                for node in item.get('data', []):
                    update_data.append(node)
            
        if len(update_data) > 0:
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
