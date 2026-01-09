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
from ..utils.graph_utils import GraphUtils
from ..utils.global_state import GraphState
from ..controllers.match_nodes_controller import MatchNodesController
from ..controllers.layout_hierarchy_controller import LayoutHierarchyController
from ..utils.global_state import NPU_PREFIX, BENCH_PREFIX, NPU, BENCH, SINGLE

logger = tb_logging.get_logger()


class GraphService:

    @staticmethod
    def load_meta_dir(is_safe_check):
        """Scan logdir for directories containing .vis files, modified to return a tuple of (run, tag)."""
      
        logdir = GraphState.get_global_value('logdir')
        runs = GraphState.get_global_value('runs', {})
        first_run_tags = GraphState.get_global_value('first_run_tags', {})
        
        meta_dir = {}
        error_list = []
        for root, _, files in GraphUtils.walk_with_max_depth(logdir, 2):
            for file in files:
                if file.endswith('.vis'):  # check for .vis extension
                    run_abs = os.path.abspath(root)
                    run = os.path.basename(run_abs)  # 不允许同名目录，否则有问题
                    tag = os.path.splitext(file)[0]  # Use the filename without extension as tag
                    _, error = GraphUtils.safe_load_data(run_abs, f"{tag}.vis", True)
                    if error and is_safe_check:
                        error_list.append({
                            'run': run,
                            'tag': tag,
                            'info': f'Error: {error}'
                        })
                        logger.error(f'Error: File run:"{run_abs},tag:{tag}" is not accessible. Error: {error}')
                        continue
                    runs[run] = run_abs
                    meta_dir.setdefault(run, []).append(tag)
        meta_dir = GraphUtils.sort_data(meta_dir)
        for run, tags in meta_dir.items():
            first_run_tags[run] = tags[0]
        GraphState.set_global_value('runs', runs)
        GraphState.set_global_value('first_run_tags', first_run_tags)
        result = {
            'data': meta_dir,
            'error': error_list
        }
        return result

    @staticmethod
    def load_graph_data(run_name, tag):
        runs = GraphState.get_global_value('runs')
        run = runs.get(run_name)
        buffer = ""
        read_bytes = 0
        chunk_size = 1024 * 1024 * 60  # 缓冲区
        json_data = None  # 最终存储的变量
        file_path = os.path.join(run, f"{tag}.vis")
        file_path = os.path.normpath(file_path)  # 标准化路径
        file_size = os.path.getsize(file_path)
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
                yield f"data: {json.dumps({'progress': current_progress, 'error': str(e)})}\n\n"

        if json_data is not None:  # 验证存储
            GraphState.set_global_value('current_file_data', json_data)
            GraphState.set_global_value('current_tag', tag)
            GraphState.set_global_value('current_run', run)
            yield f"data: {json.dumps({'done': True, 'progress': 100, 'status': 'loading'})}\n\n"
        else:
            yield f"data: {json.dumps({'progress': current_progress, 'error': 'Failed to parse JSON'})}\n\n"

    @staticmethod
    def load_graph_config_info(run, tag):
        graph_data, error_message = GraphUtils.get_graph_data({'run': run, 'tag': tag})
        if error_message or not graph_data:
            return {'success': False, 'error': error_message}
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
            config_data_run = config_data.get(run, {})
            if not config_data_run:  # 如果没有run的配置信息，则读取第一个文件中的Colors
                first_run_tags = GraphState.get_global_value("first_run_tags")
                first_tag = first_run_tags.get(run)
                if not first_tag:
                    return {'success': False, 'error': '获取配置信息失败,请检查目录中第一个文件'}
                first_graph_data, error_message = GraphUtils.get_graph_data({'run': run, 'tag': first_tag})
                config_data_run['colors'] = first_graph_data.get('Colors')
                config_data[run] = config_data_run
                GraphState.set_global_value('config_data', config_data)
                config['colors'] = first_graph_data.get('Colors')
            else:
                config['colors'] = config_data_run.get('colors')
            # 读取目录下配置文件列表
            config_files = GraphUtils.find_config_files(run)
            config['matchedConfigFiles'] = config_files or []
            config['task'] = graph_data.get('task')
            return {'success': True, 'data': config}
        except Exception as e:
            return {'success': False, 'error': '获取配置信息失败,请检查目录中第一个文件'}

    @staticmethod
    def load_graph_all_node_list(run, tag, micro_step):
        graph_data, error_message = GraphUtils.get_graph_data({'run': run, 'tag': tag})
        if error_message or not graph_data:
            return {'success': False, 'error': error_message}
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
            logger.error('获取节点列表失败:' + str(e))
            return {'success': False, 'error': '获取节点列表失败:' + str(e)}

    @staticmethod
    def change_node_expand_state(node_info, meta_data):
        graph_data, error_message = GraphUtils.get_graph_data(meta_data)
        if error_message or not graph_data:
            return {'success': False, 'error': error_message}
        graph_type = node_info.get('nodeType')
        node_name = node_info.get('nodeName')
        micro_step = meta_data.get('microStep')
        try:
            # 单图
            if not graph_data.get(NPU):
                hierarchy = LayoutHierarchyController.change_expand_state(node_name, SINGLE, graph_data, micro_step)
            # NPU
            elif (graph_type == NPU):
                hierarchy = LayoutHierarchyController.change_expand_state(node_name, graph_type,
                                                                          graph_data.get(NPU, {}), micro_step)
            # 标杆
            elif graph_type == BENCH:
                hierarchy = LayoutHierarchyController.change_expand_state(node_name, graph_type,
                                                                          graph_data.get(BENCH, {}), micro_step)
            else:
                return {'success': True, 'data': {}}
            return {'success': True, 'data': hierarchy}
        except Exception as e:
            logger.error('节点展开或收起发生错误:' + str(e))
            node_type_name = ""
            if graph_data.get(NPU): 
                node_type_name = '调试侧' if graph_type == NPU else '标杆侧'
            return {'success': False, 'error': f'{node_type_name}节点展开或收起发生错误', 'data': None}

    @staticmethod
    def update_hierarchy_data(graph_type):
        if (graph_type == NPU or graph_type == BENCH):
            hierarchy = LayoutHierarchyController.update_hierarchy_data(graph_type)
            return {'success': True, 'data': hierarchy}
        else:
            return {'success': False, 'error': '节点类型错误'}

    @staticmethod
    def get_node_info(node_info, meta_data):
        graph_data, error_message = GraphUtils.get_graph_data(meta_data)
        if error_message:
            return {'success': False, 'error': error_message}
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
            logger.error('获取节点信息失败:' + str(e))
            return {'success': False, 'error': '获取节点信息失败:' + str(e), 'data': None}

    @staticmethod
    def add_match_nodes(npu_node_name, bench_node_name, meta_data, is_match_children):
        graph_data, error_message = GraphUtils.get_graph_data(meta_data)
        if error_message:
            return {'success': False, 'error': error_message}
        task = graph_data.get('task')
        result = {}
        try:
            # 根据任务类型计算误差
            if task == 'md5' or task == 'summary':
                if is_match_children:
                    result = MatchNodesController.process_task_add_child_layer(graph_data,
                                                                    npu_node_name, bench_node_name, task)
                    return result
                else:
                    result = MatchNodesController.process_task_add(graph_data, npu_node_name, bench_node_name, task)
                    if result.get('success'):
                        config_data = GraphState.get_global_value("config_data")
                        result['data'] = {
                            'npuMatchNodes': config_data.get('npuMatchNodes', {}),
                            'benchMatchNodes': config_data.get('benchMatchNodes', {}),
                            'npuUnMatchNodes': config_data.get('npuUnMatchNodes', []),
                            'benchUnMatchNodes': config_data.get('benchUnMatchNodes', [])
                        }
                    return result
            else:
                return {'success': False, 'error': '任务类型不支持(Task type not supported) '}
        except Exception as e:
            return {'success': False, '操作失败': str(e), 'data': None}

    @staticmethod
    def add_match_nodes_by_config(config_file, meta_data):
        graph_data, error_message = GraphUtils.get_graph_data(meta_data)
        if error_message:
            return {'success': False, 'error': '读取文件失败'}
        match_node_links, error = GraphUtils.safe_load_data(meta_data.get('run'), config_file)
        if error:
            return {'success': False, 'error': '配置文件失败'}
        task = graph_data.get('task')
        try:
            # 根据任务类型计算误差
            if task == 'md5' or task == 'summary':
                result = MatchNodesController.process_task_add_child_layer_by_config(graph_data, match_node_links, task)
                return result
            else:
                return {'success': False, 'error': '任务类型不支持(Task type not supported) '}
        except Exception as e:
            return {'success': False, 'error': '操作失败', 'data': None}

    @staticmethod
    def delete_match_nodes(npu_node_name, bench_node_name, meta_data, is_unmatch_children):
        graph_data, error_message = GraphUtils.get_graph_data(meta_data)
        if error_message:
            return {'success': False, 'error': error_message}
        task = graph_data.get('task')
        result = {}
        try:
            # 根据任务类型计算误差
            if task == 'md5' or task == 'summary':
                if is_unmatch_children:
                    result = MatchNodesController.process_task_delete_child_layer(graph_data, npu_node_name,
                                                                              bench_node_name, task)
                else:
                    result = MatchNodesController.process_task_delete(graph_data, npu_node_name, bench_node_name, task)
                    if result.get('success'):
                        config_data = GraphState.get_global_value("config_data")
                        result['data'] = {
                            'npuMatchNodes': config_data.get('npuMatchNodes', {}),
                            'benchMatchNodes': config_data.get('benchMatchNodes', {}),
                            'npuUnMatchNodes': config_data.get('npuUnMatchNodes', []),
                            'benchUnMatchNodes': config_data.get('benchUnMatchNodes', [])
                        }
                return result
            else:
                return {'success': False, 'error': '任务类型不支持(Task type not supported) '}
        except Exception as e:
            return {'success': False, '操作失败': str(e), 'data': None}

    @staticmethod
    def update_colors(run, colors):
        """Set new colors in jsondata."""
        try:
            config_data = GraphState.get_global_value("config_data", {})
            first_run_tags = GraphState.get_global_value("first_run_tags")
            config_data_run = config_data.get(run, {})
            first_run_tag = first_run_tags.get(run)
            first_file_data, error = GraphUtils.safe_load_data(run, f"{first_run_tag}.vis")
            if error:
                return {'success': False, 'error': '获取配置信息失败,请检查目录中第一个文件'}
            first_file_data['Colors'] = colors
            config_data_run['colors'] = colors
            config_data[run] = config_data_run
            GraphState.set_global_value("config_data", config_data)
            GraphUtils.safe_save_data(first_file_data, run, f"{first_run_tag}.vis")
            return {'success': True, 'error': None, 'data': {}}
        except Exception as e:
            return {'success': False, 'error': str(e), 'data': None}

    @staticmethod
    def save_data(meta_data):
        if not meta_data:
            return {'success': False, 'error': '参数为空'}
        graph_data, error_message = GraphUtils.get_graph_data(meta_data)
        if error_message:
            return {'success': False, 'error': error_message}

        run = meta_data.get('run')
        tag = meta_data.get('tag')
        try:
            _, error = GraphUtils.safe_save_data(graph_data, run, f"{tag}.vis")
            if error:
                return {'success': False, 'error': error}
        except (ValueError, IOError, PermissionError) as e:
            return {'success': False, 'error': f"Error: {e}"}
        return {'success': True}

    @staticmethod
    def save_matched_relations(meta_data):
        if not meta_data:
            return {'success': False, 'error': '参数为空'}
        config_data = GraphState.get_global_value("config_data")
        # 匹配列表和未匹配列表
        npu_match_nodes_list = config_data.get('manualMatchNodes', {})
        run = meta_data.get('run')
        tag = meta_data.get('tag')
        try:
            _, error = GraphUtils.safe_save_data(npu_match_nodes_list, run, f"{tag}.vis.config")
            if error:
                return {'success': False, 'error': error}
        except (ValueError, IOError, PermissionError) as e:
            return {'success': False, 'error': f"Error: {e}"}
        return {'success': True, 'data': f"{tag}.vis.config"}
