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
import json
from pathlib import Path
from werkzeug import wrappers, Response, exceptions
from tensorboard.backend import http_util
from ..service import ServiceFactory, GraphServiceStrategy
from ..utils.file_check_wrapper import check_file_type
from ..utils.graph_utils import GraphUtils
from ..utils.constant import DataType
from ..utils.global_state import GraphState


class GraphView:
    service_factory = ServiceFactory()

    # 静态文件路由
    @staticmethod
    @wrappers.Request.application
    def static_file_route(request):
        filename = os.path.basename(request.path)

        extension = os.path.splitext(filename)[1]
        if extension == '.html':
            content_type = 'text/html'
        elif extension == '.js':
            content_type = 'application/javascript'
        else:
            content_type = 'application/octet-stream'

        try:
            # 添加白名单校验
            if filename != 'index.html' and filename != 'index.js':
                raise exceptions.NotFound('404 Not Found') from e
            current_dir = Path(__file__).resolve().parent
            server_dir = current_dir.parent.parent
            dir_path = (server_dir / "static").resolve()
            filepath = (server_dir / "static" / filename).resolve()  # resolve() 规范化路径
            if not GraphUtils.is_relative_to(filepath, dir_path):
                raise exceptions.NotFound('404 Not Found')
            # 前端打包后产生的内部输入文件，不需要要安全校验
            with open(filepath, 'rb') as infile:
                contents = infile.read()
        except IOError as e:
            raise exceptions.NotFound('404 Not Found') from e
        return Response(contents, content_type=content_type, headers={"X-Content-Type-Options": "nosniff"})

    @staticmethod
    @wrappers.Request.application
    def load_meta_dir(request):
        """Scan logdir for directories containing .vis files, modified to return a tuple of (run, tag)."""
        result = GraphServiceStrategy.load_meta_dir()
        response = http_util.Respond(request, result, "application/json")
        return response

    # 读取当前图数据
    @staticmethod
    @wrappers.Request.application
    def load_graph_data(request):
        meta_data = {
            'run': request.args.get('run'),
            'tag': request.args.get('tag'),
            'type': request.args.get('type'),
        }
        lang = request.args.get('lang')
        
        GraphState.set_global_value('lang', lang)
        strategy = GraphView._get_strategy(meta_data)
        if meta_data.get('type') == DataType.DB.value:
            result = strategy.load_graph_data()
            return http_util.Respond(request, result, "application/json")
        elif meta_data.get('type') == DataType.JSON.value:
            return Response(
                strategy.load_graph_data(),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'close',  # TCP链接不复用，请求结束释放资源
                    "X-Content-Type-Options": "nosniff",
                }
            )
        else: 
            result = {'success': False, 'error': GraphUtils.t('typeError')}
            return http_util.Respond(request, result, "application/json")

    # 获取当前图数据配置信息
    @staticmethod
    @wrappers.Request.application
    @check_file_type
    def load_graph_config_info(request):
        data = GraphUtils.safe_json_loads(request.get_data().decode('utf-8'), {})
        meta_data = data.get('metaData')
        strategy = GraphView._get_strategy(meta_data)
        result = strategy.load_graph_config_info()
        # 创建响应对象
        response = http_util.Respond(request, result, "application/json")
        return response

    # 获取当前图所有节点列表
    @staticmethod
    @wrappers.Request.application
    @check_file_type
    def load_graph_all_node_list(request):
        data = GraphUtils.safe_json_loads(request.get_data().decode('utf-8'), {})
        meta_data = data.get('metaData')
        strategy = GraphView._get_strategy(meta_data)
        result = strategy.load_graph_all_node_list(meta_data)
        response = http_util.Respond(request, result, "application/json")
        return response
    
    # 根据精度误差搜索节点
    @staticmethod
    @wrappers.Request.application
    @check_file_type
    def search_node(request):
        data = GraphUtils.safe_json_loads(request.get_data().decode('utf-8'))
        meta_data = data.get("metaData")
        search_type = data.get("type")
        values = data.get("values")
        strategy = GraphView._get_strategy(meta_data)
        if search_type == 'precision':
            result = strategy.search_node_by_precision(meta_data, values)
        elif search_type == 'overflow':
            result = strategy.search_node_by_overflow(meta_data, values)
        else:
            result = {
                'success': False,
                'error': GraphUtils.t('searchTypeError')
            }
        return http_util.Respond(request, result, "application/json") 
    
    # 更新误差节点
    @staticmethod
    @wrappers.Request.application
    @check_file_type
    def update_precision_error(request):
        data = GraphUtils.safe_json_loads(request.get_data().decode('utf-8'))
        meta_data = data.get('metaData')
        filter_value = data.get("filterValue")
        strategy = GraphView._get_strategy(meta_data)
        result = strategy.update_precision_error(meta_data, filter_value)
        return http_util.Respond(request, result, "application/json")

    # 展开关闭节点
    @staticmethod
    @wrappers.Request.application
    @check_file_type
    def change_node_expand_state(request):
        data = GraphUtils.safe_json_loads(request.get_data().decode('utf-8'), {})
        node_info = GraphUtils.safe_get_node_info(data)
        result = {'success': False, 'error': ''}
        meta_data = data.get('metaData')
        if node_info is None or not isinstance(node_info, dict):
            result['error'] = GraphUtils.t('nodeInfoError') 
        strategy = GraphView._get_strategy(meta_data)
        hierarchy = strategy.change_node_expand_state(node_info, meta_data)
        return http_util.Respond(request, json.dumps(hierarchy), "application/json")

    # 更新当前图节点信息
    @staticmethod
    @wrappers.Request.application
    @check_file_type
    def update_hierarchy_data(request):
        data = GraphUtils.safe_json_loads(request.get_data().decode('utf-8'), {})
        graph_type = data.get("graphType")
        meta_data = data.get('metaData')
        strategy = GraphView._get_strategy(meta_data)
        hierarchy = strategy.update_hierarchy_data(graph_type)
        return http_util.Respond(request, json.dumps(hierarchy), "application/json")

    # 获取当前节点对应节点的信息看板数据
    @staticmethod
    @wrappers.Request.application
    @check_file_type
    def get_node_info(request):
        data = GraphUtils.safe_json_loads(request.get_data().decode('utf-8'), {})
        node_info = GraphUtils.safe_get_node_info(data)
        result = {'success': False, 'error': ''}
        meta_data = data.get('metaData')
        if node_info is None or not isinstance(node_info, dict):
            result['error'] = GraphUtils.t('nodeInfoError') 
            return http_util.Respond(request, result, "application/json")
        
        strategy = GraphView._get_strategy(meta_data)
        node_detail = strategy.get_node_info(node_info, meta_data)
        return http_util.Respond(request, json.dumps(node_detail), "application/json")

    # 根据配置文件添加匹配节点
    @staticmethod
    @wrappers.Request.application
    @check_file_type
    def add_match_nodes_by_config(request):
        data = GraphUtils.safe_json_loads(request.get_data().decode('utf-8'), {})
        config_file = data.get('configFile')
        meta_data = data.get('metaData')
        strategy = GraphView._get_strategy(meta_data)
        match_result = strategy.add_match_nodes_by_config(config_file, meta_data)
        return http_util.Respond(request, json.dumps(match_result), "application/json")

    # 添加匹配节点
    @staticmethod
    @wrappers.Request.application
    @check_file_type
    def add_match_nodes(request):
        data = GraphUtils.safe_json_loads(request.get_data().decode('utf-8'), {})
        npu_node_name = data.get("npuNodeName")
        bench_node_name = data.get("benchNodeName")
        meta_data = data.get('metaData')
        is_match_children = data.get("isMatchChildren")
        strategy = GraphView._get_strategy(meta_data)
        match_result = strategy.add_match_nodes(npu_node_name, bench_node_name, meta_data, is_match_children)
        return http_util.Respond(request, json.dumps(match_result), "application/json")

    # 取消节点匹配
    @staticmethod
    @wrappers.Request.application
    @check_file_type
    def delete_match_nodes(request):
        data = GraphUtils.safe_json_loads(request.get_data().decode('utf-8'), {})
        npu_node_name = data.get("npuNodeName")
        bench_node_name = data.get("benchNodeName")
        meta_data = data.get('metaData')
        is_unmatch_children = data.get("isUnMatchChildren")
        strategy = GraphView._get_strategy(meta_data)
        match_result = strategy.delete_match_nodes(npu_node_name, bench_node_name, meta_data, is_unmatch_children)
        return http_util.Respond(request, json.dumps(match_result), "application/json")

    # 保存匹配节点列表
    @staticmethod
    @wrappers.Request.application
    @check_file_type
    def save_data(request):
        data = GraphUtils.safe_json_loads(request.get_data().decode('utf-8'), {})
        meta_data = data.get('metaData')
        strategy = GraphView._get_strategy(meta_data)
        save_result = strategy.save_data(meta_data)
        return http_util.Respond(request, json.dumps(save_result), "application/json")

    # 更新颜色信息
    @staticmethod
    @wrappers.Request.application
    @check_file_type
    def update_colors(request):
        data = GraphUtils.safe_json_loads(request.get_data().decode('utf-8'), {})
        meta_data = data.get('metaData')
        colors = GraphUtils.safe_json_loads(data.get('colors'))
        success, error_msg, colors = GraphUtils.validate_colors_param(colors)
        if not success:
            result = {
                'success': False,
                'error': error_msg
            }
            return http_util.Respond(request, json.dumps(result), "application/json")
        strategy = GraphView._get_strategy(meta_data, no_tag=True)
        update_result = strategy.update_colors(colors)
        return http_util.Respond(request, json.dumps(update_result), "application/json")

    # 保存匹配关系
    @staticmethod
    @wrappers.Request.application
    @check_file_type
    def save_matched_relations(request):
        data = GraphUtils.safe_json_loads(request.get_data().decode('utf-8'), {})
        meta_data = data.get('metaData')
        strategy = GraphView._get_strategy(meta_data)
        save_result = strategy.save_matched_relations(meta_data)
        return http_util.Respond(request, json.dumps(save_result), "application/json")

    @staticmethod
    def _get_strategy(meta_data, no_tag=False):
        run = meta_data.get('run')
        data_type = meta_data.get('type')
        if no_tag:
            return GraphView.service_factory.create_strategy_without_tag(data_type, run)
        
        tag = meta_data.get('tag')
        return GraphView.service_factory.create_strategy(data_type, run, tag)
