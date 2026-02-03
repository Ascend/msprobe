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
from ..utils.constant import security_headers


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
        return Response(contents, content_type=content_type, headers=security_headers)
    

    @staticmethod
    @wrappers.Request.application
    def load_meta_dir(request):
        """Scan logdir for directories containing .vis files, modified to return a tuple of (run, tag)."""
        result = GraphServiceStrategy.load_meta_dir()
        response = http_util.Respond(request, result, "application/json")
        return response
    
    @staticmethod
    @wrappers.Request.application
    def load_converted_graph_data(request):
        logdir = GraphState.get_global_value('logdir')
        result = GraphServiceStrategy.load_converted_graph_data(logdir)  
        response = http_util.Respond(request, result, "application/json")
        return response
    
    @staticmethod
    @wrappers.Request.application
    def convert_to_graph(request):
        data = GraphUtils.safe_json_loads(request.get_data().decode('utf-8'), {})
        npu_path = data.get('npu_path')
        bench_path = data.get('bench_path')
        is_print_compare_log = data.get('is_print_compare_log',False)
        parallel_merge = data.get('parallel_merge')
        layer_mapping = data.get('layer_mapping')
        overflow_check = data.get('overflow_check',False)
        fuzzy_match = data.get('fuzzy_match',False)
        # ## 入参校验
        result = {'success': True}
        logdir = GraphState.get_global_value('logdir')
        # npu_path必须存在，且必须通过安全路径校验
        abs_npu_path = os.path.join(logdir, npu_path)
        npu_path_success, _ = GraphUtils.safe_check_load_file_path(abs_npu_path, True)
        if not npu_path_success:
            result = {'success': False, 'error': f'npu_path {GraphUtils.t("convertParamsError")}'}
        # bench_path可以为''，如果不为空，必须通过安全路径校验
        if bench_path:
            abs_bench_path = os.path.join(logdir, bench_path)
            bench_path_success, _ = GraphUtils.safe_check_load_file_path(abs_bench_path, True)
            if not bench_path_success:
                result = {'success': False, 'error':f'bench_path {GraphUtils.t("convertParamsError")}'}
        # is_print_compare_log必须为布尔值
        if not isinstance(is_print_compare_log, bool):
            result = {'success': False, 'error':f'is_print_compare_log {GraphUtils.t("convertParamsError")}'}
        # bench_path可以为空，如果不为空，parallel_merge必须为字典
        if parallel_merge:
            if not isinstance(parallel_merge, dict):
                result = {'success': False, 'error': GraphUtils.t('convertParamsError')}
            npu_parallel_merge = parallel_merge.get('npu')
            bench_parallel_merge = parallel_merge.get('bench') 
            # npu_parallel_merge必须存在pp rank_size  tp vpp
            if npu_parallel_merge:
                if not isinstance(npu_parallel_merge, dict):
                    result = {'success': False, 'error':f'parallel_merge {GraphUtils.t("convertParamsError")}'} 
                required_keys = {'pp', 'rank_size', 'tp', 'vpp'}
                missing_keys = required_keys - set(npu_parallel_merge.keys())
                if missing_keys:
                    result = {'success': False, 'error':f'parallel_merge {GraphUtils.t("convertParamsError")}'}
                if npu_parallel_merge['order'] and not GraphUtils.is_safe_string(npu_parallel_merge['order']):
                    result = {'success': False, 'error': f'parallel_merge.order {GraphUtils.t("convertParamsError")}'}
                data['parallel_param_n'] = npu_parallel_merge
            else: 
                result = {'success': False, 'error': f'parallel_merge {GraphUtils.t("convertParamsError")}'}
            
            if bench_parallel_merge:
                if not isinstance(bench_parallel_merge, dict):
                    result = {'success': False, 'error':f'parallel_merge {GraphUtils.t("convertParamsError")}'} 
                required_keys = {'pp', 'rank_size', 'tp', 'vpp'}
                missing_keys = required_keys - set(bench_parallel_merge.keys())
                if missing_keys:
                    result = {'success': False, 'error':f'parallel_merge {GraphUtils.t("convertParamsError")}'}
                if bench_parallel_merge['order'] and not GraphUtils.is_safe_string(bench_parallel_merge['order']):
                    result = {'success': False, 'error': f'parallel_merge.order {GraphUtils.t("convertParamsError")}'}
                data['parallel_param_b'] = bench_parallel_merge
        # layer_mapping可以为''，如果不为空，必须通过安全路径校验
        if layer_mapping:
            abs_layer_mapping = os.path.join(logdir, layer_mapping)
            layer_mapping_success, _ = GraphUtils.safe_check_load_file_path(abs_layer_mapping)
            if not layer_mapping_success:
                result = {'success': False, 'error': f'layer_mapping {GraphUtils.t("convertParamsError")}'}
        # overflow_check必须为布尔值
        if not isinstance(overflow_check, bool):
            result = {'success': False, 'error': f'overflow_check {GraphUtils.t("convertParamsError")}'}
        # fuzzy_match必须为布尔值
        if not isinstance(fuzzy_match, bool):
            result = {'success': False, 'error': f'fuzzy_match {GraphUtils.t("convertParamsError")}'}
        if result.get('success'):
            data['npu_path'] = abs_npu_path
            data['bench_path'] = abs_bench_path if bench_path else ''
            data['layer_mapping'] = abs_layer_mapping if layer_mapping else ''
            data['output_path'] = os.path.join(logdir, data['output_path'] )
            data['is_print_compare_log'] = is_print_compare_log
            data['overflow_check'] = overflow_check
            data['fuzzy_match'] = fuzzy_match
            result = GraphServiceStrategy.convert_to_graph(data)
      
        response = http_util.Respond(request, result, "application/json")
        return response
    
    @staticmethod
    @wrappers.Request.application
    def get_convert_progress(request):
        return Response(
            GraphServiceStrategy.get_convert_progress(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'close',  # TCP链接不复用，请求结束释放资源
                "X-Content-Type-Options": "nosniff",
            }
        )
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
    
    # 获取当前图所有节点匹配关系
    @staticmethod
    @wrappers.Request.application
    @check_file_type
    def load_graph_matched_relations(request):
        data = GraphUtils.safe_json_loads(request.get_data().decode('utf-8'), {})
        meta_data = data.get('metaData')
        strategy = GraphView._get_strategy(meta_data)
        result = strategy.load_graph_matched_relations(meta_data)
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
