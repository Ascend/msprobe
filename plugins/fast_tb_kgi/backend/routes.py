# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#         http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# --------------------------------------------------------------------------------------------#

import os
import json
from werkzeug import wrappers, Response, exceptions
from pathlib import Path
from backend.kgi.api.kgi_for_tb import KGI_For_Tb

FILE_WHITE_LIST = ['main.js', 'index.html', 'index.js', 'index.css']

def get_content_type(extension: str) -> str:
    if extension == '.html':
        return 'text/html'
    elif extension == '.js':
        return 'application/javascript'
    elif extension == '.css':
        return 'text/css'
    else:
        return 'application/octet-stream'  # 二进制流

# 白名单校验
def white_list_verify(filename: str) -> bool:
    if filename in FILE_WHITE_LIST:
        return True
    return False

# 防路径遍历攻击
def path_traversal_verify(file_path: Path, dir_path: Path) -> bool:
    abs_file_path = os.path.abspath(file_path)
    abs_dir_path = os.path.abspath(dir_path)
    return os.path.commonpath([abs_file_path, abs_dir_path]) == str(abs_dir_path)

# 添加安全头部以满足TensorBoard 3.0要求
def get_secure_headers(content_type: str) -> dict:
    return {
        "X-Content-Type-Options": "nosniff",
        "Content-Security-Policy": "default-src 'self'; \
                                    script-src 'self' 'unsafe-eval'; \
                                    style-src 'self' 'unsafe-inline'",
        "Content-Type": content_type
    }

class KgiRouter:
    kgi = KGI_For_Tb()

    # 静态文件路由
    @staticmethod
    @wrappers.Request.application
    def static_file_route(request: wrappers.Request):
        filename = os.path.basename(request.path)
        extension = os.path.splitext(filename)[1]
        content_type = get_content_type(extension)

        try:
            if not white_list_verify(filename):
                raise exceptions.NotFound('404 Not Found')
            current_dir = Path(__file__).resolve().parent
            dir_path = (current_dir / "static").resolve()
            file_path = (current_dir / "static" / filename).resolve()
            if not path_traversal_verify(file_path, dir_path):
                raise exceptions.NotFound('404 Not Found')
            with open(file_path, 'rb') as f:
                contents = f.read()
        except Exception as e:
            raise exceptions.NotFound('404 Not Found') from e

        headers = get_secure_headers(content_type)
        return Response(contents, headers=headers)

    @staticmethod
    @wrappers.Request.application
    def get_controls_info(request: wrappers.Request):
        controls_info = KgiRouter.kgi.get_controls_info()
        response_data = json.dumps(controls_info)
        headers = get_secure_headers("application/json")
        return Response(response_data, headers=headers)

    # 获取图
    @staticmethod
    @wrappers.Request.application
    def get_graph(request: wrappers.Request):
        data = request.json
        if 'side' not in data:
            response_data = json.dumps({'error': '没有指定侧边'})
            return Response(response_data, status=400, content_type='application/json')
        side = data["side"]

        nodes, edges = KgiRouter.kgi.get_graph(side)

        response_data = json.dumps({
            'nodes': nodes,
            'edges': edges
        })
        headers = get_secure_headers("application/json")
        return Response(response_data, headers=headers)

    @staticmethod
    @wrappers.Request.application
    def change_compare_mode(request: wrappers.Request):
        left_nodes, right_nodes, compare_all_mode = KgiRouter.kgi.change_compare_mode()
        response_data = json.dumps({
            'left_nodes': left_nodes,
            'right_nodes': right_nodes,
            'compare_all_mode': compare_all_mode
        })
        headers = get_secure_headers("application/json")
        return Response(response_data, headers=headers)

    # 设置图
    @staticmethod
    @wrappers.Request.application
    def set_graph(request: wrappers.Request):
        if 'file' not in request.files:
            response_data = json.dumps({'error': '没有上传文件'})
            return Response(response_data, status=400, content_type='application/json')
        file = request.files['file']

        if file.filename == '':
            response_data = json.dumps({'error': '文件名为空'})
            return Response(response_data, status=400, content_type='application/json')

        if 'side' not in request.form:
            response_data = json.dumps({'error': '没有指定侧边'})
            return Response(response_data, status=400, content_type='application/json')
        side = request.form['side']

        if 'ignore_data_ops' not in request.form:
            response_data = json.dumps({'error': '没有指定是否忽略data算子'})
            return Response(response_data, status=400, content_type='application/json')
        ignore_data_ops = True if request.form['ignore_data_ops'] == 'true' else False

        if 'execute_order_type' not in request.form:
            response_data = json.dumps({'error': '没有指定执行序类型'})
            return Response(response_data, status=400, content_type='application/json')
        execute_order_type = request.form['execute_order_type']

        file_content: bytes = getattr(file, 'read')()
        content = file_content.decode('utf-8')
        left_nodes, left_edges, right_nodes, right_edges = KgiRouter.kgi.set_graph(
            side, content, ignore_data_ops, execute_order_type)

        if side == 'left':
            response_data = json.dumps({
                'left_nodes': left_nodes,
                'left_edges': left_edges,
                'right_nodes': right_nodes
            })
        else:
            response_data = json.dumps({
                'left_nodes': left_nodes,
                'right_nodes': right_nodes,
                'right_edges': right_edges
            })
        headers = get_secure_headers("application/json")
        return Response(response_data, headers=headers)

    # 切换至整图
    @staticmethod
    @wrappers.Request.application
    def change_to_whole_graph(request: wrappers.Request):
        left_nodes, left_edges, right_nodes, right_edges = KgiRouter.kgi.change_to_whole_graph()

        response_data = json.dumps({
            'left_nodes': left_nodes,
            'left_edges': left_edges,
            'right_nodes': right_nodes,
            'right_edges': right_edges
        })
        headers = get_secure_headers("application/json")
        return Response(response_data, headers=headers)

    # 设置锚点预检查
    @staticmethod
    @wrappers.Request.application
    def set_anchor_pre_check(request: wrappers.Request):
        data = request.json
        if 'side' not in data:
            response_data = json.dumps({'error': '没有指定侧边'})
            return Response(response_data, status=400, content_type='application/json')
        side = data['side']

        if 'line_id' not in data:
            response_data = json.dumps({'error': '没有指定锚点行号'})
            return Response(response_data, status=400, content_type='application/json')
        line_id = int(data['line_id'])

        pre_check_res = KgiRouter.kgi.set_anchor_pre_check(side, line_id)

        response_data = json.dumps(pre_check_res)
        headers = get_secure_headers("application/json")
        return Response(response_data, headers=headers)

    @staticmethod
    @wrappers.Request.application
    def set_anchor(request: wrappers.Request):
        data = request.json
        if 'side' not in data:
            response_data = json.dumps({'error': '没有指定侧边'})
            return Response(response_data, status=400, content_type='application/json')
        side = data['side']

        if 'line_id' not in data:
            response_data = json.dumps({'error': '没有指定锚点行号'})
            return Response(response_data, status=400, content_type='application/json')
        line_id = int(data['line_id'])

        left_nodes, left_edges, right_nodes, right_edges = KgiRouter.kgi.set_anchor(side, line_id)

        if side == 'left':
            response_data = json.dumps({
                'left_nodes': left_nodes,
                'left_edges': left_edges,
                'right_nodes': right_nodes
            })
        else:
            response_data = json.dumps({
                'left_nodes': left_nodes,
                'right_nodes': right_nodes,
                'right_edges': right_edges
            })
        headers = get_secure_headers("application/json")
        return Response(response_data, headers=headers)

    @staticmethod
    @wrappers.Request.application
    def up_compare(request: wrappers.Request):
        left_nodes, right_nodes = KgiRouter.kgi.up_compare()

        response_data = json.dumps({
            'left_nodes': left_nodes,
            'right_nodes': right_nodes
        })
        headers = get_secure_headers("application/json")
        return Response(response_data, headers=headers)

    @staticmethod
    @wrappers.Request.application
    def down_compare(request: wrappers.Request):
        left_nodes, right_nodes = KgiRouter.kgi.down_compare()

        response_data = json.dumps({
            'left_nodes': left_nodes,
            'right_nodes': right_nodes
        })
        headers = get_secure_headers("application/json")
        return Response(response_data, headers=headers)

    @staticmethod
    @wrappers.Request.application
    def replace_equal_subgraph_pre_check(request: wrappers.Request):
        data = request.json
        if 'side' not in data:
            response_data = json.dumps({'error': '没有指定侧边'})
            return Response(response_data, status=400, content_type='application/json')
        side = data['side']

        if 'nodes_id' not in data:
            response_data = json.dumps({'error': '没有指定子图节点id'})
            return Response(response_data, status=400, content_type='application/json')
        nodes_id = data["nodes_id"]

        pre_check_res = KgiRouter.kgi.replace_equal_subgraph_pre_check(side, nodes_id)

        response_data = json.dumps(pre_check_res)
        headers = get_secure_headers("application/json")
        return Response(response_data, headers=headers)

    @staticmethod
    @wrappers.Request.application
    def replace_equal_subgraph(request: wrappers.Request):
        data = request.json
        if 'left_nodes_id' not in data:
            response_data = json.dumps({'error': '没有指定左图等价子图节点id'})
            return Response(response_data, status=400, content_type='application/json')
        left_nodes_id = data["left_nodes_id"]

        if 'right_nodes_id' not in data:
            response_data = json.dumps({'error': '没有指定右图等价子图节点id'})
            return Response(response_data, status=400, content_type='application/json')
        right_nodes_id = data["right_nodes_id"]

        (left_del_nodes, left_nodes,
         left_del_edges, left_add_edges,
         right_del_nodes, right_nodes,
         right_del_edges, right_add_edges) = KgiRouter.kgi.replace_equal_subgraph(left_nodes_id, right_nodes_id)

        response_data = json.dumps({
            'left_del_nodes': left_del_nodes, 'left_nodes': left_nodes,
            'left_del_edges': left_del_edges, 'left_add_edges': left_add_edges,
            'right_del_nodes': right_del_nodes, 'right_nodes': right_nodes,
            'right_del_edges': right_del_edges, 'right_add_edges': right_add_edges
        })
        headers = get_secure_headers("application/json")
        return Response(response_data, headers=headers)

    @staticmethod
    @wrappers.Request.application
    def del_nodes_pre_check(request: wrappers.Request):
        data = request.json
        if 'side' not in data:
            response_data = json.dumps({'error': '没有指定侧边'})
            return Response(response_data, status=400, content_type='application/json')
        side = data['side']

        if 'nodes_id' not in data:
            response_data = json.dumps({'error': '没有指定节点id'})
            return Response(response_data, status=400, content_type='application/json')
        nodes_id = data["nodes_id"]

        pre_check_res = KgiRouter.kgi.del_nodes_pre_check(side, nodes_id)

        response_data = json.dumps(pre_check_res)
        headers = get_secure_headers("application/json")
        return Response(response_data, headers=headers)

    @staticmethod
    @wrappers.Request.application
    def del_nodes(request: wrappers.Request):
        data = request.json
        if 'left_nodes_id' not in data:
            response_data = json.dumps({'error': '没有指定左图等价子图节点id'})
            return Response(response_data, status=400, content_type='application/json')
        left_nodes_id = data["left_nodes_id"]

        if 'right_nodes_id' not in data:
            response_data = json.dumps({'error': '没有指定右图等价子图节点id'})
            return Response(response_data, status=400, content_type='application/json')
        right_nodes_id = data["right_nodes_id"]

        (left_del_nodes, left_nodes, left_del_edges,
         right_del_nodes, right_nodes, right_del_edges) = KgiRouter.kgi.del_nodes(left_nodes_id, right_nodes_id)

        response_data = json.dumps({
            'left_del_nodes': left_del_nodes, 'left_nodes': left_nodes, 'left_del_edges': left_del_edges,
            'right_del_nodes': right_del_nodes, 'right_nodes': right_nodes, 'right_del_edges': right_del_edges
        })
        headers = get_secure_headers("application/json")
        return Response(response_data, headers=headers)

    @staticmethod
    @wrappers.Request.application
    def del_edges(request: wrappers.Request):
        data = request.json
        if 'left_edges' not in data:
            response_data = json.dumps({'error': '没有指定左图边'})
            return Response(response_data, status=400, content_type='application/json')
        left_edges = data["left_edges"]

        if 'right_edges' not in data:
            response_data = json.dumps({'error': '没有指定右图边'})
            return Response(response_data, status=400, content_type='application/json')
        right_edges = data["right_edges"]

        (left_del_nodes, left_nodes, left_del_edges,
         right_del_nodes, right_nodes, right_del_edges) = KgiRouter.kgi.del_edges(left_edges, right_edges)

        response_data = json.dumps({
            'left_del_nodes': left_del_nodes, 'left_nodes': left_nodes, 'left_del_edges': left_del_edges,
            'right_del_nodes': right_del_nodes, 'right_nodes': right_nodes, 'right_del_edges': right_del_edges
        })
        headers = get_secure_headers("application/json")
        return Response(response_data, headers=headers)

    @staticmethod
    @wrappers.Request.application
    def set_second_level_anchor(request: wrappers.Request):
        data = request.json
        if 'left_node_id' not in data:
            response_data = json.dumps({'error': '没有指定左图节点id'})
            return Response(response_data, status=400, content_type='application/json')
        left_node_id = data["left_node_id"]

        if 'right_node_id' not in data:
            response_data = json.dumps({'error': '没有指定右图节点id'})
            return Response(response_data, status=400, content_type='application/json')
        right_node_id = data["right_node_id"]

        left_nodes, right_nodes = KgiRouter.kgi.set_second_level_anchor(left_node_id, right_node_id)

        response_data = json.dumps({
            'left_nodes': left_nodes,
            'right_nodes': right_nodes
        })
        headers = get_secure_headers("application/json")
        return Response(response_data, headers=headers)

    @staticmethod
    @wrappers.Request.application
    def del_second_level_anchor_pre_check(request: wrappers.Request):
        data = request.json
        if 'left_nodes_id' not in data:
            response_data = json.dumps({'error': '没有指定左图等价子图节点id'})
            return Response(response_data, status=400, content_type='application/json')
        left_nodes_id = data["left_nodes_id"]

        if 'right_nodes_id' not in data:
            response_data = json.dumps({'error': '没有指定右图等价子图节点id'})
            return Response(response_data, status=400, content_type='application/json')
        right_nodes_id = data["right_nodes_id"]

        pre_check_res = KgiRouter.kgi.del_second_level_anchor_pre_check(left_nodes_id, right_nodes_id)

        response_data = json.dumps(pre_check_res)
        headers = get_secure_headers("application/json")
        return Response(response_data, headers=headers)

    @staticmethod
    @wrappers.Request.application
    def del_second_level_anchor(request: wrappers.Request):
        data = request.json
        if 'left_nodes_id' not in data:
            response_data = json.dumps({'error': '没有指定左图等价子图节点id'})
            return Response(response_data, status=400, content_type='application/json')
        left_nodes_id = data["left_nodes_id"]

        if 'right_nodes_id' not in data:
            response_data = json.dumps({'error': '没有指定右图等价子图节点id'})
            return Response(response_data, status=400, content_type='application/json')
        right_nodes_id = data["right_nodes_id"]

        left_nodes, right_nodes = KgiRouter.kgi.del_second_level_anchor(left_nodes_id, right_nodes_id)

        response_data = json.dumps({
            'left_nodes': left_nodes,
            'right_nodes': right_nodes
        })
        headers = get_secure_headers("application/json")
        return Response(response_data, headers=headers)

    @staticmethod
    @wrappers.Request.application
    def get_node_info(request: wrappers.Request):
        data = request.json
        if 'side' not in data:
            response_data = json.dumps({'error': '没有指定侧边'})
            return Response(response_data, status=400, content_type='application/json')
        side = data["side"]

        if 'node_id' not in data:
            response_data = json.dumps({'error': '没有指定节点id'})
            return Response(response_data, status=400, content_type='application/json')
        node_id = data["node_id"]

        node_info = KgiRouter.kgi.get_node_info(side, node_id)

        response_data = json.dumps(node_info)
        headers = get_secure_headers("application/json")
        return Response(response_data, headers=headers)