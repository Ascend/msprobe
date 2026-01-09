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

from ..utils.constant import SINGLE, DataType
from .graph_repo_base import GraphRepo

JSON_TYPE = DataType.JSON.value


class GraphRepoVis(GraphRepo):
    
    def __init__(self, graph):
        self.graph = graph
        self.repo_type = JSON_TYPE
    
    # 查询根节点信息
    def query_root_nodes(self, graph_type, rank, step): 
        root_node = {}
        if graph_type == SINGLE:
            root_node_name = self.graph.get('root')
            root_node = self.graph.get('node', {}).get(root_node_name, {})
        else:
            root_node_name = self.graph.get(graph_type, {}).get('root')
            root_node = self.graph.get(graph_type, {}).get('node', {}).get(root_node_name, {})
        root_node['node_name'] = root_node_name
        return root_node

    # 查询所有以当前为父节点的子节点
    def query_sub_nodes(self, node_name, graph_type, rank, step):
        sub_nodes = {}
        graph_nodes = {}
        if graph_type == SINGLE:
            graph_nodes = self.graph.get('node', {})
        else:
            graph_nodes = self.graph.get(graph_type, {}).get('node', {})
        target_node = graph_nodes.get(node_name, {})
    
        target_node_children = target_node.get("subnodes", [])
        for subnode_name in target_node_children:
            node_info = graph_nodes.get(subnode_name, {})
            sub_nodes[subnode_name] = node_info
        return sub_nodes
    
    # 查询当前节点的父节点信息
    def query_up_nodes(self, node_name, graph_type, rank, step):
        graph_nodes = {}
        if graph_type == SINGLE:
            graph_nodes = self.graph.get('node', {})
        else:
            graph_nodes = self.graph.get(graph_type, {}).get('node', {})
        
        # 查询当前节点及其的所有父节点，一直到没有父节点位置{}
        up_nodes = {}
        up_nodes[node_name] = graph_nodes.get(node_name, {})
        parent_node_name = graph_nodes.get(node_name, {}).get("upnode")
        while graph_nodes.get(parent_node_name, None) is not None:
            parent_node = graph_nodes.get(parent_node_name, {})
            up_nodes[parent_node_name] = parent_node
            parent_node_name = parent_node.get("upnode")
        return up_nodes
    
    # 查询当前节点信息
    def query_node_info(self, node_name, graph_type):
        graph_nodes = {}
        if graph_type == SINGLE:
            graph_nodes = self.graph.get('node', {})
        else:
            graph_nodes = self.graph.get(graph_type, {}).get('node', {})
        return graph_nodes.get(node_name, {})
        
