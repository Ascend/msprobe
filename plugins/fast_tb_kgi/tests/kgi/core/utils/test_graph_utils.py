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

import networkx as nx
from backend.kgi.core.structs.node import Node
from backend.kgi.core.utils.graph_utils import get_bfs_layers, get_isolated_nodes, get_nodes_degree, get_opt_topo_layers
from backend.kgi.core.utils.graph_utils import get_nodes_in_degree, get_nodes_out_degree
from backend.kgi.core.utils.graph_utils import get_subgraph_with_anchor, get_topo_layers

class TestGraphUtils:
    def test_get_nodes_degree(self):
        # 添加节点
        node1 = Node()
        node1.set_node_id(1)
        node2 = Node()
        node2.set_node_id(2)
        node3 = Node()
        node3.set_node_id(3)

        graph = nx.DiGraph()

        # 添加边
        graph.add_edge(node1, node2)
        graph.add_edge(node1, node3)

        nodes_degree = get_nodes_degree(graph)

        assert nodes_degree[node1] == 2
        assert nodes_degree[node2] == 1
        assert nodes_degree[node3] == 1

    def test_get_nodes_out_degree(self):
        # 添加节点
        node1 = Node()
        node1.set_node_id(1)
        node2 = Node()
        node2.set_node_id(2)
        node3 = Node()
        node3.set_node_id(3)

        graph = nx.DiGraph()

        # 添加边
        graph.add_edge(node1, node2)
        graph.add_edge(node1, node3)

        nodes_out_degree = get_nodes_out_degree(graph)

        assert nodes_out_degree[node1] == 2
        assert nodes_out_degree[node2] == 0
        assert nodes_out_degree[node3] == 0

    def test_get_nodes_in_degree(self):
        # 添加节点
        node1 = Node()
        node1.set_node_id(1)
        node2 = Node()
        node2.set_node_id(2)
        node3 = Node()
        node3.set_node_id(3)

        graph = nx.DiGraph()

        # 添加边
        graph.add_edge(node1, node2)
        graph.add_edge(node1, node3)

        nodes_in_degree = get_nodes_in_degree(graph)

        assert nodes_in_degree[node1] == 0
        assert nodes_in_degree[node2] == 1
        assert nodes_in_degree[node3] == 1

    def test_get_isolated_nodes(self):
        graph = nx.DiGraph()

        # 添加孤立节点
        isolated_node = Node()
        isolated_node.set_node_id(1)
        graph.add_node(isolated_node)

        # 添加连接节点
        connected_node2 = Node()
        connected_node2.set_node_id(2)
        connected_node3 = Node()
        connected_node3.set_node_id(3)
        graph.add_node(connected_node2)
        graph.add_node(connected_node3)
        graph.add_edge(connected_node2, connected_node3)

        # 获取孤立节点
        isolated_nodes = get_isolated_nodes(graph)

        # 只有isolated_node是孤立节点
        assert isolated_node in isolated_nodes
        assert connected_node2 not in isolated_nodes
        assert connected_node3 not in isolated_nodes

    def test_get_subgraph_with_none_anchor(self):
        # anchor为None
        graph = nx.DiGraph()
        node1 = Node()
        node1.set_node_id(1)
        graph.add_node(node1)
        node2 = Node()
        node2.set_node_id(2)
        graph.add_node(node2)

        # 当anchor为None时，应该返回整个图
        subgraph = get_subgraph_with_anchor(graph, None, False)
        assert subgraph == graph

    def test_get_subgraph_grow_down(self):
        # 向下生长
        graph = nx.DiGraph()

        # 添加节点
        node1 = Node()
        node1.set_node_id(1)
        graph.add_node(node1)
        node2 = Node()
        node2.set_node_id(2)
        graph.add_node(node2)
        node3 = Node()
        node3.set_node_id(3)
        graph.add_node(node3)
        node4 = Node()
        node4.set_node_id(4)
        graph.add_node(node4)

        # 添加边
        graph.add_edge(node4, node1)
        graph.add_edge(node1, node2)
        graph.add_edge(node1, node3)

        sub_graph = get_subgraph_with_anchor(graph, node1, False)

        assert sub_graph.number_of_nodes() == 3
        assert node1 in sub_graph.nodes
        assert node2 in sub_graph.nodes
        assert node3 in sub_graph.nodes
        assert node4 not in sub_graph.nodes

    def test_get_subgraph_grow_up(self):
        # 向上生长
        graph = nx.DiGraph()

        # 添加节点
        node1 = Node()
        node1.set_node_id(1)
        graph.add_node(node1)
        node2 = Node()
        node2.set_node_id(2)
        graph.add_node(node2)
        node3 = Node()
        node3.set_node_id(3)
        graph.add_node(node3)
        node4 = Node()
        node4.set_node_id(4)
        graph.add_node(node4)

        # 添加边
        graph.add_edge(node1, node3)
        graph.add_edge(node2, node3)
        graph.add_edge(node3, node4)

        sub_graph = get_subgraph_with_anchor(graph, node3, True)

        assert sub_graph.number_of_nodes() == 3
        assert node1 in sub_graph.nodes
        assert node2 in sub_graph.nodes
        assert node3 in sub_graph.nodes
        assert node4 not in sub_graph.nodes

    def test_get_topo_layers(self):
        graph = nx.DiGraph()

        # 添加节点
        node1 = Node()
        node1.set_node_id(1)
        node2 = Node()
        node2.set_node_id(2)
        node3 = Node()
        node3.set_node_id(3)
        node4 = Node()
        node4.set_node_id(4)
        node5 = Node()
        node5.set_node_id(5)
        node6 = Node()
        node6.set_node_id(6)
        node7 = Node()
        node7.set_node_id(7)

        # 添加边
        graph.add_edge(node1, node2)
        graph.add_edge(node2, node4)
        graph.add_edge(node3, node4)
        graph.add_edge(node4, node5)
        graph.add_edge(node4, node6)
        graph.add_edge(node5, node7)


        topo_layers, _ = get_topo_layers(graph)

        assert topo_layers == [
            {node6, node7},
            {node5},
            {node4},
            {node2, node3},
            {node1}
        ]

    def test_get_opt_topo_layers(self):
        graph = nx.DiGraph()

        # 添加节点
        node1 = Node()
        node1.set_node_id(1)
        node2 = Node()
        node2.set_node_id(2)
        node3 = Node()
        node3.set_node_id(3)
        node4 = Node()
        node4.set_node_id(4)
        node5 = Node()
        node5.set_node_id(5)
        node6 = Node()
        node6.set_node_id(6)
        node7 = Node()
        node7.set_node_id(7)

        # 添加边
        graph.add_edge(node1, node2)
        graph.add_edge(node2, node4)
        graph.add_edge(node3, node4)
        graph.add_edge(node4, node5)
        graph.add_edge(node4, node6)
        graph.add_edge(node5, node7)


        topo_layers = get_opt_topo_layers(graph)

        assert topo_layers == [
            {node7},
            {node5, node6},
            {node4},
            {node2, node3},
            {node1}
        ]

    def test_get_bfs_layers(self):
        graph = nx.DiGraph()

        # 添加节点
        node1 = Node()
        node1.set_node_id(1)
        graph.add_node(node1)
        node2 = Node()
        node2.set_node_id(2)
        graph.add_node(node2)
        node3 = Node()
        node3.set_node_id(3)
        graph.add_node(node3)
        node4 = Node()
        node4.set_node_id(4)
        graph.add_node(node4)
        node5 = Node()
        node5.set_node_id(5)
        graph.add_node(node5)
        node6 = Node()
        node6.set_node_id(6)
        graph.add_node(node6)
        node7 = Node()
        node7.set_node_id(7)
        graph.add_node(node7)

        # 添加边
        graph.add_edge(node1, node2)
        graph.add_edge(node2, node4)
        graph.add_edge(node3, node4)
        graph.add_edge(node4, node5)
        graph.add_edge(node4, node6)
        graph.add_edge(node5, node7)


        topo_layers = get_bfs_layers(graph)

        assert topo_layers == [
            {node1, node3},
            {node2, node4},
            {node5, node6},
            {node7}
        ]