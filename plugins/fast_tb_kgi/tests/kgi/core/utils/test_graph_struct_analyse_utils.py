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
from backend.kgi.core.utils.graph_struct_analyze_utils import get_cycle_number, get_cycles, get_subgraph_number
from backend.kgi.core.utils.graph_struct_analyze_utils import get_subgraphs, get_subgraphs_node_number

class TestGraphStructAnalyzeUtils:
    def test_get_cycle_number_no_cycle(self):
        # 添加节点
        node1 = Node()
        node1.set_node_id(1)
        graph = nx.DiGraph()
        graph.add_node(node1)

        cycle_num = get_cycle_number(graph)

        assert cycle_num == 0

    def test_get_cycle_number_with_cycle(self):
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

        graph = nx.DiGraph()

        # 添加边
        graph.add_edge(node1, node2)
        graph.add_edge(node2, node3)
        graph.add_edge(node3, node1)
        graph.add_edge(node4, node5)
        graph.add_edge(node5, node6)
        graph.add_edge(node6, node4)

        cycle_num = get_cycle_number(graph)

        assert cycle_num == 2

    def test_get_cycles_no_cycle(self):
        # 添加节点
        node1 = Node()
        node1.set_node_id(1)
        graph = nx.DiGraph()
        graph.add_node(node1)

        cycles = get_cycles(graph)

        assert cycles == []

    def test_get_cycles_with_cycle(self):
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

        graph = nx.DiGraph()

        # 添加边
        graph.add_edge(node1, node2)
        graph.add_edge(node2, node3)
        graph.add_edge(node3, node1)
        graph.add_edge(node4, node5)
        graph.add_edge(node5, node6)
        graph.add_edge(node6, node4)

        cycles = get_cycles(graph)

        assert cycles == [
            [node1, node2, node3],
            [node4, node5, node6]
        ]

    def test_get_subgraph_number(self):
        # 添加节点
        node1 = Node()
        node1.set_node_id(1)
        node2 = Node()
        node2.set_node_id(2)
        graph = nx.DiGraph()
        graph.add_node(node1)
        graph.add_node(node2)

        subgraph_num = get_subgraph_number(graph)

        assert subgraph_num == 2

    def test_get_subgraphs_node_number(self):
        # 添加节点
        node1 = Node()
        node1.set_node_id(1)
        node2 = Node()
        node2.set_node_id(2)
        node3 = Node()
        node3.set_node_id(3)
        node4 = Node()
        node4.set_node_id(4)
        graph = nx.DiGraph()
        graph.add_node(node3)
        graph.add_node(node4)

        # 添加边
        graph.add_edge(node1, node2)

        subgraphs_node_number = get_subgraphs_node_number(graph)

        assert subgraphs_node_number == [1, 1, 2]

    def test_get_subgraphs(self):
        # 添加节点
        node1 = Node()
        node1.set_node_id(1)
        node2 = Node()
        node2.set_node_id(2)
        node3 = Node()
        node3.set_node_id(3)
        node4 = Node()
        node4.set_node_id(4)
        graph = nx.DiGraph()
        graph.add_node(node3)
        graph.add_node(node4)

        # 添加边
        graph.add_edge(node1, node2)

        subgraph_list = get_subgraphs(graph)

        assert subgraph_list == [
            [node3],
            [node4],
            [node1, node2]
        ]