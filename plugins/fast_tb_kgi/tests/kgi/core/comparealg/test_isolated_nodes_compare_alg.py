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
from backend.kgi.core.comparealg.isolated_nodes_compare_alg import IsolatedNodesCompareAlg
from backend.kgi.core.structs.node import Node

def _create_node(node_id: int, node_feature: str) -> Node:
    node = Node()
    node.set_node_id(node_id)
    node.set_node_feature(node_feature)
    return node

class TestIsolatedNodesCompareAlg:

    def test_feature_times_diff(self):
        # 有一种孤立节点两图出现次数不一致
        isolated_node1 = _create_node(1, "Add")
        isolated_node4 = _create_node(4, "Sub")
        left_graph = nx.DiGraph()
        left_graph.add_node(isolated_node1)
        left_graph.add_node(isolated_node4)

        isolated_node2 = _create_node(2, "Add")
        isolated_node3 = _create_node(3, "Add")
        isolated_node5 = _create_node(5, "Sub")
        right_graph = nx.DiGraph()
        right_graph.add_node(isolated_node2)
        right_graph.add_node(isolated_node3)
        right_graph.add_node(isolated_node5)

        alg = IsolatedNodesCompareAlg(left_graph, right_graph)
        alg.compare()

        assert isolated_node1 in alg._left_mismatch_nodes
        assert isolated_node2 in alg._right_mismatch_nodes
        assert isolated_node3 in alg._right_mismatch_nodes
        assert isolated_node4 in alg._left_match_nodes
        assert isolated_node5 in alg._right_match_nodes

    def test_node_only_in_left(self):
        # 有一种孤立节点只在左图出现
        isolated_node1 = _create_node(1, "Add")
        isolated_node2 = _create_node(2, "Sub")
        left_graph = nx.DiGraph()
        left_graph.add_node(isolated_node1)
        left_graph.add_node(isolated_node2)

        isolated_node3 = _create_node(3, "Sub")
        right_graph = nx.DiGraph()
        right_graph.add_node(isolated_node3)

        alg = IsolatedNodesCompareAlg(left_graph, right_graph)
        alg.compare()

        assert isolated_node1 in alg._left_mismatch_nodes
        assert isolated_node2 in alg._left_match_nodes
        assert isolated_node3 in alg._right_match_nodes

    def test_node_only_in_right(self):
        # 有一种孤立节点只在右图出现
        isolated_node2 = _create_node(2, "Sub")
        left_graph = nx.DiGraph()
        left_graph.add_node(isolated_node2)

        isolated_node1 = _create_node(1, "Add")
        isolated_node3 = _create_node(3, "Sub")
        right_graph = nx.DiGraph()
        right_graph.add_node(isolated_node1)
        right_graph.add_node(isolated_node3)

        alg = IsolatedNodesCompareAlg(left_graph, right_graph)
        alg.compare()

        assert isolated_node1 in alg._right_mismatch_nodes
        assert isolated_node2 in alg._left_match_nodes
        assert isolated_node3 in alg._right_match_nodes