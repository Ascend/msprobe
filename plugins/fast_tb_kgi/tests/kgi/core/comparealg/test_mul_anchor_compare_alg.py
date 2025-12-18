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
from backend.kgi.core.comparealg.mul_anchor_compare_alg import MulAnchorCompareAlg
from backend.kgi.core.structs.node import Node

def _create_node(node_id: int, node_feature: str) -> Node:
    node = Node()
    node.set_node_id(node_id)
    node.set_node_feature(node_feature)
    return node

def _create_left_graph() -> tuple[nx.DiGraph, list[Node]]:
    node1 = _create_node(1, "Add")
    node2 = _create_node(2, "Sub")
    node3 = _create_node(3, "Mul")
    node4 = _create_node(4, "Div")
    node5 = _create_node(5, "Add")
    node6 = _create_node(6, "Sub")
    node7 = _create_node(7, "Mul")
    node8 = _create_node(8, "Div")
    node9 = _create_node(9, "Add")
    node10 = _create_node(10, "Sub")
    node11 = _create_node(11, "Mul")
    node12 = _create_node(12, "Div")
    graph = nx.DiGraph()
    graph.add_edge(node4, node1)
    graph.add_edge(node4, node2)
    graph.add_edge(node5, node3)
    graph.add_edge(node6, node4)
    graph.add_edge(node6, node5)
    graph.add_edge(node7, node6)
    graph.add_edge(node8, node6)
    graph.add_edge(node11, node6)
    graph.add_edge(node9, node8)
    graph.add_edge(node10, node8)
    graph.add_edge(node12, node11)

    return graph, [node1, node2, node3, node4, node5, node6, node7, node8, node9, node10, node11, node12]

def _create_right_graph() -> tuple[nx.DiGraph, list[Node]]:
    node1 = _create_node(1, "Add")
    node2 = _create_node(2, "Sub")
    node3 = _create_node(3, "Mul")
    node4 = _create_node(4, "Div")
    node5 = _create_node(5, "Add")
    node6 = _create_node(6, "Sub")
    node7 = _create_node(7, "Sub")
    node8 = _create_node(8, "Div")
    node9 = _create_node(9, "Add")
    node10 = _create_node(10, "Sub")
    node11 = _create_node(11, "Mul")
    node12 = _create_node(12, "Div")
    graph = nx.DiGraph()
    graph.add_edge(node4, node1)
    graph.add_edge(node4, node2)
    graph.add_edge(node5, node3)
    graph.add_edge(node6, node4)
    graph.add_edge(node6, node5)
    graph.add_edge(node7, node6)
    graph.add_edge(node8, node6)
    graph.add_edge(node11, node6)
    graph.add_edge(node9, node8)
    graph.add_edge(node10, node8)
    graph.add_edge(node10, node11)
    graph.add_edge(node12, node11)

    return graph, [node1, node2, node3, node4, node5, node6, node7, node8, node9, node10, node11, node12]

class TestMulAnchorCompareAlg:
    def test_add_del_second_level_anchor_compare_alg(self):
        # 添加删除二级锚点
        left_graph, left_nodes = _create_left_graph()
        right_graph, right_nodes = _create_right_graph()

        alg = MulAnchorCompareAlg(left_graph, right_graph, None, None, True, True)
        alg.compare()

        assert left_nodes[0] in alg.get_left_match_nodes()
        assert right_nodes[0] in alg.get_right_match_nodes()
        assert left_nodes[1] in alg.get_left_match_nodes()
        assert right_nodes[1] in alg.get_right_match_nodes()
        assert left_nodes[2] in alg.get_left_match_nodes()
        assert right_nodes[2] in alg.get_right_match_nodes()
        assert left_nodes[3] in alg.get_left_match_nodes()
        assert right_nodes[3] in alg.get_right_match_nodes()
        assert left_nodes[4] in alg.get_left_match_nodes()
        assert right_nodes[4] in alg.get_right_match_nodes()
        assert left_nodes[5] in alg.get_left_mismatch_nodes()
        assert right_nodes[5] in alg.get_right_mismatch_nodes()

        # 添加二级锚点
        alg.add_second_level_anchor_compare_alg(left_nodes[7], right_nodes[7])
        alg.compare()

        assert left_nodes[8] in alg.get_left_match_nodes()
        assert right_nodes[8] in alg.get_right_match_nodes()
        assert left_nodes[9] in alg.get_left_match_nodes()
        assert right_nodes[9] in alg.get_right_match_nodes()

        # 删除二级锚点
        alg.del_second_level_anchor_compare_alg([left_nodes[7].get_node_id()], [right_nodes[7].get_node_id()])

        assert left_nodes[8] not in alg.get_left_match_nodes()
        assert right_nodes[8] not in alg.get_right_match_nodes()
        assert left_nodes[9] not in alg.get_left_match_nodes()
        assert right_nodes[9] not in alg.get_right_match_nodes()

    def test_match_to_mismatch(self):
        # 添加二级锚点时, 原来的二级锚点计算出的匹配节点, 在当前二级锚点计算出为不匹配节点, 信任当前二级锚点
        left_graph, left_nodes = _create_left_graph()
        right_graph, right_nodes = _create_right_graph()

        alg = MulAnchorCompareAlg(left_graph, right_graph, None, None, True, True)
        alg.compare()

        # 添加二级锚点
        alg.add_second_level_anchor_compare_alg(left_nodes[7], right_nodes[7])
        alg.compare()

        assert left_nodes[9] in alg.get_left_match_nodes()
        assert right_nodes[9] in alg.get_right_match_nodes()

        # 再添加二级锚点
        alg.add_second_level_anchor_compare_alg(left_nodes[10], right_nodes[10])
        alg.compare()

        assert left_nodes[9] in alg.get_left_match_nodes()
        assert right_nodes[9] not in alg.get_right_match_nodes()
        assert left_nodes[9] not in alg.get_left_mismatch_nodes()
        assert right_nodes[9] in alg.get_right_mismatch_nodes()
