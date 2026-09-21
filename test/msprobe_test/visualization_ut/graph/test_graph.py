# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

import unittest
from unittest.mock import MagicMock
from msprobe.visualization.graph.graph import Graph, NodeOp
from msprobe.visualization.graph.base_node import BaseNode
from msprobe.visualization.utils import GraphConst


class TestGraph(unittest.TestCase):

    def setUp(self):
        self.graph = Graph("model_name")
        self.node_id = "node_id"
        self.node_op = NodeOp.module

    def test_add_node_and_get_node(self):
        self.graph.add_node(self.node_op, self.node_id)
        node = self.graph.get_node(self.node_id)
        self.assertIsNotNone(node)
        self.assertIn(self.node_id, self.graph.node_map)

        node_id = "api"
        graph = Graph("model_name")
        for i in range(0, 9):
            graph.add_node(NodeOp.function_api, node_id, id_accumulation=True)
        self.assertEqual(len(graph.node_map), 10)
        self.assertIn("api.0", graph.node_map)
        self.assertIn("api.8", graph.node_map)
        self.assertNotIn("api", graph.node_map)

    def test_str(self):
        self.graph.add_node(self.node_op, self.node_id)
        expected_str = f'{self.node_id}'
        self.assertIn(expected_str, str(self.graph))

    def test_match(self):
        graph_a = Graph("model_name_a")
        graph_b = Graph("model_name_b")
        node_a = BaseNode(self.node_op, self.node_id)
        graph_a.add_node(NodeOp.module, "node_id_a")
        graph_b.add_node(NodeOp.module, "node_id_b")
        matched_node, ancestors = Graph.match(graph_a, node_a, graph_b)
        self.assertIsNone(matched_node)
        self.assertEqual(ancestors, [])

        graph_b.add_node(NodeOp.module, "node_id_a")
        graph_a.add_node(NodeOp.module, "node_id_a_1", graph_a.get_node("node_id_a"))
        graph_b.add_node(NodeOp.module, "node_id_a_1", graph_a.get_node("node_id_a"))
        matched_node, ancestors = Graph.match(graph_a, graph_a.get_node("node_id_a_1"), graph_b)
        self.assertIsNotNone(matched_node)
        self.assertEqual(ancestors, ['node_id_a'])

    def test_split_nodes_by_micro_step(self):
        nodes = [BaseNode(NodeOp.module, 'a.forward.0'), BaseNode(NodeOp.module, 'a.backward.0'),
                 BaseNode(NodeOp.api_collection, 'apis.0'), BaseNode(NodeOp.module, 'a.forward.1'),
                 BaseNode(NodeOp.module, 'b.forward.0'), BaseNode(NodeOp.module, 'b.backward.0'),
                 BaseNode(NodeOp.module, 'a.backward.1'), BaseNode(NodeOp.api_collection, 'apis.1')]
        result = Graph.split_nodes_by_micro_step(nodes)
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 3)

    def test_paging_by_micro_step(self):
        nodes = [BaseNode(NodeOp.module, 'a.forward.0'), BaseNode(NodeOp.module, 'a.backward.0'),
                 BaseNode(NodeOp.api_collection, 'apis.0'), BaseNode(NodeOp.module, 'a.forward.1'),
                 BaseNode(NodeOp.module, 'b.forward.0'), BaseNode(NodeOp.module, 'b.backward.0'),
                 BaseNode(NodeOp.module, 'a.backward.1'), BaseNode(NodeOp.api_collection, 'apis.1')]

        graph = Graph('Model1')
        graph.root.subnodes = nodes
        graph_other = Graph('Model2')
        graph_other.root.subnodes = nodes

        result = graph.paging_by_micro_step(graph_other)
        self.assertEqual(result, 2)
        self.assertEqual(graph.root.subnodes[0].micro_step_id, 0)
        self.assertEqual(graph_other.root.subnodes[0].micro_step_id, 0)

    def test_mapping_match(self):
        graph_a = Graph("model_name_a")
        graph_b = Graph("model_name_b")
        graph_a.add_node(NodeOp.module, "a1", BaseNode(NodeOp.module, "root"))
        graph_b.add_node(NodeOp.module, "b1", BaseNode(NodeOp.module, "root"))
        mapping_dict = {"a1": "b1"}
        node_b, ancestors_n, ancestors_b = Graph.mapping_match(graph_a.get_node("a1"), graph_b, mapping_dict)
        self.assertIsNotNone(node_b)
        self.assertEqual(ancestors_n, ["root"])
        self.assertEqual(ancestors_b, ["root"])
