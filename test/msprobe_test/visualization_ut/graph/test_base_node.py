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
from msprobe.visualization.graph.base_node import BaseNode
from msprobe.visualization.graph.node_op import NodeOp


class TestBaseNode(unittest.TestCase):

    def setUp(self):
        self.node_op = NodeOp.module
        self.node_id = "node_1"
        self.up_node = BaseNode(self.node_op, "up_node_1")
        self.node = BaseNode(self.node_op, self.node_id, self.up_node)

    def test_init_and_str(self):
        self.assertEqual(self.node.op, self.node_op)
        self.assertEqual(self.node.id, self.node_id)
        self.assertEqual(str(self.node), 'id:\tnode_1')

    def test_eq(self):
        other_node = BaseNode(self.node_op, self.node_id, self.up_node)
        self.assertEqual(self.node, other_node)

    def test_set_input_output(self):
        input_data = {'input1': 'value1'}
        output_data = {'output1': 'value2'}
        self.node.set_input_output(input_data, output_data)
        self.assertEqual(self.node.input_data, input_data)
        self.assertEqual(self.node.output_data, output_data)

    def test_add_upnode(self):
        self.node = BaseNode(self.node_op, self.node_id)
        new_up_node = BaseNode(self.node_op, "new_up_node_1")
        self.node.add_upnode(new_up_node)
        self.assertEqual(self.node.upnode, new_up_node)
        self.assertIn(self.node, new_up_node.subnodes)

    def test_add_link(self):
        other_node = BaseNode(self.node_op, "other_node_1")
        ancestors = ['a1', 'a2']
        self.node.add_link(other_node, ancestors)
        self.assertEqual(self.node.matched_node_link, ancestors)
        self.assertEqual(other_node.matched_node_link, ancestors)

    def test_get_ancestors(self):
        expected_ancestors = ['up_node_1']
        self.assertEqual(self.node.get_ancestors(), expected_ancestors)
