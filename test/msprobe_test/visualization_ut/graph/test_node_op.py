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
from msprobe.visualization.graph.node_op import NodeOp


class TestNodeOp(unittest.TestCase):

    def test_get_node_op_valid(self):
        node_name = "ModuleTest"
        self.assertEqual(NodeOp.get_node_op(node_name), NodeOp.module)

    def test_get_node_op_invalid(self):
        node_name = "InvalidNodeName"
        self.assertEqual(NodeOp.get_node_op(node_name), NodeOp.module)

    def test_get_node_op_all(self):
        test_cases = [
            ("ModuleTest", NodeOp.module),
            ("TensorTest", NodeOp.function_api),
            ("TorchTest", NodeOp.function_api),
            ("FunctionalTest", NodeOp.function_api),
            ("NPUTest", NodeOp.function_api),
            ("VFTest", NodeOp.function_api),
            ("DistributedTest", NodeOp.function_api),
            ("AtenTest", NodeOp.function_api)
        ]
        for node_name, expected_op in test_cases:
            self.assertEqual(NodeOp.get_node_op(node_name), expected_op)
