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
import os
import sys
from unittest.mock import patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from msprobe.core.compare.find_first.graph import DataNode, CommunicationNode
from msprobe.core.compare.find_first.utils import RankPath, DiffAnalyseConst
from msprobe.core.common.const import Const, CompareConst


class TestDataNode(unittest.TestCase):
    def setUp(self):
        # 创建测试数据
        self.op_name = "Torch.add.1.forward"
        self.rank = 0
        self.op_data = {
            "is_same": False,
            "op_items": [
                {
                    CompareConst.NPU_NAME: "input.0",
                    CompareConst.NPU_MAX: 1.0,
                    CompareConst.NPU_MIN: 0.0,
                    CompareConst.NPU_MEAN: 0.5,
                    CompareConst.NPU_NORM: 0.7,
                    Const.STATE: "input",
                    CompareConst.STACK: [["Torch.add.1.forward", {"file": "test.py", "line": 10}]]
                },
                {
                    CompareConst.NPU_NAME: "output.0",
                    CompareConst.NPU_MD5: "abc123",
                    Const.STATE: "output",
                    CompareConst.STACK: CompareConst.N_A
                }
            ]
        }
    
    def test_init(self):
        # 测试初始化
        node = DataNode(self.op_name, self.rank, self.op_data)
        
        # 验证基本属性
        self.assertEqual(node.op_name, self.op_name)
        self.assertEqual(node.rank, self.rank)
        self.assertTrue(node.is_diff)  # is_same为False，所以is_diff应为True
        self.assertEqual(node.layer, 0)
        self.assertEqual(node.sub_layer, 0)
        
        # 验证输入输出解析
        self.assertIn("input.0", node.inputs)
        self.assertIn("output.0", node.outputs)
        self.assertEqual(node.inputs["input.0"][CompareConst.NPU_MAX], 1.0)
        self.assertEqual(node.outputs["output.0"][CompareConst.NPU_MD5], "abc123")
        
        # 验证堆栈信息
        self.assertIsNotNone(node.stack)
    
    def test_find_stack(self):
        # 测试查找堆栈信息
        node = DataNode(self.op_name, self.rank, self.op_data)
        stack_info = node.find_stack()
        
        # 验证堆栈信息
        self.assertEqual(stack_info, {"file": "test.py", "line": 10})
    
    def test_gen_node_info(self):
        # 测试生成节点信息
        node = DataNode(self.op_name, self.rank, self.op_data)
        mock_path = MagicMock(spec=RankPath)
        
        info = node.gen_node_info(mock_path)
        
        # 验证节点信息
        self.assertEqual(info["op_name"], self.op_name)
        self.assertIn(Const.INPUT, info["data_info"])
        self.assertIn(Const.OUTPUT, info["data_info"])
        self.assertIsNotNone(info["stack_info"])


class TestCommunicationNode(unittest.TestCase):
    def setUp(self):
        # 创建测试数据
        self.op_name = "Distributed.all_reduce.1.forward"
        self.rank = 0
        self.node_id = f"{self.rank}.{self.op_name}"
        
        # 创建模拟的DataNode
        self.data_node = MagicMock(spec=DataNode)
        self.data_node.op_name = self.op_name
        self.data_node.is_diff = False
        self.data_node.inputs = {
            "input.0.dst": {CompareConst.NPU_MAX: 1},
            "input.1.group": {CompareConst.NPU_MD5: "[0,1,2]"}
        }
    
    def test_init(self):
        # 测试初始化
        node = CommunicationNode(self.node_id, self.rank, self.data_node)
        
        # 验证基本属性
        self.assertEqual(node.node_id, self.node_id)
        self.assertEqual(node.rank, self.rank)
        self.assertEqual(node.data, self.data_node)
        self.assertFalse(node.is_diff)
        self.assertEqual(node.layer, 0)
        self.assertEqual(node.api, "all_reduce")
        self.assertEqual(node.call_cnt, "1")
        self.assertFalse(node.connected)
        
        # 验证节点关系初始化
        self.assertIsNone(node.pre_node)
        self.assertEqual(node.link_nodes, {})
        self.assertEqual(node.dst_nodes, {})
        self.assertEqual(node.src_nodes, {})
        self.assertEqual(node.next_nodes, {})
        self.assertEqual(node.compute_ops, [])
    
    def test_add_next(self):
        # 测试添加下一个节点
        node = CommunicationNode(self.node_id, self.rank, self.data_node)
        next_node = CommunicationNode("1.Distributed.all_reduce.2.forward", 1, self.data_node)
        
        node.add_next(next_node)
        
        # 验证节点关系
        self.assertIn(next_node.node_id, node.next_nodes)
        self.assertEqual(next_node.pre_node, node)
        self.assertEqual(next_node.layer, node.layer + 1)
        self.assertEqual(next_node.data.layer, next_node.layer)
    
    def test_add_link(self):
        # 测试添加链接节点
        node = CommunicationNode(self.node_id, self.rank, self.data_node)
        link_node = CommunicationNode("1.Distributed.all_reduce.2.forward", 1, self.data_node)
        
        node.add_link(link_node)
        
        # 验证节点关系
        self.assertIn(link_node.node_id, node.link_nodes)
        self.assertIn(node.node_id, link_node.link_nodes)
        self.assertEqual(link_node.layer, node.layer)
        self.assertEqual(link_node.data.layer, link_node.layer)
        self.assertTrue(node.connected)
        self.assertTrue(link_node.connected)
    
    def test_add_dst(self):
        # 测试添加目标节点
        node = CommunicationNode(self.node_id, self.rank, self.data_node)
        dst_node = CommunicationNode("1.Distributed.all_reduce.2.forward", 1, self.data_node)
        
        node.add_dst(dst_node)
        
        # 验证节点关系
        self.assertIn(dst_node.node_id, node.dst_nodes)
        self.assertIn(node.node_id, dst_node.src_nodes)
        self.assertEqual(dst_node.layer, node.layer)
        self.assertEqual(dst_node.data.layer, dst_node.layer)
        self.assertTrue(node.connected)
        self.assertTrue(dst_node.connected)
    
    def test_delete(self):
        # 测试删除节点
        node = CommunicationNode(self.node_id, self.rank, self.data_node)
        next_node = CommunicationNode("1.Distributed.all_reduce.2.forward", 1, self.data_node)
        dst_node = CommunicationNode("2.Distributed.all_reduce.3.forward", 2, self.data_node)
        src_node = CommunicationNode("3.Distributed.all_reduce.4.forward", 3, self.data_node)
        link_node = CommunicationNode("4.Distributed.all_reduce.5.forward", 4, self.data_node)
        pre_node = CommunicationNode("5.Distributed.all_reduce.6.forward", 5, self.data_node)
        
        # 建立节点关系
        node.add_next(next_node)
        node.add_dst(dst_node)
        src_node.add_dst(node)
        node.add_link(link_node)
        pre_node.add_next(node)
        
        # 删除节点
        node.delete()
        
        # 验证节点关系已清除
        self.assertIsNone(next_node.pre_node)
        self.assertNotIn(node.node_id, dst_node.src_nodes)
        self.assertNotIn(node.node_id, src_node.dst_nodes)
        self.assertNotIn(node.node_id, link_node.link_nodes)
        self.assertNotIn(node.node_id, pre_node.next_nodes)
    
    def test_find_connected_nodes(self):
        # 测试查找连接节点
        node = CommunicationNode(self.node_id, self.rank, self.data_node)
        
        # 模拟输入数据
        node.data.inputs = {
            "input.0.dst": {CompareConst.NPU_MAX: 1},
            "input.1.group": {CompareConst.NPU_MD5: "[0,1,2]"}
        }
        
        result = node.find_connected_nodes()
        
        # 验证结果
        self.assertIn(1, result["ranks"])
        self.assertIn(0, result["ranks"])
        self.assertIn(2, result["ranks"])
        self.assertEqual(result["api"], "Distributed.all_reduce")
        self.assertEqual(result["type"], DiffAnalyseConst.DST)
    
    def test_resolve_type(self):
        # 测试解析节点类型
        # 测试SRC类型
        self.data_node.inputs = {"input.0.src": {CompareConst.NPU_MAX: 0}}
        node = CommunicationNode(self.node_id, 0, self.data_node)
        self.assertEqual(node.type, DiffAnalyseConst.SRC)
        
        # 测试DST类型
        self.data_node.inputs = {"input.0.dst": {CompareConst.NPU_MAX: 0}}
        node = CommunicationNode(self.node_id, 0, self.data_node)
        self.assertEqual(node.type, DiffAnalyseConst.DST)
        
        # 测试LINK类型（默认）
        self.data_node.inputs = {"input.0": {CompareConst.NPU_MAX: 0}}
        node = CommunicationNode(self.node_id, 0, self.data_node)
        self.assertEqual(node.type, DiffAnalyseConst.LINK)


if __name__ == '__main__':
    unittest.main()