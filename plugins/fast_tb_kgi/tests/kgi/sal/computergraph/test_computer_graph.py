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

from backend.kgi.sal.computergraph.computer_edge import ComputerEdge
from backend.kgi.sal.computergraph.computer_graph import DATA_NODE_OP_TYPE, ComputerGraph
from backend.kgi.sal.computergraph.computer_node import ComputerNode, MemoryAttribute

class TestComputerGraph:
    def test_add_node(self):
        # 添加节点
        graph = ComputerGraph()
        node = ComputerNode()

        graph.add_node(node)

        # 检查节点是否添加到图中
        assert graph._graph.number_of_nodes() == 1
        assert node in graph._graph.nodes()

        # 检查节点是否添加到NodeManager中
        assert graph._node_manager.get_node_by_node_id(1) == node

    def test_add_edge(self):
        # 添加边
        graph = ComputerGraph()

        # 添加两个节点
        node1 = ComputerNode()
        node2 = ComputerNode()
        graph.add_node(node1)
        graph.add_node(node2)

        # 添加边
        memory_attr = MemoryAttribute()
        memory_attr.set_dtype("float32")
        attributes = {memory_attr: 1}
        graph.add_edge(node1, node2, "%1", attributes)

        # 检查边是否添加到图中
        assert graph._graph.number_of_edges() == 1
        assert (node1, node2) in graph._graph.edges()

        # 检查边是否添加到EdgeManager中
        edge = graph._edge_manager.get_edge(node1, node2)
        assert isinstance(edge, ComputerEdge)
        assert "%1" in edge.get_memory_ids_and_attributes()

    def test_del_nodes(self):
        # 删除节点
        graph = ComputerGraph()

        # 添加节点
        node1 = ComputerNode()
        node2 = ComputerNode()
        graph.add_node(node1)
        graph.add_node(node2)

        # 添加边
        memory_attr = MemoryAttribute()
        memory_attr.set_dtype("float32")
        attributes = {memory_attr: 1}
        graph.add_edge(node1, node2, "%1", attributes)

        # 删除节点
        graph.del_nodes([1])  # 删除node1

        # 检查节点和边是否被删除
        assert graph._graph.number_of_nodes() == 1
        assert graph._graph.number_of_edges() == 0
        assert node1 not in graph._graph.nodes()
        assert node2 in graph._graph.nodes()

    def test_del_edges(self):
        # 删除边
        graph = ComputerGraph()

        # 添加节点
        node1 = ComputerNode()
        node2 = ComputerNode()
        graph.add_node(node1)
        graph.add_node(node2)

        # 添加边
        memory_attr = MemoryAttribute()
        memory_attr.set_dtype("float32")
        attributes = {memory_attr: 1}
        graph.add_edge(node1, node2, "%1", attributes)

        # 删除边
        graph.del_edges([(node1, node2)])

        # 检查边是否被删除
        assert graph._graph.number_of_edges() == 0
        assert (node1, node2) not in graph._graph.edges()

    def test_ignore_data_op(self):
        # 忽略data算子
        graph = ComputerGraph()

        # 添加data节点
        data_node = ComputerNode()
        data_node.set_op_type(DATA_NODE_OP_TYPE)
        data_node.set_output_memory_ids(["%1"])

        output_attr = MemoryAttribute()
        output_attr.set_dtype("float32")
        data_node.set_output_memory_attributes([output_attr])

        # 添加普通节点
        normal_node = ComputerNode()
        normal_node.set_node_id(2)
        normal_node.set_op_type("Add")
        normal_node.set_input_memory_ids(["%1"])
        normal_node.set_input_memory_attributes([output_attr])

        graph.add_node(data_node)
        graph.add_node(normal_node)
        graph.add_edge(data_node, normal_node, "%1", {output_attr: 1})

        # 执行忽略data算子操作
        graph.ignore_data_ops()

        # 检查data节点是否被删除
        assert graph._graph.number_of_nodes() == 1
        assert data_node not in graph._graph.nodes()
        assert normal_node in graph._graph.nodes()

        # 检查边是否被删除
        assert graph._graph.number_of_edges() == 0

        # 检查data节点关联的内存属性是否被删除
        assert normal_node.get_input_memory_ids() == []
        assert normal_node.get_input_memory_attributes() == []

    def test_fuse_nodes_has_cycle_check(self):
        # 融合节点环检测
        graph = ComputerGraph()

        # 创建5个节点
        node0 = ComputerNode()
        node1 = ComputerNode()
        node2 = ComputerNode()
        node3 = ComputerNode()
        node4 = ComputerNode()
        graph.add_node(node0)  # line num 1
        graph.add_node(node1)  # line num 2
        graph.add_node(node2)  # line num 3
        graph.add_node(node3)  # line num 4
        graph.add_node(node4)  # line num 5

        # 添加边
        memory_attr = MemoryAttribute()
        memory_attr.set_dtype("float32")
        attributes = {memory_attr: 1}
        graph.add_edge(node0, node1, "%1", attributes)
        graph.add_edge(node0, node2, "%2", attributes)
        graph.add_edge(node1, node3, "%3", attributes)
        graph.add_edge(node3, node4, "%4", attributes)
        graph.add_edge(node2, node4, "%5", attributes)

        # 检查融合节点是否会形成环
        #   0
        #  / \
        # v   v
        # 1   2           1<-
        # |   |    ==>    |  \
        # v   /           v   \
        # 3  /            3->fused
        # | /
        # v
        # 4
        has_cycle = graph.fuse_nodes_has_cycle_check([1, 3, 5])
        assert has_cycle is True

    def test_fuse_nodes(self):
        # 节点融合功能
        graph = ComputerGraph()

        # 创建三个节点
        node1 = ComputerNode()
        node1.set_op_type("Add")
        node2 = ComputerNode()
        node2.set_op_type("Mul")
        node3 = ComputerNode()
        node3.set_op_type("Sub")
        graph.add_node(node1)
        graph.add_node(node2)
        graph.add_node(node3)
        # 获取分配的节点ID
        node1_line_num = node1.get_node_id()
        node2_line_num = node2.get_node_id()

        # 添加边
        memory_attr = MemoryAttribute()
        memory_attr.set_dtype("float32")
        attributes = {memory_attr: 1}
        graph.add_edge(node1, node2, "%1", attributes)
        graph.add_edge(node2, node3, "%2", attributes)

        # 融合节点1和2
        fused_node = graph.fuse_nodes([node1_line_num, node2_line_num], node1, "FusedAddMul")

        # 检查融合结果
        # 融合后应该有2个节点：融合节点和node3
        assert fused_node in graph._graph.nodes()
        assert node3 in graph._graph.nodes()
        # node1和node2应该不再在图中
        assert node1 not in graph._graph.nodes()
        assert node2 not in graph._graph.nodes()
        assert fused_node.get_op_type() == "FusedAddMul"

        # 检查边是否正确连接
        # 应该只有 fused_node -> node3 的边
        assert graph._graph.number_of_edges() == 1
        assert (fused_node, node3) in graph._graph.edges()
        assert (node1, node2) not in graph._graph.edges()
        assert (node2, node3) not in graph._graph.edges()