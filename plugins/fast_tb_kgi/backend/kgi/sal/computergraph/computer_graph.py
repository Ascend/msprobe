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

import copy
import networkx as nx
from typing import Optional, cast, Iterator
from backend.kgi.core.utils.graph_utils import get_nodes_pos_with_connected_subgraphs
from backend.kgi.core.utils.graph_utils import get_subgraph_with_anchor
from backend.kgi.sal.computergraph.computer_node import ComputerNode, MemoryAttribute
from backend.kgi.sal.computergraph.edge_manager import EdgeManager
from backend.kgi.sal.computergraph.node_manager import NodeManager

DATA_NODE_OP_TYPE = "data"
FUSED_NODE_OP_TYPE_PREFIX = "FusedNode"

class ComputerGraph:
    def __init__(self):
        self._graph = nx.DiGraph()
        self._node_manager: NodeManager = NodeManager()
        self._edge_manager: EdgeManager = EdgeManager()

    def get_graph(self) -> nx.DiGraph:
        return self._graph

    def get_node_manager(self) -> NodeManager:
        return self._node_manager

    def get_edge_manager(self) -> EdgeManager:
        return self._edge_manager

    def add_node(self, node: ComputerNode):
        self._node_manager.add_node(node)
        self._graph.add_node(node)

    def add_edge(
            self,
            src_node: ComputerNode,
            dst_node: ComputerNode,
            memory_id: str,
            attributes: dict[MemoryAttribute, int]
    ):
        self._edge_manager.update_edge(src_node, dst_node, memory_id, attributes)
        self._graph.add_edge(src_node, dst_node)  # DiGraph有重复边添加时会自动忽略，无需额外处理重复边

    def del_node(self, node: ComputerNode):
        for predecessor in cast(Iterator[ComputerNode], self._graph.predecessors(node)):
            self._edge_manager.del_edge(predecessor, node)
        for successor in cast(Iterator[ComputerNode], self._graph.successors(node)):
            self._edge_manager.del_edge(node, successor)
        self._node_manager.del_node(node)
        self._graph.remove_node(node)

    def del_edge(self, src_node: ComputerNode, dst_node: ComputerNode):
        self._edge_manager.del_edge(src_node, dst_node)
        self._graph.remove_edge(src_node, dst_node)

    def del_nodes(self, nodes_line_id: list[int]):
        for line_id in nodes_line_id:
            node = self._node_manager.get_node_by_node_id(line_id)
            if not node:
                continue
            self.del_node(node)

    def del_edges(self, edges: list[tuple[ComputerNode, ComputerNode]]):
        for src_node, dst_node in edges:
            self.del_edge(src_node, dst_node)

    def _remove_memory_ids(self, node: ComputerNode):
        """
        data算子移除前, 先修改该节点后继节点的输入
        """
        if len(node.get_output_memory_ids()) == 0:
            return
        output_memory_id = node.get_output_memory_ids()[0]
        output_memory_attribute = node.get_output_memory_attributes()[0]
        for successor in cast(Iterator[ComputerNode], self._graph.successors(node)):
            need_remove = []
            input_memory_ids = successor.get_input_memory_ids()
            input_memory_attributes = successor.get_input_memory_attributes()
            for index in range(len(input_memory_ids)):
                if (output_memory_id == input_memory_ids[index] and
                    output_memory_attribute == input_memory_attributes[index]):
                    need_remove.append(index)
            for index in need_remove[::-1]:
                del input_memory_ids[index]
                del input_memory_attributes[index]

    def ignore_data_ops(self):
        """
        忽略data算子
        """
        for node in self._node_manager.get_nodes():
            if node.get_op_type() != DATA_NODE_OP_TYPE:
                continue
            self._remove_memory_ids(node)
            self.del_node(node)

    @staticmethod
    def _has_cycle(graph: nx.DiGraph, source: ComputerNode) -> bool:
        """
        判断是否存在从source节点到source节点的环
        """
        visited = set()
        stack = [(source, iter(graph.successors(source)))]
        while stack:
            _, successors = stack[-1]
            try:
                successor = next(successors)
            except StopIteration:
                stack.pop()
                continue

            if successor == source:
                return True

            if successor in visited:
                continue

            visited.add(successor)
            stack.append((successor, iter(graph.successors(successor))))

        return False

    def fuse_nodes_has_cycle_check(self, fuse_nodes_node_id: list[int]) -> bool:
        # 为防止影响原图，复制一份新图，在新图上做融合，再检查融合后是否成环
        new_graph = copy.deepcopy(self._graph)

        # 新图上待融合节点
        fuse_nodes = set()
        for node in cast(Iterator[ComputerNode], new_graph.nodes()):
            if node.get_node_id() in fuse_nodes_node_id:
                fuse_nodes.add(node)

        # 记录待融合节点的前驱和后继节点
        successors: set[ComputerNode] = set()
        predecessors: set[ComputerNode] = set()
        for node in fuse_nodes:
            for successor in new_graph.successors(node):
                if successor not in fuse_nodes:
                    successors.add(successor)
            for predecessor in new_graph.predecessors(node):
                if predecessor not in fuse_nodes:
                    predecessors.add(predecessor)

        # 移除待融合节点
        new_graph.remove_nodes_from(fuse_nodes)

        # 添加融合后节点，并添加与前驱和后继节点的边
        fused_node = ComputerNode()
        new_graph.add_node(fused_node)
        for successor in successors:
            new_graph.add_edge(fused_node, successor)
        for predecessor in predecessors:
            new_graph.add_edge(predecessor, fused_node)

        return ComputerGraph._has_cycle(new_graph, fused_node)

    @staticmethod
    def _set_fused_node(fused_node: ComputerNode, target_node: ComputerNode):
        fused_node.set_input_memory_ids(copy.deepcopy(target_node.get_input_memory_ids()))
        fused_node.set_output_memory_ids(copy.deepcopy(target_node.get_output_memory_ids()))
        fused_node.set_input_memory_attributes(copy.deepcopy(target_node.get_input_memory_attributes()))
        fused_node.set_output_memory_attributes(copy.deepcopy(target_node.get_output_memory_attributes()))

    def _create_fused_node(self, op_type: str, target_node: ComputerNode) -> ComputerNode:
        fused_node = ComputerNode()

        fused_node.set_op_type(op_type)
        if op_type == "":
            fused_node.set_op_type(target_node.get_op_type())

        ComputerGraph._set_fused_node(fused_node, target_node)

        self.add_node(fused_node)

        return fused_node

    def _get_fused_node_suc_pre_and_old_nodes(
            self,
            fuse_nodes_node_id: list[int]
    ) -> tuple[
        dict[ComputerNode, list[ComputerNode]],
        dict[ComputerNode, list[ComputerNode]]
    ]:
        graph = self._graph
        successors_old_nodes: dict[ComputerNode, list[ComputerNode]] = {}
        predecessors_old_nodes: dict[ComputerNode, list[ComputerNode]] = {}
        for line_id in fuse_nodes_node_id:
            node = self._node_manager.get_node_by_node_id(line_id)
            if not node:
                continue
            for successor in cast(Iterator[ComputerNode], graph.successors(node)):
                if successor.get_node_id() in fuse_nodes_node_id:
                    continue
                successors_old_nodes.setdefault(successor, []).append(node)
            for predecessor in cast(Iterator[ComputerNode], graph.predecessors(node)):
                if predecessor.get_node_id() in fuse_nodes_node_id:
                    continue
                predecessors_old_nodes.setdefault(predecessor, []).append(node)
        return successors_old_nodes, predecessors_old_nodes

    def _add_fused_node_edges(
            self,
            fused_node: ComputerNode,
            successors_old_nodes: dict[ComputerNode, list[ComputerNode]],
            predecessors_old_nodes: dict[ComputerNode, list[ComputerNode]]
    ):
        graph = self._graph
        for successor, old_nodes in successors_old_nodes.items():
            for node in old_nodes:
                edge = self._edge_manager.get_edge(node, successor)
                for memory_id, attributes in edge.get_memory_ids_and_attributes().items():
                    self._edge_manager.update_edge(fused_node, successor, memory_id, attributes)
            graph.add_edge(fused_node, successor)
        for predecessor, old_nodes in predecessors_old_nodes.items():
            for node in old_nodes:
                edge = self._edge_manager.get_edge(predecessor, node)
                for memory_id, attributes in edge.get_memory_ids_and_attributes().items():
                    self._edge_manager.update_edge(predecessor, fused_node, memory_id, attributes)
            graph.add_edge(predecessor, fused_node)

    def fuse_nodes(self, fuse_nodes_node_id: list[int], peer_node: ComputerNode, fused_node_op_type: str) -> ComputerNode:
        # 创建融合后新节点
        fused_node = self._create_fused_node(fused_node_op_type, peer_node)
        # 获取融合后节点的前驱节点和后继节点，并记录这些前驱节点和后继节点原来连接的融合前节点
        successors_old_nodes, predecessors_old_nodes = self._get_fused_node_suc_pre_and_old_nodes(fuse_nodes_node_id)
        # 添加融合后节点的边
        self._add_fused_node_edges(fused_node, successors_old_nodes, predecessors_old_nodes)
        # 删除融合前节点
        for line_id in fuse_nodes_node_id:
            node = self._node_manager.get_node_by_node_id(line_id)
            if not node:
                continue
            self.del_node(node)

        return fused_node

    def convert_to_nodes_and_edges(
            self,
            anchor: Optional[ComputerNode]
    ) -> tuple[list[dict], list[dict]]:
        """
        将ComputerGraph对象转换为节点和边格式
        """
        nodes = []
        edges = []

        graph = get_subgraph_with_anchor(self._graph, anchor, True)

        # 节点位置
        nodes_pos = get_nodes_pos_with_connected_subgraphs(graph)

        # 节点
        for node in cast(Iterator[ComputerNode], graph.nodes):
            line_id = node.get_node_id()
            label = f"{node.get_op_type()} {line_id}"

            nodes.append({
                "id": line_id,
                "x": nodes_pos[node][0],
                "y": nodes_pos[node][1],
                "label": label
            })

        # 边
        for edge in graph.edges():
            src_node, dst_node = cast(tuple[ComputerNode, ComputerNode], edge)
            edge_id = f"{src_node.get_node_id()}-{dst_node.get_node_id()}"
            edges.append({
                "id": edge_id,
                "from": src_node.get_node_id(),
                "to": dst_node.get_node_id()
            })

        return nodes, edges