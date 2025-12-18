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
from backend.kgi.sal.computergraph.computer_node import ComputerNode, MemoryAttribute

class EdgeManager:
    def __init__(self):
        self._edges: dict[str, ComputerEdge] = {}

    @staticmethod
    def _get_edge_key(src_node: ComputerNode, dst_node: ComputerNode) -> str:
        return f"{src_node.get_node_id()}->{dst_node.get_node_id()}"

    def get_edge(self, src_node: ComputerNode, dst_node: ComputerNode) -> ComputerEdge:
        key = EdgeManager._get_edge_key(src_node, dst_node)
        return self._edges[key]

    def add_edge(self, src_node: ComputerNode, dst_node: ComputerNode):
        key = EdgeManager._get_edge_key(src_node, dst_node)
        self._edges.setdefault(key, ComputerEdge())

    def update_edge(
            self,
            src_node: ComputerNode,
            dst_node: ComputerNode,
            memory_id: str,
            attributes: dict[MemoryAttribute, int]
    ):
        key = EdgeManager._get_edge_key(src_node, dst_node)
        self._edges.setdefault(key, ComputerEdge())

        edge = self._edges[key]
        edge.update_memory_ids_and_attributes(memory_id, attributes)

    def del_edge(self, src_node: ComputerNode, dst_node: ComputerNode):
        key = EdgeManager._get_edge_key(src_node, dst_node)
        del self._edges[key]