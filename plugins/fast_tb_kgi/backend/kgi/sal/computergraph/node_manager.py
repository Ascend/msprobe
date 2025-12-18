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

from typing import Optional
from backend.kgi.core.structs.node import INVALID_NODE_ID
from backend.kgi.sal.computergraph.computer_node import ComputerNode

NODE_ID_INCREMENT = 1

class NodeManager:
    def __init__(self):
        self._max_node_id = 0
        self._nodes: dict[int, ComputerNode] = {}

    def get_node_by_node_id(self, node_id: int) -> Optional[ComputerNode]:
        return self._nodes.get(node_id)

    def get_nodes(self) -> list[ComputerNode]:
        nodes = [self._nodes[node_id] for node_id in sorted(list(self._nodes.keys()))]
        return nodes

    def add_node(self, node: ComputerNode):
        node_id = node.get_node_id()
        if node_id == INVALID_NODE_ID:
            self._max_node_id += NODE_ID_INCREMENT
            node.set_node_id(self._max_node_id)
            node_id = node.get_node_id()
        else:
            self._max_node_id = node_id if node_id > self._max_node_id else self._max_node_id
        if self._nodes.get(node_id) is not None:
            raise ValueError("Node node id duplicate")
        self._nodes[node_id] = node

    def del_node(self, node: ComputerNode):
        del self._nodes[node.get_node_id()]