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

from backend.kgi.sal.computergraph.computer_node import ComputerNode, MemoryAttribute
from backend.kgi.sal.computergraph.edge_manager import EdgeManager

class TestEdgeManager:
    def test_add_edge(self):
        # 添加边
        manager = EdgeManager()
        src_node = ComputerNode()
        src_node.set_node_id(1)
        dst_node = ComputerNode()
        dst_node.set_node_id(2)

        manager.add_edge(src_node, dst_node)

        key = "1->2"
        assert key in manager._edges

    def test_update_edge(self):
        # 更新边
        manager = EdgeManager()
        src_node = ComputerNode()
        src_node.set_node_id(1)
        dst_node = ComputerNode()
        dst_node.set_node_id(2)
        memory_attr = MemoryAttribute()
        memory_attr.set_dtype("float32")
        memory_attr.set_shape([10, 20])

        attributes = {memory_attr: 1}
        manager.update_edge(src_node, dst_node, "%1", attributes)

        key = "1->2"
        assert key in manager._edges
        edge = manager._edges[key]
        assert "%1" in edge._memory_ids_and_attributes

    def test_del_edge(self):
        # 删除边
        manager = EdgeManager()
        src_node = ComputerNode()
        src_node.set_node_id(1)
        dst_node = ComputerNode()
        dst_node.set_node_id(2)

        manager.add_edge(src_node, dst_node)
        key = "1->2"
        assert key in manager._edges

        manager.del_edge(src_node, dst_node)
        assert key not in manager._edges