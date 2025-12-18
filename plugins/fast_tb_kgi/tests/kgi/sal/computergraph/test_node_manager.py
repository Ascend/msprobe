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

from backend.kgi.core.structs.node import Node
from backend.kgi.sal.computergraph.node_manager import NodeManager

class TestNodeManager:
    def test_add_node(self):
        # 添加节点
        manager = NodeManager()
        node = Node()
        manager.add_node(node)

        assert manager._max_node_id == 1
        assert node.get_node_id() == 1
        assert 1 in manager._nodes
        assert manager._nodes[1] == node

    def test_del_node(self):
        # 删除节点
        manager = NodeManager()
        node = Node()
        manager.add_node(node)

        assert manager._max_node_id == 1
        assert 1 in manager._nodes

        manager.del_node(node)
        assert 1 not in manager._nodes