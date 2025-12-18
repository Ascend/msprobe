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
from backend.kgi.sal.computergraph.computer_node import MemoryAttribute

class TestComputerEdge:
    def test_update_memory_ids_and_attributes(self):
        # 更新内存ID和属性
        edge = ComputerEdge()
        memory_attr = MemoryAttribute()
        memory_attr.set_dtype("float32")
        memory_attr.set_shape([10, 20])

        attributes = {memory_attr: 1}
        edge.update_memory_ids_and_attributes("%1", attributes)

        assert "%1" in edge._memory_ids_and_attributes
        assert memory_attr in edge._memory_ids_and_attributes["%1"]
        assert edge._memory_ids_and_attributes["%1"][memory_attr] == 1

    def test_repr(self):
        # __repr__方法
        edge = ComputerEdge()
        memory_attr = MemoryAttribute()
        memory_attr.set_dtype("float32")
        memory_attr.set_shape([10, 20])

        attributes = {memory_attr: 2}
        edge.update_memory_ids_and_attributes("%1", attributes)

        result = repr(edge)
        assert result == (f"%1\n"
                          f"2 float32:[10, 20]")