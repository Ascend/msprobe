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

class TestMemoryAttribute:
    def test_repr(self):
        # __repr__方法
        memory_attr = MemoryAttribute()
        memory_attr.set_dtype("BFloat16")
        memory_attr.set_shape([4096, 1, 32, 128])
        assert repr(memory_attr) == "BFloat16:[4096, 1, 32, 128]"

    def test_repr_with_stride_offset(self):
        # __repr__方法，包含stride和offset
        memory_attr = MemoryAttribute()
        memory_attr.set_dtype("BFloat16")
        memory_attr.set_shape([4096, 1, 32, 128])
        memory_attr.set_exist_stride_offset(True)
        memory_attr.set_stride([12288, 12288, 384, 1])
        memory_attr.set_offset(384)
        assert repr(memory_attr) == "BFloat16:[4096, 1, 32, 128]{strdes=[12288, 12288, 384, 1],offset=384}"

    def test_hash(self):
        # __hash__方法
        memory_attr1 = MemoryAttribute()
        memory_attr1.set_dtype("float32")
        memory_attr1.set_shape([10, 20])

        memory_attr2 = MemoryAttribute()
        memory_attr2.set_dtype("float32")
        memory_attr2.set_shape([10, 20])

        assert hash(memory_attr1) == hash(memory_attr2)

    def test_eq(self):
        # __eq__方法
        memory_attr1 = MemoryAttribute()
        memory_attr1.set_dtype("float32")
        memory_attr1.set_shape([10, 20])

        memory_attr2 = MemoryAttribute()
        memory_attr2.set_dtype("float32")
        memory_attr2.set_shape([10, 20])

        memory_attr3 = MemoryAttribute()
        memory_attr3.set_dtype("float64")
        memory_attr3.set_shape([10, 20])

        assert memory_attr1 == memory_attr2
        assert memory_attr1 != memory_attr3
        assert memory_attr1 != "not a MemoryAttribute"


class TestComputerNode:
    def test_repr(self):
        # __repr__方法
        node = ComputerNode()
        node.set_node_id(10)
        node.set_op_type("Add")
        expected = "id 10"
        assert repr(node) == expected

    def test_hash(self):
        # __hash__方法
        node1 = ComputerNode()
        node1.set_node_id(10)
        node1.set_op_type("Add")

        node2 = ComputerNode()
        node2.set_node_id(10)
        node2.set_op_type("Add")

        assert hash(node1) == hash(node2)

    def test_eq(self):
        # __eq__方法
        node1 = ComputerNode()
        node1.set_node_id(10)
        node1.set_op_type("Add")

        node2 = ComputerNode()
        node2.set_node_id(10)
        node2.set_op_type("Add")

        node3 = ComputerNode()
        node3.set_node_id(12)
        node3.set_op_type("Add")

        assert node1 == node2
        assert node1 != node3
        assert node1 != "not a ComputerNode"