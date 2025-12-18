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

import tempfile
import os
from backend.kgi.sal.computergraph.computer_node import MemoryAttribute
from backend.kgi.sal.parser.ms_execute_order_parser import MsExecuteOrderParser

class TestMsExecuteOrderParser:
    def test_get_scope_and_op_type_no_scope(self):
        # 没有scope
        scope, op_type = MsExecuteOrderParser._get_scope_and_op_type("Concat")
        assert scope == []
        assert op_type == "Concat"

    def test_get_scope_and_op_type(self):
        # 有scope
        scope, op_type = MsExecuteOrderParser._get_scope_and_op_type("scope1/scope2/Concat-op")
        assert scope == ["scope1", "scope2", "Concat-op"]
        assert op_type == "Concat"

    def test_get_stride_offset_NULL(self):
        # 空字符串
        memory_attr = MemoryAttribute()
        MsExecuteOrderParser._get_stride_offset("", memory_attr)
        assert memory_attr._exist_stride_offset is False

    def test_get_stride_offset_stride_NULL(self):
        # stride为空
        memory_attr = MemoryAttribute()
        MsExecuteOrderParser._get_stride_offset("strdes=[],offset=0", memory_attr)
        assert memory_attr._exist_stride_offset is True
        assert memory_attr._stride == []
        assert memory_attr._offset == 0

    def test_get_stride_offset(self):
        # stride不空
        memory_attr = MemoryAttribute()
        MsExecuteOrderParser._get_stride_offset("strdes=[12288, 12288, 384, 1],offset=384", memory_attr)
        assert memory_attr._exist_stride_offset is True
        assert memory_attr._stride == [12288, 12288, 384, 1]
        assert memory_attr._offset == 384

    def test_get_memory_attributes_NULL(self):
        # 空字符串
        result = MsExecuteOrderParser._get_memory_attributes("")
        assert result == []

    def test_get_memory_attributes_shape_NULL(self):
        # 单个memory attribute，shape为空
        result = MsExecuteOrderParser._get_memory_attributes("BFloat16:[]")
        assert len(result) == 1
        assert isinstance(result[0], MemoryAttribute)
        assert result[0]._dtype == "BFloat16"
        assert result[0]._shape == []

    def test_get_memory_attributes_shape(self):
        # 单个memory attribute，shape不空
        result = MsExecuteOrderParser._get_memory_attributes("BFloat16:[4096, 1, 4096]")
        assert len(result) == 1
        assert isinstance(result[0], MemoryAttribute)
        assert result[0]._dtype == "BFloat16"
        assert result[0]._shape == [4096, 1, 4096]

    def test_get_memory_attributes_stride_offset(self):
        # 单个memory attribute，带stride和offset
        result = MsExecuteOrderParser._get_memory_attributes("BFloat16:[4096, 1, 4096]{strdes=[4096, 4096, 1],offset=4096}")
        assert len(result) == 1
        assert isinstance(result[0], MemoryAttribute)
        assert result[0]._dtype == "BFloat16"
        assert result[0]._shape == [4096, 1, 4096]
        assert result[0]._exist_stride_offset is True
        assert result[0]._stride == [4096, 4096, 1]
        assert result[0]._offset == 4096

    def test_get_memory_attributes(self):
        # 多个memory attribute
        result = MsExecuteOrderParser._get_memory_attributes("Float32:[32, 16384], Bool:[]")
        assert len(result) == 2
        assert isinstance(result[0], MemoryAttribute)
        assert isinstance(result[1], MemoryAttribute)
        assert result[0]._dtype == "Float32"
        assert result[0]._shape == [32, 16384]
        assert result[0]._exist_stride_offset is False
        assert result[1]._dtype == "Bool"
        assert result[1]._shape == []
        assert result[1]._exist_stride_offset is False

    def test_get_in_out_memory_attributes(self):
        file_data = {}
        in_out_memory_attributes_str = "(BFloat16:[4096, 1, 4096]{strdes=[4096, 1],offset=1}) <- (BFloat16:[4096, 128])"
        MsExecuteOrderParser._get_in_out_memory_attributes(file_data, in_out_memory_attributes_str)

        assert "output_memory_attributes" in file_data
        assert "input_memory_attributes" in file_data
        assert len(file_data["output_memory_attributes"]) == 1
        assert len(file_data["input_memory_attributes"]) == 1

        output_attr = file_data["output_memory_attributes"][0]
        input_attr = file_data["input_memory_attributes"][0]
        assert isinstance(output_attr, MemoryAttribute)
        assert isinstance(input_attr, MemoryAttribute)
        assert output_attr._dtype == "BFloat16"
        assert output_attr._shape == [4096, 1, 4096]
        assert output_attr._exist_stride_offset is True
        assert output_attr._stride == [4096, 1]
        assert output_attr._offset == 1
        assert input_attr._dtype == "BFloat16"
        assert input_attr._shape == [4096, 128]

    def test_get_in_out_memory_ids_and_op(self):
        file_data = {}
        in_out_memory_ids_and_op_str = "(%1445) = Concat(%1441, %1384), task_info: 1444, attrs {stream_id=0}"
        MsExecuteOrderParser._get_in_out_memory_ids_and_op(file_data, in_out_memory_ids_and_op_str)

        assert file_data["output_memory_ids"] == ["%1445"]
        assert file_data["input_memory_ids"] == ["%1441", "%1384"]
        assert file_data["task_info"] == "1444"
        assert file_data["attrs"] == "stream_id=0"
        assert file_data["op_type"] == "Concat"

    def test_two_ops_ir_file(self):
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.ir') as f:
            f.write("(%1) = Add(%2, %3), task_info: 100, attrs {stream_id=0}\n")
            f.write("(BFloat16:[10, 10]) <- (BFloat16:[10, 10], BFloat16:[10, 10])\n")
            f.write("stack_info|\n")
            f.write("(%4) = Mul(%1, %5), task_info: 101, attrs {stream_id=1}\n")
            f.write("(BFloat16:[10, 10]) <- (BFloat16:[10, 10], BFloat16:[10, 10])\n")
            f.write("stack_info2|\n")
            temp_file_path = f.name

        try:
            # 解析器正确解析文件
            parser = MsExecuteOrderParser(temp_file_path, False)

            # 图正确构建
            computer_graph = parser.get_computer_graph()
            node_manager = computer_graph.get_node_manager()
            nodes = node_manager.get_nodes()
            assert len(nodes) == 2
            first_node = nodes[0]
            second_node = nodes[1]
            assert computer_graph.get_graph().has_edge(first_node, second_node)

            # 节点校验
            assert first_node._op_type == "Add"
            assert first_node._task_info == "100"
            assert first_node._input_memory_ids == ["%2", "%3"]
            assert first_node._output_memory_ids == ["%1"]
            assert first_node._attrs == "stream_id=0"
            assert second_node._op_type == "Mul"
            assert second_node._task_info == "101"
            assert second_node._input_memory_ids == ["%1", "%5"]
            assert second_node._output_memory_ids == ["%4"]
            assert second_node._attrs == "stream_id=1"

        finally:
            # 清理临时文件
            os.unlink(temp_file_path)