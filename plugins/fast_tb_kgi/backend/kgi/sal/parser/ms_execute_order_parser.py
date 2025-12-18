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

import re
from backend.kgi.core.utils.graph_struct_analyze_utils import print_cycle_number, print_subgraph_number
from backend.kgi.core.utils.graph_struct_analyze_utils import print_subgraphs_node_number
from backend.kgi.sal.computergraph.computer_graph import ComputerGraph
from backend.kgi.sal.computergraph.computer_node import ComputerNode, MemoryAttribute

class MsExecuteOrderParser:
    """MindSpore执行序解析器
    """
    def __init__(self, path: str, ignore_data_ops: bool):
        self._computer_graph: ComputerGraph = ComputerGraph()
        self._ignore_data_ops = ignore_data_ops

        file_datas = MsExecuteOrderParser._get_file_datas_for_path(path)
        self._get_graph(file_datas)

    def get_computer_graph(self) -> ComputerGraph:
        return self._computer_graph

    @staticmethod
    def parse_for_path(path: str, ignore_data_ops: bool) -> ComputerGraph:
        parser = MsExecuteOrderParser(path, ignore_data_ops)
        return parser.get_computer_graph()

    @staticmethod
    def parse_for_file(content: str, ignore_data_ops: bool) -> ComputerGraph:
        parser = MsExecuteOrderParser.__new__(MsExecuteOrderParser)
        parser._computer_graph = ComputerGraph()
        parser._ignore_data_ops = ignore_data_ops

        file_datas = MsExecuteOrderParser._get_file_datas_for_file(content)
        parser._get_graph(file_datas)

        return parser.get_computer_graph()

    @staticmethod
    def _get_scope_and_op_type(scope_str: str) -> tuple[list[str], str]:
        scope = []
        op_type = scope_str
        scope_op_type_pattern = r"(.*)/([^/-]+)-"
        scope_op_type_matchs = re.findall(scope_op_type_pattern, scope_str)
        if len(scope_op_type_matchs) != 0:
            scope = scope_str.split("/")
            op_type = scope_op_type_matchs[0][1]

        return scope, op_type

    @staticmethod
    def _get_in_out_memory_ids_and_op(file_data: dict, in_out_memory_ids_and_op_str: str):
        # (%1445) = Concat(%1441, %1384), task_info: 1444, attrs {stream_id=0}
        in_out_memory_ids_and_op_pattern = r"\((.*)\) = (.*)\((.*)\), task_info: (\d+), attrs \{(.*)\}"
        in_out_memory_ids_and_op_matchs = re.findall(in_out_memory_ids_and_op_pattern,
                                                          in_out_memory_ids_and_op_str)
        in_out_memory_ids_and_op_match = in_out_memory_ids_and_op_matchs[0]

        output_memory_ids = str(in_out_memory_ids_and_op_match[0])
        output_memory_ids = output_memory_ids.replace(" ", "").split(",") if output_memory_ids != "" else []

        scope_str = in_out_memory_ids_and_op_match[1]
        scope, op_type = MsExecuteOrderParser._get_scope_and_op_type(scope_str)

        input_memory_ids = str(in_out_memory_ids_and_op_match[2])
        input_memory_ids = input_memory_ids.replace(" ", "").split(",") if input_memory_ids != "" else []

        task_info = in_out_memory_ids_and_op_match[3]

        attrs = in_out_memory_ids_and_op_match[4]

        file_data["output_memory_ids"] = output_memory_ids
        file_data["scope"] = scope
        file_data["op_type"] = op_type
        file_data["input_memory_ids"] = input_memory_ids
        file_data["task_info"] = task_info
        file_data["attrs"] = attrs

    @staticmethod
    def _get_stride_offset(stride_offset_str: str, memory_attribute: MemoryAttribute):
        if stride_offset_str == "":
            return

        memory_attribute.set_exist_stride_offset()

        # strdes=[12288, 12288, 384, 1],offset=0
        stride_offset_pattern = r"strdes=\[([\d, ]*)\],offset=(\d*)"
        stride_offset_matchs = re.findall(stride_offset_pattern, stride_offset_str)
        stride_offset_match = stride_offset_matchs[0]
        stride_str = str(stride_offset_match[0])
        if stride_str != "":
            stride = stride_str.split(",")
            memory_attribute.set_stride([int(x.strip()) for x in stride])
        memory_attribute.set_offset(int(stride_offset_match[1]))

    @staticmethod
    def _get_memory_attributes(memory_attributes_str: str) -> list[MemoryAttribute]:
        if memory_attributes_str == "":
            return []

        # BFloat16:[4096, 1, 32, 128]{strdes=[12288, 12288, 384, 1],offset=0}
        memory_attribute_pattern = r"(\w*):\[([\d, ]*)\](?:\{(.*?)\})?"
        memory_attribute_matchs = re.findall(memory_attribute_pattern, memory_attributes_str)
        memory_attributes = []
        for memory_attribute_match in memory_attribute_matchs:
            memory_attribute = MemoryAttribute()
            memory_attribute.set_dtype(memory_attribute_match[0])
            if memory_attribute_match[1] != "":
                shape_str = str(memory_attribute_match[1])
                shape = shape_str.split(",")
                memory_attribute.set_shape([int(x.strip()) for x in shape])
            MsExecuteOrderParser._get_stride_offset(memory_attribute_match[2], memory_attribute)
            memory_attributes.append(memory_attribute)

        return memory_attributes

    @staticmethod
    def _get_in_out_memory_attributes(file_data: dict, in_out_memory_attributes_str: str):
        # (BFloat16:[4096, 1, 4096]{strdes=[4096, 4096, 1],offset=0}) <- (BFloat16:[4096, 1, 32, 128])
        in_out_memory_attributes_pattern = r"\((.*)\) <- \((.*)\)"
        in_out_memory_attributes_matchs = re.findall(in_out_memory_attributes_pattern, in_out_memory_attributes_str)
        in_out_memory_attributes_match = in_out_memory_attributes_matchs[0]
        out_memory_attributes_str = in_out_memory_attributes_match[0]
        in_memory_attributes_str = in_out_memory_attributes_match[1]

        file_data["output_memory_attributes"] = MsExecuteOrderParser._get_memory_attributes(out_memory_attributes_str)
        file_data["input_memory_attributes"] = MsExecuteOrderParser._get_memory_attributes(in_memory_attributes_str)

    @staticmethod
    def _get_file_datas_for_path(file_path: str) -> list[dict]:
        with open(file_path, "r") as f:
            lines = f.readlines()

        file_datas = []
        for line_id in range(0, len(lines) - 1, 3):
            file_data = {}

            in_out_memory_ids_and_op_str = lines[line_id]
            MsExecuteOrderParser._get_in_out_memory_ids_and_op(file_data, in_out_memory_ids_and_op_str)

            in_out_memory_attributes_str = lines[line_id + 1]
            MsExecuteOrderParser._get_in_out_memory_attributes(file_data, in_out_memory_attributes_str)

            stack_str = lines[line_id + 2]
            file_data["stack"] = stack_str.split("|")
            file_data["line_id"] = line_id + 1

            file_datas.append(file_data)

        return file_datas

    @staticmethod
    def _get_file_datas_for_file(content: str) -> list[dict]:
        lines = content.strip().split('\n')

        file_datas = []
        for line_id in range(0, len(lines) - 1, 3):
            file_data = {}

            in_out_memory_ids_and_op_str = lines[line_id]
            MsExecuteOrderParser._get_in_out_memory_ids_and_op(file_data, in_out_memory_ids_and_op_str)

            in_out_memory_attributes_str = lines[line_id + 1]
            MsExecuteOrderParser._get_in_out_memory_attributes(file_data, in_out_memory_attributes_str)

            stack_str = lines[line_id + 2]
            file_data["stack"] = stack_str.split("|")
            file_data["line_id"] = line_id + 1

            file_datas.append(file_data)

        return file_datas

    def _add_nodes(self, file_datas: list[dict]):
        for file_data in file_datas:
            node = ComputerNode()
            node.set_node_id(file_data["line_id"])
            node.set_op_type(file_data["op_type"])
            node.set_scope(file_data["scope"])
            node.set_stack(file_data["stack"])
            node.set_input_memory_ids(file_data["input_memory_ids"])
            node.set_output_memory_ids(file_data["output_memory_ids"])
            node.set_input_memory_attributes(file_data["input_memory_attributes"])
            node.set_output_memory_attributes(file_data["output_memory_attributes"])
            node.set_task_info(file_data["task_info"])
            node.set_attrs(file_data["attrs"])

            self._computer_graph.add_node(node)

    @staticmethod
    def _get_build_edge_key(memory_id: str, memory_attribute: MemoryAttribute):
        return f"{memory_id}+{memory_attribute}"

    @staticmethod
    def _can_build_edge(
            memory_id: str,
            memory_attribute: MemoryAttribute,
            predecessors: dict[str, ComputerNode]
    ) -> bool:
        build_edge_key = MsExecuteOrderParser._get_build_edge_key(memory_id, memory_attribute)
        if predecessors.get(build_edge_key) is None:
            return False
        return True

    @staticmethod
    def _get_build_edge_predecessor(
            memory_id: str,
            memory_attribute: MemoryAttribute,
            predecessors: dict[str, ComputerNode]
    ) -> ComputerNode:
        build_edge_key = MsExecuteOrderParser._get_build_edge_key(memory_id, memory_attribute)
        return predecessors[build_edge_key]

    @staticmethod
    def _update_build_edge_predecessors(
            memory_id: str,
            memory_attribute: MemoryAttribute,
            predecessors: dict[str, ComputerNode],
            predecessor: ComputerNode
    ):
        build_edge_key = MsExecuteOrderParser._get_build_edge_key(memory_id, memory_attribute)
        predecessors[build_edge_key] = predecessor

    def _add_edges(self):
        build_edge_predecessors: dict[str, ComputerNode] = {}  # 记录前驱
        for node in self._computer_graph.get_node_manager().get_nodes():
            input_memory_ids = node.get_input_memory_ids()
            if input_memory_ids:
                for input_memory_id, input_memory_attribute in zip(input_memory_ids,
                                                                   node.get_input_memory_attributes()):
                    if not MsExecuteOrderParser._can_build_edge(
                        input_memory_id,
                        input_memory_attribute,
                        build_edge_predecessors
                    ):
                        continue
                    predecessor = MsExecuteOrderParser._get_build_edge_predecessor(
                        input_memory_id,
                        input_memory_attribute,
                        build_edge_predecessors
                    )
                    self._computer_graph.add_edge(predecessor, node, input_memory_id, {input_memory_attribute: 1})

            output_memory_ids = node.get_output_memory_ids()
            if output_memory_ids:
                for output_memory_id, output_memory_attribute in zip(output_memory_ids,
                                                                     node.get_output_memory_attributes()):
                    MsExecuteOrderParser._update_build_edge_predecessors(
                        output_memory_id,
                        output_memory_attribute,
                        build_edge_predecessors,
                        node
                    )

    def _get_graph(self, file_datas: list[dict]):
        self._add_nodes(file_datas)
        self._add_edges()
        if self._ignore_data_ops:
            self._computer_graph.ignore_data_ops()
        print_cycle_number(self._computer_graph.get_graph())
        print_subgraph_number(self._computer_graph.get_graph())
        print_subgraphs_node_number(self._computer_graph.get_graph())
        print(f"Get graph, node num is {len(self._computer_graph.get_node_manager().get_nodes())}.")