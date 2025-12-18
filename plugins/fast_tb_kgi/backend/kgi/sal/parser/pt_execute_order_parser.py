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

import json
import os
import re
from backend.kgi.core.utils.graph_struct_analyze_utils import print_cycle_number, print_subgraph_number
from backend.kgi.core.utils.graph_struct_analyze_utils import print_subgraphs_node_number
from backend.kgi.sal.computergraph.computer_graph import ComputerGraph
from backend.kgi.sal.computergraph.computer_node import ComputerNode, MemoryAttribute

ACLNN_FILE_PATH = os.path.join(os.path.dirname(__file__), 'aclnns.json')

class PtExecuteOrderParser:
    """PyTorch执行序解析器
    """
    def __init__(self, path: str):
        self._computer_graph: ComputerGraph = ComputerGraph()
        self._aclnns_input_arg_id, self._aclnns_output_arg_id = PtExecuteOrderParser.get_aclnns_input_output_arg_id()

        file_datas = PtExecuteOrderParser._get_file_datas_for_path(path)
        self._get_graph(file_datas)

    def get_computer_graph(self) -> ComputerGraph:
        return self._computer_graph

    @staticmethod
    def get_aclnns_input_output_arg_id() -> tuple[dict[str, list[int]], dict[str, list[int]]]:
        with open(ACLNN_FILE_PATH, 'r', encoding='utf-8') as f:
            aclnns_arg = json.load(f)
        aclnns_input_arg_id = dict()
        aclnns_output_arg_id = dict()
        for aclnn_name, args in aclnns_arg.items():
            aclnns_input_arg_id[aclnn_name] = \
                list(arg_id for arg_id, arg in enumerate(args) if '输入' in arg)
            aclnns_output_arg_id[aclnn_name] = \
                list(arg_id for arg_id, arg in enumerate(args) if '输出' in arg)
        return aclnns_input_arg_id, aclnns_output_arg_id

    @staticmethod
    def parse_for_path(path: str) -> ComputerGraph:
        parser = PtExecuteOrderParser(path)
        return parser.get_computer_graph()

    @staticmethod
    def parse_for_file(content: str) -> ComputerGraph:
        parser = PtExecuteOrderParser.__new__(PtExecuteOrderParser)
        parser._computer_graph = ComputerGraph()
        parser._aclnns_input_arg_id, parser._aclnns_output_arg_id = PtExecuteOrderParser.get_aclnns_input_output_arg_id()

        file_datas = PtExecuteOrderParser._get_file_datas_for_file(content)
        parser._get_graph(file_datas)

        return parser.get_computer_graph()

    @staticmethod
    def _get_op_type_from_line(line: str) -> str:
        pattern = r"(.*) (.*) EXEC_NPU_CMD(.*)"
        return re.findall(pattern, line)[0][1]

    @staticmethod
    def _get_data_dtype_shape_offset_stride(line: str, line_data: dict, arg_id: int):
        pattern = r"(.*?):(.*)Tensor size: \[(.*)\], stride: \[(.*)\], offset: (\d+), dtype: (.*), " \
            r"device ID: (.*), data_ptr: (.*)"
        matchs = re.findall(pattern, line)
        dtype = matchs[0][5]
        shape = [int(x.strip()) for x in str(matchs[0][2]).split(",")]
        stride = [int(x.strip()) for x in str(matchs[0][3]).split(",")]
        offset = int(matchs[0][4])
        data = matchs[0][7]
        line_data[f"arg{arg_id}"] = {
            "dtype": dtype,
            "shape": shape,
            "stride": stride,
            "offset": offset,
            "data": data
        }

    @staticmethod
    def _get_file_datas(lines: list[str]) -> list[dict]:
        file_datas = []
        last_line_id = len(lines) - 1
        cur_line_id = 0
        while True:
            if cur_line_id > last_line_id:
                break
            line = lines[cur_line_id]
            if 'EXEC_NPU_CMD' in line:
                file_data = dict()
                file_datas.append(file_data)
                file_data["line_id"] = cur_line_id + 1
                op_type = PtExecuteOrderParser._get_op_type_from_line(line)
                file_data["op_type"] = op_type

                cur_line_id += 1
                arg_id = 0
                while True:
                    if cur_line_id > last_line_id:
                        break
                    line = lines[cur_line_id]
                    if 'torch_npu.op_plugin' in line:
                        break
                    if 'arg_names info' in line:
                        cur_line_id += 1
                        continue
                    if 'data_ptr' in line:
                        PtExecuteOrderParser._get_data_dtype_shape_offset_stride(line, file_data, arg_id)
                        cur_line_id += 1
                        arg_id += 1
                    else:
                        cur_line_id += 1
                        arg_id += 1
            else:
                cur_line_id += 1

        return file_datas

    @staticmethod
    def _get_file_datas_for_path(file_path: str) -> list[dict]:
        with open(file_path, "r") as f:
            lines = f.readlines()

        return PtExecuteOrderParser._get_file_datas(lines)

    @staticmethod
    def _get_file_datas_for_file(content: str) -> list[dict]:
        lines = content.strip().split('\n')

        return PtExecuteOrderParser._get_file_datas(lines)

    def _add_nodes(self, file_datas: list[dict]):
        for file_data in file_datas:
            node = ComputerNode()
            node.set_node_id(file_data["line_id"])
            op_type = file_data["op_type"]
            node.set_op_type(op_type)

            input_memory_ids = []
            input_memory_attrs = []
            if op_type not in self._aclnns_input_arg_id:
                raise ValueError(f"Operator type {op_type} is not supported")
            for arg_id in self._aclnns_input_arg_id[op_type]:
                if not file_data.get(f"arg{arg_id}"):
                    continue
                input_memory_ids.append(file_data[f"arg{arg_id}"]["data"])
                mem_attr = MemoryAttribute()
                mem_attr.set_dtype(file_data[f"arg{arg_id}"]["dtype"])
                mem_attr.set_shape(file_data[f"arg{arg_id}"]["shape"])
                mem_attr.set_exist_stride_offset(True)
                mem_attr.set_stride(file_data[f"arg{arg_id}"]["stride"])
                mem_attr.set_offset(file_data[f"arg{arg_id}"]["offset"])
                input_memory_attrs.append(mem_attr)
            node.set_input_memory_ids(input_memory_ids)
            node.set_input_memory_attributes(input_memory_attrs)

            output_memory_ids = []
            output_memory_attrs = []
            for arg_id in self._aclnns_output_arg_id[op_type]:
                if not file_data.get(f"arg{arg_id}"):
                    continue
                output_memory_ids.append(file_data[f"arg{arg_id}"]["data"])
                mem_attr = MemoryAttribute()
                mem_attr.set_dtype(file_data[f"arg{arg_id}"]["dtype"])
                mem_attr.set_shape(file_data[f"arg{arg_id}"]["shape"])
                mem_attr.set_exist_stride_offset(True)
                mem_attr.set_stride(file_data[f"arg{arg_id}"]["stride"])
                mem_attr.set_offset(file_data[f"arg{arg_id}"]["offset"])
                output_memory_attrs.append(mem_attr)
            node.set_output_memory_ids(output_memory_ids)
            node.set_output_memory_attributes(output_memory_attrs)

            self._computer_graph.add_node(node)

    @staticmethod
    def _get_build_edge_key(memory_id: str, memory_attribute: MemoryAttribute):
        return f"{memory_id}"

    @staticmethod
    def _can_build_edge(
            memory_id: str,
            memory_attribute: MemoryAttribute,
            predecessors: dict[str, ComputerNode]
    ) -> bool:
        build_edge_key = PtExecuteOrderParser._get_build_edge_key(memory_id, memory_attribute)
        if predecessors.get(build_edge_key) is None:
            return False
        return True

    @staticmethod
    def _get_build_edge_predecessor(
            memory_id: str,
            memory_attribute: MemoryAttribute,
            predecessors: dict[str, ComputerNode]
    ) -> ComputerNode:
        build_edge_key = PtExecuteOrderParser._get_build_edge_key(memory_id, memory_attribute)
        return predecessors[build_edge_key]

    @staticmethod
    def _update_build_edge_predecessors(
            memory_id: str,
            memory_attribute: MemoryAttribute,
            predecessors: dict[str, ComputerNode],
            predecessor: ComputerNode
    ):
        build_edge_key = PtExecuteOrderParser._get_build_edge_key(memory_id, memory_attribute)
        predecessors[build_edge_key] = predecessor

    def _add_edges(self):
        build_edge_predecessors: dict[str, ComputerNode] = {}  # 记录前驱
        for node in self._computer_graph.get_node_manager().get_nodes():
            input_memory_ids = node.get_input_memory_ids()
            if input_memory_ids:
                for input_memory_id, input_memory_attribute in zip(input_memory_ids,
                                                                   node.get_input_memory_attributes()):
                    if not PtExecuteOrderParser._can_build_edge(
                        input_memory_id,
                        input_memory_attribute,
                        build_edge_predecessors
                    ):
                        continue
                    predecessor = PtExecuteOrderParser._get_build_edge_predecessor(
                        input_memory_id,
                        input_memory_attribute,
                        build_edge_predecessors
                    )
                    self._computer_graph.add_edge(predecessor, node, input_memory_id, {input_memory_attribute: 1})

            output_memory_ids = node.get_output_memory_ids()
            if output_memory_ids:
                for output_memory_id, output_memory_attribute in zip(output_memory_ids,
                                                                     node.get_output_memory_attributes()):
                    PtExecuteOrderParser._update_build_edge_predecessors(
                        output_memory_id,
                        output_memory_attribute,
                        build_edge_predecessors,
                        node
                    )

    def _get_graph(self, file_datas: list[dict]):
        self._add_nodes(file_datas)
        self._add_edges()
        print_cycle_number(self._computer_graph.get_graph())
        print_subgraph_number(self._computer_graph.get_graph())
        print_subgraphs_node_number(self._computer_graph.get_graph())
        print(f"Get graph, node num is {len(self._computer_graph.get_node_manager().get_nodes())}.")