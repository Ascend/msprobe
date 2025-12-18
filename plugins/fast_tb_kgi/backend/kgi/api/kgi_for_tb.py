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
from backend.kgi.core.comparealg.anchor_compare_all_alg import AnchorCompareAllAlg
from backend.kgi.core.comparealg.mul_anchor_compare_alg import MulAnchorCompareAlg
from backend.kgi.sal.computergraph.computer_graph import ComputerGraph
from backend.kgi.sal.computergraph.computer_node import ComputerNode
from backend.kgi.sal.parser.ms_execute_order_parser import MsExecuteOrderParser
from backend.kgi.sal.parser.pt_execute_order_parser import PtExecuteOrderParser

class KGI_For_Tb:
    def __init__(self):
        self._left_computer_graph = ComputerGraph()
        self._right_computer_graph = ComputerGraph()
        self._compare_all_mode = False
        self._alg: Optional[MulAnchorCompareAlg | AnchorCompareAllAlg] = None
        self._left_anchor: Optional[ComputerNode] = None
        self._right_anchor: Optional[ComputerNode] = None
        self._left_second_level_anchors: list[ComputerNode] = []
        self._right_second_level_anchors: list[ComputerNode] = []
        self._left_nodes: list[dict] = []
        self._right_nodes: list[dict] = []
        self._left_edges: list[dict] = []
        self._right_edges: list[dict] = []
        self._fused_node_id = 1

    def _get_fused_node_op_type(self) -> str:
        fused_node_id = self._fused_node_id
        self._fused_node_id += 1
        return f'FusedNode-{fused_node_id}'

    def _set_mismatch_flag(self):
        if not self._alg:
            return

        mismatch_nodes_id = {node.get_node_id() for node in self._alg.get_left_mismatch_nodes()}
        for node in self._left_nodes:
            if node['id'] in mismatch_nodes_id:
                node['mismatch'] = True

        mismatch_nodes_id = {node.get_node_id() for node in self._alg.get_right_mismatch_nodes()}
        for node in self._right_nodes:
            if node['id'] in mismatch_nodes_id:
                node['mismatch'] = True

    def _clear_mismatch_flag(self):
        for node in self._left_nodes:
            if "mismatch" in node:
                del node["mismatch"]

        for node in self._right_nodes:
            if "mismatch" in node:
                del node["mismatch"]

    def _set_match_flag(self):
        if not self._alg:
            return

        match_nodes_id = {node.get_node_id() for node in self._alg.get_left_match_nodes()}
        for node in self._left_nodes:
            if node['id'] in match_nodes_id:
                node['match'] = True

        match_nodes_id = {node.get_node_id() for node in self._alg.get_right_match_nodes()}
        for node in self._right_nodes:
            if node['id'] in match_nodes_id:
                node['match'] = True

    def _clear_match_flag(self):
        for node in self._left_nodes:
            if "match" in node:
                del node["match"]

        for node in self._right_nodes:
            if "match" in node:
                del node["match"]

    def _set_anchor_flag(self):
        left_anchor_nodes_id = []
        if self._left_anchor:
            left_anchor_nodes_id.append(self._left_anchor.get_node_id())
        left_anchor_nodes_id.extend([node.get_node_id() for node in self._left_second_level_anchors])
        if left_anchor_nodes_id:
            for node in self._left_nodes:
                if node['id'] in left_anchor_nodes_id:
                    node['is_anchor'] = True

        right_anchor_nodes_id = []
        if self._right_anchor:
            right_anchor_nodes_id.append(self._right_anchor.get_node_id())
        right_anchor_nodes_id.extend([node.get_node_id() for node in self._right_second_level_anchors])
        if right_anchor_nodes_id:
            for node in self._right_nodes:
                if node['id'] in right_anchor_nodes_id:
                    node['is_anchor'] = True

    def _clear_anchor_flag(self):
        for node in self._left_nodes:
            if "is_anchor" in node:
                del node["is_anchor"]

        for node in self._right_nodes:
            if "is_anchor" in node:
                del node["is_anchor"]

    @staticmethod
    def _get_node(nodes: list[dict], node_id: int) -> dict:
        for node in nodes:
            if node["id"] == node_id:
                return node

    def get_controls_info(self) -> dict:
        return {
            'compare_all_mode': self._compare_all_mode,
            'has_compared': True if self._alg else False
        }

    def get_graph(self, side: str) -> tuple[list[dict], list[dict]]:
        if side != 'left' and side != 'right':
            raise ValueError(f'Invalid side: {side}')

        if side == 'left':
            return self._left_nodes, self._left_edges
        else:
            return self._right_nodes, self._right_edges

    def change_compare_mode(self) -> tuple[list[dict], list[dict], bool]:
        self._compare_all_mode = not self._compare_all_mode
        self._alg = None
        self._left_second_level_anchors.clear()
        self._right_second_level_anchors.clear()
        self._clear_match_flag()
        self._clear_mismatch_flag()
        self._clear_anchor_flag()
        self._set_anchor_flag()

        return self._left_nodes, self._right_nodes, self._compare_all_mode

    def set_graph(
            self,
            side: str,
            content: str,
            ignore_data_ops: bool,
            execute_order_type: str
    ) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
        if side != 'left' and side != 'right':
            raise ValueError(f'Invalid side: {side}')
        if content == '':
            raise ValueError('Invalid content')

        try:
            if execute_order_type == 'MindSpore':
                computer_graph = MsExecuteOrderParser.parse_for_file(content, ignore_data_ops)
            elif execute_order_type == 'PyTorch':
                computer_graph = PtExecuteOrderParser.parse_for_file(content)
            else:
                raise ValueError(f'Invalid execute_order_type: {execute_order_type}')
        except Exception as e:
            raise IOError(f'Failed to parse content: {str(e)}')
        nodes, edges = computer_graph.convert_to_nodes_and_edges(None)

        self._alg = None
        self._left_second_level_anchors.clear()
        self._right_second_level_anchors.clear()
        if side == 'left':
            self._left_computer_graph = computer_graph
            self._left_anchor = None
            self._left_nodes = nodes
            self._left_edges = edges
        else:
            self._right_computer_graph = computer_graph
            self._right_anchor = None
            self._right_nodes = nodes
            self._right_edges = edges
        self._clear_match_flag()
        self._clear_mismatch_flag()
        self._clear_anchor_flag()
        self._set_anchor_flag()

        return self._left_nodes, self._left_edges, self._right_nodes, self._right_edges

    def change_to_whole_graph(self) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
        self._left_nodes, self._left_edges = self._left_computer_graph.convert_to_nodes_and_edges(None)
        self._right_nodes, self._right_edges = self._right_computer_graph.convert_to_nodes_and_edges(None)

        self._alg = None
        self._left_second_level_anchors.clear()
        self._right_second_level_anchors.clear()
        self._left_anchor = None
        self._right_anchor = None

        return self._left_nodes, self._left_edges, self._right_nodes, self._right_edges

    def set_anchor_pre_check(
            self,
            side: str,
            node_id: int
    ) -> dict:
        if side != 'left' and side != 'right':
            raise ValueError(f'Invalid side: {side}')

        if side == 'left':
            computer_graph = self._left_computer_graph
        else:
            computer_graph = self._right_computer_graph
        node_exist = computer_graph.get_node_manager().get_node_by_node_id(node_id) is not None

        return {
            'node_exist': node_exist,
        }

    def set_anchor(
            self,
            side: str,
            node_id: int
    ) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
        if side != 'left' and side != 'right':
            raise ValueError(f'Invalid side: {side}')

        if side == 'left':
            computer_graph = self._left_computer_graph
        else:
            computer_graph = self._right_computer_graph

        anchor = computer_graph.get_node_manager().get_node_by_node_id(node_id)
        if anchor is None:
            raise ValueError(f'Invalid anchor node line id: {node_id}')
        nodes, edges = computer_graph.convert_to_nodes_and_edges(anchor)

        self._alg = None
        self._left_second_level_anchors.clear()
        self._right_second_level_anchors.clear()
        if side == 'left':
            self._left_nodes = nodes
            self._left_edges = edges
            self._left_anchor = anchor
        else:
            self._right_nodes = nodes
            self._right_edges = edges
            self._right_anchor = anchor
        self._clear_match_flag()
        self._clear_mismatch_flag()
        self._clear_anchor_flag()
        self._set_anchor_flag()

        return self._left_nodes, self._left_edges, self._right_nodes, self._right_edges

    def up_compare(self) -> tuple[list[dict], list[dict]]:
        if not self._alg or not self._alg.get_compare_up():
            if self._compare_all_mode:
                self._alg = AnchorCompareAllAlg(
                    self._left_computer_graph.get_graph(),
                    self._right_computer_graph.get_graph(),
                    self._left_anchor,
                    self._right_anchor,
                    True,
                    True
                )
            else:
                self._alg = MulAnchorCompareAlg(
                    self._left_computer_graph.get_graph(),
                    self._right_computer_graph.get_graph(),
                    self._left_anchor,
                    self._right_anchor,
                    True,
                    True
                )
        self._alg.compare()
        self._clear_match_flag()
        self._clear_mismatch_flag()
        self._clear_anchor_flag()
        self._set_match_flag()
        self._set_mismatch_flag()
        self._set_anchor_flag()

        return self._left_nodes, self._right_nodes

    def down_compare(self) -> tuple[list[ComputerNode], list[ComputerNode]]:
        if not self._alg or self._alg.get_compare_up():
            if self._compare_all_mode:
                self._left_second_level_anchors.clear()
                self._right_second_level_anchors.clear()
                self._alg = AnchorCompareAllAlg(
                    self._left_computer_graph.get_graph(),
                    self._right_computer_graph.get_graph(),
                    self._left_anchor,
                    self._right_anchor,
                    True,
                    False
                )
            else:
                self._left_second_level_anchors.clear()
                self._right_second_level_anchors.clear()
                self._alg = MulAnchorCompareAlg(
                    self._left_computer_graph.get_graph(),
                    self._right_computer_graph.get_graph(),
                    self._left_anchor,
                    self._right_anchor,
                    True,
                    False
                )
        self._alg.compare()
        self._clear_match_flag()
        self._clear_mismatch_flag()
        self._clear_anchor_flag()
        self._set_match_flag()
        self._set_mismatch_flag()
        self._set_anchor_flag()

        return self._left_nodes, self._right_nodes

    def replace_equal_subgraph_pre_check(self, side: str, nodes_id: list[int]) -> dict:
        if side != 'left' and side != 'right':
            raise ValueError(f'Invalid side: {side}')

        if side == 'left':
            has_cycle = self._left_computer_graph.fuse_nodes_has_cycle_check(nodes_id)
        else:
            has_cycle = self._right_computer_graph.fuse_nodes_has_cycle_check(nodes_id)

        return {
            'has_cycle': has_cycle
        }

    def _nodes_edges_diff_cal(
            self,
            left_nodes: list[dict],
            left_edges: list[dict],
            right_nodes: list[dict],
            right_edges: list[dict]
    ) -> tuple[
        list[dict], list[dict], list[dict],
        list[dict], list[dict], list[dict]
    ]:
        old_left_nodes_id = set(node["id"] for node in self._left_nodes)
        new_left_nodes_id = set(node["id"] for node in left_nodes)
        left_del_nodes_id = old_left_nodes_id - new_left_nodes_id
        left_del_nodes = [node for node in self._left_nodes if node["id"] in left_del_nodes_id]
        self._left_nodes = left_nodes
        old_left_edges_id = set(edge["id"] for edge in self._left_edges)
        new_left_edges_id = set(edge["id"] for edge in left_edges)
        left_del_edges_id = old_left_edges_id - new_left_edges_id
        left_add_edges_id = new_left_edges_id - old_left_edges_id
        left_del_edges = [edge for edge in self._left_edges if edge["id"] in left_del_edges_id]
        left_add_edges = [edge for edge in left_edges if edge["id"] in left_add_edges_id]
        self._left_edges = left_edges
        old_right_nodes_id = set(node["id"] for node in self._right_nodes)
        new_right_nodes_id = set(node["id"] for node in right_nodes)
        right_del_nodes_id = old_right_nodes_id - new_right_nodes_id
        right_del_nodes = [node for node in self._right_nodes if node["id"] in right_del_nodes_id]
        self._right_nodes = right_nodes
        old_right_edges_id = set(edge["id"] for edge in self._right_edges)
        new_right_edges_id = set(edge["id"] for edge in right_edges)
        right_del_edges_id = old_right_edges_id - new_right_edges_id
        right_add_edges_id = new_right_edges_id - old_right_edges_id
        right_del_edges = [edge for edge in self._right_edges if edge["id"] in right_del_edges_id]
        right_add_edges = [edge for edge in right_edges if edge["id"] in right_add_edges_id]
        self._right_edges = right_edges

        return (
            left_del_nodes, left_del_edges, left_add_edges,
            right_del_nodes, right_del_edges, right_add_edges
        )

    def replace_equal_subgraph(
            self,
            left_nodes_id: list[int],
            right_nodes_id: list[int]
    ) -> tuple[
        list[dict], list[dict],
        list[dict], list[dict],
        list[dict], list[dict],
        list[dict], list[dict]
    ]:
        if len(left_nodes_id) == 1:
            left_node_id = left_nodes_id[0]
            left_fused_node = self._left_computer_graph.get_node_manager().get_node_by_node_id(left_node_id)
            if left_fused_node is None:
                raise ValueError(f'Invalid left node id: {left_node_id}')
            right_fused_node = self._right_computer_graph.fuse_nodes(
                right_nodes_id,
                left_fused_node,
                left_fused_node.get_op_type()
            )
        elif len(right_nodes_id) == 1:
            right_node_id = right_nodes_id[0]
            right_fused_node = self._right_computer_graph.get_node_manager().get_node_by_node_id(right_node_id)
            if right_fused_node is None:
                raise ValueError(f'Invalid right node id: {right_node_id}')
            left_fused_node = self._left_computer_graph.fuse_nodes(
                left_nodes_id,
                right_fused_node,
                right_fused_node.get_op_type()
            )
        else:
            fused_node_op_type = self._get_fused_node_op_type()
            left_node_id = left_nodes_id[0]
            peer_node = self._left_computer_graph.get_node_manager().get_node_by_node_id(left_node_id)
            if peer_node is None:
                    raise ValueError(f'Invalid left node id: {left_node_id}')
            right_fused_node = self._right_computer_graph.fuse_nodes(
                right_nodes_id,
                peer_node,
                fused_node_op_type
            )
            left_fused_node = self._left_computer_graph.fuse_nodes(
                left_nodes_id,
                right_fused_node,
                fused_node_op_type
            )

        if self._left_anchor:
            if self._left_anchor.get_node_id() in left_nodes_id:
                self._left_anchor = left_fused_node
        if self._right_anchor:
            if self._right_anchor.get_node_id() in right_nodes_id:
                self._right_anchor = right_fused_node

        left_nodes, left_edges = self._left_computer_graph.convert_to_nodes_and_edges(self._left_anchor)
        right_nodes, right_edges = self._right_computer_graph.convert_to_nodes_and_edges(self._right_anchor)

        (
            left_del_nodes, left_del_edges, left_add_edges,
            right_del_nodes, right_del_edges, right_add_edges
        ) = self._nodes_edges_diff_cal(left_nodes, left_edges, right_nodes, right_edges)
        self._alg = None
        self._left_second_level_anchors.clear()
        self._right_second_level_anchors.clear()
        self._clear_match_flag()
        self._clear_mismatch_flag()
        self._clear_anchor_flag()
        self._set_anchor_flag()

        return (
            left_del_nodes, self._left_nodes,
            left_del_edges, left_add_edges,
            right_del_nodes, self._right_nodes,
            right_del_edges, right_add_edges
        )

    def del_nodes_pre_check(self, side: str, nodes_id: list[int]) -> dict:
        if side != 'left' and side != 'right':
            raise ValueError(f'Invalid side: {side}')

        if side == 'left':
            has_anchor = self._left_anchor.get_node_id() in nodes_id if self._left_anchor else False
        else:
            has_anchor = self._right_anchor.get_node_id() in nodes_id if self._right_anchor else False

        return {
            'has_anchor': has_anchor
        }

    def del_nodes(
            self,
            left_nodes_id: list[int],
            right_nodes_id: list[int]
    ) -> tuple[
        list[dict], list[dict], list[dict],
        list[dict], list[dict], list[dict]
    ]:
        self._left_computer_graph.del_nodes(left_nodes_id)
        self._right_computer_graph.del_nodes(right_nodes_id)

        left_nodes, left_edges = self._left_computer_graph.convert_to_nodes_and_edges(self._left_anchor)
        right_nodes, right_edges = self._right_computer_graph.convert_to_nodes_and_edges(self._right_anchor)

        (
            left_del_nodes, left_del_edges, _,
            right_del_nodes, right_del_edges, _
        ) = self._nodes_edges_diff_cal(left_nodes, left_edges, right_nodes, right_edges)
        self._alg = None
        self._left_second_level_anchors.clear()
        self._right_second_level_anchors.clear()
        self._clear_match_flag()
        self._clear_mismatch_flag()
        self._clear_anchor_flag()
        self._set_anchor_flag()

        return (left_del_nodes, self._left_nodes, left_del_edges,
                right_del_nodes, self._right_nodes, right_del_edges)

    def del_edges(
            self,
            left_edges: list[list[int, int]],
            right_edges: list[list[int, int]]
    ) -> tuple[
        list[dict], list[dict], list[dict],
        list[dict], list[dict], list[dict]
    ]:
        left_edges = [
            [
                self._left_computer_graph.get_node_manager().get_node_by_node_id(from_node_id),
                self._left_computer_graph.get_node_manager().get_node_by_node_id(to_node_id)
            ] for from_node_id, to_node_id in left_edges
        ]
        right_edges = [
            [
                self._right_computer_graph.get_node_manager().get_node_by_node_id(from_node_id),
                self._right_computer_graph.get_node_manager().get_node_by_node_id(to_node_id)
            ] for from_node_id, to_node_id in right_edges
        ]
        self._left_computer_graph.del_edges(left_edges)
        self._right_computer_graph.del_edges(right_edges)

        left_nodes, left_edges = self._left_computer_graph.convert_to_nodes_and_edges(self._left_anchor)
        right_nodes, right_edges = self._right_computer_graph.convert_to_nodes_and_edges(self._right_anchor)

        (
            left_del_nodes, left_del_edges, _,
            right_del_nodes, right_del_edges, _
        ) = self._nodes_edges_diff_cal(left_nodes, left_edges, right_nodes, right_edges)
        self._alg = None
        self._left_second_level_anchors.clear()
        self._right_second_level_anchors.clear()
        self._clear_match_flag()
        self._clear_mismatch_flag()
        self._clear_anchor_flag()
        self._set_anchor_flag()

        return (left_del_nodes, self._left_nodes, left_del_edges,
                right_del_nodes, self._right_nodes, right_del_edges)


    def set_second_level_anchor(self, left_node_id: int, right_node_id: int) -> tuple[list[dict], list[dict]]:
        if self._alg is None:
            raise ValueError('Alg is None')

        left_anchor = self._left_computer_graph.get_node_manager().get_node_by_node_id(left_node_id)
        right_anchor = self._right_computer_graph.get_node_manager().get_node_by_node_id(right_node_id)
        self._left_second_level_anchors.append(left_anchor)
        self._right_second_level_anchors.append(right_anchor)
        self._alg.add_second_level_anchor_compare_alg(left_anchor, right_anchor)

        self._set_anchor_flag()
        left_node = KGI_For_Tb._get_node(self._left_nodes, left_node_id)
        right_node = KGI_For_Tb._get_node(self._right_nodes, right_node_id)

        return [left_node], [right_node]

    def del_second_level_anchor_pre_check(self, left_nodes_id: list[int], right_nodes_id: list[int]) -> dict:
        if self._alg is None:
            raise ValueError('No algorithm')

        left_second_level_anchors_id = set(node.get_node_id() for node in self._left_second_level_anchors)
        right_second_level_anchors_id = set(node.get_node_id() for node in self._right_second_level_anchors)
        left_not_anchor_nodes_id = set(left_nodes_id) - left_second_level_anchors_id
        right_not_anchor_nodes_id = set(right_nodes_id) - right_second_level_anchors_id

        return {
            'left_not_anchor_nodes_id': list(left_not_anchor_nodes_id),
            'right_not_anchor_nodes_id': list(right_not_anchor_nodes_id)
        }

    def del_second_level_anchor(
            self,
            left_nodes_id: list[int],
            right_nodes_id: list[int]
    ) -> tuple[list[dict], list[dict]]:
        self._alg.del_second_level_anchor_compare_alg(left_nodes_id, right_nodes_id)

        need_del_ids = []
        for need_del_id in range(len(self._left_second_level_anchors)):
            left_anchor = self._left_second_level_anchors[need_del_id]
            right_anchor = self._right_second_level_anchors[need_del_id]
            if (left_anchor.get_node_id() in left_nodes_id or
                right_anchor.get_node_id() in right_nodes_id):
                need_del_ids.append(need_del_id)
        for need_del_id in need_del_ids[::-1]:
            del self._left_second_level_anchors[need_del_id]
            del self._right_second_level_anchors[need_del_id]

        self._clear_match_flag()
        self._clear_mismatch_flag()
        self._clear_anchor_flag()
        self._set_match_flag()
        self._set_mismatch_flag()
        self._set_anchor_flag()

        return self._left_nodes, self._right_nodes

    def get_node_info(self, side: str, node_id: int) -> dict:
        if side != 'left' and side != 'right':
            raise ValueError(f'Invalid side: {side}')

        if side == "left":
            node = self._left_computer_graph.get_node_manager().get_node_by_node_id(node_id)
        else:
            node = self._right_computer_graph.get_node_manager().get_node_by_node_id(node_id)

        node_info = dict()
        node_info["line_id"] = node.get_node_id()
        node_info["op_type"] = node.get_op_type()
        input_memory_ids = list(str(memory_id)
                                for memory_id in node.get_input_memory_ids())
        node_info["input_memory_num"] = len(input_memory_ids)
        node_info["input_memory_ids"] = input_memory_ids
        input_memory_attrs = list(str(memory_attribute)
                                       for memory_attribute in node.get_input_memory_attributes())
        node_info["input_memory_attrs"] = input_memory_attrs
        output_memory_ids = list(str(memory_id)
                                 for memory_id in node.get_output_memory_ids())
        node_info["output_memory_num"] = len(output_memory_ids)
        node_info["output_memory_ids"] = output_memory_ids
        output_memory_attrs = list(str(memory_attribute)
                                        for memory_attribute in node.get_output_memory_attributes())
        node_info["output_memory_attrs"] = output_memory_attrs
        scope = node.get_scope()
        if scope:
            node_info["scope"] = node.get_scope()
        stack = node.get_stack()
        if stack:
            node_info["stack"] = node.get_stack()

        return node_info