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

import networkx as nx
from typing import Iterator, Optional, cast
from backend.kgi.core.comparealg.compare_alg import CompareAlg
from backend.kgi.core.structs.node import Node
from backend.kgi.core.utils import graph_utils

class AnchorCompareAllAlg(CompareAlg):
    """基于锚点比较算法, 比较所有节点
    """
    def __init__(
            self,
            left_graph: nx.DiGraph,
            right_graph: nx.DiGraph,
            left_anchor: Optional[Node],
            right_anchor: Optional[Node],
            grow_up: bool,
            compare_up: bool
    ):
        """_summary_

        Parameters
        ----------
        left_graph : nx.DiGraph
            待比较的左图
        right_graph : nx.DiGraph
            待比较的右图
        left_anchor : Optional[Node]
            左图锚点, 如果为None, 表示整图
        right_anchor : Optional[Node]
            右图锚点, 如果为None, 表示整图
        grow_up : bool
            基于锚点生成子图的方向,
            True表示基于锚点向上生成子图, 即锚点及其依赖节点生成的子图,
            False表示基于锚点向下生成子图, 即锚点及其影响节点生成的子图
        compare_up : bool
            比较方向, True表示从下向上比较, False表示从上向下比较
        """
        super().__init__(left_graph, right_graph)
        self._left_anchor: Optional[Node] = left_anchor
        self._right_anchor: Optional[Node] = right_anchor
        self._grow_up: bool = grow_up
        self._compare_up: bool = compare_up
        self._left_subgraph: nx.DiGraph  # 基于锚点生成的子图
        self._right_subgraph: nx.DiGraph
        self._left_topo_layers: list[set[Node]] = []  # 基于锚点生成的子图的拓扑分层
        self._right_topo_layers: list[set[Node]] = []
        self._left_key_times: list[dict[str, int]] = []  # 记录每一拓扑层key值出现的次数
        self._right_key_times: list[dict[str, int]] = []
        self._left_key_nodes: list[dict[str, set[Node]]] = []  # 记录每一拓扑层key值对应节点
        self._right_key_nodes: list[dict[str, set[Node]]] = []
        self._mismatch_keys: list[set[str]] = []  # 记录每一拓扑层不匹配的key值
        self._left_match_nodes: set[Node] = set()  # 记录匹配的节点
        self._right_match_nodes: set[Node] = set()
        self._left_mismatch_nodes: set[Node] = set()  # 记录不匹配的节点
        self._right_mismatch_nodes: set[Node] = set()

    def get_left_anchor(self) -> Optional[Node]:
        return self._left_anchor

    def get_right_anchor(self) -> Optional[Node]:
        return self._right_anchor

    def get_grow_up(self) -> bool:
        return self._grow_up

    def get_compare_up(self) -> bool:
        return self._compare_up

    def get_left_subgraph(self) -> nx.DiGraph:
        return self._left_subgraph

    def get_right_subgraph(self) -> nx.DiGraph:
        return self._right_subgraph

    def get_left_topo_layers(self) -> list[set[Node]]:
        return self._left_topo_layers

    def get_right_topo_layers(self) -> list[set[Node]]:
        return self._right_topo_layers

    def get_left_key_times(self) -> list[dict[str, int]]:
        return self._left_key_times
    
    def get_right_key_times(self) -> list[dict[str, int]]:
        return self._right_key_times

    def get_left_key_nodes(self) -> list[dict[str, set[Node]]]:
        return self._left_key_nodes

    def get_right_key_nodes(self) -> list[dict[str, set[Node]]]:
        return self._right_key_nodes

    def get_mismatch_keys(self) -> list[set[str]]:
        return self._mismatch_keys

    def get_left_match_nodes(self) -> set[Node]:
        return self._left_match_nodes

    def get_right_match_nodes(self) -> set[Node]:
        return self._right_match_nodes

    def get_left_mismatch_nodes(self) -> set[Node]:
        return self._left_mismatch_nodes

    def get_right_mismatch_nodes(self) -> set[Node]:
        return self._right_mismatch_nodes

    @staticmethod
    def _get_subgraph_and_topo_layers(
            graph: nx.DiGraph,
            anchor: Optional[Node],
            grow_up: bool
    ) -> tuple[nx.DiGraph, list[set[Node]]]:
        """获取基于锚点生成的子图和子图的拓扑分层

        Parameters
        ----------
        graph : nx.DiGraph
            原图
        anchor : Optional[Node]
            锚点
        grow_up : bool
            基于锚点生成子图的方向

        Returns
        -------
        nx.DiGraph
            基于锚点生成的子图
        list[set[Node]]
            子图的拓扑分层
        """
        subgraph = graph_utils.get_subgraph_without_anchor(graph, anchor, grow_up)  # 不需要比较锚点，所以扣除锚点
        topo_layers = graph_utils.get_opt_topo_layers(subgraph)

        return subgraph, topo_layers

    @staticmethod
    def _get_node_key(cur_node: Node, graph: nx.DiGraph, compare_up: bool) -> str:
        """获取节点key值

        Parameters
        ----------
        cur_node : Node
            当前节点
        graph : nx.DiGraph
            节点所在图
        compare_up : bool
            比较方向

        Returns
        -------
        str
            key值
        """
        get_around = graph.successors
        if compare_up:
            get_around = graph.predecessors

        nodes_feature = [cur_node.get_node_feature()]
        around_nodes = sorted(
            cast(Iterator[Node], get_around(cur_node)),
            key=lambda node: node.get_node_feature()
        )
        for around_node in around_nodes:
            nodes_feature.append(around_node.get_node_feature())

        return "@".join(nodes_feature)

    @staticmethod
    def _record_key_times(record: dict[str, int], key: str):
        """记录key值出现次数

        Parameters
        ----------
        record : dict[str, int]
            key值出现次数记录
        key : str
            key值
        """
        record.setdefault(key, 0)
        record[key] += 1

    @staticmethod
    def _record_key_nodes(record: dict[str, set[Node]], key: str, node: Node):
        """记录key值对应节点

        Parameters
        ----------
        record : dict[str, set[Node]]
            key值对应节点记录
        key : str
            key值
        node : Node
            节点
        """
        record.setdefault(key, set())
        record[key].add(node)

    @staticmethod
    def _in_compare_area(node: Node, graph: nx.DiGraph, compare_up: bool, mismatch_nodes: set[Node]) -> bool:
        """判断节点是否需要参与key值比较

        Parameters
        ----------
        node : Node
            节点
        graph : nx.DiGraph
            节点所在图
        compare_up : bool
            比较方向
        mismatch_nodes : set[Node]
            不匹配节点

        Returns
        -------
        bool
            节点是否需要参与key值比较, 如果为True, 则需要参与key值比较, 否则不参与
        """
        get_around = graph.predecessors
        if compare_up:
            get_around = graph.successors
        around_nodes = list(get_around(node))

        if len(around_nodes) == 0:
            return True

        # around节点都为不匹配节点, 则不参与key值比较
        return not all(node in mismatch_nodes for node in around_nodes)

    def _cal_layer_nodes_key_info(self, layer_id: int):
        """计算本层节点key值信息

        Parameters
        ----------
        layer_id : int
            层号
        """
        left_layer_nodes = set()
        if layer_id < len(self._left_topo_layers):
            left_layer_nodes = self._left_topo_layers[layer_id]
        right_layer_nodes = set()
        if layer_id < len(self._right_topo_layers):
            right_layer_nodes = self._right_topo_layers[layer_id]

        compare_up = self._compare_up
        left_subgraph = self._left_subgraph
        right_subgraph = self._right_subgraph

        for node in left_layer_nodes:
            if not AnchorCompareAllAlg._in_compare_area(node, left_subgraph, compare_up, self._left_mismatch_nodes):
                # 不参与key值比较的节点记为不匹配节点
                self._left_mismatch_nodes.add(node)
                continue
            key = AnchorCompareAllAlg._get_node_key(node, left_subgraph, compare_up)
            AnchorCompareAllAlg._record_key_times(self._left_key_times[layer_id], key)
            AnchorCompareAllAlg._record_key_nodes(self._left_key_nodes[layer_id], key, node)
            

        for node in right_layer_nodes:
            if not AnchorCompareAllAlg._in_compare_area(node, right_subgraph, compare_up, self._right_mismatch_nodes):
                self._right_mismatch_nodes.add(node)
                continue
            key = AnchorCompareAllAlg._get_node_key(node, right_subgraph, compare_up)
            AnchorCompareAllAlg._record_key_times(self._right_key_times[layer_id], key)
            AnchorCompareAllAlg._record_key_nodes(self._right_key_nodes[layer_id], key, node)

    def _get_layer_diff_nodes(self, layer_id: int):
        """获取本层差异节点

        Parameters
        ----------
        layer_id : int
            层号
        """
        self._cal_layer_nodes_key_info(layer_id)

        left_key_times = self._left_key_times[layer_id]
        right_key_times = self._right_key_times[layer_id]
        left_key_nodes = self._left_key_nodes[layer_id]
        right_key_nodes = self._right_key_nodes[layer_id]
        mismatch_keys = self._mismatch_keys[layer_id]
        for key in set(left_key_times.keys()) | set(right_key_times.keys()):
            if left_key_times.get(key) is None:  # key值只在right图中存在
                mismatch_keys.add(key)
                self._right_mismatch_nodes.update(right_key_nodes[key])
                continue
            if right_key_times.get(key) is None:  # key值只在left图中存在
                mismatch_keys.add(key)
                self._left_mismatch_nodes.update(left_key_nodes[key])
                continue
            if left_key_times[key] != right_key_times[key]:  # key值在两图中出现次数不一致
                mismatch_keys.add(key)
                self._left_mismatch_nodes.update(left_key_nodes[key])
                self._right_mismatch_nodes.update(right_key_nodes[key])
                continue
            self._left_match_nodes.update(left_key_nodes[key])
            self._right_match_nodes.update(right_key_nodes[key])

    def compare(self):
        """比较
        """
        self._left_subgraph, self._left_topo_layers = AnchorCompareAllAlg._get_subgraph_and_topo_layers(
            self._left_graph,
            self._left_anchor,
            self._grow_up
        )
        self._right_subgraph, self._right_topo_layers = AnchorCompareAllAlg._get_subgraph_and_topo_layers(
            self._right_graph,
            self._right_anchor,
            self._grow_up
        )
        if not self._compare_up:
            self._left_topo_layers = self._left_topo_layers[::-1]
            self._right_topo_layers = self._right_topo_layers[::-1]

        topo_layer_num = max(len(self._left_topo_layers), len(self._right_topo_layers))
        for layer_id in range(topo_layer_num):
            self._left_key_times.append({})
            self._right_key_times.append({})
            self._left_key_nodes.append({})
            self._right_key_nodes.append({})
            self._mismatch_keys.append(set())

        layer_id = 0
        while True:
            if layer_id == topo_layer_num:  # 到达最后一层
                break
            self._get_layer_diff_nodes(layer_id)
            layer_id += 1