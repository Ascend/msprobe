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
from typing import Optional
from backend.kgi.core.comparealg.anchor_compare_alg import AnchorCompareAlg
from backend.kgi.core.comparealg.compare_alg import CompareAlg
from backend.kgi.core.structs.node import Node

class MulAnchorCompareAlg(CompareAlg):
    """多锚点比较算法
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
        """多锚点比较算法

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
        self._first_level_anchor_compare_alg: AnchorCompareAlg = AnchorCompareAlg(
            left_graph,
            right_graph,
            left_anchor,
            right_anchor,
            grow_up,
            compare_up
        )  # 一级锚点比较算法
        self._second_level_anchor_compare_algs: list[AnchorCompareAlg] = []  # 二级锚点比较算法
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

    def get_first_level_anchor_compare_alg(self) -> AnchorCompareAlg:
        return self._first_level_anchor_compare_alg

    def get_second_level_anchor_compare_algs(self) -> list[AnchorCompareAlg]:
        return self._second_level_anchor_compare_algs

    def get_left_match_nodes(self) -> set[Node]:
        return self._left_match_nodes

    def get_right_match_nodes(self) -> set[Node]:
        return self._right_match_nodes

    def get_left_mismatch_nodes(self) -> set[Node]:
        return self._left_mismatch_nodes

    def get_right_mismatch_nodes(self) -> set[Node]:
        return self._right_mismatch_nodes

    def add_second_level_anchor_compare_alg(self, left_anchor: Node, right_anchor: Node):
        """新增二级锚点比较算法

        Parameters
        ----------
        left_anchor : Node
            左图锚点
        right_anchor : Node
            右图锚点
        """
        self._second_level_anchor_compare_algs.append(AnchorCompareAlg(
            self._left_graph,
            self._right_graph,
            left_anchor,
            right_anchor,
            self._compare_up,  # 二级锚点的生长方向要与一级锚点的比较方向一致
            self._compare_up
        ))

    def _update_match_mismatch_nodes(self, alg: AnchorCompareAlg):
        """根据子比较算法比较结果更新匹配不匹配节点

        Parameters
        ----------
        alg : AnchorCompareAlg
            子比较算法, 可能为一级锚点比较算法或二级锚点比较算法
        """
        self._left_match_nodes.update(alg._left_match_nodes)
        self._left_mismatch_nodes.update(alg._left_mismatch_nodes)
        self._right_match_nodes.update(alg._right_match_nodes)
        self._right_mismatch_nodes.update(alg._right_mismatch_nodes)
        # 当前锚点计算的匹配结果与之前的锚点计算结果不一致，信任当前锚点的计算结果
        # mismatch -> match
        diff_nodes = alg._left_match_nodes & self._left_mismatch_nodes
        self._left_mismatch_nodes.difference_update(diff_nodes)
        diff_nodes = alg._right_match_nodes & self._right_mismatch_nodes
        self._right_mismatch_nodes.difference_update(diff_nodes)
        # match -> mismatch
        diff_nodes = alg._left_mismatch_nodes & self._left_match_nodes
        self._left_match_nodes.difference_update(diff_nodes)
        diff_nodes = alg._right_mismatch_nodes & self._right_match_nodes
        self._right_match_nodes.difference_update(diff_nodes)

    def del_second_level_anchor_compare_alg(self, left_nodes_id: list[int], right_nodes_id: list[int]):
        """删除二级锚点比较算法, 支持一次删除多个

        Parameters
        ----------
        left_nodes_id : list[int]
            左图锚点节点id
        right_nodes_id : list[int]
            右图锚点节点id
        """
        need_del_ids = []
        for alg_id in range(len(self._second_level_anchor_compare_algs)):
            left_anchor = self._second_level_anchor_compare_algs[alg_id].get_left_anchor()
            right_anchor = self._second_level_anchor_compare_algs[alg_id].get_right_anchor()
            if (left_anchor.get_node_id() in left_nodes_id or
                right_anchor.get_node_id() in right_nodes_id):
                need_del_ids.append(alg_id)
        for alg_id in need_del_ids[::-1]:
            del self._second_level_anchor_compare_algs[alg_id]

        # 重新计算匹配不匹配节点
        self._left_match_nodes.clear()
        self._left_mismatch_nodes.clear()
        self._right_match_nodes.clear()
        self._right_mismatch_nodes.clear()
        for alg in [self._first_level_anchor_compare_alg] + self._second_level_anchor_compare_algs:
            if alg.get_compared():
                self._update_match_mismatch_nodes(alg)

    def compare(self):
        """比较
        """
        for alg in [self._first_level_anchor_compare_alg] + self._second_level_anchor_compare_algs:
            if not alg.get_compared():  # 已比较过，不在重新比较
                alg.compare()
                self._update_match_mismatch_nodes(alg)