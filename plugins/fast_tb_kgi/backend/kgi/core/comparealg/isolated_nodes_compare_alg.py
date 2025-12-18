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
from backend.kgi.core.comparealg.compare_alg import CompareAlg
from backend.kgi.core.structs.node import Node
from backend.kgi.core.utils import graph_utils

class IsolatedNodesCompareAlg(CompareAlg):
    """孤立节点比较算法
    """
    def __init__(
            self,
            left_graph: nx.DiGraph,
            right_graph: nx.DiGraph
    ):
        super().__init__(left_graph, right_graph)
        self._left_feature_times: dict[str, int] = {}  # 记录特征值出现的次数
        self._right_feature_times: dict[str, int] = {}
        self._left_feature_nodes: dict[str, set[Node]] = {}  # 记录特征值对应节点
        self._right_feature_nodes: dict[str, set[Node]] = {}
        self._mismatch_features: set[str] = set()  # 记录不匹配的特征值
        self._left_match_nodes: set[Node] = set()  # 记录匹配的节点
        self._right_match_nodes: set[Node] = set()
        self._left_mismatch_nodes: set[Node] = set()  # 记录不匹配的节点
        self._right_mismatch_nodes: set[Node] = set()

    def get_left_feature_times(self) -> dict[str, int]:
        return self._left_feature_times

    def get_right_feature_times(self) -> dict[str, int]:
        return self._right_feature_times

    def get_left_feature_nodes(self) -> dict[str, set[Node]]:
        return self._left_feature_nodes

    def get_right_feature_nodes(self) -> dict[str, set[Node]]:
        return self._right_feature_nodes

    def get_mismatch_features(self) -> set[str]:
        return self._mismatch_features

    def get_left_match_nodes(self) -> set[Node]:
        return self._left_match_nodes

    def get_right_match_nodes(self) -> set[Node]:
        return self._right_match_nodes

    def get_left_mismatch_nodes(self) -> set[Node]:
        return self._left_mismatch_nodes

    def get_right_mismatch_nodes(self) -> set[Node]:
        return self._right_mismatch_nodes

    @staticmethod
    def _record_feature_times(record: dict[str, int], feature: str):
        """记录特征值出现次数

        Parameters
        ----------
        record : dict[str, int]
            特征值出现次数记录
        feature : str
            特征值
        """
        record.setdefault(feature, 0)
        record[feature] += 1

    @staticmethod
    def _record_feature_nodes(record: dict[str, set[Node]], feature: str, node: Node):
        """记录特征值对应节点

        Parameters
        ----------
        record : dict[str, set[Node]]
            特征值对应节点记录
        feature : str
            特征值
        node : Node
            节点
        """
        record.setdefault(feature, set())
        record[feature].add(node)

    def compare(self):
        """比较
        """
        left_isolated_nodes = graph_utils.get_isolated_nodes(self._left_graph)
        for node in left_isolated_nodes:
            feature = node.get_node_feature()
            IsolatedNodesCompareAlg._record_feature_times(self._left_feature_times, feature)
            IsolatedNodesCompareAlg._record_feature_nodes(self._left_feature_nodes, feature, node)

        right_isolated_nodes = graph_utils.get_isolated_nodes(self._right_graph)
        for node in right_isolated_nodes:
            feature = node.get_node_feature()
            IsolatedNodesCompareAlg._record_feature_times(self._right_feature_times, feature)
            IsolatedNodesCompareAlg._record_feature_nodes(self._right_feature_nodes, feature, node)

        for feature in set(self._left_feature_times.keys()) | set(self._right_feature_times.keys()):
            if self._left_feature_times.get(feature) is None:  # 特征值只在right图中存在
                self._mismatch_features.add(feature)
                self._right_mismatch_nodes.update(self._right_feature_nodes[feature])
                continue
            if self._right_feature_times.get(feature) is None:  # 特征值只在left图中存在
                self._mismatch_features.add(feature)
                self._left_mismatch_nodes.update(self._left_feature_nodes[feature])
                continue
            if self._left_feature_times[feature] != self._right_feature_times[feature]:  # 特征值在两图中出现次数不一致
                self._mismatch_features.add(feature)
                self._left_mismatch_nodes.update(self._left_feature_nodes[feature])
                self._right_mismatch_nodes.update(self._right_feature_nodes[feature])
                continue
            self._left_match_nodes.update(self._left_feature_nodes[feature])
            self._right_match_nodes.update(self._right_feature_nodes[feature])