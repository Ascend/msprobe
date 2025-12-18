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

INVALID_NODE_ID = 0  # 无效的节点id

class Node:
    """节点
    """
    def __init__(self):
        self._id: int = INVALID_NODE_ID
        self._feature: str = ""
        self._hash: str = ""

    def set_node_id(self, node_id: int):
        self._id = node_id

    def get_node_id(self) -> int:
        return self._id

    def set_node_feature(self, feature: str):
        self._feature = feature

    def get_node_feature(self) -> str:
        return self._feature

    def set_node_hash(self, node_hash: str):
        self._hash = node_hash

    def get_node_hash(self) -> str:
        return self._hash

    # id 1
    def __repr__(self):
        return f"id {self._id}"

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.__repr__() == other.__repr__()
        return False

    def __lt__(self, other):
        if isinstance(other, Node):
            return self._id < other._id
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, Node):
            return self._id > other._id
        return NotImplemented