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

from backend.kgi.sal.computergraph.computer_node import MemoryAttribute

class ComputerEdge:
    def __init__(self):
        self._memory_ids_and_attributes: dict[str, dict[MemoryAttribute, int]] = {}  # 记录边上的内存id和属性

    def update_memory_ids_and_attributes(self, memory_id: str, attributes: dict[MemoryAttribute, int]):
        self._memory_ids_and_attributes.setdefault(memory_id, {})
        for attribute, attribute_num in attributes.items():
            self._memory_ids_and_attributes[memory_id].setdefault(attribute, 0)
            self._memory_ids_and_attributes[memory_id][attribute] += attribute_num

    def get_memory_ids_and_attributes(self) -> dict[str, dict[MemoryAttribute, int]]:
        return self._memory_ids_and_attributes

    def __repr__(self):
        return "\n".join(
            (
                memory_id + "\n" +
                "\n".join([f"{attribute_num} {attribute}" for attribute, attribute_num in attributes.items()])
            )
            for memory_id, attributes in self._memory_ids_and_attributes.items()
        )
