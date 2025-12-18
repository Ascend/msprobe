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

from backend.kgi.core.structs.node import Node

class MemoryAttribute:
    def __init__(self):
        self._dtype = ""
        self._shape: list[int] = []
        self._exist_stride_offset = False  # stride和offset是否存在
        self._stride: list[int] = []
        self._offset = 0

    def set_dtype(self, dtype: str):
        self._dtype = dtype

    def get_dtype(self) -> str:
        return self._dtype

    def set_shape(self, shape: list[int]):
        self._shape = shape

    def get_shape(self) -> list[int]:
        return self._shape

    def set_exist_stride_offset(self, exist_stride_offset: bool = True):
        self._exist_stride_offset = exist_stride_offset

    def get_exist_stride_offset(self) -> bool:
        return self._exist_stride_offset

    def set_stride(self, stride: list[int]):
        self._stride = stride

    def get_stride(self) -> list[int]:
        return self._stride

    def set_offset(self, offset: int):
        self._offset = offset

    def get_offset(self) -> int:
        return self._offset

    # BFloat16:[4096, 1, 32, 128]{strdes=[12288, 12288, 384, 1],offset=0}
    def __repr__(self):
        shape_str = ", ".join(map(str, self._shape))
        res_str = f"{self._dtype}:[{shape_str}]"

        if self._exist_stride_offset:
            stride_str = ", ".join(map(str, self._stride))
            res_str += f"{{strdes=[{stride_str}],offset={self._offset}}}"

        return res_str

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        if isinstance(other, MemoryAttribute):
            return self.__repr__() == other.__repr__()
        return False

class ComputerNode(Node):
    def __init__(self):
        super().__init__()
        self._op_type: str = ""
        self._scope: list[str] = []
        self._stack: list[str] = []
        self._input_memory_ids: list[str] = []
        self._output_memory_ids: list[str] = []
        self._input_memory_attributes: list[MemoryAttribute] = []
        self._output_memory_attributes: list[MemoryAttribute] = []
        self._task_info: str = ""
        self._attrs: str = ""

    def set_op_type(self, op_type: str):
        self._op_type = op_type

    def get_op_type(self) -> str:
        return self._op_type

    def set_scope(self, scope: list[str]):
        self._scope = scope

    def get_scope(self) -> list[str]:
        return self._scope

    def set_stack(self, stack: list[str]):
        self._stack = stack

    def get_stack(self) -> list[str]:
        return self._stack

    def set_input_memory_ids(self, input_memory_ids: list[str]):
        self._input_memory_ids = input_memory_ids

    def get_input_memory_ids(self) -> list[str]:
        return self._input_memory_ids

    def set_output_memory_ids(self, output_memory_ids: list[str]):
        self._output_memory_ids = output_memory_ids

    def get_output_memory_ids(self) -> list[str]:
        return self._output_memory_ids

    def set_input_memory_attributes(self, input_memory_attributes: list[MemoryAttribute]):
        self._input_memory_attributes = input_memory_attributes

    def get_input_memory_attributes(self) -> list[MemoryAttribute]:
        return self._input_memory_attributes

    def set_output_memory_attributes(self, output_memory_attributes: list[MemoryAttribute]):
        self._output_memory_attributes = output_memory_attributes

    def get_output_memory_attributes(self) -> list[MemoryAttribute]:
        return self._output_memory_attributes

    def set_task_info(self, task_info: str):
        self._task_info = task_info

    def get_task_info(self) -> str:
        return self._task_info

    def set_attrs(self, attrs: str):
        self._attrs = attrs

    def get_attrs(self) -> str:
        return self._attrs

    def get_node_feature(self) -> str:
        input_memory_attributes = list(str(memory_attribute)
                                       for memory_attribute in self.get_input_memory_attributes())
        input_memory_attributes_str = ", ".join(input_memory_attributes)

        output_memory_attributes = list(str(memory_attribute)
                                        for memory_attribute in self.get_output_memory_attributes())
        output_memory_attributes_str = ", ".join(output_memory_attributes)

        return f"({output_memory_attributes_str})<{self.get_op_type()}-({input_memory_attributes_str})"