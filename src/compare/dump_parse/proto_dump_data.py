# coding=utf-8
# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

import json
from enum import Enum
import base64
from typing import List, Any, Dict


class EnumEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Enum):
            return o.value
        if isinstance(o, bytes):
            return base64.b64encode(o).decode("ascii")
        return super().default(o)


class RepeatedField(list):
    """模仿 protobuf repeated 字段，支持 add()"""
    def __init__(self, cls_type):
        super().__init__()
        self._cls_type = cls_type

    def add(self, **kwargs):
        obj = self._cls_type(**kwargs)
        self.append(obj)
        return obj


class Shape:
    def __init__(self, dim=None):
        self._dim = []
        if dim:
            self.extend(dim)

    @property
    def dim(self):
        return self._dim

    def __setattr__(self, name, value):
        if name == "dim":
            object.__setattr__(self, "_dim", [int(v) for v in value])
        else:
            object.__setattr__(self, name, value)

    def append(self, value):
        self._dim.append(int(value))

    def extend(self, values):
        self._dim.extend(int(v) for v in values)

    def Clear(self):
        self._dim.clear()

    def to_dict(self):
        return {"dim": self._dim}

    def __repr__(self):
        return f"Shape(dim={self._dim})"


class DimRange:
    def __init__(self, dim_start=0, dim_end=0):
        self.dim_start = dim_start
        self.dim_end = dim_end

    def to_dict(self):
        return {"dim_start": self.dim_start, "dim_end": self.dim_end}


class OriginalOp:
    def __init__(self, name="", output_index=0, data_type=0, data_format=0):
        self.name = name
        self.output_index = output_index
        self.data_type = data_type
        self.format = data_format

    def to_dict(self):
        return {
            "name": self.name,
            "output_index": self.output_index,
            "data_type": self.data_type,
            "format": self.format,
        }


class OpOutput():
    def __init__(self):
        self.data_type = 0
        self.format = 0
        self.shape = Shape()
        self.original_op = OriginalOp()
        self.data = b""
        self.size = 0
        self.original_shape = Shape()
        self.sub_format = 0
        self.address = 0
        self.dim_range: List[DimRange] = []
        self.offset = 0

    def to_dict(self):
        return {
            "data_type": self.data_type,
            "format": self.format,
            "shape": self.shape.to_dict(),
            "original_op": self.original_op.to_dict(),
            "data": self.data,
            "size": self.size,
            "original_shape": self.original_shape.to_dict(),
            "sub_format": self.sub_format,
            "address": self.address,
            "dim_range": [d.to_dict() for d in self.dim_range],
            "offset": self.offset,
        }


class OpInput():
    def __init__(self):
        self.data_type = 0
        self.format = 0
        self.shape = Shape()
        self.data = b""
        self.size = 0
        self.original_shape = Shape()
        self.sub_format = 0
        self.address = 0
        self.offset = 0

    def to_dict(self):
        return {
            "data_type": self.data_type,
            "format": self.format,
            "shape": self.shape.to_dict(),
            "data": self.data,
            "size": self.size,
            "original_shape": self.original_shape.to_dict(),
            "sub_format": self.sub_format,
            "address": self.address,
            "offset": self.offset,
        }


class OpBuffer:
    def __init__(self):
        self.buffer_type = 0
        self.data = b""
        self.size = 0

    def to_dict(self):
        return {"buffer_type": self.buffer_type, "data": self.data, "size": self.size}


class OpAttr:
    def __init__(self, name="", value=""):
        self.name = name
        self.value = value

    def to_dict(self):
        return {"name": self.name, "value": self.value}


class Workspace:
    def __init__(self, data_type=0):
        self.type = data_type
        self.data = b""
        self.size = 0

    def to_dict(self):
        return {"type": self.type, "data": self.data, "size": self.size}


class DumpData:
    def __init__(self):
        self.version = ""
        self.dump_time = 0
        self.output = RepeatedField(OpOutput)
        self.input = RepeatedField(OpInput)
        self.buffer = RepeatedField(OpBuffer)
        self.op_name = ""
        self.attr = RepeatedField(OpAttr)
        self.space = RepeatedField(Workspace)
    
    @classmethod
    def from_dict(cls, data: Dict):
        obj = cls()
        fill_dump_data(obj, data)
        return obj

    def to_dict(self):
        return {
            "version": self.version,
            "dump_time": self.dump_time,
            "op_name": self.op_name,
            "output": [data.to_dict() for data in self.output],
            "input": [data.to_dict() for data in self.input],
            "buffer": [data.to_dict() for data in self.buffer],
            "attr": [data.to_dict() for data in self.attr],
            "space": [data.to_dict() for data in self.space],
        }

    def SerializeToString(self) -> bytes:
        """模拟 protobuf 的序列化，返回 JSON bytes"""
        return json.dumps(self.to_dict(), cls=EnumEncoder).encode("utf-8")

    def ParseFromString(self, data: bytes):
        """
        模拟 protobuf 的 ParseFromString
        :param data: bytes (JSON 格式)
        :return: self
        """
        obj_dict = json.loads(data.decode("utf-8"))
        # 用 from_dict 填充当前实例
        fill_dump_data(self, obj_dict)
        return self


TYPE_MAP = {
    "input": OpInput,
    "output": OpOutput,
    "buffer": OpBuffer,
    "attr": OpAttr,
    "space": Workspace,
}


def _handle_repeated_field(attr: Any, key: str, value_list: list) -> None:
    """处理 repeated 字段：当 TYPE_MAP 有对应类时用 add() 并递归填充，否则 append 原始元素。"""
    cls_type = TYPE_MAP.get(key)
    if cls_type:
        for item in value_list:
            # 保持原行为：使用 add() 创建子对象
            sub_obj = attr.add()  
            # 只有 dict 才递归填充，否则跳过（与原行为一致）
            if isinstance(item, dict):
                fill_dump_data(sub_obj, item)
    else:
        # 保持原行为：逐项 append（而不是 extend），以保留原先每项可能的非 list 语义
        for item in value_list:
            attr.append(item)


def _handle_shape(attr: "Shape", value: Any) -> None:
    """填充 Shape 类型：支持 dict {'dim': [...]} 或直接 list."""
    if isinstance(value, dict):
        dims = value.get("dim")
        if dims is None:
            return
        # 保证全部转换为 int
        attr.dim = [int(v) for v in dims]
    elif isinstance(value, list):
        # 如果直接给了 list，也允许赋值（向后兼容）
        attr.dim = [int(v) for v in value]


def _handle_original_op(attr: "OriginalOp", value: Any) -> None:
    """按字段赋值 OriginalOp（仅处理 dict）。"""
    if not isinstance(value, dict):
        return
    for sub_key, sub_val in value.items():
        if hasattr(attr, sub_key):
            setattr(attr, sub_key, sub_val)


def fill_dump_data(obj: Any, data: Dict) -> Any:
    """
    将字典 data 递归填充到 obj（DumpData 或子消息）上。
    """
    for key, value in data.items():
        if not hasattr(obj, key):
            # 字典中含有类没有的字段，跳过
            continue

        attr = getattr(obj, key)

        # 1) repeated field (RepeatedField)
        if isinstance(attr, RepeatedField) and isinstance(value, list):
            _handle_repeated_field(attr, key, value)
            continue

        # 2) Shape 嵌套
        if isinstance(attr, Shape):
            _handle_shape(attr, value)
            continue

        # 3) OriginalOp 嵌套
        if isinstance(attr, OriginalOp):
            _handle_original_op(attr, value)
            continue

        # 4) 其它基本类型（bytes / int / str / list 等），直接赋值
        setattr(obj, key, value)

    return obj
