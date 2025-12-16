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

import base64
import unittest
from dump_parse.proto_dump_data import (
    DumpData, OpOutput, OpInput, OpBuffer, OpAttr, Workspace,
    Shape, RepeatedField, OriginalOp, DimRange, fill_dump_data
)


class TestDumpDataFull(unittest.TestCase):

    def test_shape_operations(self):
        s = Shape([1, 2])
        self.assertEqual(s.dim, [1, 2])
        s.append(3)
        s.extend([4, 5])
        self.assertEqual(s.dim, [1, 2, 3, 4, 5])
        s.dim = [6, 7]
        self.assertEqual(s.dim, [6, 7])
        s.Clear()
        self.assertEqual(s.dim, [])

    def test_original_op_dict(self):
        op = OriginalOp(name="conv", output_index=0, data_type=1, data_format=2)
        d = op.to_dict()
        self.assertEqual(d["name"], "conv")
        self.assertEqual(d["data_type"], 1)
        self.assertEqual(d["format"], 2)

    def test_repeated_field_add_and_append(self):
        rf = RepeatedField(OpOutput)
        out = rf.add()           # 先创建对象
        out.data_type = 1        # 再赋值
        self.assertIsInstance(out, OpOutput)
        self.assertEqual(out.data_type, 1)
        # 测试 append 原始元素
        rf.append(OpOutput())
        self.assertEqual(len(rf), 2)

    def test_opoutput_to_dict(self):
        out = OpOutput()
        out.data_type = 1
        out.format = 2
        out.shape.dim = [1, 2]
        out.original_op.name = "op"
        out.dim_range.append(DimRange(dim_start=0, dim_end=1)) 
        d = out.to_dict()
        self.assertEqual(d["data_type"], 1)
        self.assertEqual(d["format"], 2)
        self.assertEqual(d["shape"]["dim"], [1, 2])
        self.assertEqual(d["original_op"]["name"], "op")
        self.assertEqual(d["dim_range"][0]["dim_start"], 0)
        self.assertEqual(d["dim_range"][0]["dim_end"], 1)

    def test_opinput_to_dict(self):
        inp = OpInput()
        inp.data_type = 3
        inp.shape.dim = [3, 4]
        d = inp.to_dict()
        self.assertEqual(d["data_type"], 3)
        self.assertEqual(d["shape"]["dim"], [3, 4])

    def test_opbuffer_to_dict(self):
        buf = OpBuffer()
        buf.buffer_type = 1
        buf.size = 100
        d = buf.to_dict()
        self.assertEqual(d["buffer_type"], 1)
        self.assertEqual(d["size"], 100)

    def test_opattr_to_dict(self):
        attr = OpAttr(name="key", value="val")
        d = attr.to_dict()
        self.assertEqual(d["name"], "key")
        self.assertEqual(d["value"], "val")

    def test_workspace_to_dict(self):
        ws = Workspace(data_type=2)
        ws.size = 50
        d = ws.to_dict()
        self.assertEqual(d["type"], 2)
        self.assertEqual(d["size"], 50)

    def test_dumpdata_to_from_dict_full(self):
        dd = DumpData()
        dd.version = "1.0"
        dd.dump_time = 123
        dd.op_name = "test_op"

        out = dd.output.add()
        out.data_type = 5
        out.shape.dim = [1,2]

        inp = dd.input.add()
        inp.data_type = 6
        inp.shape.dim = [3,4]

        buf = dd.buffer.add()
        buf.buffer_type = 2
        buf.size = 10

        attr = dd.attr.add()
        attr.name = "attr1"
        attr.value = "val1"

        ws = dd.space.add()
        ws.type = 1
        ws.size = 20

        d = dd.to_dict()
        dd2 = DumpData.from_dict(d)

        self.assertEqual(dd2.version, "1.0")
        self.assertEqual(dd2.dump_time, 123)
        self.assertEqual(dd2.op_name, "test_op")
        self.assertEqual(dd2.output[0].data_type, 5)
        self.assertEqual(dd2.output[0].shape.dim, [1,2])
        self.assertEqual(dd2.input[0].data_type, 6)
        self.assertEqual(dd2.input[0].shape.dim, [3,4])
        self.assertEqual(dd2.buffer[0].buffer_type, 2)
        self.assertEqual(dd2.attr[0].name, "attr1")
        self.assertEqual(dd2.space[0].size, 20)

    def test_serialize_parse_string_full(self):
        dd = DumpData()
        dd.version = "v2"
        dd.dump_time = 456
        out = dd.output.add()
        out.data_type = 9
        out.data = b"hello"

        s = dd.SerializeToString()
        self.assertIsInstance(s, bytes)

        dd2 = DumpData()
        dd2.ParseFromString(s)

        # base64 decode 后再比对
        decoded_data = base64.b64decode(dd2.output[0].data)
        self.assertEqual(decoded_data, b"hello")
        self.assertEqual(dd2.version, "v2")
        self.assertEqual(dd2.dump_time, 456)
        self.assertEqual(dd2.output[0].data_type, 9)

    def test_fill_dump_data_with_extra_fields(self):
        dd = DumpData()
        data = {
            "version": "v1",
            "non_exist": 123,  # 多余字段
            "output": [{"data_type": 10, "shape": {"dim":[1,2]}}]
        }
        fill_dump_data(dd, data)
        self.assertEqual(dd.version, "v1")
        self.assertEqual(dd.output[0].data_type, 10)
        self.assertEqual(dd.output[0].shape.dim, [1,2])
