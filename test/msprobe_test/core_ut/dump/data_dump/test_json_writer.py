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

import csv
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from msprobe.core.common.const import Const
from msprobe.core.common.utils import DumpPathAggregation
from msprobe.core.common.file_utils import FileOpen, remove_path, load_json
from msprobe.core.dump.data_dump.json_writer import DataWriter


class TestDataWriter(unittest.TestCase):
    @patch("msprobe.core.dump.data_dump.json_writer.save_json")
    def test_nan_check_without_detail_does_not_create_or_cache_json(self, save):
        writer = DataWriter()
        writer.config = SimpleNamespace(nan_check_detail=False)
        writer.register_pre_flush_callback(lambda: self.fail("unexpected metadata flush"))
        writer.initialize_json_file(task=Const.NAN_CHECK, level=Const.LEVEL_L1)
        writer.update_data({"Torch.add.0.forward": {Const.OUTPUT: []}})
        writer.data_updated = True
        writer.write_json()
        save.assert_not_called()
        self.assertEqual(writer.cache_data, {})

    def setUp(self):
        self.data_writer = DataWriter()
        self.data_content = {"task": "tensor", "level": "L1", "data": {"Tensor.add": 1}}
        self.cur_path = os.path.dirname(os.path.realpath(__file__))
        self.stat_vector = [1.0, 2.0, 3.0, 4.0]  # Example stat_vector for tests
        self.data_writer.stat_stack_list = [self.stat_vector]  # Mock the stat_stack_list

    def test_replace_stat_placeholders(self):
        stat_result = [[1.0, 2.0, 3.0, 4.0]]  # Mocking stat_result with a dummy value
        data = {"type": "Tensor", "dtype": "float32", "shape": [1, 2, 3], Const.TENSOR_STAT_INDEX: 0}

        # Call _replace_stat_placeholders directly
        self.data_writer._replace_stat_placeholders(data, stat_result)

        # Check that the function processed the placeholders correctly
        self.assertEqual(data["Max"], 1.0)
        self.assertEqual(data["Min"], 2.0)
        self.assertEqual(data["Mean"], 3.0)
        self.assertEqual(data["Norm"], 4.0)

    def test_append_stat_to_buffer(self):
        index = self.data_writer.append_stat_to_buffer(self.stat_vector)
        self.assertEqual(index, 1)  # The first append will return index 0
        self.assertEqual(self.data_writer.stat_stack_list[0],
                         self.stat_vector)  # Check if the stat is appended correctly

    def test_get_buffer_values_max(self):
        max_value = self.data_writer.get_buffer_values_max(0)
        self.assertEqual(max_value, 1.0)  # The max value of stat_vector is 1.0

        # Test when index is out of range
        max_value_invalid = self.data_writer.get_buffer_values_max(1)
        self.assertIsNone(max_value_invalid)  # Should return None for invalid index

    def test_get_buffer_values_min(self):
        min_value = self.data_writer.get_buffer_values_min(0)
        self.assertEqual(min_value, 2.0)  # The min value of stat_vector is 2.0

        # Test when index is out of range
        min_value_invalid = self.data_writer.get_buffer_values_min(1)
        self.assertIsNone(min_value_invalid)  # Should return None for invalid index

    def test_flush_stat_stack(self):
        # Ensure that flush_stat_stack works and clears the stat_stack_list
        result = self.data_writer.flush_stat_stack()
        self.assertEqual(result, [[1.0, 2.0, 3.0, 4.0]])  # Returns the flushed stats
        self.assertEqual(self.data_writer.stat_stack_list, [])  # Ensure the list is cleared after flush

    def test_reset_cache(self):
        self.data_writer.cache_data = {"data": 1}
        self.data_writer.cache_stack = {"stack": 2}
        self.data_writer.cache_construct = {"construct": 3}
        self.data_writer.reset_cache()
        self.assertEqual(self.data_writer.cache_data, {})
        self.assertEqual(self.data_writer.cache_stack, {})
        self.assertEqual(self.data_writer.cache_construct, {})

    def test_initialize_json_file(self):
        self.data_writer.dump_tensor_data_dir = "./dump_tensor_data"
        self.data_writer.dump_file_path = os.path.join(self.cur_path, "dump.json")
        self.data_writer.stack_file_path = os.path.join(self.cur_path, "stack.json")
        self.data_writer.construct_file_path = os.path.join(self.cur_path, "construct.json")

        self.data_writer.initialize_json_file(task="tensor", level="L1")
        load_data = load_json(self.data_writer.dump_file_path)
        expected = {"task": "tensor", "level": "L1", "dump_data_dir": "./dump_tensor_data", "data": {}}
        self.assertEqual(load_data, expected)

        expected = {}
        load_data = load_json(self.data_writer.stack_file_path)
        self.assertEqual(load_data, expected)

        load_data = load_json(self.data_writer.construct_file_path)
        self.assertEqual(load_data, expected)

        remove_path(self.data_writer.dump_file_path)
        remove_path(self.data_writer.stack_file_path)
        remove_path(self.data_writer.construct_file_path)

    def test_update_dump_paths(self):
        self.assertIsNone(self.data_writer.dump_file_path)
        test_path = os.path.join(self.cur_path, "test1.json")

        dump_path_aggregation = DumpPathAggregation()
        dump_path_aggregation.dump_file_path = test_path
        dump_path_aggregation.stack_file_path = test_path
        dump_path_aggregation.construct_file_path = test_path
        dump_path_aggregation.dump_tensor_data_dir = test_path
        dump_path_aggregation.debug_file_path = test_path
        dump_path_aggregation.dump_error_info_path = test_path

        self.data_writer.update_dump_paths(dump_path_aggregation)
        self.assertTrue(self.data_writer.dump_file_path == test_path)
        self.assertTrue(self.data_writer.stack_file_path == test_path)
        self.assertTrue(self.data_writer.construct_file_path == test_path)
        self.assertTrue(self.data_writer.dump_tensor_data_dir == test_path)
        self.assertTrue(self.data_writer.debug_file_path == test_path)

    @patch.object(DataWriter, "write_json")
    def test_flush_data_periodically(self, mock_write_json):
        self.data_writer.cache_data["data"] = {"Tensor.add": 1, "Tensor.mul": 2}
        self.data_writer.flush_size = 2
        self.data_writer.flush_data_periodically()
        mock_write_json.assert_called_once()

    def test_update_data(self):
        self.data_writer.cache_data["data"] = {}

        new_data = {"Tensor.mul1": {"input": 1}}
        expected = {"data": {"Tensor.mul1": {"input": 1}}}
        self.data_writer.update_data(new_data)
        self.assertEqual(self.data_writer.cache_data, expected)

        new_data = {"Tensor.mul1": {"output": 2}}
        expected = {"data": {"Tensor.mul1": {"input": 1, "output": 2}}}
        self.data_writer.update_data(new_data)
        self.assertEqual(self.data_writer.cache_data, expected)

        new_data = {"Tensor.mul2": 2, "Tensor.mul3": 3}
        self.data_writer.update_data(new_data)
        self.assertEqual(self.data_writer.cache_data, expected)

    def test_update_stack(self):
        self.data_writer.cache_stack = {"stack1": ["test1"]}
        self.data_writer.update_stack("test2", "stack1")
        self.assertEqual(self.data_writer.cache_stack, {"stack1": ["test1", "test2"]})

    def test_update_construct(self):
        self.data_writer.update_construct(self.data_content)
        self.assertEqual(self.data_writer.cache_construct, self.data_content)

    def test_write_data_json(self):
        self.data_writer.cache_data = self.data_content
        file_path = os.path.join(self.cur_path, "dump.json")
        self.data_writer.write_data_json(file_path)
        load_result = load_json(file_path)

        try:
            self.assertEqual(load_result, self.data_content)
        finally:
            os.remove(file_path)

    def test_write_stack_info_json(self):
        self.data_writer.cache_stack = {("api1", "api2"): ["stack1"]}
        file_path = os.path.join(self.cur_path, "stack.json")
        self.data_writer.write_stack_info_json(file_path)
        load_result = load_json(file_path)

        try:
            self.assertEqual(load_result, {"0": [["stack1"], ["api1", "api2"]]})
        finally:
            os.remove(file_path)

    def test_write_construct_info_json(self):
        self.data_writer.cache_construct = self.data_content
        file_path = os.path.join(self.cur_path, "construct.json")
        self.data_writer.write_construct_info_json(file_path)
        load_result = load_json(file_path)

        try:
            self.assertEqual(load_result, self.data_content)
        finally:
            os.remove(file_path)

    def test_replace_stat_placeholders_invalid_index(self):
        data = {
            "type": "Tensor",
            "dtype": "float32",
            "shape": [1, 2],
            Const.TENSOR_STAT_INDEX: 10  # 超出索引
        }
        stat_result = [[1.0, 2.0, 3.0, 4.0]]
        self.data_writer._replace_stat_placeholders(data, stat_result)
        self.assertIsNone(data.get(Const.TENSOR_STAT_INDEX))
        self.assertIn(Const.MAX, data)
        self.assertIsNone(data[Const.MAX])  # 越界填 None

    def test_append_stat_to_buffer_multiple(self):
        for i in range(5):
            idx = self.data_writer.append_stat_to_buffer([i, i+1, i+2, i+3])
            self.assertEqual(idx, i + 1)
        self.assertEqual(len(self.data_writer.stat_stack_list), 6)  # 包含 setUp 中那一条

    def test_get_buffer_values_max_invalid_data(self):
        self.data_writer.stat_stack_list = [["not-a-number"]]  # 非预期格式
        max_val = self.data_writer.get_buffer_values_max(0)
        self.assertEqual(max_val, "not-a-number")  # 仍然返回第一位

        max_val = self.data_writer.get_buffer_values_max(-1)
        self.assertIsNone(max_val)

    def test_flush_stat_stack_empty(self):
        self.data_writer.stat_stack_list = []
        result = self.data_writer.flush_stat_stack()
        self.assertEqual(result, [])

    def test_flush_stat_stack_with_tensor_like_items(self):
        class DummyTensor:
            def __init__(self, v): self.v = v
            def item(self): return self.v

        self.data_writer.stat_stack_list = [
            [DummyTensor(1), DummyTensor(2), DummyTensor(3), DummyTensor(4)],
            [5.5, 6.6, 7.7, 8.8]  # 混合类型
        ]
        result = self.data_writer.flush_stat_stack()
        self.assertEqual(result, [[1, 2, 3, 4], [5.5, 6.6, 7.7, 8.8]])
        self.assertEqual(self.data_writer.stat_stack_list, [])
