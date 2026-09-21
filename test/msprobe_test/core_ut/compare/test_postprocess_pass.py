#!/usr/bin/env python3
# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
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

from dataclasses import dataclass
from unittest import TestCase
from msprobe.core.compare.layer_mapping.postprocess_pass import extract_next_item_last_number, \
    replace_next_item_index, renumber_index_pass


@dataclass
class DataItem:
    """Class for keeping track of an item in inventory"""
    type_name: str
    full_scope: str
    layer_scope: str


class TestPostProcessPass(TestCase):

    def test_check_path_type_None(self):
        input_data = "conv1.Conv2d.forward.input"
        prefix = "Conv2d"
        none_result = extract_next_item_last_number(input_data, prefix)
        self.assertEqual(none_result, None)

    def test_check_path_type_find_result(self):
        input_data = "conv1.Conv2d.forward.input.conv1"
        prefix = "conv1"
        result_2 = extract_next_item_last_number(input_data, prefix)
        self.assertEqual(result_2, 2)

    def test_replace_next_item_index(self):
        input_data = "conv1.Conv2d.forward.input.conv1"
        prefix = "conv1"
        replace_result = replace_next_item_index(input_data, prefix, 1)
        self.assertEqual(replace_result, "conv1.1.forward.input.conv1")

    def test_replace_next_item_index_with_inf(self):
        input_data = "conv1.Conv2d.forward.input.conv1"
        prefix = "conv1"
        inf_value = float("inf")
        replace_result = replace_next_item_index(input_data, prefix, inf_value)
        self.assertEqual(replace_result, input_data)

    def test_renumber_index_pass(self):
        a = DataItem("ParallelTransformer", "fake_data.layers.10", "fake_data.layers")
        b = DataItem("ParallelTransformer", "fake_data.layers.12", "fake_data.layers")
        c = DataItem("FakeLayer", "fake_data.layers.10.a.b.c", "fake_data.layers.a.b")
        data_items = [a, b, c]
        renumber_index_pass(data_items, "ParallelTransformer")
        self.assertEqual(a.full_scope, "fake_data.layers.0")
        self.assertEqual(b.full_scope, "fake_data.layers.2")
        self.assertEqual(c.full_scope, "fake_data.layers.0.a.b.c")
