# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UTs for python/msprobe/mindspore/dump/mindtorch/mindtorch_adaptor.py
"""

import unittest
from collections import OrderedDict
from unittest.mock import patch, MagicMock

from msprobe.mindspore.dump.mindtorch import mindtorch_adaptor as adaptor


class DummyCell:
    """Helper dummy cell object for testing adaptor functions."""

    def __init__(self):
        self.__ms_class__ = False
        self._backward_hooks = {}
        self._backward_pre_hooks = OrderedDict()
        self._forward_hooks = {}
        self._forward_pre_hooks = {}
        self._forward_hooks_with_kwargs = set()
        self._forward_hooks_always_called = set()
        self._is_full_backward_hook = None
        self.forward = MagicMock()


class TestMindtorchAdaptor(unittest.TestCase):
    def test__call_impl_when_ms_class_then_pass(self):
        cell = DummyCell()
        cell.__ms_class__ = True
        cell.forward.return_value = "ok"

        result = adaptor._call_impl(cell, 1, 2, key="v")

        cell.forward.assert_called_once_with(1, 2, key="v")
        self.assertEqual(result, "ok")

    @patch.object(adaptor, "_global_forward_pre_hooks", {})
    @patch.object(adaptor, "_global_forward_hooks", {})
    @patch.object(adaptor, "_global_backward_pre_hooks", {})
    @patch.object(adaptor, "_global_backward_hooks", {})
    def test__call_impl_when_no_hooks_then_pass(self, *_):
        cell = DummyCell()
        cell.__ms_class__ = False
        cell.forward.return_value = "no_hooks"

        result = adaptor._call_impl(cell, "arg1")

        cell.forward.assert_called_once_with("arg1")
        self.assertEqual(result, "no_hooks")

    @patch("msprobe.mindspore.dump.mindtorch.mindtorch_adaptor.is_backward_hook_output_a_view", return_value=True)
    def test_apply_backward_hook_on_tensors_when_output_is_view_then_pass(self, mock_is_view):
        hook = MagicMock(return_value="hooked")
        args = ("a", "b")

        result = adaptor.apply_backward_hook_on_tensors(hook, args)

        hook.assert_called_once_with(args)
        self.assertEqual(result, "hooked")
        mock_is_view.assert_called_once()

    @patch("msprobe.mindspore.dump.mindtorch.mindtorch_adaptor.is_backward_hook_output_a_view", return_value=False)
    def test_apply_backward_hook_on_tensors_when_single_non_tuple_then_pass(self, mock_is_view):
        hook = MagicMock(return_value="processed")
        args = "x"

        result = adaptor.apply_backward_hook_on_tensors(hook, args)

        hook.assert_called_once_with(args)
        self.assertEqual(result, "processed")
        mock_is_view.assert_called_once()

    @patch("msprobe.mindspore.dump.mindtorch.mindtorch_adaptor.is_backward_hook_output_a_view", return_value=False)
    def test_apply_backward_hook_on_tensors_when_single_tuple_then_pass(self, mock_is_view):
        hook = MagicMock(return_value="processed")
        args = ("x",)

        result = adaptor.apply_backward_hook_on_tensors(hook, args)

        hook.assert_called_once_with("x")
        self.assertIsInstance(result, tuple)
        self.assertEqual(result, ("processed",))
        mock_is_view.assert_called_once()

    def test__get_backward_pre_hooks_when_global_hooks_exist_then_pass(self):
        cell = DummyCell()
        cell._backward_pre_hooks = OrderedDict({1: "local"})

        with patch.object(adaptor, "_global_backward_pre_hooks", {2: "global"}):
            adaptor._get_backward_pre_hooks(cell)

        self.assertEqual(cell._backward_pre_hooks[1], "local")
        self.assertEqual(cell._backward_pre_hooks[2], "global")

    def test__get_backward_hooks_when_flag_true_then_pass(self):
        cell = DummyCell()
        cell._backward_hooks = {1: "local"}

        with patch.object(adaptor, "_global_is_full_backward_hook", True), \
             patch.object(adaptor, "_global_backward_hooks", {2: "global"}):
            adaptor._get_backward_hooks(cell)

        self.assertEqual(cell._backward_hooks[1], "local")
        self.assertEqual(cell._backward_hooks[2], "global")

    def test__get_backward_hooks_when_flag_false_then_pass(self):
        cell = DummyCell()
        cell._backward_hooks = {1: "local"}

        with patch.object(adaptor, "_global_is_full_backward_hook", False), \
             patch.object(adaptor, "_global_backward_hooks", {2: "global"}):
            adaptor._get_backward_hooks(cell)

        # should keep original hooks unchanged when flag is False
        self.assertEqual(cell._backward_hooks, {1: "local"})

    def test_register_full_backward_pre_hook_when_prepend_false_then_pass(self):
        cell = DummyCell()
        cell._backward_pre_hooks = OrderedDict()

        class FakeHandle:
            _next_id = 0

            def __init__(self, mapping):
                FakeHandle._next_id += 1
                self.id = FakeHandle._next_id
                self.mapping = mapping

        hook = MagicMock()
        with patch.object(adaptor, "RemovableHandle", FakeHandle):
            handle = adaptor.register_full_backward_pre_hook(cell, hook, prepend=False)

        self.assertIn(handle.id, cell._backward_pre_hooks)
        self.assertIs(cell._backward_pre_hooks[handle.id], hook)

    def test_register_full_backward_pre_hook_when_prepend_true_then_pass(self):
        cell = DummyCell()
        cell._backward_pre_hooks = OrderedDict([(100, "existing")])

        class FakeHandle:
            _next_id = 0

            def __init__(self, mapping):
                FakeHandle._next_id += 1
                self.id = FakeHandle._next_id
                self.mapping = mapping

        hook = MagicMock()
        with patch.object(adaptor, "RemovableHandle", FakeHandle):
            handle = adaptor.register_full_backward_pre_hook(cell, hook, prepend=True)

        # new hook should be moved to the front
        first_key = next(iter(cell._backward_pre_hooks.keys()))
        self.assertEqual(first_key, handle.id)

    def test_register_full_backward_hook_when_is_full_backward_hook_false_then_raise(self):
        cell = DummyCell()
        cell._is_full_backward_hook = False

        with self.assertRaises(RuntimeError) as cm:
            adaptor.register_full_backward_hook(cell, MagicMock())

        self.assertIn("Cannot use both regular backward hooks and full backward hooks", str(cm.exception))

    def test_register_full_backward_hook_when_normal_then_pass(self):
        cell = DummyCell()
        cell._backward_hooks = OrderedDict()
        cell._is_full_backward_hook = None

        class FakeHandle:
            _next_id = 0

            def __init__(self, mapping):
                FakeHandle._next_id += 1
                self.id = FakeHandle._next_id
                self.mapping = mapping

        hook = MagicMock()
        with patch.object(adaptor, "RemovableHandle", FakeHandle):
            handle = adaptor.register_full_backward_hook(cell, hook, prepend=True)

        self.assertTrue(cell._is_full_backward_hook)
        self.assertIn(handle.id, cell._backward_hooks)
        # new hook should be at the front when prepend=True
        first_key = next(iter(cell._backward_hooks.keys()))
        self.assertEqual(first_key, handle.id)


if __name__ == "__main__":
    unittest.main()

