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

import threading

import unittest
from unittest.mock import MagicMock, patch

from msprobe.core.common.const import Const
from msprobe.core.common.runtime import Runtime
from msprobe.core.dump.custom_tag import CustomTagManager
from msprobe.core.dump.hook_manager import BaseHookManager


class TestBaseHookManager(unittest.TestCase):
    class MockBaseHookManager(BaseHookManager):
        @staticmethod
        def _no_grad_context():
            return MagicMock()

        @staticmethod
        def _add_count(name):
            pass

        @staticmethod
        def _get_count(name):
            pass

        @staticmethod
        def _process_kwargs_and_output(module, tid, hook_type, kwargs_or_output, output_or_kwargs):
            return {"kwargs": kwargs_or_output}, output_or_kwargs

        @staticmethod
        def _get_current_rank():
            pass

        def build_hook(self):
            pass

        def _register_forward_hook(self, module, api_name):
            pass

        def _register_backward_hook(self, module, full_backward_name, args):
            pass

        def _register_backward_pre_hook(self, module, full_backward_name, args, kwargs, output):
            pass

        def _get_params_dict(self, module):
            return {}

        def _need_exchange(self, module):
            return False

        def _register_param_hook(self, name, module, params_dict):
            pass

        def _init_specific_components(self):
            """初始化框架特定组件"""
            pass

    def setUp(self):
        self.mock_data_collector = MagicMock()
        self.mock_config = MagicMock()
        self.mock_config.level = Const.LEVEL_MIX
        self.mock_config.data_mode = ["all"]
        self.manager = self.MockBaseHookManager(
            self.mock_data_collector,
            self.mock_config
        )
        BaseHookManager.inner_switch[threading.get_ident()] = False
        BaseHookManager.hook_handle_dict = {}
        BaseHookManager.grad_hook_call = {}

    @staticmethod
    def _reset_api_call_counters():
        """与 setUp 一致显式置空本线程的类级计数器。

        pre-hook 仅在 inner_api_count 自增后等于 1(本线程第一次 API 调用)时才会
        进入名称与自定义标签的登记分支, 置零可避免前序用例残留影响本用例。
        """
        tid = threading.get_ident()
        BaseHookManager.inner_switch[tid] = False
        BaseHookManager.inner_api_count[tid] = 0

    def test_init(self):
        self.assertEqual(self.manager.data_collector, self.mock_data_collector)
        self.assertEqual(self.manager.config, self.mock_config)

    def test_should_execute_hook_conditions(self):
        tid = threading.get_ident()
        Runtime.is_running = True
        BaseHookManager.inner_switch[tid] = False
        self.mock_data_collector.data_processor.is_terminated = False
        self.assertTrue(self.manager._should_execute_hook(Const.MODULE, tid))
        self.assertTrue(self.manager._should_execute_hook(Const.API, tid))
        self.assertTrue(self.manager._should_execute_hook(Const.API, tid, is_forward=False))

        Runtime.is_running = False
        self.assertFalse(self.manager._should_execute_hook(Const.MODULE, tid))
        self.assertFalse(self.manager._should_execute_hook(Const.API, tid))
        self.assertTrue(self.manager._should_execute_hook(Const.API, tid, is_forward=False))

        Runtime.is_running = True
        BaseHookManager.inner_switch[tid] = True
        self.assertFalse(self.manager._should_execute_hook(Const.MODULE, tid))
        self.assertFalse(self.manager._should_execute_hook(Const.API, tid))
        self.assertFalse(self.manager._should_execute_hook(Const.API, tid, is_forward=False))

        self.mock_data_collector.data_processor.is_terminated = True
        BaseHookManager.inner_switch[tid] = False
        self.assertFalse(self.manager._should_execute_hook(Const.MODULE, tid))
        self.assertFalse(self.manager._should_execute_hook(Const.API, tid))
        self.assertFalse(self.manager._should_execute_hook(Const.API, tid, is_forward=False))

    def test_clear_input_kwargs(self):
        module = MagicMock()
        tid = threading.get_ident()
        module.msprobe_input_kwargs[tid] = {"key": "value"}
        self.manager._clear_input_kwargs(module, tid)
        self.assertFalse(tid in module.msprobe_input_kwargs)

    @patch.object(BaseHookManager, "_should_execute_hook")
    def test_forward_pre_hook_behavior(self, mock_should_execute_hook):
        Runtime.current_rank = None
        mock_should_execute_hook.return_value = True
        hook = self.manager._build_forward_pre_hook(Const.API, "api_name")
        module = MagicMock()
        module.msprobe_input_kwargs = {"kwarg": "value"}
        args = (1, 2)

        Runtime.is_running = True
        self.mock_data_collector.data_processor.is_terminated = False
        with patch.object(self.manager, '_no_grad_context') as mock_ctx:
            hook(module, args)
            self.mock_data_collector.forward_input_data_collect.assert_called_once()

    @patch.object(BaseHookManager, "_should_execute_hook")
    def test_forward_pre_hook_registers_custom_tag(self, mock_should_execute_hook):
        self._reset_api_call_counters()  # 确保本次调用是本线程第一次 API 调用
        Runtime.current_rank = None
        mock_should_execute_hook.return_value = True
        hook = self.manager._build_forward_pre_hook(Const.API, "api_name")
        module = MagicMock()
        module.msprobe_input_kwargs = {"kwarg": "value"}
        args = (1, 2)

        Runtime.is_running = True
        self.mock_data_collector.data_processor.is_terminated = False
        with patch.object(self.manager, '_no_grad_context'), \
                patch.object(CustomTagManager, 'record_tags') as mock_record:
            CustomTagManager.push("ut_api_tag")
            try:
                hook(module, args)
            finally:
                CustomTagManager.pop()

            mock_record.assert_called_once()
            names, label = mock_record.call_args[0]
            self.assertEqual(label, "ut_api_tag")
            self.assertEqual(len(names), 2)  # 前向名 + 反向名
            self.assertTrue(any(n.endswith(Const.FORWARD) for n in names))
            self.assertTrue(any(n.endswith(Const.BACKWARD) for n in names))

    @patch.object(BaseHookManager, "_should_execute_hook")
    def test_forward_pre_hook_without_tag_registers_nothing(self, mock_should_execute_hook):
        self._reset_api_call_counters()  # 确保本次调用是本线程第一次 API 调用
        Runtime.current_rank = None
        mock_should_execute_hook.return_value = True
        hook = self.manager._build_forward_pre_hook(Const.API, "api_name")
        module = MagicMock()
        module.msprobe_input_kwargs = {"kwarg": "value"}
        args = (1, 2)

        Runtime.is_running = True
        self.mock_data_collector.data_processor.is_terminated = False
        with patch.object(self.manager, '_no_grad_context'), \
                patch.object(CustomTagManager, 'record_tags') as mock_record:
            hook(module, args)
            mock_record.assert_not_called()  # 无活跃标签时零登记

    @patch.object(BaseHookManager, "_should_execute_hook")
    def test_forward_pre_hook_tag_after_untagged_call_registers_once(self, mock_should_execute_hook):
        # 覆盖真实 step 场景: 本线程首个 API 调用发生在标签作用域外(如投影内部的
        # F.linear), 该调用完整结束后才进入 with tag 执行第二个 API 调用。
        self._reset_api_call_counters()  # 确保本次调用是本线程第一次 API 调用
        Runtime.current_rank = None
        mock_should_execute_hook.return_value = True
        pre_hook = self.manager._build_forward_pre_hook(Const.API, "untagged_api")
        forward_hook = self.manager._build_forward_hook(Const.API, "untagged_api")
        tagged_pre_hook = self.manager._build_forward_pre_hook(Const.API, "tagged_api")
        module = MagicMock()
        module.msprobe_input_kwargs = {"kwarg": "value"}
        args = (1, 2)

        Runtime.is_running = True
        self.mock_data_collector.data_processor.is_terminated = False
        with patch.object(self.manager, '_no_grad_context'), \
                patch.object(CustomTagManager, 'record_tags') as mock_record:
            # 第一次 API 调用: 无活跃标签(发生在标签作用域之前)
            pre_hook(module, args)
            mock_record.assert_not_called()
            # 走完 forward_hook 使 inner_api_count 回落到 0, 与真实 step 中一次完整调用一致
            forward_hook(module, args, {"kwarg": "value"}, MagicMock())
            self.assertEqual(BaseHookManager.inner_api_count[threading.get_ident()], 0)

            # 第二次 API 调用: push 标签后执行
            CustomTagManager.push("ut_api_tag")
            try:
                tagged_pre_hook(module, args)
            finally:
                CustomTagManager.pop()

            mock_record.assert_called_once()  # 仅带标签的第二次调用发生登记
            names, label = mock_record.call_args[0]
            self.assertEqual(label, "ut_api_tag")
            self.assertTrue(any("tagged_api" in n for n in names))

    @patch.object(BaseHookManager, "_should_execute_hook")
    def test_forward_hook_behavior(self, mock_should_execute_hook):
        Runtime.current_rank = None
        mock_should_execute_hook.return_value = True
        hook = self.manager._build_forward_hook(Const.MODULE, "module_name")
        module = MagicMock()
        args = (1, 2)
        kwargs = {"kwargs": []}
        output = MagicMock()

        self.mock_data_collector.if_return_forward_new_output.return_value = False
        with patch.object(self.manager, '_get_params_dict', return_value={}):
            result = hook(module, args, kwargs, output)
            self.assertEqual(result, output)
            self.mock_data_collector.forward_output_data_collect.assert_called_once()
            self.mock_data_collector.get_forward_new_output.assert_not_called()

        self.mock_data_collector.if_return_forward_new_output.return_value = True
        with patch.object(self.manager, '_get_params_dict', return_value={}):
            result = hook(module, args, output)
            self.assertEqual(result, None)

    @patch.object(BaseHookManager, "_should_execute_hook")
    def test_backward_hook_behavior(self, mock_should_execute_hook):
        Runtime.current_rank = None
        mock_should_execute_hook.return_value = True
        hook = self.manager._build_backward_hook(Const.API, "api_name")
        module = MagicMock()
        grad_input = (MagicMock(),)
        grad_output = (MagicMock(),)

        module.forward_data_collected = True
        Runtime.is_running = True
        hook(module, grad_input, grad_output)

        self.mock_data_collector.backward_data_collect.assert_called_once()

        with patch.object(self.manager, '_need_exchange', return_value=True):
            hook(module, grad_input, grad_output)
            args, _ = self.mock_data_collector.backward_data_collect.call_args_list[1]
            self.assertEqual(args[3].grad_input, grad_output)
            self.assertEqual(args[3].grad_output, grad_input)
