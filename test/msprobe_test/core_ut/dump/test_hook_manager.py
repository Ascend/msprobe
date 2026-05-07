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
        BaseHookManager.params_grad_info = {}

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
            self.mock_data_collector.forward_data_collect.assert_called_once()
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
