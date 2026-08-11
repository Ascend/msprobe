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
from msprobe.core.dump.hook_manager import HookSet, BaseHookManager
from msprobe.mindspore.dump.dump_processor.hook_cell.ms_hook_manager import MindsporeHookManager


class TestMindsporeHookManager(unittest.TestCase):
    def setUp(self):
        self.mock_data_collector = MagicMock()
        self.mock_config = MagicMock()
        self.mock_config.data_mode = ["all"]
        self.mock_config.task = "statistics"
        self.mock_config.level = Const.LEVEL_L1
        self.manager = MindsporeHookManager(
            self.mock_data_collector,
            self.mock_config
        )
        BaseHookManager.inner_switch[threading.get_ident()] = False

    def test_properties(self):
        with patch('msprobe.mindspore.dump.dump_processor.hook_cell.ms_hook_manager._no_grad') as mock_no_grad:
            ctx = self.manager._no_grad_context()
            mock_no_grad.assert_called_once()

    def test_add_count(self):
        with patch('msprobe.mindspore.dump.dump_processor.hook_cell.ms_hook_manager.HOOKCell.add_cell_count') as mock_add:
            self.manager._add_count("test_module")
            mock_add.assert_called_once_with("test_module")

    def test_process_kwargs_and_output(self):
        tid = threading.get_ident()
        mock_module = MagicMock()
        mock_module.msprobe_input_kwargs = {tid: {"kw1": "v1"}}

        kwargs, output = self.manager._process_kwargs_and_output(
            mock_module,
            tid,
            Const.API,
            "output_value",
            "ignored"
        )
        self.assertEqual(kwargs, {"kw1": "v1"})
        self.assertEqual(output, "output_value")

        with patch('msprobe.mindspore.dump.dump_processor.hook_cell.ms_hook_manager.has_kwargs_in_forward_hook', return_value=True):
            kwargs, output = self.manager._process_kwargs_and_output(
                mock_module,
                tid,
                Const.MODULE,
                "kwargs_value",
                "output_value"
            )
            self.assertEqual(kwargs, "kwargs_value")
            self.assertEqual(output, "output_value")

    def test_build_hook(self):
        hook_set = self.manager.build_hook(Const.API, "test_api")
        self.assertIsInstance(hook_set, HookSet)
        self.assertEqual(hook_set.forward_pre_hook.__name__, "forward_pre_hook")

        hook_set = self.manager.build_hook(Const.MODULE, "test_module")
        self.assertEqual(hook_set.module_forward_pre_hook.__name__, "module_forward_pre_hook")
        self.assertEqual(hook_set.forward_hook.__name__, "forward_hook")
        self.assertEqual(hook_set.backward_pre_hook.__name__, "backward_pre_hook")
        self.assertEqual(hook_set.backward_hook.__name__, "backward_hook")

    def test_need_exchange(self):
        mock_module = MagicMock()
        del mock_module.has_pre_hook_called
        self.assertFalse(self.manager._need_exchange(mock_module))

        mock_module.has_pre_hook_called = False
        self.assertFalse(self.manager._need_exchange(mock_module))

        mock_module.has_pre_hook_called = True
        self.assertTrue(self.manager._need_exchange(mock_module))

    def test_get_params_dict(self):
        mock_module = MagicMock()

        self.mock_config.task = Const.STRUCTURE
        params_dict = self.manager._get_params_dict(mock_module)
        self.assertEqual(params_dict, {})

        self.mock_config.task = "statistics"
        mock_params = {
            "test_module.weight": "w1",
            "test_module.bias": "b1"
        }
        mock_module.parameters_dict.return_value = mock_params
        params_dict = self.manager._get_params_dict(mock_module)
        mock_module.parameters_dict.assert_called_once_with(recurse=False)
        self.assertEqual(params_dict, {"weight": "w1", "bias": "b1"})

    def test_build_backward_pre_hook(self):
        hook_fn = self.manager._build_backward_pre_hook(Const.MODULE, "test_module_backward")

        mock_module = MagicMock()
        mock_grad_input = ("grad1", "grad2")

        with patch.object(self.manager, '_should_execute_hook', return_value=False):
            hook_fn(mock_module, mock_grad_input)
            self.mock_data_collector.backward_input_data_collect.assert_not_called()

        self.mock_config.level = Const.LEVEL_L2
        with patch.object(self.manager, '_should_execute_hook', return_value=True):
            hook_fn(mock_module, mock_grad_input)

            self.mock_data_collector.update_api_or_module_name.assert_called_with("test_module_backward")
            self.mock_data_collector.backward_input_data_collect.assert_called_once()

            call_args = self.mock_data_collector.backward_input_data_collect.call_args[0]
            module_input = call_args[3]
            self.assertEqual(module_input.grad_input, mock_grad_input)

            self.assertFalse(BaseHookManager.inner_switch[threading.get_ident()])

        self.mock_config.level = Const.LEVEL_L1
        with patch.object(self.manager, '_should_execute_hook', return_value=True):
            hook_fn(mock_module, mock_grad_input)
            self.mock_data_collector.backward_input_data_collect.assert_called_once()

    def test_register_param_hook(self):
        BaseHookManager.hook_handle_dict.clear()
        module = MagicMock()
        param1 = MagicMock(requires_grad=True)
        param1.register_hook.return_value = MagicMock()
        param2 = MagicMock(requires_grad=False)
        params = {"param1": param1, "param2": param2}
        full_name = "module.forward"

        # data_mode包含backward时，应该注册hook
        self.mock_config.data_mode = ["all"]
        self.manager._register_param_hook(full_name, module, params)
        self.assertEqual(len(BaseHookManager.hook_handle_dict), 1)
        self.assertTrue("module.param1" in BaseHookManager.hook_handle_dict)
        param1.register_hook.assert_called_once()

        # data_mode只有forward时，不注册hook
        BaseHookManager.hook_handle_dict.clear()
        self.mock_config.data_mode = ["forward"]
        self.manager._register_param_hook(full_name, module, params)
        self.assertEqual(len(BaseHookManager.hook_handle_dict), 0)
        self.assertEqual(param1.register_hook.call_count, 1)
