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
from contextlib import nullcontext
from msprobe.pytorch.dump.api_dump.pt_hook_manager import PytorchHookManager
from msprobe.core.common.const import Const
from msprobe.core.dump.hook_manager import HookSet, BaseHookManager

class TestPytorchHookManager(unittest.TestCase):
    def setUp(self):
        self.mock_data_collector = MagicMock()
        self.mock_config = MagicMock()
        self.mock_config.data_mode = ["all"]
        self.mock_config.task = "statistics"
        self.manager = PytorchHookManager(
            self.mock_data_collector,
            self.mock_config
        )
        BaseHookManager.inner_switch[threading.get_ident()] = False

    def test_no_grad_context(self):
        """当 _disable_current_modes 不可用时应回退到 nullcontext"""
        with patch('msprobe.pytorch.dump.api_dump.pt_hook_manager._disable_current_modes', None):
            ctx = self.manager._no_grad_context()
            self.assertIsInstance(ctx, nullcontext)

    def test_no_grad_context_disable_modes(self):
        """当 _disable_current_modes 可用时应调用它"""
        mock_ctx = MagicMock()
        with patch('msprobe.pytorch.dump.api_dump.pt_hook_manager._disable_current_modes',
                   MagicMock(return_value=mock_ctx)) as mock_disable:
            result = self.manager._no_grad_context()
            mock_disable.assert_called_once()
            self.assertIs(result, mock_ctx)

    def test_init_sets_logger(self):
        """__init__ 应设置 self.logger（从 _init_specific_components 移入）"""
        self.assertIsNotNone(self.manager.logger)
        from msprobe.pytorch.dump.api_dump.pt_hook_manager import logger
        self.assertIs(self.manager.logger, logger)

    def test_init_specific_components_is_noop(self):
        """_init_specific_components 应为空操作"""
        result = self.manager._init_specific_components()
        self.assertIsNone(result)

    def test_add_count(self):
        with patch('msprobe.pytorch.dump.api_dump.pt_hook_manager.HOOKModule.add_module_count') as mock_add:
            self.manager._add_count("test_layer")
            mock_add.assert_called_once_with("test_layer")

    def test_process_kwargs_and_output(self):
        kwargs, output = self.manager._process_kwargs_and_output(
            None,
            None,
            "API",
            "kwargs_value",
            "output_value"
        )
        self.assertEqual(kwargs, "kwargs_value")
        self.assertEqual(output, "output_value")

        with patch('msprobe.pytorch.dump.api_dump.pt_hook_manager.torch_version_above_or_equal_2', new=True):
            kwargs, output = self.manager._process_kwargs_and_output(
                None,
                None,
                None,
                "kwargs_value",
                "output_value"
            )
            self.assertEqual(kwargs, "kwargs_value")
            self.assertEqual(output, "output_value")

        with patch('msprobe.pytorch.dump.api_dump.pt_hook_manager.torch_version_above_or_equal_2', new=False):
            kwargs, output = self.manager._process_kwargs_and_output(
                None,
                None,
                None,
                "kwargs_value",
                "output_value"
            )
            self.assertEqual(kwargs, {})
            self.assertEqual(output, "kwargs_value")

    def test_build_hook(self):
        hook_set = self.manager.build_hook(Const.API, "test_api")
        self.assertIsInstance(hook_set, HookSet)
        self.assertTrue(callable(hook_set.forward_pre_hook))
        self.assertTrue(callable(hook_set.distributed_forward_hook))
        self.assertIsNone(hook_set.forward_hook)
        self.assertIsNone(hook_set.backward_pre_hook)
        self.assertIsNone(hook_set.backward_hook)

        hook_set = self.manager.build_hook(Const.MODULE, "test_module")
        self.assertEqual(hook_set.forward_hook.__name__, "forward_hook")
        self.assertEqual(hook_set.backward_hook.__name__, "backward_hook")

    def test_need_exchange(self):
        self.assertTrue(self.manager._need_exchange(None))
        self.assertTrue(self.manager._need_exchange(MagicMock()))

    def test_get_params_dict(self):
        mock_module = MagicMock()

        self.mock_config.task = Const.STRUCTURE
        params_dict = self.manager._get_params_dict(mock_module)
        self.assertEqual(params_dict, {})

        self.mock_config.task = "statistics"

        mock_named_params = [
            ("conv.weight", MagicMock()),
            ("bn.bias", MagicMock())
        ]
        mock_module.named_parameters.return_value = mock_named_params
        params_dict = self.manager._get_params_dict(mock_module)
        mock_module.named_parameters.assert_called_once_with(recurse=False)

        self.assertEqual(set(params_dict.keys()), {"weight", "bias"})
        self.assertEqual(params_dict["weight"], mock_named_params[0][1])
        self.assertEqual(params_dict["bias"], mock_named_params[1][1])

    def test_register_param_hook(self):
        BaseHookManager.hook_handle_dict.clear()
        module = MagicMock()
        param1 = MagicMock(requires_grad=True)
        param1.expand_as.return_value.grad_fn = None
        param2 = MagicMock(requires_grad=False)
        params = {"param1": param1, "param2": param2}
        full_name = "module.forward"

        # data_mode包含backward时，应该注册hook
        self.mock_config.data_mode = ["all"]
        self.manager._register_param_hook(full_name, module, params)
        # param1.requires_grad=True，但grad_fn为None，所以不会注册
        self.assertEqual(len(BaseHookManager.hook_handle_dict), 0)

        # 测试有grad_fn的情况
        mock_grad_acc = MagicMock()
        mock_grad_fn = MagicMock()
        mock_grad_fn.next_functions = [(mock_grad_acc, None)]
        param1.expand_as.return_value.grad_fn = mock_grad_fn
        mock_grad_acc.register_hook.return_value = MagicMock()

        self.manager._register_param_hook(full_name, module, params)
        self.assertEqual(len(BaseHookManager.hook_handle_dict), 1)
        self.assertTrue("module.param1" in BaseHookManager.hook_handle_dict)

        # data_mode只有forward时，不注册hook
        BaseHookManager.hook_handle_dict.clear()
        self.mock_config.data_mode = ["forward"]
        self.manager._register_param_hook(full_name, module, params)
        self.assertEqual(len(BaseHookManager.hook_handle_dict), 0)
