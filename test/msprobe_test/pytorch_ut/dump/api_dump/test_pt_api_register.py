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

import unittest
from unittest.mock import MagicMock, patch

import msprobe.pytorch.dump.api_dump.api_register as api_register
from msprobe.pytorch.dump.api_dump.api_register import (
    tensor_module_forward,
    dist_module_forward,
    npu_module_forward,
    get_api_register,
    ApiTemplate
)


class TestAPIRegister(unittest.TestCase):
    def setUp(self):
        api_register.api_register = None

    def test_tensor_module_forward(self):
        mock_module = MagicMock()
        mock_module.api_name = "test_name"
        mock_module.api_func.return_value = "test_result"

        args = (1, 2, 3)
        kwargs = {"key": "value"}
        result = tensor_module_forward(mock_module, *args, **kwargs)

        mock_module.api_func.assert_called_once_with(*args, **kwargs)
        self.assertEqual(result, "test_result")

    @patch('msprobe.pytorch.dump.api_dump.api_register.logger.warning')
    def test_basic_dist_module_forward(self, mock_logger):
        mock_module = MagicMock()
        mock_module.api_func.return_value = "test_handle"
        mock_module.api_name = "test_api"

        result = dist_module_forward(mock_module, 1, 2, key="value")
        mock_module.api_func.assert_called_once_with(1, 2, key="value")
        self.assertEqual(result, "test_handle")
        mock_logger.assert_not_called()

    @patch('msprobe.pytorch.dump.api_dump.api_register.ApiRegistry')
    def test_get_api_register_with_new_obj(self, mock_api_registry):
        get_api_register(return_new=True)
        mock_api_registry.assert_called_once()
        self.assertIsNone(api_register.api_register)

    @patch('msprobe.pytorch.dump.api_dump.api_register.ApiRegistry')
    def test_get_api_register_with_not_new_obj(self, mock_api_registry):
        get_api_register()
        mock_api_registry.assert_called_once()
        self.assertIsNotNone(api_register.api_register)


class TestNpuModuleForward(unittest.TestCase):
    def setUp(self):
        self.npu_custom_functions = {
            "custom_func": MagicMock(return_value="custom_result"),
            "npu_fusion_attention": MagicMock(return_value="nfa_result"),
            "gpu_fusion_attention": MagicMock(return_value="gfa_result")
        }

        self.module = MagicMock()
        self.module.api_func.return_value = "test_result"

    @patch('msprobe.pytorch.dump.api_dump.api_register.get_npu_custom_functions')
    def test_with_hook_enabled(self, mock_get_npu_custom_functions):
        self.module.need_hook = True
        result = npu_module_forward(self.module, 1, 2, key="value")
        self.module.api_func.assert_called_once_with(1, 2, key="value")
        self.assertEqual(result, "test_result")
        mock_get_npu_custom_functions.assert_not_called()

    def test_with_unknown_api(self):
        self.module.need_hook = False
        self.module.api_name = "unknown_func"
        with patch(
            'msprobe.pytorch.dump.api_dump.api_register.get_npu_custom_functions',
            return_value=self.npu_custom_functions,
        ):
            with self.assertRaises(Exception) as context:
                npu_module_forward(self.module, 1, 2, key="value")
        self.assertIn("There is not bench function unknown_func", str(context.exception))

    def test_cuda_device_with_mapping(self):
        self.module.need_hook = False
        self.module.api_name = "npu_fusion_attention"
        self.module.device = 'cuda'

        with patch(
            'msprobe.pytorch.dump.api_dump.api_register.get_npu_custom_functions',
            return_value=self.npu_custom_functions,
        ):
            result = npu_module_forward(self.module, 1, 2, key="value")
        self.npu_custom_functions["gpu_fusion_attention"].assert_called_once_with(1, 2, key="value")
        self.assertEqual(result, "gfa_result")

    def test_cpu_device(self):
        self.module.need_hook = False
        self.module.api_name = "custom_func"
        self.module.device = "cpu"

        with patch(
            'msprobe.pytorch.dump.api_dump.api_register.get_npu_custom_functions',
            return_value=self.npu_custom_functions,
        ):
            result = npu_module_forward(self.module, 1, 2, key="value")
        self.npu_custom_functions["custom_func"].assert_called_once_with(1, 2, key="value")
        self.assertEqual(result, "custom_result")

    def test_unsupported_device(self):
        self.module.need_hook = False
        self.module.api_name = "custom_func"
        self.module.device = "unsupported_device"

        with patch(
            'msprobe.pytorch.dump.api_dump.api_register.get_npu_custom_functions',
            return_value=self.npu_custom_functions,
        ):
            result = npu_module_forward(self.module, 1, 2, key="value")
        self.module.api_func.assert_called_once_with(1, 2, key="value")
        self.assertEqual(result, "test_result")


class TestApiTemplate(unittest.TestCase):
    def setUp(self):
        self.api_name = "Tensor.test_api"
        self.api_func = MagicMock(return_value="test_result")
        self.prefix = "test_prefix"
        self.hook_build_func = MagicMock()
        self.mock_hook_module = MagicMock()

    def test_init(self):
        with patch('msprobe.pytorch.dump.api_dump.api_register.HOOKModule') as mock_hook_module:
            template = ApiTemplate(
                self.api_name,
                self.api_func,
                self.prefix,
                self.hook_build_func,
                need_hook=False
            )

            self.assertEqual(template.api_name, self.api_name)
            self.assertEqual(template.api_func, self.api_func)
            self.assertEqual(template.prefix, self.prefix)
            self.assertEqual(template.prefix_api_name, "test_prefix.test_api.")
            self.assertEqual(template.device, "cpu")
            self.assertFalse(template.op_is_distributed)
            self.assertFalse(template.need_hook)

    def test_init_with_distributed_prefix(self):
        with patch('msprobe.pytorch.dump.api_dump.api_register.HOOKModule'):
            self.prefix = "Distributed"
            template = ApiTemplate(
                self.api_name,
                self.api_func,
                self.prefix,
                self.hook_build_func,
                need_hook=False,
                device="npu"
            )

            self.assertEqual(template.device, "npu")
            self.assertEqual(template.prefix_api_name, "Distributed.test_api.")
            self.assertTrue(template.op_is_distributed)

    def test_init_without_hook(self):
        with patch('msprobe.pytorch.dump.api_dump.api_register.HOOKModule') as mock_hook_module:
            template = ApiTemplate(
                self.api_name,
                self.api_func,
                self.prefix,
                self.hook_build_func,
                need_hook=False,
                device="npu"
            )

            self.assertFalse(template.need_hook)
            self.mock_hook_module.assert_not_called()

    def test_forward_with_prefix_match(self):
        with patch('msprobe.pytorch.dump.api_dump.api_register.HOOKModule'):
            self.prefix = "Tensor"
            template = ApiTemplate(
                self.api_name,
                self.api_func,
                self.prefix,
                self.hook_build_func,
                need_hook=False,
                device="npu"
            )

            result = template.forward("arg1", key="value")

            self.assertEqual(result, "test_result")

    def test_forward_without_prefix_match(self):
        with patch('msprobe.pytorch.dump.api_dump.api_register.HOOKModule'):
            template = ApiTemplate(
                self.api_name,
                self.api_func,
                self.prefix,
                self.hook_build_func,
                need_hook=False,
                device="npu"
            )

            result = template.forward("arg1", key="value")

            self.api_func.assert_called_once_with("arg1", key="value")
            self.assertEqual(result, "test_result")
