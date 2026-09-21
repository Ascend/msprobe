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

import unittest
import torch
from unittest.mock import patch

from msprobe.pytorch.api_accuracy_checker.acc_check.data_generate import fp32_to_hf32_to_fp32, gen_random_tensor, gen_atten_mask


class TestFP32ToHF32ToFP32(unittest.TestCase):
    """Test cases for fp32_to_hf32_to_fp32 function."""

    def setUp(self):
        """Set up test fixtures."""
        self.torch_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)

    def test_basic_conversion(self):
        """Test basic conversion with simple tensor."""
        result = fp32_to_hf32_to_fp32(self.torch_tensor)

        # Check result type and shape
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(result.shape, self.torch_tensor.shape)


class TestGenRandomTensor(unittest.TestCase):
    """Test cases for gen_random_tensor function."""

    def test_zero_in_shape_branch(self):
        """Test the if 0 in shape branch."""
        # Test case 1: shape contains 0 in first position
        info1 = {
            'Min': 1.0,
            'Max': 10.0,
            'Min_except_inf_nan': 1.0,
            'Max_except_inf_nan': 10.0,
            'dtype': 'torch.float32',
            'shape': [0, 3, 4]
        }

        with patch('msprobe.pytorch.api_accuracy_checker.acc_check.data_generate.gen_common_tensor') as mock_gen_common:
            mock_gen_common.return_value = torch.tensor([]).reshape(0, 3, 4)

            result = gen_random_tensor(info1, convert_type=False)

            # Verify that low and high were set to 0
            # The mock should have been called with [0, 0] for both low_info and high_info
            mock_gen_common.assert_called_once()
            call_args = mock_gen_common.call_args[0]
            self.assertEqual(call_args[0], [0, 0])  # low_info
            self.assertEqual(call_args[1], [0, 0])  # high_info
            self.assertEqual(call_args[2], (0, 3, 4))  # shape


class TestGenAttenMask(unittest.TestCase):
    """Test cases for gen_atten_mask function."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_real_data_path = "/tmp/test_data"

    def test_tensor_type_with_data_path(self):
        """Test tensor type with data path."""
        info = {
            'type': 'torch.Tensor',
            'datapath': '/tmp/mask_data',
            'data_name': 'mask_name.pt'  # datapath should take precedence
        }

        with patch('msprobe.pytorch.api_accuracy_checker.acc_check.data_generate.gen_real_tensor') as mock_gen_real:
            expected_tensor = torch.tensor([[1, 0], [0, 1]], dtype=torch.bool)
            mock_gen_real.return_value = expected_tensor

            result = gen_atten_mask(info, convert_type=True, real_data_path=self.test_real_data_path)

            # Verify result
            self.assertIsNotNone(result)
            torch.testing.assert_close(result, expected_tensor)
