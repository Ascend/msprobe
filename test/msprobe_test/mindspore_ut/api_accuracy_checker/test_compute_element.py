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
import sys
import logging
import os
import mindspore
import torch
import numpy as np

from msprobe.mindspore.api_accuracy_checker.compute_element import ComputeElement
from msprobe.mindspore.api_accuracy_checker.type_mapping import (FLOAT32, FLOAT_TYPE_STR, INT_TYPE_STR,
                                                                 TUPLE_TYPE_STR, STR_TYPE_STR, SLICE_TYPE_STR)
from msprobe.mindspore.api_accuracy_checker.utils import global_context
from msprobe.core.common.const import Const

logging.basicConfig(stream = sys.stdout, level = logging.INFO, format = '[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

file_path = os.path.abspath(__file__)
directory = os.path.dirname(file_path)


class TestComputeElement(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        class level setup_class
        """
        cls.init(TestComputeElement)

    def init(self):
        global_context.init(False, os.path.join(directory, "files"), "mindspore")
        self.ndarray = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.float32)
        self.ms_tensor = mindspore.Tensor(self.ndarray)
        self.torch_tensor = torch.Tensor(self.ndarray)
        self.tensor_shape = (2, 3)


    def test_init_with_parameter_tensor(self):
        # input_parameter, origin_parameter, mstensor_parameter, torchtensor_parameter, shape, dtype_str
        parameter_results_mapping = [
            [self.ms_tensor, self.ms_tensor, self.ms_tensor, self.torch_tensor, self.tensor_shape, FLOAT32],
            [self.torch_tensor, self.torch_tensor, self.ms_tensor, self.torch_tensor, self.tensor_shape, FLOAT32],
        ]
        for parameter_result in parameter_results_mapping:
            input_parameter, origin_parameter, mstensor_parameter, torchtensor_parameter, shape, dtype_str = parameter_result

            compute_element = ComputeElement(parameter=input_parameter)

            self.assertTrue((compute_element.get_parameter(get_origin=True) == origin_parameter).all())
            self.assertTrue((compute_element.get_parameter(get_origin=False, tensor_platform=Const.MS_FRAMEWORK) == mstensor_parameter).all())
            self.assertTrue((compute_element.get_parameter(get_origin=False, tensor_platform=Const.PT_FRAMEWORK) == torchtensor_parameter).all())
            self.assertEqual(compute_element.get_shape(), shape)
            self.assertEqual(compute_element.get_dtype(), dtype_str)

    def test_init_with_parameter_other_type(self):
        # input_parameter, origin_parameter, shape, dtype_str
        parameter_results_mapping = [
            [1, 1, tuple(), INT_TYPE_STR],
            [1.0, 1.0, tuple(), FLOAT_TYPE_STR],
            ["string", "string", tuple(), STR_TYPE_STR],
            [slice(1, 10, 2), slice(1, 10, 2), tuple(), SLICE_TYPE_STR],
            [tuple([1, 2]), tuple([1, 2]), tuple(), TUPLE_TYPE_STR],
        ]

        for parameter_result in parameter_results_mapping:
            input_parameter, origin_parameter, shape, dtype_str = parameter_result

            compute_element = ComputeElement(parameter=input_parameter)

            self.assertEqual(compute_element.get_parameter(get_origin=True), origin_parameter)
            self.assertEqual(compute_element.get_shape(), shape)
            self.assertEqual(compute_element.get_dtype(), dtype_str)

    def test_init_with_compute_element_info_mstensor(self):
        global_context.is_constructed = False
        compute_element_info = {
            "type": "mindspore.Tensor",
            "dtype": "Float32",
            "shape":[2, 3],
            "Max": 3.0,
            "Min": 1.0,
            "data_name": "2_3_input.npy"
        }
        compute_element = ComputeElement(compute_element_info=compute_element_info)
        self.assertTrue((compute_element.get_parameter(get_origin=True) == self.ms_tensor).all())
        self.assertEqual(compute_element.get_shape(), self.tensor_shape)
        self.assertEqual(compute_element.get_dtype(), FLOAT32)

    def test_init_with_compute_element_info_mstensor_constructed(self):
        global_context.is_constructed = True
        compute_element_info = {
            "type": "mindspore.Tensor",
            "dtype": "Float32",
            "shape":[2, 3],
            "Max": 3.0,
            "Min": 1.0,
            "data_name": "2_3_input.npy"
        }
        compute_element = ComputeElement(compute_element_info=compute_element_info)
        parameter = compute_element.get_parameter(get_origin=True)
        self.assertTrue((parameter <= 3.0).all())
        self.assertTrue((parameter >= 1.0).all())
        self.assertEqual(compute_element.get_shape(), self.tensor_shape)
        self.assertEqual(compute_element.get_dtype(), FLOAT32)

    def test_init_with_compute_element_info_tuple(self):
        global_context.is_constructed = False
        compute_element_info = [
            {
                "type": "mindspore.Tensor",
                "dtype": "Float32",
                "shape":[2, 3],
                "Max": 3.0,
                "Min": 1.0,
                "data_name": "2_3_input.npy"
            },
            {
                "type": "mindspore.Tensor",
                "dtype": "Float32",
                "shape":[2, 3],
                "Max": 3.0,
                "Min": 1.0,
                "data_name": "2_3_input.npy"
            },
        ]
        compute_element = ComputeElement(compute_element_info=compute_element_info)
        mindspore_parameter = compute_element.get_parameter(get_origin=False, tensor_platform="mindspore")
        self.assertTrue((mindspore_parameter[0] == self.ms_tensor).all())
        self.assertTrue((mindspore_parameter[1] == self.ms_tensor).all())
        torch_parameter = compute_element.get_parameter(get_origin=False, tensor_platform="pytorch")
        self.assertTrue((torch_parameter[0] == self.torch_tensor).all())
        self.assertTrue((torch_parameter[1] == self.torch_tensor).all())
        self.assertEqual(compute_element.get_shape(), tuple())
        self.assertEqual(compute_element.get_dtype(), TUPLE_TYPE_STR)

    def test_init_with_compute_element_info_int(self):
        compute_element_info = {
            "type": "int",
            "value": -1,
        }
        compute_element = ComputeElement(compute_element_info=compute_element_info)
        parameter = compute_element.get_parameter(get_origin=True)
        self.assertEqual(parameter, -1)
        self.assertEqual(compute_element.get_shape(), tuple())
        self.assertEqual(compute_element.get_dtype(), INT_TYPE_STR)

    def test_init_with_compute_element_info_mindspore_dtype(self):
        compute_element_info = {
            "type": "mindspore.dtype",
            "value": "Float32",
        }
        compute_element = ComputeElement(compute_element_info=compute_element_info)
        ms_parameter = compute_element.get_parameter(tensor_platform=Const.MS_FRAMEWORK)
        pt_parameter = compute_element.get_parameter(tensor_platform=Const.PT_FRAMEWORK)
        self.assertEqual(ms_parameter, mindspore.float32)
        self.assertEqual(pt_parameter, torch.float32)

    def test_transfer_to_torch_tensor(self):
        ms_tensor_2_torch_tensor_mapping = {
            mindspore.Tensor([1, 2, 3], dtype=mindspore.uint8): torch.tensor([1, 2, 3], dtype=torch.uint8),
            mindspore.Tensor([1, 2, 3], dtype=mindspore.float32): torch.tensor([1, 2, 3], dtype=torch.float32)
        }
        for ms_tensor, torch_tensor in ms_tensor_2_torch_tensor_mapping.items():
            real_torch_tensor = ComputeElement.transfer_to_torch_tensor(ms_tensor)
            self.assertTrue((real_torch_tensor == torch_tensor).all())
            self.assertEqual(real_torch_tensor.dtype, torch_tensor.dtype)

    def test_transfer_to_mindspore_tensor(self):
        ms_tensor_2_torch_tensor_mapping = {
            mindspore.Tensor([1, 2, 3], dtype=mindspore.uint8): torch.tensor([1, 2, 3], dtype=torch.uint8),
            mindspore.Tensor([1, 2, 3], dtype=mindspore.float32): torch.tensor([1, 2, 3], dtype=torch.float32)
        }
        for ms_tensor, torch_tensor in ms_tensor_2_torch_tensor_mapping.items():
            real_ms_tensor = ComputeElement.transfer_to_mindspore_tensor(torch_tensor)
            self.assertTrue((real_ms_tensor == ms_tensor).all())
            self.assertEqual(real_ms_tensor.dtype, ms_tensor.dtype)


if __name__ == '__main__':
    unittest.main()
