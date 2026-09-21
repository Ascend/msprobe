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
from unittest.mock import MagicMock

from msprobe.mindspore.api_accuracy_checker.api_runner import api_runner, ApiInputAggregation
from msprobe.core.common.const import Const

logging.basicConfig(stream = sys.stdout, level = logging.INFO, format = '[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

file_path = os.path.abspath(__file__)
directory = os.path.dirname(file_path)

def func(x_1, x_2, opt="opt1"):
    y_1 = x_1*2 + 1
    if opt == "opt1":
        y_2 = x_1 + x_2
    else:
        y_2 = x_1*2 + x_2
    return y_1, y_2

def side_effect_forward_input_1(**kwargs):
    if kwargs.get("tensor_platform") == Const.MS_FRAMEWORK:
        return mindspore.Tensor([1., 2., 3.])
    else:
        return torch.Tensor([1., 2., 3.])

def side_effect_forward_input_2(**kwargs):
    if kwargs.get("tensor_platform") == Const.MS_FRAMEWORK:
        return mindspore.Tensor([1.1, 2., 3.])
    else:
        return torch.Tensor([1.1, 2., 3.])

def side_effect_forward_output_1(**kwargs):
    if kwargs.get("tensor_platform") == Const.MS_FRAMEWORK:
        return mindspore.Tensor([3., 5., 7.])
    else:
        return torch.Tensor([3., 5., 7.])

def side_effect_forward_output_2(**kwargs):
    if kwargs.get("tensor_platform") == Const.MS_FRAMEWORK:
        return mindspore.Tensor([2.1, 4., 6.])
    else:
        return torch.Tensor([2.1, 4., 6.])

def side_effect_backward_input_1(**kwargs):
    if kwargs.get("tensor_platform") == Const.MS_FRAMEWORK:
        return mindspore.Tensor([1., 2., 3.])
    else:
        return torch.Tensor([1., 2., 3.])

def side_effect_backward_input_2(**kwargs):
    if kwargs.get("tensor_platform") == Const.MS_FRAMEWORK:
        return mindspore.Tensor([1.11, 2., 3.])
    else:
        return torch.Tensor([1.11, 2., 3.])

def side_effect_backward_output_1(**kwargs):
    if kwargs.get("tensor_platform") == Const.MS_FRAMEWORK:
        return mindspore.Tensor([3.11, 6., 9.])
    else:
        return torch.Tensor([3.11, 6., 9.])

def side_effect_backward_output_2(**kwargs):
    if kwargs.get("tensor_platform") == Const.MS_FRAMEWORK:
        return mindspore.Tensor([1.11, 2., 3.])
    else:
        return torch.Tensor([1.11, 2., 3.])


class TestApiRunner(unittest.TestCase):

    def setUp(self):
        self.mock_compute_element_kwargs_instance = MagicMock()
        self.mock_compute_element_kwargs_instance.get_parameter.return_value = "opt1"

        self.mock_compute_element_forward_input_1_instance = MagicMock()
        self.mock_compute_element_forward_input_1_instance.get_parameter.side_effect = side_effect_forward_input_1

        self.mock_compute_element_forward_input_2_instance = MagicMock()
        self.mock_compute_element_forward_input_2_instance.get_parameter.side_effect = side_effect_forward_input_2

        self.mock_compute_element_backward_input_1_instance = MagicMock()
        self.mock_compute_element_backward_input_1_instance.get_parameter.side_effect = side_effect_backward_input_1

        self.mock_compute_element_backward_input_2_instance = MagicMock()
        self.mock_compute_element_backward_input_2_instance.get_parameter.side_effect = side_effect_backward_input_2

        self.mock_compute_element_forward_output_1_instance = MagicMock()
        self.mock_compute_element_forward_output_1_instance.get_parameter.side_effect = side_effect_forward_output_1

        self.mock_compute_element_forward_output_2_instance = MagicMock()
        self.mock_compute_element_forward_output_2_instance.get_parameter.side_effect = side_effect_forward_output_2

        self.mock_compute_element_backward_output_1_instance = MagicMock()
        self.mock_compute_element_backward_output_1_instance.get_parameter.side_effect = side_effect_backward_output_1

        self.mock_compute_element_backward_output_2_instance = MagicMock()
        self.mock_compute_element_backward_output_2_instance.get_parameter.side_effect = side_effect_backward_output_2

    def test_run_api(self):
        kwargs = {"opt": self.mock_compute_element_kwargs_instance}
        inputs = [self.mock_compute_element_forward_input_1_instance, self.mock_compute_element_forward_input_2_instance]
        gradient_inputs = [self.mock_compute_element_backward_input_1_instance, self.mock_compute_element_backward_input_2_instance]
        forward_result = [self.mock_compute_element_forward_output_1_instance, self.mock_compute_element_forward_output_2_instance]
        backward_result = [self.mock_compute_element_backward_output_1_instance, self.mock_compute_element_backward_output_2_instance]

        forward_api_input_aggregation = ApiInputAggregation(inputs, kwargs, None)
        backward_api_input_aggregation = ApiInputAggregation(inputs, {}, gradient_inputs)

        # api_instance, api_input_aggregation, forward_or_backward, api_platform, result
        test_cases = [
            [func, forward_api_input_aggregation, Const.FORWARD, Const.MS_FRAMEWORK, forward_result],
            [func, backward_api_input_aggregation, Const.BACKWARD, Const.MS_FRAMEWORK, backward_result],
            [func, forward_api_input_aggregation, Const.FORWARD, Const.PT_FRAMEWORK, forward_result],
            [func, backward_api_input_aggregation, Const.BACKWARD, Const.PT_FRAMEWORK, backward_result],
        ]
        for test_case in test_cases:
            api_instance, api_input_aggregation, forward_or_backward, api_platform, results_target = test_case
            output = api_runner.run_api(api_instance, api_input_aggregation, forward_or_backward,
                                                    api_platform)

            # 如果返回的是 tuple，就拿第一个元素；否则直接当 list 用
            if isinstance(output, tuple):
                results_real = output[0]
            else:
                results_real = output
            # 下面跟原来测试逻辑一模一样
            for res_real, res_target in zip(results_real, results_target):
                assert (abs(
                    res_real.get_parameter()
                    - res_target.get_parameter(tensor_platform=api_platform)
                ) < 1e-5).all()

    def test_get_api_instance(self):
        #api_type_str, api_sub_name, api_platform, result_api
        test_cases = [
            ["MintFunctional", "relu", Const.MS_FRAMEWORK, mindspore.mint.nn.functional.relu],
            ["MintFunctional", "relu", Const.PT_FRAMEWORK, torch.nn.functional.relu]
        ]
        for test_case in test_cases:
            api_type_str, api_sub_name, api_platform, result_api = test_case
            assert api_runner.get_api_instance(api_type_str, api_sub_name, api_platform) == result_api

    def test_get_info_from_name(self):
        api_name = "MintFunctional.relu.0"
        api_type_str, api_sub_name = api_runner.get_info_from_name(api_name_str=api_name)
        assert api_type_str == "MintFunctional"
        assert api_sub_name == "relu"
