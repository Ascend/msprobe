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

# coding=utf-8
import unittest
from unittest.mock import patch

from msprobe.core.compare.check import check_dump_json_str, check_json_key_value, valid_key_value, \
    check_stack_json_str, check_consistent_param
from msprobe.core.common.utils import CompareException
from msprobe.core.common.const import Const

# test_check_struct_match
npu_dict = {'op_name': ['Functional.conv2d.0.forward.input.0', 'Functional.conv2d.0.forward.input.1',
                        'Functional.conv2d.0.forward.input.2', 'Functional.conv2d.0.forward.output'],
           'input_struct': [('torch.float32', [1, 1, 28, 28]), ('torch.float32', [16, 1, 5, 5]),
                             ('torch.float32', [16])],
            'output_struct': [('torch.float32', [1, 16, 28, 28])],
            'summary': [[3.029174327850342, -2.926689624786377, -0.06619918346405029],
                        [0.19919930398464203, -0.19974489510059357, 0.006269412115216255],
                        [0.19734230637550354, -0.18177609145641327, 0.007903944700956345],
                        [2.1166646480560303, -2.190781354904175, -0.003579073818400502]], 'stack_info': []}

bench_dict = {'op_name': ['Functional.conv2d.0.forward.input.0', 'Functional.conv2d.0.forward.input.1',
                          'Functional.conv2d.0.forward.input.2', 'Functional.conv2d.0.forward.output'],
             'input_struct': [('torch.float32', [1, 1, 28, 28]), ('torch.float32', [16, 1, 5, 5]),
                               ('torch.float32', [16])],
              'output_struct': [('torch.float32', [1, 16, 28, 28])],
              'summary': [[3.029174327850342, -2.926689624786377, -0.06619918346405029],
                          [0.19919930398464203, -0.19974489510059357, 0.006269412115216255],
                          [0.19734230637550354, -0.18177609145641327, 0.007903944700956345],
                          [2.1166646480560303, -2.190781354904175, -0.003579073818400502]], 'stack_info': []}


# test_check_type_shape_match
npu_struct = [('torch.float32', [1, 1, 28, 28]), ('torch.float32', [16, 1, 5, 5]), ('torch.float32', [16])]
bench_struct = [('torch.float32', [1, 1, 28, 28]), ('torch.float32', [16, 1, 5, 5]), ('torch.float32', [16])]


# test_check_dump_json_str
op_data = {'input_args': [
    {
        'type': '@torch.Tensor',
        'dtype': 'torch.float32',
        'shape': [2, 2],
        'Max': 0.5,
        'Min': -0.7,
        'Mean': 0.2,
        'Norm': 0.9,
        'requires_grad': True,
        'data_name': 'Function.linear.0.forward.output.0.pt'
    }
]}


# test_check_json_key_value
input_output = {
    'type': '@torch.Tensor',
    'dtype': 'torch.float32',
    'shape': [2, 2],
    'Max': 0.5,
    'Min': -0.7,
    'Mean': 0.2,
    'Norm': 0.9,
    'requires_grad': True,
    'data_name': 'Function.linear.0.forward.output.0.pt'}


op_name = 'Functional.conv2d.0.backward.input.0'


class TestUtilsMethods(unittest.TestCase):
    def test_check_dump_json_str(self):
        with self.assertRaises(CompareException) as context:
            check_dump_json_str(op_data, op_name)
        self.assertEqual(context.exception.code, CompareException.INVALID_CHAR_ERROR)

    def test_check_json_key_value(self):
        with self.assertRaises(CompareException) as context:
            check_json_key_value(input_output, op_name)
        self.assertEqual(context.exception.code, CompareException.INVALID_CHAR_ERROR)

    def test_check_json_key_value_max_depth(self):
        result = check_json_key_value(input_output, op_name, depth=401)
        self.assertEqual(result, None)

    def test_valid_key_value_type_shape(self):
        key = 'shape'
        value = 'abc'
        with self.assertRaises(CompareException) as context:
            valid_key_value(key, value, op_name)
        self.assertEqual(context.exception.code, CompareException.INVALID_OBJECT_TYPE_ERROR)

    def test_valid_key_value_type_requires_grad(self):
        key = 'requires_grad'
        value = 'abc'
        with self.assertRaises(CompareException) as context:
            valid_key_value(key, value, op_name)
        self.assertEqual(context.exception.code, CompareException.INVALID_OBJECT_TYPE_ERROR)

    def test_check_stack_json_str_type_stack_info(self):
        stack_info = 'File'
        with self.assertRaises(CompareException) as context:
            check_stack_json_str(stack_info, op_name)
        self.assertEqual(context.exception.code, CompareException.INVALID_OBJECT_TYPE_ERROR)

    def test_check_stack_json_str_2(self):
        stack_info = ['=File', 'File']
        with self.assertRaises(CompareException) as context:
            check_stack_json_str(stack_info, op_name)
        self.assertEqual(context.exception.code, CompareException.INVALID_CHAR_ERROR)

    def test_check_stack_json_str_3(self):
        stack_info = ['+File', 'File']
        with self.assertRaises(CompareException) as context:
            check_stack_json_str(stack_info, op_name)
        self.assertEqual(context.exception.code, CompareException.INVALID_CHAR_ERROR)

    def test_check_stack_json_str_4(self):
        stack_info = ['-File', 'File']
        with self.assertRaises(CompareException) as context:
            check_stack_json_str(stack_info, op_name)
        self.assertEqual(context.exception.code, CompareException.INVALID_CHAR_ERROR)

    def test_check_stack_json_str_5(self):
        stack_info = ['@File', 'File']
        with self.assertRaises(CompareException) as context:
            check_stack_json_str(stack_info, op_name)
        self.assertEqual(context.exception.code, CompareException.INVALID_CHAR_ERROR)


class TestCheckConsistentParam(unittest.TestCase):

    @patch('msprobe.core.compare.check.logger')
    def test_consistent_check_false_skip(self, mock_logger):
        """
        场景 1: consistent_check 为 False
        预期：无论 backend 是什么，都不应抛出异常，也不应记录错误日志
        """
        # 执行
        check_consistent_param(consistent_check=False, backend="ANY_BACKEND")

        # 断言：没有抛出异常 (隐式)
        # 断言：logger.error 没有被调用
        mock_logger.error.assert_not_called()

    @patch('msprobe.core.compare.check.logger')
    def test_consistent_check_true_valid_fsdp(self, mock_logger):
        """
        场景 2: consistent_check 为 True, backend 为 FSDP (合法)
        预期：不抛出异常，不记录错误日志
        """
        # 执行
        check_consistent_param(consistent_check=True, backend=Const.FSDP)

        # 断言
        mock_logger.error.assert_not_called()

    @patch('msprobe.core.compare.check.logger')
    def test_consistent_check_true_valid_megatron(self, mock_logger):
        """
        场景 3: consistent_check 为 True, backend 为 MEGATRON (合法)
        预期：不抛出异常，不记录错误日志
        """
        # 执行
        check_consistent_param(consistent_check=True, backend=Const.MEGATRON)

        # 断言
        mock_logger.error.assert_not_called()

    @patch('msprobe.core.compare.check.logger')
    def test_consistent_check_true_invalid_backend(self, mock_logger):
        """
        场景 4: consistent_check 为 True, backend 非法
        预期：抛出 CompareException，且 logger.error 被调用
        """
        invalid_backend = "other"

        # 断言：抛出特定异常
        with self.assertRaises(CompareException) as context:
            check_consistent_param(consistent_check=True, backend=invalid_backend)

        # 断言：异常中的错误码正确
        self.assertEqual(context.exception.code, CompareException.INVALID_PARAM_ERROR)

        # 断言：logger.error 被调用了一次
        mock_logger.error.assert_called_once()

        # 断言：日志内容包含关键信息 (可选，确保日志内容符合预期)
        call_args = mock_logger.error.call_args[0][0]
        self.assertIn("consistent_check", call_args)
        self.assertIn(invalid_backend, call_args)


if __name__ == '__main__':
    unittest.main()
