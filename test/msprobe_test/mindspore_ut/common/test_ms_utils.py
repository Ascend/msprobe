#!/usr/bin/env python3
# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
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
from unittest.mock import MagicMock, patch, call
import numpy as np
import mindspore as ms
import os
import random

from msprobe.core.common.exceptions import DistributedNotInitializedError
from msprobe.mindspore.common.utils import (get_rank_if_initialized,
    convert_bf16_to_fp32,
    save_tensor_as_npy,
    convert_to_int,
    list_lowest_level_directories,
    seed_all,
    remove_dropout,
    MsprobeStep)

class MockCell:
    def __init__(self):
        self.mindstudio_reserved_name = None


class TestMsprobeStep(unittest.TestCase):
    def setUp(self):
        class Debugger:
            def __init__(self):
                self.start_called = False
                self.stop_called = False
                self.step_called = False
                self.stop_called_first = False

            def start(self):
                self.start_called = True

            def stop(self):
                self.stop_called = True

            def step(self):
                if self.stop_called:
                    self.stop_called_first = True
                self.step_called = True

        debugger = Debugger()
        self.msprobe_step = MsprobeStep(debugger)

    def test_on_train_step_begin(self):
        self.msprobe_step.on_train_step_begin("run_context")
        self.assertTrue(self.msprobe_step.debugger.start_called)
        self.assertFalse(self.msprobe_step.debugger.stop_called)
        self.assertFalse(self.msprobe_step.debugger.step_called)

    def test_on_train_step_end(self):
        self.msprobe_step.on_train_step_end("run_context")
        self.assertFalse(self.msprobe_step.debugger.start_called)
        self.assertTrue(self.msprobe_step.debugger.stop_called)
        self.assertTrue(self.msprobe_step.debugger.step_called)
        self.assertTrue(self.msprobe_step.debugger.stop_called_first)


class TestMsprobeFunctions(unittest.TestCase):

    @patch('mindspore.communication.GlobalComm.INITED', True)
    @patch('mindspore.communication.get_rank', return_value=0)
    def test_get_rank_if_initialized(self, mock_get_rank):
        rank = get_rank_if_initialized()
        self.assertEqual(rank, 0)
        mock_get_rank.assert_called_once()

    def test_convert_bf16_to_fp32(self):
        original_tensor = ms.Tensor(np.array([1.5, 2.5, 3.5]), dtype=ms.bfloat16)
        converted_tensor = convert_bf16_to_fp32(original_tensor)
        self.assertEqual(converted_tensor.dtype, ms.float32)
        np.testing.assert_array_almost_equal(
            converted_tensor.asnumpy(), np.array([1.5, 2.5, 3.5], dtype=np.float32))

    def test_convert_to_int(self):
        self.assertEqual(convert_to_int("123"), 123)
        self.assertEqual(convert_to_int("abc"), -1)

    @patch('os.listdir', return_value=['dir1', 'dir2'])
    @patch('os.path.isdir', side_effect=lambda x: x in ['root/dir1', 'root/dir2'])
    @patch('os.path.exists', side_effect=lambda x: x == 'root')
    @patch('msprobe.core.common.file_utils.check_path_exists')
    def test_list_lowest_level_directories(self, mock_check_exists, mock_exists, mock_isdir, mock_listdir):
        mock_check_exists.return_value = None

        # 执行函数并验证结果
        lowest_dirs = list_lowest_level_directories('root')
        self.assertEqual(lowest_dirs, ['root/dir1', 'root/dir2'])

    @patch('os.environ', new_callable=dict)
    @patch('mindspore.set_seed')
    @patch('random.seed')
    @patch('mindspore.set_context')
    @patch('msprobe.mindspore.common.utils.check_seed_all')
    def test_seed_all(self, mock_check_seed_all, mock_set_context, mock_random_seed, mock_set_seed, mock_environ):
        seed_all(42, True)

        # 验证 check_seed_all 的调用
        mock_check_seed_all.assert_called_once_with(42, True, False)
        # 验证环境变量是否设置正确
        self.assertEqual(mock_environ.get('PYTHONHASHSEED'), '42')
        # 验证其他函数是否正确调用
        mock_set_seed.assert_called_once_with(42)
        mock_random_seed.assert_called_once_with(42)
        mock_set_context.assert_called_once_with(deterministic="ON")

    def test_remove_dropout(self):
        remove_dropout()
        from mindspore import Tensor
        x1d = Tensor(np.ones([5, 5]), ms.float32)
        x2d = Tensor(np.ones([5, 5, 5, 5]), ms.float32)
        x3d = Tensor(np.ones([5, 5, 5, 5, 5]), ms.float32)
        from mindspore.ops import Dropout, Dropout2D, Dropout3D
        self.assertTrue((Dropout(0.5)(x1d)[0].numpy() == x1d.numpy()).all())
        self.assertTrue((Dropout2D(0.5)(x2d)[0].numpy() == x2d.numpy()).all())
        self.assertTrue((Dropout3D(0.5)(x3d)[0].numpy() == x3d.numpy()).all())

        from mindspore.mint.nn import Dropout
        from mindspore.mint.nn.functional import dropout
        self.assertTrue((Dropout(0.5)(x1d).numpy() == x1d.numpy()).all())
        self.assertTrue((dropout(x1d, p=0.5).numpy() == x1d.numpy()).all())



if __name__ == "__main__":
    unittest.main()