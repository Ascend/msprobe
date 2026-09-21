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

import os
import shutil
import threading
import unittest
from unittest import mock
from unittest.mock import patch

import numpy as np

from msprobe.pytorch.compare import utils
from msprobe.pytorch.compare.utils import read_pt_data
from msprobe.core.common.utils import CompareException
from ...core_ut.compare.test_acc_compare import generate_pt


base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'test_pt_compare_utils_data')
pt_dir = os.path.join(base_dir, f'dump_data_dir')


class TestReadPtData(unittest.TestCase):

    def setUp(self):
        os.makedirs(base_dir, mode=0o750, exist_ok=True)
        os.makedirs(pt_dir, mode=0o750, exist_ok=True)

        self.lock = threading.Lock()

    def tearDown(self):
        if os.path.exists(pt_dir):
            shutil.rmtree(pt_dir)
        if os.path.exists(base_dir):
            shutil.rmtree(base_dir)

    def test_read_pt_data_normal(self):
        generate_pt(pt_dir)
        result = read_pt_data(pt_dir, 'Functional.linear.0.forward.input.0.pt')
        expected = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertTrue(np.array_equal(result, expected))

    def test_read_pt_data_no_file_name(self):
        result = read_pt_data(pt_dir, None)
        self.assertEqual(result, None)

    @patch.object(utils, 'load_pt')
    @patch.object(utils, 'FileChecker')
    def test_read_pt_data_runtime_error(self, mock_file_checker_class, mock_load_pt):
        mock_file_checker = mock.Mock()
        mock_file_checker.common_check.return_value = 'fake/path/file.pt'
        mock_file_checker_class.return_value = mock_file_checker

        mock_load_pt.side_effect = RuntimeError('failed to load')

        with self.assertRaises(CompareException) as context:
            read_pt_data('fake/path', 'file.pt')
        self.assertEqual(context.exception.code, CompareException.INVALID_FILE_ERROR)

    @patch.object(utils, 'load_pt')
    @patch.object(utils, 'FileChecker')
    def test_read_pt_data_attribute_error(self, mock_file_checker_class, mock_load_pt):
        mock_file_checker = mock.Mock()
        mock_file_checker.common_check.return_value = 'fake/path/file.pt'
        mock_file_checker_class.return_value = mock_file_checker

        class FakeTensor:
            def detach(self):
                raise AttributeError('no detach')

        mock_load_pt.return_value = FakeTensor()

        with self.assertRaises(CompareException) as context:
            read_pt_data('fake/path', 'file.pt')
        self.assertEqual(context.exception.code, CompareException.DETACH_ERROR)
