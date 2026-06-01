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

from msprobe.pytorch.dump.api_dump.utils import get_ops, dynamic_import_op


class MockPackage:
    __name__ = "mock_package"
    __file__ = "/fake_path/__init__.py"


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.yaml_content = {
            'functional': ['func1', 'func2'],
            'tensor': ['tensor_op1'],
            'torch': ['torch_op1', 'torch_op2'],
            'torch_npu': ['npu_op1']
        }

        self.mock_listdir = patch('msprobe.pytorch.dump.api_dump.utils.os.listdir').start()

    def tearDown(self):
        patch.stopall()

    def test_get_ops(self):
        with patch('msprobe.pytorch.dump.api_dump.utils.load_yaml') as mock_load:
            mock_load.return_value = self.yaml_content
            result = get_ops()
            self.assertEqual(
                result,
                {
                    'func1',
                    'func2',
                    'tensor_op1',
                    'torch_op1',
                    'torch_op2',
                    'npu_op1'
                }
            )

    @patch('msprobe.pytorch.dump.api_dump.utils.inspect')
    def test_dynamic_import_op_success(self, mock_inspect):
        mock_func = lambda x: x
        mock_inspect.getmembers = MagicMock()
        mock_inspect.getmembers.return_value = [['test_func', mock_func]]

        self.mock_listdir.return_value = ['valid.py', 'invalid.py']
        mock_module = MagicMock()

        with patch('importlib.import_module', return_value=mock_module) as mock_import:
            ops = dynamic_import_op(MockPackage(), white_list=['valid.py'])
            self.assertEqual(ops, {'valid.test_func': mock_func})
            mock_import.assert_called_once_with('mock_package.valid')

    def test_dynamic_import_op_failure(self):
        self.mock_listdir.return_value = ['fail.py']
        with patch('importlib.import_module') as mock_import:
            mock_import.side_effect = ImportError("Fake error")
            with patch('msprobe.pytorch.dump.api_dump.utils.logger.warning') as mock_logger:
                ops = dynamic_import_op(MockPackage(), white_list=['fail.py'])
                self.assertEqual(ops, {})
                mock_logger.assert_called_once_with("import mock_package.fail failed!")