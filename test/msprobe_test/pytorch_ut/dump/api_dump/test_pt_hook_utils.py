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

        self._mock_listdir_patcher = patch('msprobe.pytorch.dump.api_dump.utils.os.listdir')
        self.mock_listdir = self._mock_listdir_patcher.start()

    def tearDown(self):
        self._mock_listdir_patcher.stop()

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

    def test_get_ops_with_missing_category(self):
        # 用户按文档修改 support_wrap_ops.yaml 后只保留部分分类，缺失的分类按空处理，不应抛异常
        yaml_content = {
            'functional': ['func1', 'func2'],
            'tensor': ['tensor_op1'],
            'torch': ['torch_op1', 'torch_op2']
        }
        with patch('msprobe.pytorch.dump.api_dump.utils.load_yaml') as mock_load:
            mock_load.return_value = yaml_content
            result = get_ops()
        self.assertEqual(result, {'func1', 'func2', 'tensor_op1', 'torch_op1', 'torch_op2'})

    def test_get_ops_with_missing_category_warns(self):
        # 缺失的分类按空处理时需打出 WARNING 日志，方便用户定位
        yaml_content = {
            'functional': ['func1'],
            'tensor': ['tensor_op1'],
            'torch': ['torch_op1']
        }
        with patch('msprobe.pytorch.dump.api_dump.utils.load_yaml') as mock_load:
            mock_load.return_value = yaml_content
            with patch('msprobe.pytorch.dump.api_dump.utils.logger.warning') as mock_warning:
                result = get_ops()
        self.assertEqual(result, {'func1', 'tensor_op1', 'torch_op1'})
        mock_warning.assert_called_once_with(
            "[msprobe] support_wrap_ops.yaml has no 'torch_npu' category, treated as empty"
        )

    def test_get_ops_with_none_category(self):
        # 分类存在但无条目（如 "functional:" 后为空）时 YAML 解析为 None，按空处理
        yaml_content = {
            'functional': None,
            'tensor': ['tensor_op1'],
            'torch': ['torch_op1', 'torch_op2'],
            'torch_npu': ['npu_op1']
        }
        with patch('msprobe.pytorch.dump.api_dump.utils.load_yaml') as mock_load:
            mock_load.return_value = yaml_content
            result = get_ops()
        self.assertEqual(result, {'tensor_op1', 'torch_op1', 'torch_op2', 'npu_op1'})

    def test_get_ops_with_non_list_category(self):
        # 分类值不是列表（误写成标量）时忽略该分类，避免按字符拆分
        yaml_content = {
            'functional': 'conv1d',
            'tensor': ['tensor_op1'],
            'torch': ['torch_op1', 'torch_op2'],
            'torch_npu': ['npu_op1']
        }
        with patch('msprobe.pytorch.dump.api_dump.utils.load_yaml') as mock_load:
            mock_load.return_value = yaml_content
            result = get_ops()
        self.assertEqual(result, {'tensor_op1', 'torch_op1', 'torch_op2', 'npu_op1'})

    def test_get_ops_with_non_list_category_warns(self):
        # 分类值不是列表时忽略该分类，并打出 WARNING 日志
        yaml_content = {
            'functional': 'conv1d',
            'tensor': ['tensor_op1'],
            'torch': ['torch_op1', 'torch_op2'],
            'torch_npu': ['npu_op1']
        }
        with patch('msprobe.pytorch.dump.api_dump.utils.load_yaml') as mock_load:
            mock_load.return_value = yaml_content
            with patch('msprobe.pytorch.dump.api_dump.utils.logger.warning') as mock_warning:
                result = get_ops()
        self.assertEqual(result, {'tensor_op1', 'torch_op1', 'torch_op2', 'npu_op1'})
        mock_warning.assert_called_once_with(
            "[msprobe] 'functional' in support_wrap_ops.yaml is not a list, ignored"
        )

    def test_get_ops_with_empty_yaml(self):
        # 整个 yaml 文件为空时 safe_load 返回 None，返回空集合而非抛异常
        with patch('msprobe.pytorch.dump.api_dump.utils.load_yaml') as mock_load:
            mock_load.return_value = None
            result = get_ops()
        self.assertEqual(result, set())


    @patch('msprobe.pytorch.dump.api_dump.utils.inspect')
    def test_dynamic_import_op_success(self, mock_inspect):
        def mock_func(x):
            return x

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
        with patch('msprobe.pytorch.dump.api_dump.utils.logger.warning') as mock_logger:
            with patch('importlib.import_module') as mock_import:
                mock_import.side_effect = ImportError("Fake error")
                ops = dynamic_import_op(MockPackage(), white_list=['fail.py'])
                self.assertEqual(ops, {})
                mock_logger.assert_called_once_with("import mock_package.fail failed!")
