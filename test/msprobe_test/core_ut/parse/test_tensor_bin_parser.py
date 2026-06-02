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
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock

from msprobe.core.parse.tensor_bin_parser import TensorBinFileParser


class TestTensorBinFileParser(unittest.TestCase):
    """测试 TensorBinFileParser 类"""

    def setUp(self):
        """创建解析器实例和临时目录"""
        self.parser = TensorBinFileParser()
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """清理临时目录"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch('msprobe.core.parse.tensor_bin_parser.torch_available', False)
    def test_parse_file_without_torch(self):
        """测试没有 torch 时解析文件会报错"""
        with self.assertRaises(RuntimeError) as context:
            self.parser.parse('/path/to/file.bin', '/output', 'npy')

        self.assertIn('torch is required', str(context.exception))

    @patch('msprobe.core.parse.tensor_bin_parser.torch_available', True)
    @patch('msprobe.core.parse.tensor_bin_parser.os.path.isfile')
    @patch.object(TensorBinFileParser, '_parse_single_bin_file')
    def test_parse_single_file(self, mock_parse_single, mock_isfile):
        """测试解析单个文件"""
        mock_isfile.return_value = True

        self.parser.parse('/path/to/file.bin', '/output', 'npy')

        mock_parse_single.assert_called_once_with('/path/to/file.bin', '/output', 'npy')

    @patch('msprobe.core.parse.tensor_bin_parser.torch_available', True)
    @patch('msprobe.core.parse.tensor_bin_parser.os.path.isdir')
    @patch('msprobe.core.parse.tensor_bin_parser.os.path.isfile')
    @patch.object(TensorBinFileParser, '_find_bin_files')
    @patch.object(TensorBinFileParser, '_parse_single_bin_file')
    def test_parse_directory(self, mock_parse_single, mock_find_bin, mock_isfile, mock_isdir):
        """测试解析目录"""
        mock_isfile.return_value = False
        mock_isdir.return_value = True
        mock_find_bin.return_value = ['/path/to/file1.bin', '/path/to/file2.bin']

        with patch('msprobe.core.parse.tensor_bin_parser.logger') as mock_logger:
            self.parser.parse('/path/to/dir', '/output', 'npy')

        mock_find_bin.assert_called_once_with('/path/to/dir')
        self.assertEqual(mock_parse_single.call_count, 2)
        mock_parse_single.assert_any_call('/path/to/file1.bin', '/output', 'npy')
        mock_parse_single.assert_any_call('/path/to/file2.bin', '/output', 'npy')

    def test_find_bin_files(self):
        """测试查找目录中的 .bin 文件"""
        # 创建测试文件
        file1 = os.path.join(self.test_dir, 'file1.bin')
        file2 = os.path.join(self.test_dir, 'file2.txt')
        file3 = os.path.join(self.test_dir, 'file3.BIN')  # 测试大小写不敏感

        with open(file1, 'w') as f:
            f.write('test')
        with open(file2, 'w') as f:
            f.write('test')
        with open(file3, 'w') as f:
            f.write('test')

        with patch('msprobe.core.parse.tensor_bin_parser.check_file_or_directory_path'):
            result = self.parser._find_bin_files(self.test_dir)

        # 应该找到所有 .bin 文件（大小写不敏感）
        self.assertEqual(len(result), 2)
        self.assertIn(file1, result)
        self.assertIn(file3, result)
        self.assertNotIn(file2, result)

    def test_find_bin_files_empty_directory(self):
        """测试空目录返回空列表"""
        with patch('msprobe.core.parse.tensor_bin_parser.check_file_or_directory_path'):
            result = self.parser._find_bin_files(self.test_dir)
        self.assertEqual(result, [])

    def test_find_bin_files_with_subdir(self):
        """测试只查找第一层目录，不递归"""
        # 创建第一层文件
        file1 = os.path.join(self.test_dir, 'file1.bin')
        with open(file1, 'w') as f:
            f.write('test')

        # 创建子目录和文件
        subdir = os.path.join(self.test_dir, 'subdir')
        os.makedirs(subdir)
        file2 = os.path.join(subdir, 'file2.bin')
        with open(file2, 'w') as f:
            f.write('test')

        with patch('msprobe.core.parse.tensor_bin_parser.check_file_or_directory_path'):
            result = self.parser._find_bin_files(self.test_dir)

        # 应该只找到第一层的文件
        self.assertEqual(len(result), 1)
        self.assertIn(file1, result)
        self.assertNotIn(file2, result)

    @patch('msprobe.core.parse.tensor_bin_parser.torch_available', True)
    @patch('msprobe.core.parse.tensor_bin_parser.TensorBinFile')
    @patch('msprobe.core.parse.tensor_bin_parser.save_npy')
    @patch('msprobe.core.parse.tensor_bin_parser.BaseParser.get_output_file_path')
    def test_parse_single_bin_file_npy(self, mock_get_path, mock_save_npy, mock_tensor_bin):
        """测试解析单个 .bin 文件为 npy 格式"""
        # 准备 mock 对象
        mock_tensor = MagicMock()
        mock_tensor.dtype = MagicMock()
        mock_tensor.numpy.return_value = MagicMock()
        mock_tensor_bin_instance = MagicMock()
        mock_tensor_bin_instance.get_data.return_value = mock_tensor
        mock_tensor_bin_instance.is_valid = True
        mock_tensor_bin.return_value = mock_tensor_bin_instance
        mock_get_path.return_value = '/output/file.npy'

        self.parser._parse_single_bin_file('/path/to/file.bin', '/output', 'npy')

        mock_tensor_bin.assert_called_once_with('/path/to/file.bin')
        mock_get_path.assert_called_once_with('/path/to/file.bin', '/output', 'npy')
        mock_save_npy.assert_called_once()
        
    @patch('msprobe.core.parse.tensor_bin_parser.torch_available', True)
    @patch('msprobe.core.parse.tensor_bin_parser.TensorBinFile')
    def test_parse_single_bin_file_none_data(self, mock_tensor_bin):
        """测试 get_data() 返回 None 时抛出 AttributeError"""
        mock_tensor_bin_instance = MagicMock()
        mock_tensor_bin_instance.is_valid = True
        mock_tensor_bin_instance.get_data.return_value = None
        mock_tensor_bin.return_value = mock_tensor_bin_instance

        with self.assertRaises(AttributeError):
            self.parser._parse_single_bin_file('/path/to/file.bin', '/output', 'npy')

    @patch('msprobe.core.parse.tensor_bin_parser.torch_available', True)
    @patch('msprobe.core.parse.tensor_bin_parser.TensorBinFile')
    @patch('msprobe.core.parse.tensor_bin_parser.torch')
    @patch('msprobe.core.parse.tensor_bin_parser.logger')
    def test_parse_single_bin_file_bfloat16_to_float(self, mock_logger, mock_torch, mock_tensor_bin):
        """测试 bfloat16 类型转换为 float"""
        mock_tensor = MagicMock()
        mock_tensor.dtype = mock_torch.bfloat16
        mock_tensor.float.return_value = MagicMock()
        mock_tensor.numpy.return_value = MagicMock()

        mock_tensor_bin_instance = MagicMock()
        mock_tensor_bin_instance.get_data.return_value = mock_tensor
        mock_tensor_bin_instance.is_valid = True
        mock_tensor_bin.return_value = mock_tensor_bin_instance

        with patch('msprobe.core.parse.tensor_bin_parser.save_npy'), \
             patch('msprobe.core.parse.tensor_bin_parser.BaseParser.get_output_file_path'):
            self.parser._parse_single_bin_file('/path/to/file.bin', '/output', 'npy')

        # 验证 bfloat16 被转换为 float
        mock_tensor.float.assert_called_once()
        # 验证日志被打印
        mock_logger.info.assert_any_call("Converting bfloat16 tensor to float32 for numpy save")

    @patch('msprobe.core.parse.tensor_bin_parser.torch_available', True)
    @patch('msprobe.core.parse.tensor_bin_parser.TensorBinFile')
    @patch.object(TensorBinFileParser, '_save_tensor_to_pt')
    @patch('msprobe.core.parse.tensor_bin_parser.BaseParser.get_output_file_path')
    def test_parse_single_bin_file_pt(self, mock_get_path, mock_save_pt, mock_tensor_bin):
        """测试解析单个 .bin 文件为 pt 格式"""
        mock_tensor = MagicMock()
        mock_tensor_bin_instance = MagicMock()
        mock_tensor_bin_instance.get_data.return_value = mock_tensor
        mock_tensor_bin_instance.is_valid = True
        mock_tensor_bin.return_value = mock_tensor_bin_instance
        mock_get_path.return_value = '/output/file.pt'

        self.parser._parse_single_bin_file('/path/to/file.bin', '/output', 'pt')

        mock_get_path.assert_called_once_with('/path/to/file.bin', '/output', 'pt')
        mock_save_pt.assert_called_once_with(mock_tensor, '/output/file.pt')

    @patch('msprobe.core.parse.tensor_bin_parser.torch_available', True)
    @patch('msprobe.core.parse.tensor_bin_parser.TensorBinFile')
    def test_parse_single_bin_file_invalid(self, mock_tensor_bin):
        """测试无效的 tensor 数据"""
        mock_tensor_bin_instance = MagicMock()
        mock_tensor_bin_instance.is_valid = False
        mock_tensor_bin.return_value = mock_tensor_bin_instance

        # 不应该抛出异常，只是返回
        self.parser._parse_single_bin_file('/path/to/file.bin', '/output', 'npy')

    @patch('msprobe.core.parse.tensor_bin_parser.torch_available', True)
    @patch('msprobe.core.parse.tensor_bin_parser.torch.save')
    @patch('msprobe.core.parse.tensor_bin_parser.check_path_before_create')
    @patch('msprobe.core.parse.tensor_bin_parser.os.path.realpath')
    def test_save_tensor_to_pt(self, mock_realpath, mock_check_path, mock_torch_save):
        """测试保存 tensor 为 .pt 文件"""
        mock_tensor = MagicMock()
        mock_tensor.contiguous.return_value = mock_tensor
        mock_tensor.detach.return_value = mock_tensor
        mock_realpath.return_value = '/output/file.pt'

        self.parser._save_tensor_to_pt(mock_tensor, '/output/file.pt')

        mock_check_path.assert_called_once_with('/output/file.pt')
        mock_realpath.assert_called_once_with('/output/file.pt')
        mock_tensor.contiguous.assert_called_once()
        mock_tensor.detach.assert_called_once()
        mock_torch_save.assert_called_once_with(mock_tensor, '/output/file.pt')

    @patch('msprobe.core.parse.tensor_bin_parser.torch_available', False)
    def test_save_tensor_to_pt_without_torch(self):
        """测试没有 torch 时保存 .pt 文件会报错"""
        with self.assertRaises(RuntimeError) as context:
            self.parser._save_tensor_to_pt(MagicMock(), '/output/file.pt')

        self.assertIn('torch is required', str(context.exception))

    @patch('msprobe.core.parse.tensor_bin_parser.torch_available', True)
    @patch('msprobe.core.parse.tensor_bin_parser.TensorBinFile')
    def test_parse_single_bin_file_exception(self, mock_tensor_bin):
        """测试解析文件时抛出异常"""
        mock_tensor_bin.side_effect = Exception("Test error")

        with self.assertRaises(Exception) as context:
            self.parser._parse_single_bin_file('/path/to/file.bin', '/output', 'npy')

        self.assertIn('Test error', str(context.exception))


if __name__ == '__main__':
    unittest.main()
