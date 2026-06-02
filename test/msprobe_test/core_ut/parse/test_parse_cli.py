# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
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
import argparse
from unittest.mock import patch, MagicMock

from msprobe.core.parse.parse_cli import _parse_parser, parse_cli


class TestParseParser(unittest.TestCase):
    """测试 _parse_parser 函数"""

    def test_parse_parser_adds_arguments(self):
        """测试参数解析器是否正确添加参数"""
        parser = argparse.ArgumentParser()
        _parse_parser(parser)

        # 测试必选参数 dump_path
        args = parser.parse_args(['-d', '/path/to/dump'])
        self.assertEqual(args.dump_path, '/path/to/dump')

        # 测试可选参数 type，默认值为 pt
        args = parser.parse_args(['-d', '/path/to/dump'])
        self.assertEqual(args.parse_type, 'pt')

        # 测试指定 type 参数
        args = parser.parse_args(['-d', '/path/to/dump', '-t', 'npy'])
        self.assertEqual(args.parse_type, 'npy')

        # 测试可选参数 output_path，默认值为 ./output
        args = parser.parse_args(['-d', '/path/to/dump'])
        self.assertEqual(args.output_path, './output')

        # 测试指定 output_path 参数
        args = parser.parse_args(['-d', '/path/to/dump', '-o', '/path/to/output'])
        self.assertEqual(args.output_path, '/path/to/output')

        # 测试使用长参数名
        args = parser.parse_args(['--dump_path', '/path/to/dump', '--type', 'npy', '--output_path', '/path/to/output'])
        self.assertEqual(args.dump_path, '/path/to/dump')
        self.assertEqual(args.parse_type, 'npy')
        self.assertEqual(args.output_path, '/path/to/output')

    def test_parse_parser_requires_dump_path(self):
        """测试 dump_path 参数是必选的"""
        parser = argparse.ArgumentParser()
        _parse_parser(parser)

        with self.assertRaises(SystemExit):
            parser.parse_args([])

    def test_parse_parser_type_choices(self):
        """测试 type 参数只能选择 npy 或 pt"""
        parser = argparse.ArgumentParser()
        _parse_parser(parser)

        # 测试有效的选择
        args = parser.parse_args(['-d', '/path/to/dump', '-t', 'npy'])
        self.assertEqual(args.parse_type, 'npy')

        args = parser.parse_args(['-d', '/path/to/dump', '-t', 'pt'])
        self.assertEqual(args.parse_type, 'pt')

        # 测试无效的选择
        with self.assertRaises(SystemExit):
            parser.parse_args(['-d', '/path/to/dump', '-t', 'invalid'])


class TestParseCli(unittest.TestCase):
    """测试 parse_cli 函数"""

    @patch('msprobe.core.parse.parse_cli.create_directory')
    @patch('msprobe.core.parse.parse_cli._parser_factory')
    def test_parse_cli_with_valid_args(self, mock_factory, mock_create_dir):
        """测试 parse_cli 使用有效参数"""
        # 准备 mock 对象
        mock_parser = MagicMock()
        mock_factory.get_parser.return_value = mock_parser

        # 创建 args 对象
        args = argparse.Namespace()
        args.dump_path = '/path/to/dump'
        args.parse_type = 'npy'
        args.output_path = '/path/to/output'

        # 执行函数
        parse_cli(args)

        # 验证调用
        mock_create_dir.assert_called_once_with('/path/to/output')
        mock_factory.get_parser.assert_called_once_with('/path/to/dump')
        mock_parser.parse.assert_called_once_with('/path/to/dump', '/path/to/output', 'npy')

    @patch('msprobe.core.parse.parse_cli.create_directory')
    @patch('msprobe.core.parse.parse_cli._parser_factory')
    def test_parse_cli_with_default_type(self, mock_factory, mock_create_dir):
        """测试 parse_cli 使用默认类型 pt"""
        mock_parser = MagicMock()
        mock_factory.get_parser.return_value = mock_parser

        args = argparse.Namespace()
        args.dump_path = '/path/to/dump'
        args.parse_type = 'pt'  # 默认值
        args.output_path = './output'  # 默认值

        parse_cli(args)

        mock_parser.parse.assert_called_once_with('/path/to/dump', './output', 'pt')

    @patch('msprobe.core.parse.parse_cli.create_directory')
    @patch('msprobe.core.parse.parse_cli._parser_factory')
    def test_parse_cli_with_default_output_path(self, mock_factory, mock_create_dir):
        """测试 parse_cli 使用默认输出路径"""
        mock_parser = MagicMock()
        mock_factory.get_parser.return_value = mock_parser

        args = argparse.Namespace()
        args.dump_path = '/path/to/dump'
        args.parse_type = 'npy'
        args.output_path = './output'  # 默认值

        parse_cli(args)

        mock_create_dir.assert_called_once_with('./output')
        mock_parser.parse.assert_called_once_with('/path/to/dump', './output', 'npy')

    @patch('msprobe.core.parse.parse_cli.create_directory')
    @patch('msprobe.core.parse.parse_cli._parser_factory')
    def test_parse_cli_parser_raises_exception(self, mock_factory, mock_create_dir):
        """测试 parse_cli 处理解析器抛出的异常"""
        mock_parser = MagicMock()
        mock_parser.parse.side_effect = RuntimeError("Parse error")
        mock_factory.get_parser.return_value = mock_parser

        args = argparse.Namespace()
        args.dump_path = '/path/to/dump'
        args.parse_type = 'npy'
        args.output_path = '/path/to/output'

        with self.assertRaises(RuntimeError) as context:
            parse_cli(args)

        self.assertIn('Parse error', str(context.exception))
        mock_create_dir.assert_called_once()
        mock_parser.parse.assert_called_once()


if __name__ == '__main__':
    unittest.main()

