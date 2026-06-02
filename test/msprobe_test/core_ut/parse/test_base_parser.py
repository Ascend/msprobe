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

from msprobe.core.parse.base import BaseParser


class ConcreteParser(BaseParser):
    """用于测试的具体解析器实现"""

    def parse(self, dump_path, output_path, parse_type):
        raise NotImplementedError("Subclasses must implement parse")


class TestBaseParser(unittest.TestCase):
    """测试 BaseParser 基类"""

    def setUp(self):
        """创建测试用的解析器实例"""
        self.parser = ConcreteParser()

    def test_get_output_file_path_default_extension(self):
        """测试生成输出文件路径，使用默认扩展名"""
        input_file = '/path/to/input/file.bin'
        output_dir = '/path/to/output'
        parse_type = 'npy'

        result = BaseParser.get_output_file_path(input_file, output_dir, parse_type)

        expected = os.path.join(output_dir, 'file.npy')
        self.assertEqual(result, expected)

    def test_get_output_file_path_custom_extension(self):
        """测试生成输出文件路径，使用自定义扩展名"""
        input_file = '/path/to/input/file.bin'
        output_dir = '/path/to/output'
        parse_type = 'npy'
        extension = 'custom'

        result = BaseParser.get_output_file_path(input_file, output_dir, parse_type, extension)

        expected = os.path.join(output_dir, 'file.custom')
        self.assertEqual(result, expected)

    def test_get_output_file_path_without_extension(self):
        """测试输入文件没有扩展名的情况"""
        input_file = '/path/to/input/file'
        output_dir = '/path/to/output'
        parse_type = 'pt'

        result = BaseParser.get_output_file_path(input_file, output_dir, parse_type)

        expected = os.path.join(output_dir, 'file.pt')
        self.assertEqual(result, expected)

    def test_abstract_method_raises_error(self):
        """测试抽象方法不能被直接实例化"""
        with self.assertRaises(TypeError):
            BaseParser()


if __name__ == '__main__':
    unittest.main()
