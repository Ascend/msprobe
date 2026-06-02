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
from unittest.mock import DEFAULT, patch, MagicMock

from msprobe.core.parse.msaccucmp_parser import MsaccucmpParser
from msprobe.core.common.exceptions import FileCheckException


class TestMsaccucmpParser(unittest.TestCase):
    """测试 MsaccucmpParser 类"""

    def setUp(self):
        """创建解析器实例"""
        self.parser = MsaccucmpParser()

    @patch('msprobe.core.parse.msaccucmp_parser.subprocess.Popen')
    @patch.object(MsaccucmpParser, '_get_msaccucmp_script_path')
    @patch('msprobe.core.parse.msaccucmp_parser.sys.executable', '/usr/bin/python3')
    @patch('msprobe.core.parse.msaccucmp_parser.logger')
    def test_parse_success(self, mock_logger, mock_get_path, mock_popen):
        """测试成功解析"""
        mock_get_path.return_value = '/path/to/msaccucmp.py'

        # 模拟进程成功执行
        mock_process = MagicMock()
        mock_process.stdout.readline.side_effect = ['line1\n', 'line2\n', '']
        mock_process.returncode = 0
        mock_process.stdout.close = MagicMock()
        mock_process.wait = MagicMock()
        mock_popen.return_value = mock_process

        self.parser.parse('/path/to/dump', '/path/to/output', 'npy')

        # 验证命令参数
        expected_cmd = [
            '/usr/bin/python3',
            '/path/to/msaccucmp.py',
            'convert',
            '-d', '/path/to/dump',
            '-t', 'npy',
            '-out', '/path/to/output'
        ]
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args[0][0]
        self.assertEqual(call_args, expected_cmd)
        # 验证日志被调用
        mock_logger.info.assert_any_call("msaccucmp convert execution completed successfully")

    @patch('msprobe.core.parse.msaccucmp_parser.subprocess.Popen')
    @patch.object(MsaccucmpParser, '_get_msaccucmp_script_path')
    @patch('msprobe.core.parse.msaccucmp_parser.sys.executable', '/usr/bin/python3')
    @patch('msprobe.core.parse.msaccucmp_parser.logger')
    def test_parse_failure(self, mock_logger, mock_get_path, mock_popen):
        """测试解析失败"""
        mock_get_path.return_value = '/path/to/msaccucmp.py'

        # 模拟进程失败
        mock_process = MagicMock()
        mock_process.stdout.readline.side_effect = ['error line\n', '']
        mock_process.returncode = 1
        mock_process.stdout.close = MagicMock()
        mock_process.wait = MagicMock()
        mock_popen.return_value = mock_process

        with self.assertRaises(RuntimeError) as context:
            self.parser.parse('/path/to/dump', '/path/to/output', 'npy')

        self.assertIn('msaccucmp convert execution failed', str(context.exception))
        mock_logger.error.assert_any_call("msaccucmp convert execution failed with return code 1")

    @patch.multiple(
        'msprobe.core.parse.msaccucmp_parser',
        check_file_or_directory_path=DEFAULT,
        os=DEFAULT
    )
    def test_get_msaccucmp_script_path_success(self, check_file_or_directory_path, os):
        """测试成功获取 msaccucmp.py 路径"""
        os.path.abspath.side_effect = [
            '/path/to/msaccucmp_parser.py',  # 第一次调用：获取当前文件路径
            '/path/to/msprobe/msaccucmp/msaccucmp.py'  # 第二次调用：获取脚本绝对路径
        ]
        os.path.dirname.side_effect = [
            '/path/to/parse',  # 第一次调用
            '/path/to/core',   # 第二次调用
            '/path/to/msprobe' # 第三次调用
        ]
        os.path.join.return_value = '/path/to/msprobe/msaccucmp/msaccucmp.py'
        os.path.exists.return_value = True

        result = self.parser._get_msaccucmp_script_path()

        self.assertEqual(result, '/path/to/msprobe/msaccucmp/msaccucmp.py')
        check_file_or_directory_path.assert_called_once_with('/path/to/msprobe/msaccucmp/msaccucmp.py', isdir=False)

    @patch.multiple(
        'msprobe.core.parse.msaccucmp_parser',
        check_file_or_directory_path=DEFAULT,
        os=DEFAULT
    )
    @patch('msprobe.core.parse.msaccucmp_parser.logger')
    def test_get_msaccucmp_script_path_not_found(self, mock_logger, check_file_or_directory_path, os):
        """测试 msaccucmp.py 文件不存在"""
        check_file_or_directory_path.return_value = None
        os.path.abspath.return_value = '/path/to/msaccucmp_parser.py'
        os.path.dirname.side_effect = [
            '/path/to/parse',
            '/path/to/core',
            '/path/to/msprobe'
        ]
        os.path.join.return_value = '/path/to/msprobe/msaccucmp/msaccucmp.py'
        os.path.exists.return_value = False

        with self.assertRaises(FileNotFoundError) as context:
            self.parser._get_msaccucmp_script_path()

        self.assertIn('msaccucmp.py file not found', str(context.exception))
        check_file_or_directory_path.assert_called_once_with('/path/to/msprobe/msaccucmp/msaccucmp.py', isdir=False)
        mock_logger.error.assert_called_once()

    @patch('msprobe.core.parse.msaccucmp_parser.subprocess.Popen')
    @patch.object(MsaccucmpParser, '_get_msaccucmp_script_path')
    @patch('msprobe.core.parse.msaccucmp_parser.logger')
    def test_parse_file_not_found_error(self, mock_logger, mock_get_path, mock_popen):
        """测试 FileNotFoundError 异常处理"""
        mock_get_path.side_effect = FileNotFoundError("msaccucmp.py not found")

        with self.assertRaises(FileCheckException) as context:
            self.parser.parse('/path/to/dump', '/path/to/output', 'npy')

        self.assertIn('Failed to find msaccucmp.py', str(context.exception))
        mock_logger.error.assert_called_once()

    @patch('msprobe.core.parse.msaccucmp_parser.subprocess.Popen')
    @patch.object(MsaccucmpParser, '_get_msaccucmp_script_path')
    @patch('msprobe.core.parse.msaccucmp_parser.logger')
    def test_parse_general_exception(self, mock_logger, mock_get_path, mock_popen):
        """测试一般异常处理"""
        mock_get_path.return_value = '/path/to/msaccucmp.py'
        mock_popen.side_effect = Exception("Unexpected error")

        with self.assertRaises(RuntimeError) as context:
            self.parser.parse('/path/to/dump', '/path/to/output', 'npy')

        self.assertIn('Failed to execute msaccucmp convert', str(context.exception))
        mock_logger.error.assert_called_once()

    @patch('msprobe.core.parse.msaccucmp_parser.subprocess.Popen')
    @patch.object(MsaccucmpParser, '_get_msaccucmp_script_path')
    @patch('msprobe.core.parse.msaccucmp_parser.sys.executable', '/usr/bin/python3')
    @patch('msprobe.core.parse.msaccucmp_parser.logger')
    def test_parse_output_logging(self, mock_logger, mock_get_path, mock_popen):
        """测试输出日志记录"""
        mock_get_path.return_value = '/path/to/msaccucmp.py'

        mock_process = MagicMock()
        # iter(process.stdout.readline, '') 会在 readline 返回 '' 时停止
        mock_process.stdout.readline.side_effect = ['output line 1\n', 'output line 2\n', '']
        mock_process.returncode = 0
        mock_process.stdout.close = MagicMock()
        mock_process.wait = MagicMock()
        mock_popen.return_value = mock_process

        self.parser.parse('/path/to/dump', '/path/to/output', 'npy')

        # 验证 logger.raw 被调用（通过 Popen 的 stdout 读取）
        # 代码中会调用 line.rstrip()，所以换行符会被去掉
        self.assertEqual(mock_logger.raw.call_count, 2)
        # 验证调用的参数（去掉换行符后的内容）
        calls = [call[0][0] for call in mock_logger.raw.call_args_list]
        self.assertIn('output line 1', calls)
        self.assertIn('output line 2', calls)

    @patch('msprobe.core.parse.msaccucmp_parser.subprocess.Popen')
    @patch.object(MsaccucmpParser, '_get_msaccucmp_script_path')
    @patch('msprobe.core.parse.msaccucmp_parser.sys.executable', '/usr/bin/python3')
    @patch('msprobe.core.parse.msaccucmp_parser.logger')
    def test_parse_with_pt_type(self, mock_logger, mock_get_path, mock_popen):
        """测试使用 pt 类型解析"""
        mock_get_path.return_value = '/path/to/msaccucmp.py'

        mock_process = MagicMock()
        mock_process.stdout.readline.side_effect = ['']
        mock_process.returncode = 0
        mock_process.stdout.close = MagicMock()
        mock_process.wait = MagicMock()
        mock_popen.return_value = mock_process

        self.parser.parse('/path/to/dump', '/path/to/output', 'pt')

        # 验证命令参数中包含 pt
        call_args = mock_popen.call_args[0][0]
        self.assertIn('pt', call_args)
        self.assertIn('-t', call_args)
        pt_index = call_args.index('-t')
        self.assertEqual(call_args[pt_index + 1], 'pt')


if __name__ == '__main__':
    unittest.main()
