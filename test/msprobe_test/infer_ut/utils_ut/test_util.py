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
from unittest.mock import patch, MagicMock
from itertools import product

from msprobe.infer.utils.util import (
    confirmation_interaction,
    check_file_ext,
    check_file_size_based_on_ext,
    filter_cmd,
    load_file_to_read_common_check
)


class TestUtil(unittest.TestCase):

    def test_confirmation_interaction_yes(self):
        yes_input = ['y', 'Y', 'yes', 'YES', 'Yes', 'yES']

        for i in yes_input:
            with self.subTest(i):
                with patch('builtins.input', return_value=i):
                    self.assertTrue(confirmation_interaction(""))

    def test_confirmation_interaction_no(self):
        no_input = ['n', 'no', 'abc', 'EOF']

        for i in no_input:
            with self.subTest(i):
                with patch('builtins.input', return_value=i):
                    self.assertFalse(confirmation_interaction(""))

    def test_check_file_ext_type_error(self):
        paths = [1, 2.5, -3 + 1j]
        exts = [range(10), set(), dict()]

        for path, ext in product(paths, exts):
            with self.subTest(path=path, ext=ext):
                self.assertRaises(TypeError, check_file_ext, path, ext)

    def test_check_file_ext_not_match(self):
        path = 'model.onnx'
        exts = ['onnx', 'py', '.py', '.cpp']

        for ext in exts:
            with self.subTest(path=path, ext=ext):
                self.assertFalse(check_file_ext(path, ext))

    def test_check_file_size_based_on_ext_type_error(self):
        paths = [1, 2.5, -3 + 1j]

        for path in paths:
            with self.subTest(path=path):
                self.assertRaises(TypeError, check_file_size_based_on_ext, path)

    def test_check_file_size_based_on_ext_large_size(self):
        exts = ['.csv', '.json', '.txt', '.onnx', '.ini', '.py', '.pth', '.bin']

        with patch('os.path.getsize', return_value=500 * 1024 * 1024 * 1024):
            for ext in exts:
                with self.subTest(ext=ext):
                    self.assertFalse(check_file_size_based_on_ext('random_file', ext))
                    self.assertFalse(check_file_size_based_on_ext('random_file' + ext))

            with patch('builtins.input', return_value='n'):
                self.assertFalse(check_file_size_based_on_ext('random_file'))

    def test_check_file_size_based_on_ext_normal_size(self):
        config_file_size = 8 * 1024
        text_file_size = 8 * 1024 * 1024
        onnx_model_size = 1 * 1024 * 1024 * 1024
        model_weigtht_size = 8 * 1024 * 1024 * 1024

        exts = ['.ini', '.csv', '.json', '.txt', '.py', '.onnx', '.pth', '.bin']
        sizes = [config_file_size] + [text_file_size] * 4 + [onnx_model_size] + [model_weigtht_size] * 2

        for ext, size in zip(exts, sizes):
            with patch('os.path.getsize', return_value=size):
                with self.subTest(ext=ext, size=size):
                    self.assertTrue(check_file_size_based_on_ext('random_file', ext))
                    self.assertTrue(check_file_size_based_on_ext('random_file' + ext))
                    self.assertTrue(check_file_size_based_on_ext('random_file'))

    @patch('msprobe.infer.utils.util.check_file_ext')
    @patch('msprobe.infer.utils.util.is_legal_path_length')
    @patch('msprobe.infer.utils.util.check_file_size_based_on_ext')
    @patch('os.path.realpath')
    @patch('os.stat')
    @patch('re.search')
    def test_load_file_to_read_common_check(self, mock_re_search, mock_os_stat,
                                            mock_realpath, mock_check_file_size_based_on_ext,
                                            mock_is_legal_path_length, mock_check_file_ext):

        # 1. Normal case: Valid path and file
        mock_check_file_ext.return_value = True
        mock_is_legal_path_length.return_value = True
        mock_check_file_size_based_on_ext.return_value = True
        mock_realpath.return_value = '/mock/real/path'

        mock_stat_result = MagicMock()
        mock_stat_result.st_mode = 0o100644  # Regular file mode
        mock_os_stat.return_value = mock_stat_result

        mock_re_search.return_value = None

        path = '/mock/path/to/file.txt'
        exts = ['.txt', '.md']
        result = load_file_to_read_common_check(path, exts)
        self.assertEqual(result, '/mock/real/path')

        # 2. Test invalid path type (should raise TypeError)
        with self.assertRaises(TypeError):
            load_file_to_read_common_check(123, exts=['.txt'])

        # 3. Test valid file extension
        path = '/mock/path/to/file.pdf'
        load_file_to_read_common_check(path, exts=['.txt', '.md'])

        # 4. Test invalid character in path (should raise ValueError)
        mock_re_search.return_value = True
        with self.assertRaises(ValueError):
            load_file_to_read_common_check('/mock/path/with@invalid/char.txt', exts=['.txt'])

        # 5. Test directory instead of file (should raise ValueError)
        mock_stat_result.st_mode = 0o040000  # Directory mode
        with patch('os.stat', return_value=mock_stat_result):
            with self.assertRaises(ValueError):
                load_file_to_read_common_check('/mock/path/to/directory', exts=['.txt'])

        # 6. Test file with other-writeable permission (should raise PermissionError)
        mock_stat_result.st_mode = 0o666  # Other-writeable mode
        mock_re_search.return_value = False
        with patch('os.stat', return_value=mock_stat_result), patch('os.st.S_ISREG', return_value=True):
            with self.assertRaises(PermissionError):
                load_file_to_read_common_check('/mock/path/to/writeable/file.txt', exts=['.txt'])


class TestFilterCmd(unittest.TestCase):
    def test_valid_characters(self):
        input_args = ["hello", "world123", "file_name.txt", "path/to/file", "a-b-c", "A_B_C", "1 2 3", "var=value"]
        expected = input_args.copy()
        self.assertEqual(filter_cmd(input_args), expected)

    def test_invalid_characters_raises_error(self):
        input_args = ["hello!", "world@123", "file$name", "path/to|file", "a{b}c", "A#B#C"]
        for arg in input_args:
            with self.assertRaises(ValueError, msg=f"Expected ValueError for input: {arg}"):
                filter_cmd([arg])

    def test_mixed_valid_invalid_raises_error(self):
        input_args = ["valid", "inval!d", "good123", "bad@arg", "ok"]
        with self.assertRaises(ValueError):
            filter_cmd(input_args)

    def test_empty_input(self):
        self.assertEqual(filter_cmd([]), [])

    def test_non_string_input(self):
        input_args = [123, 45.67, True, None]
        expected = ["123", "45.67", "True", "None"]
        self.assertEqual(filter_cmd(input_args), expected)

    def test_whitespace_only(self):
        self.assertEqual(filter_cmd([" ", "   "]), [" ", "   "])

    def test_edge_cases(self):
        input_args = ["", "-._ /=", "a" * 1000]
        with self.assertRaises(ValueError):
            filter_cmd(input_args)

    def test_non_ascii_raises_error(self):
        input_args = ["héllo", "世界", "café"]
        for arg in input_args:
            with self.assertRaises(ValueError, msg=f"Expected ValueError for input: {arg}"):
                filter_cmd([arg])
