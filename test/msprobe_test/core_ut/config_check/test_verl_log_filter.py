import json
import os
import re
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch, mock_open

from msprobe.core.config_check.verl_param_compare.verl_log_filter import (
    check_is_end_of_list,
    detect_prefix,
    split_line_by_prefix,
    is_generic_format,
    get_in_string_value,
    get_config_start_idx,
    update_depth_square_bracket,
    extract_dict,
    verl_get_config_file_path,
    check_log_extension,
    verl_filter_config_info,
)


class TestCheckIsEndOfList(unittest.TestCase):
    def test_valid_end_of_list_string(self):
        s = "'value']},"
        stack = []
        pre_stack = [']']
        result = check_is_end_of_list(s, stack, pre_stack)
        self.assertTrue(result)

    def test_valid_numeric_end_of_list(self):
        s = "123]},"
        stack = []
        pre_stack = [']']
        result = check_is_end_of_list(s, stack, pre_stack)
        self.assertTrue(result)

    def test_invalid_format(self):
        s = "not_a_valid_format"
        stack = []
        pre_stack = [']']
        result = check_is_end_of_list(s, stack, pre_stack)
        self.assertFalse(result)

    def test_bracket_count_mismatch(self):
        s = "'value']]},"
        stack = []
        pre_stack = [']']
        result = check_is_end_of_list(s, stack, pre_stack)
        self.assertFalse(result)

    def test_float_value_end_of_list(self):
        s = "3.14]},"
        stack = []
        pre_stack = [']']
        result = check_is_end_of_list(s, stack, pre_stack)
        self.assertTrue(result)

    def test_scientific_notation_end_of_list(self):
        s = "1e-5]},"
        stack = []
        pre_stack = [']']
        result = check_is_end_of_list(s, stack, pre_stack)
        self.assertTrue(result)

    def test_negative_number_end_of_list(self):
        s = "-42]},"
        stack = []
        pre_stack = [']']
        result = check_is_end_of_list(s, stack, pre_stack)
        self.assertTrue(result)

    def test_empty_pre_stack(self):
        s = "'value']},"
        stack = []
        pre_stack = []
        result = check_is_end_of_list(s, stack, pre_stack)
        self.assertFalse(result)


class TestDetectPrefix(unittest.TestCase):
    def test_detect_prefix_success(self):
        lines = [
            "prefix{'actor_rollout_ref':",
            "prefix{'key': 'value'}",
        ]
        pattern = detect_prefix(lines)
        self.assertIsNotNone(pattern)
        self.assertTrue(pattern.search(lines[0]))

    def test_detect_prefix_custom_target_key(self):
        lines = [
            "log_prefix{'actor_rollout_ref':",
            "log_prefix{'key': 'value'}",
        ]
        pattern = detect_prefix(lines)
        self.assertIsNotNone(pattern)

    def test_detect_prefix_target_key_not_found(self):
        lines = [
            "no_target_key_here",
            "another_line",
        ]
        with self.assertRaises(ValueError):
            detect_prefix(lines)

    def test_detect_prefix_mismatched_prefix(self):
        lines = [
            "prefix1{'actor_rollout_ref':",
            "prefix2{'actor_rollout_ref':",
        ]
        with self.assertRaises(ValueError):
            detect_prefix(lines)

    def test_detect_prefix_single_line(self):
        lines = [
            "prefix{'actor_rollout_ref':",
        ]
        pattern = detect_prefix(lines)
        self.assertIsNotNone(pattern)


class TestSplitLineByPrefix(unittest.TestCase):
    def test_split_single_match(self):
        prefix_pattern = re.compile(re.escape("prefix"))
        line = "prefixcontent1"
        result = split_line_by_prefix(line, prefix_pattern)
        self.assertEqual(result, ["content1"])

    def test_split_multiple_matches(self):
        prefix_pattern = re.compile(re.escape("prefix"))
        line = "prefixcontent1prefixcontent2"
        result = split_line_by_prefix(line, prefix_pattern)
        self.assertEqual(result, ["content1", "content2"])

    def test_split_no_match(self):
        prefix_pattern = re.compile(re.escape("prefix"))
        line = "no_match_here"
        result = split_line_by_prefix(line, prefix_pattern)
        self.assertEqual(result, ["no_match_here"])

    def test_split_empty_line(self):
        prefix_pattern = re.compile(re.escape("prefix"))
        line = ""
        result = split_line_by_prefix(line, prefix_pattern)
        self.assertEqual(result, [])

    def test_split_whitespace_only_content(self):
        prefix_pattern = re.compile(re.escape("prefix"))
        line = "prefix   "
        result = split_line_by_prefix(line, prefix_pattern)
        self.assertEqual(result, [])


class TestIsGenericFormat(unittest.TestCase):
    def test_string_value(self):
        self.assertTrue(is_generic_format("'value',"))

    def test_integer_value(self):
        self.assertTrue(is_generic_format(" 42,"))

    def test_negative_integer_value(self):
        self.assertTrue(is_generic_format(" -5,"))

    def test_float_value(self):
        self.assertTrue(is_generic_format(" 3.14,"))

    def test_not_generic_format(self):
        self.assertFalse(is_generic_format("{'key':"))

    def test_empty_string(self):
        self.assertFalse(is_generic_format(""))

    def test_string_without_comma(self):
        self.assertFalse(is_generic_format("'value'"))


class TestGetInStringValue(unittest.TestCase):
    def test_enter_string_single_quote(self):
        in_string, quote_char = get_in_string_value("'", False, None)
        self.assertTrue(in_string)
        self.assertEqual(quote_char, "'")

    def test_exit_string_single_quote(self):
        in_string, quote_char = get_in_string_value("'", True, "'")
        self.assertFalse(in_string)
        self.assertIsNone(quote_char)

    def test_enter_string_double_quote(self):
        in_string, quote_char = get_in_string_value('"', False, None)
        self.assertTrue(in_string)
        self.assertEqual(quote_char, '"')

    def test_exit_string_double_quote(self):
        in_string, quote_char = get_in_string_value('"', True, '"')
        self.assertFalse(in_string)
        self.assertIsNone(quote_char)

    def test_different_quote_char_not_exit(self):
        in_string, quote_char = get_in_string_value('"', True, "'")
        self.assertTrue(in_string)
        self.assertEqual(quote_char, "'")


class TestGetConfigStartIdx(unittest.TestCase):
    def test_single_config_start(self):
        lines = [
            "some line",
            "{'actor_rollout_ref': 'value'}",
            "another line",
        ]
        idx, key = get_config_start_idx(lines, "{'actor_rollout_ref':")
        self.assertEqual(idx, 1)

    def test_multiple_config_starts(self):
        lines = [
            "{'actor_rollout_ref': 'value1'}",
            "some line",
            "{'actor_rollout_ref': 'value2'}",
        ]
        idx, key = get_config_start_idx(lines, "{'actor_rollout_ref':")
        self.assertEqual(idx, 2)

    def test_json_format_config_start(self):
        lines = [
            "some line",
            '{"actor_rollout_ref": "value"}',
        ]
        idx, key = get_config_start_idx(lines, "{'actor_rollout_ref':")
        self.assertEqual(idx, 1)
        self.assertEqual(key, '{"actor_rollout_ref":')

    def test_no_config_start(self):
        lines = [
            "some line",
            "another line",
        ]
        with self.assertRaises(ValueError):
            get_config_start_idx(lines, "{'actor_rollout_ref':")


class TestUpdateDepthSquareBracket(unittest.TestCase):
    def test_open_brace(self):
        depth, stack = update_depth_square_bracket(False, '{', 0, [])
        self.assertEqual(depth, 1)
        self.assertEqual(stack, [])

    def test_close_brace(self):
        depth, stack = update_depth_square_bracket(False, '}', 1, [])
        self.assertEqual(depth, 0)
        self.assertEqual(stack, [])

    def test_open_square_bracket(self):
        depth, stack = update_depth_square_bracket(False, '[', 0, [])
        self.assertEqual(depth, 0)
        self.assertEqual(stack, ['['])

    def test_close_square_bracket(self):
        depth, stack = update_depth_square_bracket(False, ']', 0, ['['])
        self.assertEqual(depth, 0)
        self.assertEqual(stack, [])

    def test_in_string_ignored(self):
        depth, stack = update_depth_square_bracket(True, '{', 0, [])
        self.assertEqual(depth, 0)
        self.assertEqual(stack, [])

    def test_in_string_square_bracket_ignored(self):
        depth, stack = update_depth_square_bracket(True, '[', 0, [])
        self.assertEqual(depth, 0)
        self.assertEqual(stack, [])

    def test_nested_brackets(self):
        depth, stack = update_depth_square_bracket(False, '[', 0, ['['])
        self.assertEqual(depth, 0)
        self.assertEqual(stack, ['[', '['])


class TestExtractDict(unittest.TestCase):
    def test_simple_dict(self):
        text = "[INFO] {'actor_rollout_ref': 'value'}"
        result = extract_dict(text)
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

    def test_dict_with_list(self):
        text = "[INFO] {'actor_rollout_ref': [1, 2, 3]}"
        result = extract_dict(text)
        self.assertIsInstance(result, list)

    def test_unclosed_dict(self):
        text = "[INFO] {'actor_rollout_ref': 'value'"
        with self.assertRaises(ValueError):
            extract_dict(text)

    def test_no_actor_rollout_ref(self):
        text = "{'other_key': 'value'}"
        with self.assertRaises(ValueError):
            extract_dict(text)

    def test_multiline_dict(self):
        text = """[INFO] {'actor_rollout_ref': 'value',
[INFO] 'key2': 'value2'}"""
        result = extract_dict(text)
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

    def test_dict_with_prefix(self):
        text = """[INFO] {'actor_rollout_ref': 'value'}
[INFO] 'key2': 'value2'}"""
        result = extract_dict(text)
        self.assertTrue(len(result) > 0)
        self.assertIn("'actor_rollout_ref':", result[0])


class TestVerlGetConfigFilePath(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_existing_directory(self):
        npu_path, bench_path = verl_get_config_file_path(self.temp_dir)
        self.assertTrue(npu_path.endswith("NPU_config.json"))
        self.assertTrue(bench_path.endswith("bench_config.json"))

    def test_non_existing_directory(self):
        new_dir = os.path.join(self.temp_dir, "new_subdir")
        npu_path, bench_path = verl_get_config_file_path(new_dir)
        self.assertTrue(os.path.isdir(new_dir))
        self.assertTrue(npu_path.endswith("NPU_config.json"))
        self.assertTrue(bench_path.endswith("bench_config.json"))


class TestCheckLogExtension(unittest.TestCase):
    def test_log_extension(self):
        self.assertTrue(check_log_extension("test.log"))

    def test_txt_extension(self):
        self.assertTrue(check_log_extension("test.txt"))

    def test_uppercase_log_extension(self):
        self.assertTrue(check_log_extension("test.LOG"))

    def test_json_extension(self):
        self.assertFalse(check_log_extension("test.json"))

    def test_no_extension(self):
        self.assertFalse(check_log_extension("testfile"))

    def test_py_extension(self):
        self.assertFalse(check_log_extension("test.py"))


class TestVerlFilterConfigInfo(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.temp_dir, "test.log")
        self.out_file = os.path.join(self.temp_dir, "output.json")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_filter_simple_config(self):
        with open(self.log_file, 'w') as f:
            f.write("[INFO] {'actor_rollout_ref': 'value'}\n")

        verl_filter_config_info(self.log_file, self.out_file)

        self.assertTrue(os.path.exists(self.out_file))
        with open(self.out_file, 'r') as f:
            config = json.load(f)
        self.assertEqual(config.get('actor_rollout_ref'), 'value')

    def test_filter_config_with_list(self):
        with open(self.log_file, 'w') as f:
            f.write("[INFO] {'actor_rollout_ref': [1, 2, 3]}\n")

        verl_filter_config_info(self.log_file, self.out_file)

        self.assertTrue(os.path.exists(self.out_file))
        with open(self.out_file, 'r') as f:
            config = json.load(f)
        self.assertEqual(config.get('actor_rollout_ref'), [1, 2, 3])

    def test_filter_config_with_prefix(self):
        with open(self.log_file, 'w') as f:
            f.write("[INFO] {'actor_rollout_ref': 'value'}\n")

        verl_filter_config_info(self.log_file, self.out_file)

        self.assertTrue(os.path.exists(self.out_file))
        with open(self.out_file, 'r') as f:
            config = json.load(f)
        self.assertEqual(config.get('actor_rollout_ref'), 'value')

    def test_filter_invalid_log_raises(self):
        with open(self.log_file, 'w') as f:
            f.write("no valid config here\n")

        with self.assertRaises(ValueError):
            verl_filter_config_info(self.log_file, self.out_file)

    @patch('msprobe.core.config_check.verl_param_compare.verl_log_filter.check_file_or_directory_path')
    def test_filter_checks_file_path(self, mock_check):
        mock_check.side_effect = Exception("file check failed")
        with self.assertRaises(Exception):
            verl_filter_config_info(self.log_file, self.out_file)

    def test_filter_python_dict_format_config(self):
        with open(self.log_file, 'w') as f:
            f.write('[INFO] {\'actor_rollout_ref\': \'value\'}\n')

        verl_filter_config_info(self.log_file, self.out_file)

        self.assertTrue(os.path.exists(self.out_file))
        with open(self.out_file, 'r') as f:
            config = json.load(f)
        self.assertEqual(config.get('actor_rollout_ref'), 'value')

    def test_filter_complex_config(self):
        with open(self.log_file, 'w') as f:
            f.write("[INFO] {'actor_rollout_ref': {'sub_key': 'sub_value'}, 'key2': 42}\n")

        verl_filter_config_info(self.log_file, self.out_file)

        self.assertTrue(os.path.exists(self.out_file))
        with open(self.out_file, 'r') as f:
            config = json.load(f)
        self.assertIn('actor_rollout_ref', config)
