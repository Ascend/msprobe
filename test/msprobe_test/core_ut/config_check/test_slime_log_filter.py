import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from msprobe.core.config_check.slime_param_compare.slime_log_filter import (
    get_args_raw_text,
    parse_argument_lines,
    slime_get_config_file_path,
    slime_filter_config_info,
    START_MARKER,
    END_MARKER
)


class TestGetArgsRawText(unittest.TestCase):
    def test_extract_success_normal_block(self):
        content = (
            f"some log before\n"
            f"{START_MARKER}\n"
            f"arg1 value1\n"
            f"{END_MARKER}\n"
            f"some log after"
        )
        result = get_args_raw_text(content)
        self.assertIn("arg1 value1", result)

    def test_start_marker_missing(self):
        content = f"random log\n{END_MARKER}"
        with self.assertRaises(ValueError) as ctx:
            get_args_raw_text(content)
        self.assertIn("Start arguments marker not found", str(ctx.exception))

    def test_end_marker_missing(self):
        content = f"{START_MARKER}\narg1=123\nother log"
        with self.assertRaises(ValueError) as ctx:
            get_args_raw_text(content)
        self.assertIn("End arguments marker not found", str(ctx.exception))

    def test_empty_content_between_markers(self):
        content = f"{START_MARKER}\n{END_MARKER}"
        res = get_args_raw_text(content)
        self.assertEqual("", res.strip())


class TestParseArgumentLines(unittest.TestCase):
    def test_parse_bool_true(self):
        block = "  use_amp ......... True"
        res = parse_argument_lines(block)
        self.assertEqual(res["use_amp"], True)
        self.assertIsInstance(res["use_amp"], bool)

    def test_parse_bool_false(self):
        block = "  enable_checkpoint ......... False"
        res = parse_argument_lines(block)
        self.assertEqual(res["enable_checkpoint"], False)
        self.assertIsInstance(res["enable_checkpoint"], bool)

    def test_parse_int_positive(self):
        block = "  global_batch_size ......... 32"
        res = parse_argument_lines(block)
        self.assertEqual(res["global_batch_size"], 32)
        self.assertIsInstance(res["global_batch_size"], int)

    def test_parse_int_negative(self):
        block = "  offset ......... -10"
        res = parse_argument_lines(block)
        self.assertEqual(res["offset"], -10)
        self.assertIsInstance(res["offset"], int)

    def test_parse_float_normal(self):
        block = "  lr ......... 0.001"
        res = parse_argument_lines(block)
        self.assertEqual(res["lr"], 0.001)
        self.assertIsInstance(res["lr"], float)

    def test_parse_float_scientific(self):
        block = "  lr ......... 1e-5"
        res = parse_argument_lines(block)
        self.assertEqual(res["lr"], 1e-5)
        self.assertIsInstance(res["lr"], float)

    def test_parse_string_value(self):
        block = "  model_name ......... llama-7b"
        res = parse_argument_lines(block)
        self.assertEqual(res["model_name"], "llama-7b")
        self.assertIsInstance(res["model_name"], str)

    def test_skip_unmatched_lines(self):
        block = """
random garbage text
  max_epoch ......... 10
another invalid‑line
  device ......... npu
"""
        res = parse_argument_lines(block)
        self.assertEqual(res["max_epoch"], 10)
        self.assertEqual(res["device"], "npu")
        self.assertEqual(len(res), 2)

    def test_empty_input_string(self):
        res = parse_argument_lines("")
        self.assertEqual(res, {})

    def test_multiple_params(self):
        block = """
  lr ......... 0.01
  epoch ......... 20
  fp16 ......... True
  name ......... slime_model
"""
        res = parse_argument_lines(block)
        self.assertEqual(res["lr"], 0.01)
        self.assertEqual(res["epoch"], 20)
        self.assertEqual(res["fp16"], True)
        self.assertEqual(res["name"], "slime_model")


class TestSlimeGetConfigFilePath(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_existing_directory(self):
        npu_path, bench_path = slime_get_config_file_path(self.temp_dir)
        self.assertTrue(npu_path.endswith("NPU_config.json"))
        self.assertTrue(bench_path.endswith("bench_config.json"))

    def test_non_existing_directory_auto_create(self):
        new_sub = os.path.join(self.temp_dir, "sub_config")
        self.assertFalse(os.path.exists(new_sub))
        npu_path, bench_path = slime_get_config_file_path(new_sub)
        self.assertTrue(os.path.isdir(new_sub))
        self.assertTrue(npu_path.endswith("NPU_config.json"))
        self.assertTrue(bench_path.endswith("bench_config.json"))


class TestSlimeFilterConfigInfo(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.log_path = os.path.join(self.temp_dir, "slime_train.log")
        self.out_json = os.path.join(self.temp_dir, "result_config.json")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_filter_normal_log_success(self):
        log_content = (
            "pre‑log info\n"
            f"{START_MARKER}\n"
            "  lr ......... 0.005\n"
            "  max_step ......... 1000\n"
            "  use_npu ......... True\n"
            "  exp_name ......... slime_run01\n"
            f"{END_MARKER}\n"
            "post‑training log"
        )
        with open(self.log_path, "w", encoding="utf‑8") as f:
            f.write(log_content)

        result_dict = slime_filter_config_info(self.log_path, self.out_json)
        self.assertTrue(os.path.exists(self.out_json))

        with open(self.out_json, "r", encoding="utf‑8") as f:
            loaded = json.load(f)

        self.assertEqual(loaded["lr"], 0.005)
        self.assertEqual(loaded["max_step"], 1000)
        self.assertEqual(loaded["use_npu"], True)
        self.assertEqual(loaded["exp_name"], "slime_run01")
        self.assertEqual(result_dict, loaded)

    def test_extracted_block_empty_raise_value_error(self):
        log_content = (
            f"{START_MARKER}\n"
            f"{END_MARKER}"
        )
        with open(self.log_path, "w", encoding="utf‑8") as f:
            f.write(log_content)

        with self.assertRaises(ValueError):
            slime_filter_config_info(self.log_path, self.out_json)

    def test_missing_start_marker_raise(self):
        log_content = f"some log text\n{END_MARKER}"
        with open(self.log_path, "w", encoding="utf‑8") as f:
            f.write(log_content)
        with self.assertRaises(ValueError):
            slime_filter_config_info(self.log_path, self.out_json)

    @patch("msprobe.core.config_check.slime_param_compare.slime_log_filter.check_file_or_directory_path")
    def test_file_check_failed_raise_exception(self, mock_check: MagicMock):
        mock_check.side_effect = Exception("path invalid")
        with self.assertRaises(Exception):
            slime_filter_config_info(self.log_path, self.out_json)

