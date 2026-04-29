#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import numpy as np
import torch

from msprobe.core.compare import torchair_cmp_utils as utils


class TestBasicDataInfo(unittest.TestCase):
    @patch.object(utils.BasicDataInfo, "_check_path")
    def test_BasicDataInfo_get_token_id_when_history_dump_structure_then_pass(self, mock_check):
        # msit_dump/tensors/xxx/123 so that:
        # dirseg[-3] == 'tensors' and dirseg[-4].startswith('msit_dump')
        path = os.path.join("msit_dump", "tensors", "xxx", "123")
        info = utils.BasicDataInfo.__new__(utils.BasicDataInfo)
        token_id = info.get_token_id(path)
        self.assertEqual(token_id, 123)

    def test_BasicDataInfo_get_token_id_when_depth_too_deep_then_raise(self):
        info = utils.BasicDataInfo.__new__(utils.BasicDataInfo)
        deep_path = os.sep.join(["a"] * 17)
        with self.assertRaises(RecursionError):
            info.get_token_id(deep_path)

    def test_BasicDataInfo__validate_op_type_when_valid_tuple_then_pass(self):
        op_type = ("add", "add_ref")
        my_type, golden_type = utils.BasicDataInfo._validate_op_type(op_type)
        self.assertEqual(my_type, "add")
        self.assertEqual(golden_type, "add_ref")

    def test_BasicDataInfo__validate_op_type_when_invalid_then_raise(self):
        with self.assertRaises(ValueError):
            utils.BasicDataInfo._validate_op_type("add")


class TestFillRowDataAndLoad(unittest.TestCase):
    @patch.object(utils.BasicDataInfo, "_check_path")
    def test_fill_row_data_when_golden_path_not_file_then_pass(self, mock_check):
        data_info = utils.BasicDataInfo.__new__(utils.BasicDataInfo)
        data_info.golden_data_path = "/not_exist_golden"
        data_info.my_data_path = "/not_exist_my"
        data_info.token_id = 0
        data_info.data_id = 0
        data_info.my_op_type = None
        data_info.golden_op_type = None
        data_info.to_dict = lambda: {
            utils.TOKEN_ID: "0",
            utils.DATA_ID: "0",
            utils.GOLDEN_DATA_PATH: data_info.golden_data_path,
            utils.MY_DATA_PATH: data_info.my_data_path,
        }

        row = utils.fill_row_data(data_info)
        self.assertIn(utils.CMP_FAIL_REASON, row)
        self.assertIn("golden_data_path", row[utils.CMP_FAIL_REASON])

    @patch.object(utils.BasicDataInfo, "_check_path")
    def test_fill_row_data_when_my_path_not_file_then_pass(self, mock_check):
        data_info = utils.BasicDataInfo.__new__(utils.BasicDataInfo)
        data_info.golden_data_path = __file__  # existing file
        data_info.my_data_path = "/not_exist_my"
        data_info.token_id = 0
        data_info.data_id = 0
        data_info.my_op_type = None
        data_info.golden_op_type = None
        data_info.to_dict = lambda: {
            utils.TOKEN_ID: "0",
            utils.DATA_ID: "0",
            utils.GOLDEN_DATA_PATH: data_info.golden_data_path,
            utils.MY_DATA_PATH: data_info.my_data_path,
        }

        row = utils.fill_row_data(data_info)
        self.assertIn(utils.CMP_FAIL_REASON, row)
        self.assertIn("my_data_path", row[utils.CMP_FAIL_REASON])

    @patch.object(utils, "set_tensor_basic_info_in_row_data", return_value={"golden_dtype": "torch.float32"})
    @patch.object(utils, "compare_data", return_value={"cmp_metric": True})
    def test_fill_row_data_when_loaded_tensors_provided_then_pass(self, mock_compare, mock_basic_info):
        data_info = utils.BasicDataInfo.__new__(utils.BasicDataInfo)
        data_info.golden_data_path = "/any_golden"
        data_info.my_data_path = "/any_my"
        data_info.token_id = 1
        data_info.data_id = 2
        data_info.my_op_type = "add"
        data_info.golden_op_type = "add_ref"
        data_info.to_dict = lambda: {
            utils.TOKEN_ID: "1",
            utils.DATA_ID: "2",
            utils.GOLDEN_DATA_PATH: data_info.golden_data_path,
            utils.MY_DATA_PATH: data_info.my_data_path,
            utils.GOLDEN_OP_TYPE: data_info.golden_op_type,
            utils.MY_OP_TYPE: data_info.my_op_type,
        }

        golden = torch.ones((2, 2), dtype=torch.float32)
        my = torch.ones((2, 2), dtype=torch.float32)
        row = utils.fill_row_data(data_info, loaded_my_data=my, loaded_golden_data=golden)

        self.assertEqual(row[utils.TOKEN_ID], "1")
        self.assertEqual(row["cmp_metric"], True)
        self.assertEqual(row["golden_dtype"], "torch.float32")
        mock_compare.assert_called_once()
        mock_basic_info.assert_called_once()

    def test_load_as_torch_tensor_when_numpy_with_unsupported_dtype_then_pass(self):
        arr = np.array([1, 2], dtype=np.uint16)
        tensor = utils.load_as_torch_tensor("/dummy", loaded_data=arr)
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(str(tensor.dtype), "torch.int32")

    def test_load_as_torch_tensor_when_torch_tensor_then_pass(self):
        t = torch.ones(3, dtype=torch.float16)
        tensor = utils.load_as_torch_tensor("/dummy", loaded_data=t)
        self.assertIs(tensor, t)

    @patch.object(utils, "read_data")
    def test_load_as_torch_tensor_when_loaded_data_is_none_then_pass(self, mock_read):
        mock_read.return_value = torch.zeros(1)
        tensor = utils.load_as_torch_tensor("/dummy", loaded_data=None)
        self.assertTrue(torch.equal(tensor, torch.zeros(1)))
        mock_read.assert_called_once_with("/dummy")


class TestTensorInfoAndCsv(unittest.TestCase):
    def test_get_tensor_basic_info_when_non_empty_tensor_then_pass(self):
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        info = utils.get_tensor_basic_info(t)
        self.assertEqual(info[utils.DTYPE], "torch.float32")
        self.assertEqual(info[utils.SHAPE], str([2, 2]))
        self.assertEqual(info[utils.MAX_VALUE], 4.0)
        self.assertEqual(info[utils.MIN_VALUE], 1.0)
        self.assertAlmostEqual(info[utils.MEAN_VALUE], 2.5)

    def test_get_tensor_basic_info_when_zero_dim_tensor_then_pass(self):
        t = torch.empty((0, 3), dtype=torch.float32)
        info = utils.get_tensor_basic_info(t)
        self.assertEqual(info[utils.DTYPE], "torch.float32")
        self.assertEqual(info[utils.SHAPE], str([0, 3]))
        self.assertNotIn(utils.MAX_VALUE, info)

    def test_set_tensor_basic_info_in_row_data_when_two_tensors_then_pass(self):
        golden = torch.ones((1, 1), dtype=torch.float32)
        my = torch.ones((1, 1), dtype=torch.float32) * 2
        row = utils.set_tensor_basic_info_in_row_data(golden, my)
        self.assertIn("golden_dtype", row)
        self.assertIn("my_dtype", row)

    @patch("msprobe.core.compare.torchair_cmp_utils.write_df_to_csv")
    @patch("msprobe.core.compare.torchair_cmp_utils.create_directory")
    def test_save_compare_result_to_csv_when_filter_and_int8_excluded_then_pass(
        self, mock_create_dir, mock_write_csv
    ):
        rows = [
            {  # shape mismatch row should be filtered out
                utils.GOLDEN_DTYPE: "torch.float32",
                utils.MY_DTYPE: "torch.float32",
                utils.CMP_FAIL_REASON: "data shape doesn't match.",
            },
            {  # mixed int8 row should be removed
                utils.GOLDEN_DTYPE: "torch.int8",
                utils.MY_DTYPE: "torch.float32",
                utils.CMP_FAIL_REASON: "",
            },
            {  # valid row
                utils.GOLDEN_DTYPE: "torch.float32",
                utils.MY_DTYPE: "torch.float32",
                utils.CMP_FAIL_REASON: "",
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = utils.save_compare_result_to_csv(rows, output_path=tmp, rank_id=0)

        mock_create_dir.assert_called_once_with(tmp)
        self.assertIn("msprobe_cmp_report_rank0_", csv_path)
        # capture the DataFrame passed into write_df_to_csv
        df_arg = mock_write_csv.call_args[0][0]
        self.assertEqual(len(df_arg), 1)


class TestCompareDataAndRead(unittest.TestCase):
    @patch.object(utils, "compare_tensor", return_value={"cmp": True})
    @patch.object(utils, "logger")
    def test_compare_data_when_non_fp32_inputs_then_pass(self, mock_logger, mock_compare_tensor):
        golden = torch.ones((2, 2), dtype=torch.float16)
        my = torch.ones((2, 2), dtype=torch.int32)

        result = utils.compare_data(golden, my)

        self.assertEqual(result, {"cmp": True})
        mock_compare_tensor.assert_called_once()
        # logger.debug should have been called for both tensors
        self.assertGreaterEqual(mock_logger.debug.call_count, 1)

    @patch("msprobe.core.compare.torchair_cmp_utils.Rule")
    def test_read_data_when_npy_file_then_pass(self, mock_rule):
        mock_rule.input_file.return_value.check.return_value = None
        with tempfile.TemporaryDirectory() as tmp:
            npy_path = os.path.join(tmp, "arr.npy")
            np.save(npy_path, np.array([1, 2], dtype=np.float32))

            tensor = utils.read_data(npy_path)

        self.assertIsInstance(tensor, torch.Tensor)
        self.assertTrue(torch.equal(tensor, torch.tensor([1.0, 2.0])))

    @patch("msprobe.core.compare.torchair_cmp_utils.Rule")
    @patch("msprobe.core.compare.torchair_cmp_utils.safe_torch_load")
    def test_read_data_when_pt_file_then_pass(self, mock_safe_load, mock_rule):
        mock_rule.input_file.return_value.check.return_value = None
        data = np.array([3, 4], dtype=np.float32)
        mock_safe_load.return_value = data
        with tempfile.TemporaryDirectory() as tmp:
            pt_path = os.path.join(tmp, "arr.pt")
            open(pt_path, "wb").close()

            tensor = utils.read_data(pt_path)

        self.assertTrue(torch.equal(tensor, torch.tensor([3.0, 4.0])))

    @patch("msprobe.core.compare.torchair_cmp_utils.Rule")
    @patch.object(utils, "logger")
    def test_read_data_when_unsupported_suffix_then_raise(self, mock_logger, mock_rule):
        mock_rule.input_file.return_value.check.return_value = None
        with tempfile.TemporaryDirectory() as tmp:
            bin_path = os.path.join(tmp, "arr.bin")
            open(bin_path, "wb").close()

            with self.assertRaises(TypeError):
                utils.read_data(bin_path)

        mock_logger.error.assert_called_once()

    @patch("msprobe.core.compare.torchair_cmp_utils.Rule")
    @patch("msprobe.core.compare.torchair_cmp_utils.safe_torch_load")
    @patch.object(utils, "logger")
    def test_read_data_when_unsupported_content_type_then_raise(self, mock_logger, mock_safe_load, mock_rule):
        mock_rule.input_file.return_value.check.return_value = None
        mock_safe_load.return_value = {"not": "tensor"}
        with tempfile.TemporaryDirectory() as tmp:
            pt_path = os.path.join(tmp, "arr.pt")
            open(pt_path, "wb").close()

            with self.assertRaises(TypeError):
                utils.read_data(pt_path)

        mock_logger.error.assert_called_once()


class TestCompareTensorAndCheck(unittest.TestCase):
    @patch.object(utils, "check_tensor", return_value=(False, "data shape doesn't match."))
    @patch.object(utils, "logger")
    def test_compare_tensor_when_check_tensor_fail_then_pass(self, mock_logger, mock_check):
        golden = torch.ones(3)
        my = torch.ones(4)

        result = utils.compare_tensor(golden, my)

        self.assertEqual(result[utils.CMP_FAIL_REASON], "data shape doesn't match.")
        mock_logger.debug.assert_called_once()

    @patch(
        "msprobe.core.compare.torchair_cmp_utils._lazy_cmp_alg_maps",
        return_value=({"alg": lambda a, b: (False, "alg_warn")}, {"cust": lambda a, b: (True, "custom_warn")}),
    )
    def test_compare_tensor_when_all_algorithms_run_then_pass(self, *_):
        golden = torch.ones(3)
        my = torch.ones(3)

        result = utils.compare_tensor(golden, my)

        self.assertIn("alg", result)
        self.assertIn("cust", result)
        self.assertIn("alg_warn", result[utils.CMP_FAIL_REASON])
        self.assertIn("custom_warn", result[utils.CMP_FAIL_REASON])

    def test_check_tensor_when_all_conditions_then_pass(self):
        golden = torch.tensor([1.0, 2.0])
        my = torch.tensor([1.0, 2.0])

        ok, msg = utils.check_tensor(golden, my)
        self.assertTrue(ok)
        self.assertEqual(msg, "")

        # length mismatch
        ok, msg = utils.check_tensor(golden, my[:1])
        self.assertFalse(ok)
        self.assertIn("data shape doesn't match.", msg)

        # nan in golden
        golden_nan = torch.tensor([1.0, float("nan")])
        ok, msg = utils.check_tensor(golden_nan, my)
        self.assertFalse(ok)
        self.assertIn("golden_data includes NAN or inf.", msg)

        # inf in my
        my_inf = torch.tensor([1.0, float("inf")])
        ok, msg = utils.check_tensor(golden, my_inf)
        self.assertFalse(ok)
        self.assertIn("my_data includes NAN or inf.", msg)


if __name__ == "__main__":
    unittest.main()
