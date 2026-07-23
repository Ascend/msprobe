#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile
import unittest
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import numpy as np

from msprobe.core.common.utils import CompareException
from msprobe.core.compare import torchair_acc_cmp as tac


class TestTorchairAccCmp(unittest.TestCase):
    def test_get_rank_id_from_torchair_data_when_valid_worldsize_rank_then_pass(self):
        self.assertEqual(tac.get_rank_id_from_torchair_data("worldsize8_rank3"), 3)
        self.assertEqual(tac.get_rank_id_from_torchair_data("worldsize2_rank0"), 0)

    def test_get_rank_id_from_torchair_data_when_invalid_name_then_pass(self):
        self.assertEqual(tac.get_rank_id_from_torchair_data("rank3"), -1)
        self.assertEqual(tac.get_rank_id_from_torchair_data("worldsize_rankX"), -1)
        self.assertEqual(tac.get_rank_id_from_torchair_data("foo_worldsize8_rank"), -1)

    def test_get_unique_key_when_key_conflicts_then_pass(self):
        cur_dict = {"k": 1, "k#1": 2}
        new_key = tac.get_unique_key(cur_dict, "k")
        self.assertEqual(new_key, "k#2")
        cur_dict[new_key] = 3
        self.assertIn("k#2", cur_dict)

    def test_parse_pbtxt_to_dict_when_nested_blocks_then_pass(self):
        content = """
node {
  name: "node_a"
  attr {
    key: "k"
    value: "v"
  }
}
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.pbtxt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

            result = tac.parse_pbtxt_to_dict(path)

        self.assertIsInstance(result, list)
        self.assertGreaterEqual(len(result), 1)
        root = result[0]
        self.assertIn("node", root)
        self.assertEqual(root["node"]["name"], "node_a")
        self.assertEqual(root["node"]["attr"]["key"], "k")
        self.assertEqual(root["node"]["attr"]["value"], "v")

    def test_judge_single_or_multi_device_when_multi_device_in_root_then_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "device0"))
            os.makedirs(os.path.join(tmp, "device1"))

            self.assertTrue(tac.judge_single_or_multi_device(tmp))

    def test_judge_single_or_multi_device_when_time_based_multi_device_then_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            time_dir = "20250101120000"
            time_path = os.path.join(tmp, time_dir)
            os.makedirs(os.path.join(time_path, "0"))
            os.makedirs(os.path.join(time_path, "1"))

            self.assertTrue(tac.judge_single_or_multi_device(tmp))

    def test_judge_single_or_multi_device_when_single_device_then_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            time_dir = "20250101120000"
            time_path = os.path.join(tmp, time_dir)
            os.makedirs(os.path.join(time_path, "0"))

            self.assertFalse(tac.judge_single_or_multi_device(tmp))

    def test__has_rank_directory_when_rank_subdir_exists_then_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "worldsize2_rank0"))
            os.makedirs(os.path.join(tmp, "foo"))

            self.assertTrue(tac._has_rank_directory(tmp))

    def test__has_rank_directory_when_no_rank_subdir_then_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "foo"))

            self.assertFalse(tac._has_rank_directory(tmp))

    @patch("msprobe.core.compare.torchair_acc_cmp.FileChecker")
    @patch("msprobe.core.compare.torchair_acc_cmp.os.path.isdir")
    def test__validate_read_path_when_dir_and_file_then_pass(self, mock_isdir, mock_filechecker):
        instance = MagicMock()
        mock_filechecker.return_value = instance

        # directory path
        mock_isdir.return_value = True
        tac._validate_read_path("/tmp/dir")
        mock_filechecker.assert_called_with("/tmp/dir", tac.FileCheckConst.DIR, ability=tac.FileCheckConst.READ_ABLE)
        instance.common_check.assert_called_once()

        # file path
        instance.common_check.reset_mock()
        mock_isdir.return_value = False
        tac._validate_read_path("/tmp/file")
        mock_filechecker.assert_called_with("/tmp/file", tac.FileCheckConst.FILE, ability=tac.FileCheckConst.READ_ABLE)
        instance.common_check.assert_called_once()

    def test_get_torchair_ge_graph_path_when_rank_filtered_then_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            subdir = os.path.join(tmp, "sub")
            os.makedirs(subdir)
            # matching rank 0 file
            f_match = os.path.join(subdir, f"{tac.GE_GRAPH_FILE_PREFIX}20250101120000_rank_0_x.txt")
            with open(f_match, "w", encoding="utf-8"):
                pass
            # non-matching rank 1 file
            f_other = os.path.join(subdir, f"{tac.GE_GRAPH_FILE_PREFIX}20240101120000_rank_1_x.txt")
            with open(f_other, "w", encoding="utf-8"):
                pass

            result = tac.get_torchair_ge_graph_path(tmp, rank=0)

        self.assertEqual(result, [f_match])

    def test_get_torchair_ge_graph_path_when_not_directory_then_pass(self):
        with tempfile.NamedTemporaryFile() as tmp_file:
            result = tac.get_torchair_ge_graph_path(tmp_file.name)
        self.assertIsNone(result)

    @patch("msprobe.core.compare.torchair_acc_cmp.os.walk")
    @patch("msprobe.core.compare.torchair_acc_cmp.os.listdir")
    def test_gather_data_with_token_id_fx_when_rank_info_existed_then_pass(self, mock_listdir, mock_walk):
        # simulate data_path containing subdirs "0" and "1", each with one .npy file
        data_path = "/root"
        mock_walk.return_value = [(data_path, ["0", "1"], [])]

        def listdir_side_effect(path):
            if path == os.path.join(data_path, "0"):
                return ["a.npy"]
            if path == os.path.join(data_path, "1"):
                return ["b.npy"]
            return []

        mock_listdir.side_effect = listdir_side_effect

        gathered = tac.gather_data_with_token_id_fx(data_path, [], rank_info_existed=True)

        self.assertEqual(len(gathered), 1)
        gathered_map = gathered[0]
        # token id is basename(token_dir)+1
        self.assertIn(1, gathered_map)
        self.assertIn(2, gathered_map)
        self.assertEqual(gathered_map[1], [os.path.join(data_path, "0", "a.npy")])
        self.assertEqual(gathered_map[2], [os.path.join(data_path, "1", "b.npy")])

    def test_gather_data_with_token_id_fx_when_multiple_rank_token_groups_then_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            for image_name in ["image0", "image1"]:
                for token_id in ["0", "1"]:
                    token_path = os.path.join(tmp, image_name, token_id)
                    os.makedirs(token_path)
                    open(os.path.join(token_path, f"{image_name}_{token_id}.npy"), "wb").close()

            gathered = tac.gather_data_with_token_id_fx(tmp, [], rank_info_existed=True)

            self.assertEqual(len(gathered), 2)
            self.assertIn(1, gathered[0])
            self.assertIn(2, gathered[0])
            self.assertIn(os.path.join(tmp, "image0", "0", "image0_0.npy"), gathered[0][1])
            self.assertIn(os.path.join(tmp, "image0", "1", "image0_1.npy"), gathered[0][2])
            self.assertIn(1, gathered[1])
            self.assertIn(2, gathered[1])
            self.assertIn(os.path.join(tmp, "image1", "0", "image1_0.npy"), gathered[1][1])
            self.assertIn(os.path.join(tmp, "image1", "1", "image1_1.npy"), gathered[1][2])

    def test_gather_data_with_token_id_fx_when_multiple_dump_groups_then_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            for image_name in ["image0", "image1"]:
                for token_id in ["1", "2"]:
                    dump_path = os.path.join(tmp, image_name, token_id, "dump0")
                    os.makedirs(dump_path)
                    open(os.path.join(dump_path, f"{image_name}_{token_id}.npy"), "wb").close()

            gathered = tac.gather_data_with_token_id_fx(tmp, [], rank_info_existed=False)

            self.assertEqual(len(gathered), 2)
            self.assertIn(1, gathered[0])
            self.assertIn(2, gathered[0])
            self.assertIn(os.path.join(tmp, "image0", "1", "dump0", "image0_1.npy"), gathered[0][1])
            self.assertIn(os.path.join(tmp, "image0", "2", "dump0", "image0_2.npy"), gathered[0][2])
            self.assertIn(1, gathered[1])
            self.assertIn(2, gathered[1])
            self.assertIn(os.path.join(tmp, "image1", "1", "dump0", "image1_1.npy"), gathered[1][1])
            self.assertIn(os.path.join(tmp, "image1", "2", "dump0", "image1_2.npy"), gathered[1][2])

    def test_gather_data_with_token_id_fx_when_empty_token_dir_then_skip_token(self):
        # rank_info_existed=True: token dir with no .npy files should be skipped
        with tempfile.TemporaryDirectory() as tmp:
            token_empty = os.path.join(tmp, "0")
            token_full = os.path.join(tmp, "1")
            os.makedirs(token_empty)
            os.makedirs(token_full)
            open(os.path.join(token_full, "x.npy"), "wb").close()

            gathered = tac.gather_data_with_token_id_fx(tmp, [], rank_info_existed=True)

            self.assertEqual(len(gathered), 1)
            self.assertNotIn(1, gathered[0])  # empty token dir is skipped
            self.assertIn(2, gathered[0])
            self.assertEqual(gathered[0][2], [os.path.join(token_full, "x.npy")])

    def test_gather_data_with_token_id_fx_when_empty_dump_dir_then_skip_dump(self):
        # rank_info_existed=False: dump dir with no .npy files should be skipped
        with tempfile.TemporaryDirectory() as tmp:
            for token_id in ["1", "2"]:
                token_path = os.path.join(tmp, token_id)
                os.makedirs(token_path)
                if token_id == "1":
                    dump_path = os.path.join(token_path, "dump0")
                    os.makedirs(dump_path)
                    open(os.path.join(dump_path, "x.npy"), "wb").close()
                else:
                    empty_dump = os.path.join(token_path, "dump0")
                    os.makedirs(empty_dump)

            gathered = tac.gather_data_with_token_id_fx(tmp, [], rank_info_existed=False)

            self.assertEqual(len(gathered), 1)
            self.assertNotIn(2, gathered[0])  # token 2's only dump is empty, skipped
            self.assertIn(1, gathered[0])
            self.assertEqual(gathered[0][1], [os.path.join(tmp, "1", "dump0", "x.npy")])

    def test_gather_data_with_token_id_fx_when_inconsistent_dump_counts_then_use_min(self):
        # When tokens have different numbers of dump dirs, gather_data should
        # use the smallest count and emit a warning instead of silently skipping.
        with tempfile.TemporaryDirectory() as tmp:
            # token "1" has 2 dump dirs, token "2" has 1 dump dir
            for token_id, dump_count in [("1", 2), ("2", 1)]:
                token_path = os.path.join(tmp, token_id)
                os.makedirs(token_path)
                for i in range(dump_count):
                    dump_path = os.path.join(token_path, f"dump{i}")
                    os.makedirs(dump_path)
                    open(os.path.join(dump_path, f"{token_id}_dump{i}.npy"), "wb").close()

            gathered = tac.gather_data_with_token_id_fx(tmp, [], rank_info_existed=False)

            # num_dumps = min(2, 1) = 1
            self.assertEqual(len(gathered), 1)
            self.assertIn(1, gathered[0])
            self.assertIn(2, gathered[0])
            self.assertEqual(gathered[0][1], [os.path.join(tmp, "1", "dump0", "1_dump0.npy")])
            self.assertEqual(gathered[0][2], [os.path.join(tmp, "2", "dump0", "2_dump0.npy")])

    def test_gather_data_with_token_id_when_rank_info_existed_true_then_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = os.path.join(tmp, "parent")
            os.makedirs(parent)
            token0 = os.path.join(parent, "0")
            token1 = os.path.join(parent, "1")
            os.makedirs(token0)
            os.makedirs(token1)
            f0 = os.path.join(token0, "x.bin")
            f1 = os.path.join(token1, "y.bin")
            open(f0, "wb").close()
            open(f1, "wb").close()

            gathered = tac.gather_data_with_token_id(parent, fx=False, rank_info_existed=True)

            self.assertEqual(len(gathered), 1)
            subdir_map = gathered[0]
            self.assertIn(0, subdir_map)
            self.assertIn(1, subdir_map)
            self.assertIn(f0, subdir_map[0])
            self.assertIn(f1, subdir_map[1])

    def test_sort_ge_dump_data_when_reorder_ops_then_pass(self):
        dump_data = OrderedDict([("opB", "pathB"), ("opA", "pathA")])
        graph_map = [
            {"op": {"name": "opA"}},
            {"op": {"name": "opB"}},
        ]
        sorted_data = tac.sort_ge_dump_data(dump_data, graph_map)
        self.assertEqual(list(sorted_data.keys()), ["opA", "opB"])

    def test_sort_by_timestamp_when_paths_with_timestamps_then_pass(self):
        rows = [
            {"my_data_path": "Op.Op.1.2.200,inputs,0"},
            {"my_data_path": "Op.Op.1.2.100,inputs,0"},
        ]
        sorted_rows = tac.sort_by_timestamp(rows)
        self.assertEqual(sorted_rows[0]["my_data_path"], "Op.Op.1.2.100,inputs,0")

    def test_filter_valid_fx_desc_tensor_info_when_valid_and_invalid_then_pass(self):
        valid = {
            "key": "_fx_tensor_name",
            "value": {"s": "add_1-aten.add.Tensor.OUTPUT.0"},
        }
        self.assertTrue(tac.filter_valid_fx_desc_tensor_info("attr", valid))
        self.assertFalse(tac.filter_valid_fx_desc_tensor_info("other", valid))
        self.assertFalse(tac.filter_valid_fx_desc_tensor_info("attr", {"key": "wrong"}))

    def test_get_all_op_input_names_when_multiple_inputs_then_pass(self):
        op_info = {
            "input": "name1:0",
            "input#1": "name2:1",
            "other": "ignored",
        }
        names = tac.get_all_op_input_names(op_info)
        self.assertEqual(names, ["name1", "name2"])

    def test_find_longest_name_when_exact_and_prefix_then_pass(self):
        op_map = {"abc": "v1", "ab": "v2"}
        fused = {"xyz": "p"}
        ge = {"abc": "p"}

        self.assertEqual(tac.find_longest_name("abc", op_map, fused, ge), "abc")
        self.assertEqual(tac.find_longest_name("abx", op_map, fused, ge), "ab")

        # name hits fused/geo data but not op_map -> None
        fused2 = {"xyz": "p"}
        self.assertIsNone(tac.find_longest_name("xyzK", op_map, fused2, ge))

    @patch("msprobe.core.compare.torchair_acc_cmp.gather_data_with_token_id")
    def test_init_ge_dump_data_from_bin_path_when_valid_files_then_pass(self, mock_gather):
        mock_gather.return_value = [
            {
                0: [
                    "d/1/Add.Add_2.44.6.1706596912161941",
                    "d/1/Cast.Cast_9.19.6.1706596911887829",
                ]
            }
        ]
        result = tac.init_ge_dump_data_from_bin_path("dummy_path")

        self.assertEqual(len(result), 1)
        token_map = result[0]
        self.assertIn(0, token_map)
        op_map = token_map[0]
        self.assertEqual(op_map["Add_2"], "d/1/Add.Add_2.44.6.1706596912161941")
        self.assertEqual(op_map["Cast_9"], "d/1/Cast.Cast_9.19.6.1706596911887829")

    @patch("msprobe.core.compare.torchair_acc_cmp.gather_data_with_token_id")
    def test_init_fx_dump_data_from_path_when_valid_files_then_pass(self, mock_gather):
        mock_gather.return_value = [
            {
                1: [
                    "d/1/mm-aten.mm.default.INPUT.0.20240125031118787351.npy",
                    "d/1/mm-aten.mm.default.INPUT.1.20240125031118787351.npy",
                    "d/1/mm-aten.mm.default.OUTPUT.0.20240125031118787351.npy",
                ]
            }
        ]
        result = tac.init_fx_dump_data_from_path("dummy_path", rank_info_existed=False)

        self.assertEqual(len(result), 1)
        token_map = result[0]
        # FX token id is decremented by 1
        self.assertIn(0, token_map)
        op_map = token_map[0]
        self.assertIn("mm-aten.mm.default", op_map)
        self.assertEqual(len(op_map["mm-aten.mm.default"]["input"]), 2)
        self.assertEqual(len(op_map["mm-aten.mm.default"]["output"]), 1)

    @patch("msprobe.core.compare.torchair_acc_cmp.fill_row_data", return_value={"ok": True})
    @patch("msprobe.core.compare.torchair_acc_cmp.BasicDataInfo")
    def test_compare_single_data_when_basic_flow_then_pass(self, mock_basic, mock_fill):
        result = tac.compare_single_data("/golden", "/my", token_id=2, golden_data="g", my_data="m")

        mock_basic.assert_called_once_with("/golden", "/my", 2)
        mock_fill.assert_called_once()
        self.assertEqual(result, {"ok": True})

    @patch("msprobe.core.compare.torchair_acc_cmp.compare_single_data")
    def test_compare_specials_private_ops_when_inputs_outputs_then_pass(self, mock_compare_single):
        ge_inputs = ["gi0", "gi1"]
        ge_outputs = ["go0", "go1"]
        mock_compare_single.side_effect = ["row0", "row1"]

        result = tac.compare_specials_private_ops(ge_inputs, ge_outputs, token_id=3, my_path="/my")

        self.assertEqual(result, ["row0", "row1"])
        mock_compare_single.assert_any_call("/my,inputs,0", "/my,outputs,0", 3, "gi0", "go0")
        mock_compare_single.assert_any_call("/my,inputs,1", "/my,outputs,1", 3, "gi1", "go1")

    @patch("msprobe.core.compare.torchair_acc_cmp.compare_single_data")
    def test_compare_ops_when_inputs_outputs_then_pass(self, mock_compare_single):
        fx_inputs = ["fi0"]
        fx_outputs = ["fo0"]
        ge_inputs = ["gi0"]
        ge_outputs = ["go0"]
        mock_compare_single.side_effect = ["row_in", "row_out"]

        result = tac.compare_ops((fx_inputs, fx_outputs), (ge_inputs, ge_outputs), token_id=4, my_path="/my")

        self.assertEqual(result, ["row_in", "row_out"])
        mock_compare_single.assert_any_call("fi0", "/my,inputs,0", 4, my_data="gi0")
        mock_compare_single.assert_any_call("fo0", "/my,outputs,0", 4, my_data="go0")

    @patch("msprobe.core.compare.torchair_acc_cmp.compare_ops", return_value=["ok_row"])
    @patch("msprobe.core.compare.torchair_acc_cmp.parse_torchair_dump_data", return_value=(["gi"], ["go"]))
    def test_compare_ge_with_fx_single_op_when_valid_desc_then_pass(self, mock_parse, mock_compare_ops):
        op_info = {
            "output_desc": {
                "attr": {
                    "key": "_fx_tensor_name",
                    "value": {"s": "fx_tensor.OUTPUT.0"},
                }
            }
        }
        fx_dump_data = {
            "fx_tensor": {"input": ["fi"], "output": ["fo"]},
        }

        rows = tac.compare_ge_with_fx_single_op(op_info, fx_dump_data, "ge_op", "/my", token_id=1)

        self.assertEqual(rows, ["ok_row"])
        mock_compare_ops.assert_called_once()
        mock_parse.assert_called_once_with("/my")

    @patch("msprobe.core.compare.torchair_acc_cmp.compare_single_data", return_value="row")
    @patch("msprobe.core.compare.torchair_acc_cmp.parse_torchair_dump_data", return_value=(["gi"], ["go"]))
    def test_compare_ge_with_fx_multiple_ops_details_when_input_then_pass(self, mock_parse, mock_compare_single):
        op_info = {
            "output_desc": {
                "attr": {
                    "key": "_fx_tensor_name",
                    "value": {"s": "fx_tensor.OUTPUT.0"},
                }
            }
        }
        fx_dump_data = {
            "fx_tensor": {"input": ["fi"], "output": ["fo"]},
        }

        rows = tac.compare_ge_with_fx_multiple_ops_details(op_info, fx_dump_data, "ge_op", "/my", "input", 1)

        self.assertEqual(rows, ["row"])
        mock_compare_single.assert_called_once()
        mock_parse.assert_called_once_with("/my")

    @patch("msprobe.core.compare.torchair_acc_cmp.acc_compare")
    @patch("msprobe.core.compare.torchair_acc_cmp._has_rank_directory", return_value=False)
    @patch("msprobe.core.compare.torchair_acc_cmp._validate_read_path")
    @patch("msprobe.core.compare.torchair_acc_cmp.logger")
    def test_compare_torchair_mode_when_invalid_rank_then_raise(
        self, mock_logger, mock_validate, mock_has_rank_dir, mock_acc_compare
    ):
        args = SimpleNamespace(target_path="/tmp/my", golden_path="/tmp/golden", output_path="/tmp/out", rank="abc")

        with self.assertRaises(CompareException) as cm:
            tac.compare_torchair_mode(args)

        self.assertEqual(cm.exception.code, CompareException.INVALID_PARAM_ERROR)
        mock_logger.error.assert_called_once()
        mock_acc_compare.assert_not_called()
        mock_validate.assert_any_call(os.path.realpath(args.target_path))
        mock_validate.assert_any_call(os.path.realpath(args.golden_path))

    @patch("msprobe.core.compare.torchair_acc_cmp.acc_compare", return_value="ok")
    @patch("msprobe.core.compare.torchair_acc_cmp._has_rank_directory", return_value=False)
    @patch("msprobe.core.compare.torchair_acc_cmp._validate_read_path")
    @patch("msprobe.core.compare.torchair_acc_cmp.logger")
    def test_compare_torchair_mode_when_rank_without_rank_dirs_then_pass(
        self, mock_logger, mock_validate, mock_has_rank_dir, mock_acc_compare
    ):
        args = SimpleNamespace(target_path="/tmp/my", golden_path="/tmp/golden", output_path="/tmp/out", rank="1")

        result = tac.compare_torchair_mode(args)

        self.assertEqual(result, "ok")
        mock_logger.warning.assert_called_once()
        mock_acc_compare.assert_called_once()
        _, kwargs = mock_acc_compare.call_args
        # acc_compare(golden_path, my_path, output_path, rank_id, rank_info_existed)
        self.assertEqual(len(mock_acc_compare.call_args[0]), 5)

    @patch("msprobe.core.compare.torchair_acc_cmp.acc_compare", return_value="ok")
    @patch("msprobe.core.compare.torchair_acc_cmp._has_rank_directory", return_value=True)
    @patch("msprobe.core.compare.torchair_acc_cmp._validate_read_path")
    @patch("msprobe.core.compare.torchair_acc_cmp.logger")
    def test_compare_torchair_mode_when_no_rank_then_pass(
        self, mock_logger, mock_validate, mock_has_rank_dir, mock_acc_compare
    ):
        args = SimpleNamespace(target_path="/tmp/my", golden_path="/tmp/golden", output_path="/tmp/out")

        result = tac.compare_torchair_mode(args)

        self.assertEqual(result, "ok")
        mock_logger.warning.assert_not_called()
        mock_acc_compare.assert_called_once()
        # rank_id should be None when args.rank is missing
        self.assertIsNone(mock_acc_compare.call_args[0][3])

    @patch("msprobe.core.compare.torchair_acc_cmp.get_torchair_ge_graph_path", return_value=[])
    def test_acc_compare_once_when_no_ge_graph_then_raise(self, mock_get_graph):
        args = ("/golden", "/my", "/out", -1)

        with self.assertRaises(Exception) as cm:
            tac.acc_compare_once(args)

        self.assertIn("Can not get ge graph", str(cm.exception))
        mock_get_graph.assert_called_once_with("/my", -1)

    @patch("msprobe.core.compare.torchair_acc_cmp.set_msaccucmp_path_from_cann")
    @patch("msprobe.core.compare.torchair_acc_cmp.get_torchair_ge_graph_path", return_value=None)
    def test_acc_compare_when_no_ge_graph_then_raise(self, mock_get_graph, mock_set_path):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(Exception) as cm:
                tac.acc_compare(tmp, tmp, "/out", rank_id=None, rank_info_existed=False)
        self.assertIn("Can not get ge graph", str(cm.exception))
        mock_get_graph.assert_called_once()
        mock_set_path.assert_called_once()

    @patch("msprobe.core.compare.torchair_acc_cmp.set_msaccucmp_path_from_cann")
    @patch("msprobe.core.compare.torchair_acc_cmp.acc_compare_once")
    @patch("msprobe.core.compare.torchair_acc_cmp.get_torchair_ge_graph_path", return_value=["graph.pbtxt"])
    def test_acc_compare_when_single_rank_and_rank_info_existed_then_pass(self, mock_get_graph, mock_acc_once, mock_set_path):
        with tempfile.TemporaryDirectory() as golden, tempfile.TemporaryDirectory() as my:
            os.makedirs(os.path.join(golden, "worldsize2_rank0"))
            os.makedirs(os.path.join(my, "worldsize2_rank0"))

            out = tac.acc_compare(golden, my, "/out", rank_id=None, rank_info_existed=True)

        self.assertEqual(out, "/out")
        mock_acc_once.assert_called_once()
        mock_set_path.assert_called_once()

    @patch("msprobe.core.compare.torchair_acc_cmp.acc_compare_once", side_effect=RuntimeError("inner_error"))
    def test_save_compare_once_when_acc_compare_once_raises_then_pass(self, mock_acc_once):
        with self.assertRaises(ValueError) as cm:
            tac.save_compare_once(("/golden", "/my", "/out", -1))
        self.assertIn("Error in acc_compare_once: inner_error", str(cm.exception))

    def test_gather_fused_op_data_when_partial_ge_and_missing_ge_then_pass(self):
        fused_op_name = "OpAOpB"
        op_map = {
            "OpA": {
                "input": "x:0",
                "input#1": "OpB:0",
            },
            "OpB": {
                "input": "y:0",
            },
        }
        fused_ge_dump_data = {}
        ge_dump_data = {
            "OpA": "/path/opA.bin",
        }
        with patch(
            "msprobe.core.compare.torchair_acc_cmp.parse_torchair_dump_data",
            return_value=([np.array([1.0], dtype="float32")], [np.array([2.0], dtype="float32")]),
        ) as mock_parse, patch("msprobe.core.compare.torchair_acc_cmp.logger") as mock_logger:
            golden_inputs_tuple, golden_outputs_tuple = tac.gather_fused_op_data(
                fused_op_name, op_map, fused_ge_dump_data, ge_dump_data
            )

        inputs, input_paths = golden_inputs_tuple
        outputs, output_path = golden_outputs_tuple
        # Should contain two inputs: one from OpA (non-fused) and one empty for OpB
        self.assertEqual(len(inputs), 2)
        self.assertEqual(len(input_paths), 2)
        self.assertTrue(input_paths[0].startswith("/path/opA.bin,inputs,"))
        self.assertEqual(input_paths[1], "")
        self.assertEqual(len(outputs), 1)
        self.assertEqual(output_path, "/path/opA.bin")
        mock_parse.assert_called_once_with("/path/opA.bin")
        mock_logger.warning.assert_called()

    @patch("msprobe.core.compare.torchair_acc_cmp.compare_specials_private_ops", return_value=["sp_row"])
    @patch("msprobe.core.compare.torchair_acc_cmp.compare_ge_with_fx_single_op", return_value=["single_row"])
    @patch("msprobe.core.compare.torchair_acc_cmp.compare_ge_with_fx_multiple_ops", return_value=["multi_row"])
    @patch("msprobe.core.compare.torchair_acc_cmp.get_all_ops_from_fusion_op")
    @patch("msprobe.core.compare.torchair_acc_cmp.sort_ge_dump_data")
    def test_compare_ge_with_fx_when_single_multi_and_special_ops_then_pass(
        self, mock_sort, mock_get_all_ops, mock_multi, mock_single, mock_special
    ):
        graph_map = [{"op": {"name": "op1"}}, {"op": {"name": "op2"}}, {"op": {"name": "Cast_0"}}]
        ge_dump_data = OrderedDict(
            [
                ("op1", "/ge/op1.bin"),
                ("op2op3", "/ge/op2op3.bin"),
                ("Cast_0", "/ge/Cast_0.bin"),
            ]
        )
        fx_dump_data = {"fx": "dummy"}
        mock_sort.return_value = ge_dump_data

        def get_all_ops_side_effect(op_name, *_):
            if op_name == "op1":
                return ["op1"]
            if op_name == "op2op3":
                return ["op2", "op3"]
            return []

        mock_get_all_ops.side_effect = get_all_ops_side_effect
        with patch(
            "msprobe.core.compare.torchair_acc_cmp.parse_torchair_dump_data",
            return_value=(["gi"], ["go"]),
        ) as mock_parse:
            rows = tac.compare_ge_with_fx(graph_map, ge_dump_data, fx_dump_data, token_id=5)

        # At least single and multi-op branches should contribute rows; Cast branch is also exercised
        self.assertIn("single_row", rows)
        self.assertIn("multi_row", rows)
        mock_single.assert_called_once()
        mock_multi.assert_called_once()
        mock_special.assert_called_once()
        mock_parse.assert_called_once_with("/ge/Cast_0.bin")

    @patch("msprobe.core.compare.torchair_acc_cmp.save_compare_result_to_csv")
    @patch("msprobe.core.compare.torchair_acc_cmp.sort_by_timestamp", side_effect=lambda x: x)
    @patch("msprobe.core.compare.torchair_acc_cmp.compare_ge_with_ge", return_value=["ge_row"])
    @patch("msprobe.core.compare.torchair_acc_cmp.compare_ge_with_fx", return_value=["fx_row"])
    @patch("msprobe.core.compare.torchair_acc_cmp.init_ge_dump_data_from_bin_path")
    @patch("msprobe.core.compare.torchair_acc_cmp.init_fx_dump_data_from_path")
    @patch("msprobe.core.compare.torchair_acc_cmp.parse_pbtxt_to_dict", return_value=[{"op": {"name": "op1"}}])
    @patch("msprobe.core.compare.torchair_acc_cmp.get_torchair_ge_graph_path")
    @patch("msprobe.core.compare.torchair_acc_cmp.logger")
    def test_acc_compare_once_when_rank_and_ge_vs_fx_and_ge_vs_ge_then_pass(
        self,
        mock_logger,
        mock_get_graph,
        mock_parse_pbtxt,
        mock_init_fx,
        mock_init_ge,
        mock_compare_fx,
        mock_compare_ge,
        mock_sort_ts,
        mock_save_csv,
    ):
        golden_root = "/golden"
        my_root = "/my"
        output_path = "/out"
        rank_id = -1

        mock_get_graph.side_effect = [
            ["/graph_rank.pbtxt"],  # for dir_of_my_path with rank_id
            None,  # for dir_of_golden_path -> is_golden_fx True
        ]

        mock_init_ge.return_value = [{0: {"op1": "/my/op1.bin"}}]
        mock_init_fx.return_value = [{0: {"op1": "/golden/op1.bin"}}]

        args = (golden_root, my_root, output_path, rank_id)
        tac.acc_compare_once(args)

        mock_compare_fx.assert_called_once()
        mock_compare_ge.assert_not_called()
        mock_save_csv.assert_called_once()

        # Now test is_golden_fx False path
        mock_get_graph.reset_mock()
        mock_get_graph.side_effect = [
            ["/graph_rank.pbtxt"],  # for dir_of_my_path with rank_id
            ["/graph_rank_golden.pbtxt"],  # for dir_of_golden_path -> is_golden_fx False
        ]
        mock_compare_fx.reset_mock()
        mock_compare_ge.reset_mock()

        tac.acc_compare_once(args)

        mock_compare_ge.assert_called_once()
        mock_compare_fx.assert_not_called()


if __name__ == "__main__":
    unittest.main()
