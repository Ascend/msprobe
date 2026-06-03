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

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch, mock_open

import numpy as np
import pandas as pd

from msprobe.core.common.const import Const as CoreConst
from msprobe.core.common.const import FileCheckConst
from msprobe.mindspore.dump.dump_processor.cell_dump_process import (
    CellDumpConfig,
    gen_file_path,
    need_tensordump_in,
    sort_filenames,
    rename_filename,
    get_cell_name,
    get_data_mode,
    check_relation,
    get_parent_cell_name,
    get_construct,
    generate_construct,
    process_file,
    custom_sort,
    convert_special_values,
    process_csv,
    generate_dump_info,
    generate_stack_info,
    is_download_finished,
    process_step,
    remove_trailing_commas,
    merge_file,
    process_statistics_step,
    get_yaml_keys,
    get_tensordump_mode,
    str_to_list,
    set_tensordump_mode,
    create_kbyk_json,
    start,
    dump_task,
    construct,
    cell_list,
    free_cells,
    parent_cell_types,
)


class TestCellDumpConfig(unittest.TestCase):
    def test_init_with_defaults(self):
        mock_net = MagicMock()
        config = CellDumpConfig(
            net=mock_net,
            dump_path="/tmp/dump",
            data_mode="all"
        )
        self.assertIs(config.net, mock_net)
        self.assertEqual(config.dump_path, "/tmp/dump")
        self.assertEqual(config.data_mode, "all")
        self.assertEqual(config.task, CoreConst.STATISTICS)
        self.assertIsNone(config.summary_mode)
        self.assertEqual(config.step, 0)

    def test_init_with_all_params(self):
        config = CellDumpConfig(
            net=MagicMock(),
            dump_path="/tmp/dump",
            data_mode="forward",
            task=CoreConst.TENSOR,
            summary_mode=["max", "min"],
            step=5
        )
        self.assertEqual(config.task, CoreConst.TENSOR)
        self.assertEqual(config.summary_mode, ["max", "min"])
        self.assertEqual(config.step, 5)
        self.assertEqual(config.data_mode, "forward")


class TestGenFilePath(unittest.TestCase):
    def test_gen_file_path_tensor(self):
        with patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.dump_task", CoreConst.TENSOR):
            result = gen_file_path("/tmp/dump", "Cell.net1.Class1", "forward", "input", 0)
        expected_suffix = "forward.input.0"
        self.assertIn("{step}", result)
        self.assertIn("{rank}", result)
        self.assertIn(CoreConst.DUMP_TENSOR_DATA, result)
        self.assertIn("Cell.net1.Class1", result)
        self.assertIn("forward.input.0", result)

    def test_gen_file_path_statistics(self):
        with patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.dump_task", CoreConst.STATISTICS):
            result = gen_file_path("/tmp/dump", "Cell.net1.Class1", "forward", "output", 1)
        self.assertIn(CoreConst.HYPHEN, result)
        self.assertIn("forward-output-1", result)


class TestNeedTensorDumpIn(unittest.TestCase):
    def test_no_attr(self):
        cell = MagicMock(spec=[])
        self.assertFalse(need_tensordump_in(cell, "input_dump_mode", 0))

    def test_index_out_of_range(self):
        cell = MagicMock()
        cell.input_dump_mode = ["out"]
        self.assertFalse(need_tensordump_in(cell, "input_dump_mode", 1))

    def test_value_not_in(self):
        cell = MagicMock()
        cell.input_dump_mode = ["out", "out"]
        self.assertFalse(need_tensordump_in(cell, "input_dump_mode", 0))

    def test_value_is_in(self):
        cell = MagicMock()
        cell.input_dump_mode = ["in", "out"]
        self.assertTrue(need_tensordump_in(cell, "input_dump_mode", 0))


class TestSortFilenames(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_sort_filenames_by_id(self):
        filenames = [
            "Cell.net1.Class1.forward.0.input.0_float32_10.npy",
            "Cell.net1.Class1.forward.0.input.0_float32_2.npy",
            "Cell.net1.Class1.forward.0.input.0_float32_5.npy",
        ]
        for f in filenames:
            open(os.path.join(self.temp_dir, f), 'w').close()

        with patch("os.listdir", return_value=filenames):
            result = sort_filenames(self.temp_dir)
        self.assertEqual(result[0], filenames[1])
        self.assertEqual(result[1], filenames[2])
        self.assertEqual(result[2], filenames[0])

    def test_sort_filenames_ignore_invalid(self):
        filenames = [
            "invalid_file.txt",
            "Cell.net1.Class1.forward.0.input.0_float32_2.npy",
        ]
        with patch("os.listdir", return_value=filenames):
            with patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.logger") as mock_logger:
                result = sort_filenames(self.temp_dir)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], filenames[1])
        mock_logger.warning.assert_called_once()


class TestRenameFilename(unittest.TestCase):
    def test_rename_filename_tensor(self):
        with patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.dump_task", CoreConst.TENSOR):
            filenames = [
                "Cell.net1.Class1.forward.0.input.0_float32_1.npy",
                "Cell.net1.Class1.forward.0.output.0_float32_2.npy",
            ]
            with patch("os.listdir", return_value=filenames), \
                 patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.move_file") as mock_move:
                rename_filename(path="/fake/path")
            self.assertEqual(mock_move.call_count, 2)

    def test_rename_filename_statistics(self):
        with patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.dump_task", CoreConst.STATISTICS):
            df = pd.DataFrame({
                CoreConst.OP_NAME: [
                    "Cell.net1.Class1.forward.0.input.0",
                    "Cell.net1.Class1.backward.0.output.1",
                ]
            })
            rename_filename(data_df=df)
        self.assertIn("forward.0.0", df[CoreConst.OP_NAME].iloc[0])
        self.assertIn("backward.0.0", df[CoreConst.OP_NAME].iloc[1])

    def test_rename_filename_statistics_duplicate_name(self):
        with patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.dump_task", CoreConst.STATISTICS), \
             patch("os.listdir", return_value=[]):
            df = pd.DataFrame({
                CoreConst.OP_NAME: [
                    "Cell.net1.Class1.forward.0.input.0",
                    "Cell.net1.Class1.forward.0.input.1",
                ]
            })
            rename_filename(data_df=df)
        self.assertIn("forward.0.0", df[CoreConst.OP_NAME].iloc[0])
        self.assertIn("forward.0.0", df[CoreConst.OP_NAME].iloc[1])

    def test_rename_filename_tensor_duplicate(self):
        with patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.dump_task", CoreConst.TENSOR):
            filenames = [
                "Cell.net1.Class1.forward.0.input.0_float32_1.npy",
                "Cell.net1.Class1.forward.0.input.0_float32_2.npy",
            ]
            with patch("os.listdir", return_value=filenames), \
                 patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.move_file") as mock_move:
                rename_filename(path="/fake/path")
            self.assertEqual(mock_move.call_count, 2)
            first_dst = mock_move.call_args_list[0][0][1]
            self.assertIn("forward.0.0", first_dst)
            second_dst = mock_move.call_args_list[1][0][1]
            self.assertIn("forward.1.0", second_dst)


class TestGetCellName(unittest.TestCase):
    def test_get_cell_name_valid(self):
        result = get_cell_name("Cell.net1.subnet.Class1.forward.0.input.0")
        self.assertEqual(result, "net1.subnet.Class1.forward")

    def test_get_cell_name_short(self):
        result = get_cell_name("Cell.net1.forward.0")
        self.assertEqual(result, "")

    def test_get_cell_name_too_short(self):
        result = get_cell_name("Cell.net1.forward")
        self.assertIsNone(result)

    def test_get_cell_name_none(self):
        with self.assertRaises(AttributeError):
            get_cell_name(None)


class TestGetDataMode(unittest.TestCase):
    def test_get_data_mode_forward(self):
        result = get_data_mode("Cell.net1.Class1.forward.0")
        self.assertEqual(result, "forward")

    def test_get_data_mode_backward(self):
        result = get_data_mode("Cell.net1.Class1.backward.0")
        self.assertEqual(result, "backward")


class TestCheckRelation(unittest.TestCase):
    def test_direct_parent(self):
        self.assertTrue(check_relation("net1.subnet.class1", "net1.subnet"))

    def test_layers_pattern_parent(self):
        self.assertTrue(check_relation("net1.subnet.class1.layers.0", "net1.subnet.class1"))

    def test_not_related(self):
        self.assertFalse(check_relation("net1.subnet1", "net1.subnet2"))

    def test_no_dot(self):
        self.assertFalse(check_relation("net1", "parent"))


class TestGetParentCellName(unittest.TestCase):
    def test_layers_pattern(self):
        result = get_parent_cell_name("net1.subnet.class1.layers.0")
        self.assertEqual(result, "net1.subnet.class1")

    def test_normal_pattern(self):
        result = get_parent_cell_name("net1.subnet.class1")
        self.assertEqual(result, "net1.subnet")


class TestGetConstruct(unittest.TestCase):
    def test_get_construct_with_parent_child_relation(self):
        cell_list_input = [
            "Cell.net1.Class1.forward.0.input.0",
            "Cell.net1.Class1.sub.forward.0.input.0",
        ]
        with patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.get_cell_name",
                   side_effect=lambda x: "net1.Class1" if ".sub." not in x else "net1.Class1.sub"), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.get_data_mode",
                   return_value="forward"), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.check_relation",
                   side_effect=[False, False, True]), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.get_parent_cell_name",
                   return_value=""):
            get_construct(cell_list_input)
            import msprobe.mindspore.dump.dump_processor.cell_dump_process as cdp
            self.assertIn(cell_list_input[1], cdp.construct)
            self.assertEqual(cdp.construct[cell_list_input[1]], cell_list_input[0])

    def test_get_construct_with_free_cells(self):
        cell_list_input = [
            "Cell.FreeCell0.forward.0",
            "Cell.FreeCell1.forward.0",
            "Cell.FreeCell2.forward.0",
        ]
        with patch.dict("msprobe.mindspore.dump.dump_processor.cell_dump_process.free_cells", {
                "FreeCell0.forward": "Cell.FreeCell0.forward.0",
                "FreeCell1.forward": "Cell.FreeCell1.forward.0",
                "FreeCell2.forward": "Cell.FreeCell2.forward.0",
            }, clear=True), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.get_cell_name",
                   side_effect=lambda x: x.split('.')[1]), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.get_data_mode",
                   return_value="forward"), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.check_relation",
                   return_value=False), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.get_parent_cell_name",
                   return_value=""):
            get_construct(cell_list_input)
            import msprobe.mindspore.dump.dump_processor.cell_dump_process as cdp
            self.assertEqual(len(cdp.construct), len(cell_list_input))
            for cell in cell_list_input:
                self.assertEqual(cdp.construct[cell], cell)

    def test_get_construct_with_parent_cell_types(self):
        cell_list_input = [
            "Cell.net1.Class1.forward.0.input.0",
        ]
        with patch.dict("msprobe.mindspore.dump.dump_processor.cell_dump_process.parent_cell_types", {
                "net1.Class1": "TestClass",
            }, clear=True), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.get_cell_name",
                   return_value="net1.Class1"), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.get_data_mode",
                   return_value="forward"), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.check_relation",
                   return_value=False), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.get_parent_cell_name",
                   return_value="net1"):
            get_construct(cell_list_input)
            import msprobe.mindspore.dump.dump_processor.cell_dump_process as cdp
            cell = cell_list_input[0]
            self.assertIn(cell, cdp.construct)
            self.assertIn("Cell.net1.TestClass", cdp.construct[cell])


class TestProcessFile(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_process_file_input(self):
        npy_path = os.path.join(self.temp_dir, "Cell.net1.Class1.forward.0.input.0_float32_1.npy")
        np.save(npy_path, np.array([1.0, 2.0, 3.0], dtype=np.float32))

        with patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.move_file"):
            op_name, key, tensor_json = process_file(npy_path)
        self.assertEqual(op_name, "Cell.net1.Class1.forward.0")
        self.assertEqual(key, CoreConst.INPUT_ARGS)
        self.assertIsNotNone(tensor_json)
        self.assertEqual(tensor_json[CoreConst.TYPE], "mindspore.Tensor")
        self.assertEqual(tensor_json[CoreConst.SHAPE], [3])

    def test_process_file_output(self):
        npy_path = os.path.join(self.temp_dir, "Cell.net1.Class1.forward.0.output.0_float32_1.npy")
        np.save(npy_path, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))

        with patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.move_file"):
            op_name, key, tensor_json = process_file(npy_path)
        self.assertEqual(op_name, "Cell.net1.Class1.forward.0")
        self.assertEqual(key, CoreConst.OUTPUT)
        self.assertEqual(tensor_json[CoreConst.SHAPE], [2, 2])
        self.assertAlmostEqual(tensor_json[CoreConst.MAX], 4.0)
        self.assertAlmostEqual(tensor_json[CoreConst.MIN], 1.0)

    def test_process_file_invalid_parts(self):
        npy_path = os.path.join(self.temp_dir, "invalid.npy")
        np.save(npy_path, np.array([1.0]))
        with patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.move_file"):
            with patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.logger") as mock_logger:
                op_name, key, tensor_json = process_file(npy_path)
        self.assertIsNone(op_name)
        self.assertIsNone(key)
        self.assertIsNone(tensor_json)

    def test_process_file_unknown_dtype(self):
        npy_path = os.path.join(self.temp_dir, "Cell.net1.Class1.forward.0.input.0_customdtype_1.npy")
        np.save(npy_path, np.array([1.0, 2.0, 3.0], dtype=np.float32))

        with patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.move_file"), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.logger") as mock_logger:
            op_name, key, tensor_json = process_file(npy_path)
        self.assertIsNotNone(tensor_json)
        self.assertEqual(tensor_json[CoreConst.DTYPE], 'None')

    def test_process_file_unknown_io_type(self):
        npy_path = os.path.join(self.temp_dir, "Cell.net1.Class1.forward.0.unknown.0_float32_1.npy")
        np.save(npy_path, np.array([5.0, 6.0], dtype=np.float32))
        with patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.move_file"):
            op_name, key, tensor_json = process_file(npy_path)
        self.assertIsNone(op_name)
        self.assertIsNone(key)
        self.assertIsNone(tensor_json)


class TestCustomSort(unittest.TestCase):
    def test_custom_sort_existing_key(self):
        key_to_index = {"a": 0, "b": 1, "c": 2}
        self.assertEqual(custom_sort(("a", "val"), key_to_index), 0)
        self.assertEqual(custom_sort(("b", "val"), key_to_index), 1)
        self.assertEqual(custom_sort(("c", "val"), key_to_index), 2)

    def test_custom_sort_missing_key(self):
        key_to_index = {"a": 0}
        result = custom_sort(("unknown", "val"), key_to_index)
        self.assertEqual(result, float('inf'))


class TestConvertSpecialValues(unittest.TestCase):
    def test_true_string(self):
        self.assertTrue(convert_special_values("true"))
        self.assertTrue(convert_special_values("True"))

    def test_false_string(self):
        self.assertFalse(convert_special_values("false"))
        self.assertFalse(convert_special_values("False"))

    def test_float_string(self):
        self.assertEqual(convert_special_values("3.14"), 3.14)
        self.assertEqual(convert_special_values("42"), 42.0)

    def test_plain_string(self):
        self.assertEqual(convert_special_values("hello"), "hello")

    def test_nan(self):
        result = convert_special_values(float('nan'))
        self.assertIsNone(result)

    def test_none_value(self):
        self.assertIsNone(convert_special_values(None))

    def test_pandas_na(self):
        result = convert_special_values(pd.NA)
        self.assertIsNone(result)


class TestProcessCsv(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.temp_dir, "statistic.csv")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_process_csv(self):
        data = {
            "Op Name": [
                "Cell.net1.Class1.forward.0.input.0",
                "Cell.net1.Class1.forward.0.output.0",
            ],
            "Shape": ["(1, 4096)", "(1, 1000)"],
            "Data Type": ["float32", "float32"],
            "Max Value": [1.0, 2.0],
            "Min Value": [0.0, -1.0],
            "Avg Value": [0.5, 0.5],
            "L2Norm Value": [64.0, 31.6],
        }
        df = pd.DataFrame(data)
        with patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.read_csv",
                   return_value=df):
            results = process_csv(self.csv_path)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], "Cell.net1.Class1.forward.0")
        self.assertEqual(results[0][1], CoreConst.INPUT_ARGS)
        self.assertEqual(results[1][0], "Cell.net1.Class1.forward.0")
        self.assertEqual(results[1][1], CoreConst.OUTPUT)

    def test_process_csv_invalid_io_key(self):
        data = {
            "Op Name": ["Cell.net1.Class1.forward.0.unknown.0"],
            "Shape": ["(1,)"],
            "Data Type": ["float32"],
        }
        df = pd.DataFrame(data)
        with patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.read_csv",
                   return_value=df):
            results = process_csv(self.csv_path)
        self.assertEqual(len(results), 1)
        self.assertIsNone(results[0][0])


class TestGenerateDumpInfo(unittest.TestCase):
    def setUp(self):
        global cell_list
        cell_list.clear()
        self._original_cell_list = list(cell_list)
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        global cell_list
        cell_list.clear()
        cell_list.extend(self._original_cell_list)
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_generate_dump_info_tensor(self):
        npy_path = os.path.join(self.temp_dir, "Cell.net1.Class1.forward.0.input.0_float32_1.npy")
        np.save(npy_path, np.array([1.0, 2.0], dtype=np.float32))

        with patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.dump_task", CoreConst.TENSOR), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.move_file"), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.save_json") as mock_save, \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.Pool") as mock_pool, \
             patch("os.listdir", return_value=["Cell.net1.Class1.forward.0.input.0_float32_1.npy"]), \
             patch("os.path.exists", return_value=True):
            generate_dump_info(self.temp_dir)
        mock_save.assert_called_once()

    def test_generate_dump_info_path_not_exist(self):
        with patch("os.path.exists", return_value=False), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.logger") as mock_logger:
            generate_dump_info("/nonexistent/path")
        mock_logger.error.assert_called()

    def test_generate_dump_info_statistics(self):
        csv_path = os.path.join(self.temp_dir, "statistic.csv")
        data = {
            "Op Name": ["Cell.net1.Class1.forward.0.input.0"],
            "Shape": ["(1, 2)"],
            "Data Type": ["float32"],
        }
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)

        with patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.dump_task", CoreConst.STATISTICS), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.save_json") as mock_save, \
             patch("os.path.exists", return_value=True):
            generate_dump_info(csv_path)
        mock_save.assert_called_once()


class TestGenerateStackInfo(unittest.TestCase):
    def setUp(self):
        global cell_list
        self._original_cell_list = list(cell_list)

    def tearDown(self):
        global cell_list
        cell_list.clear()
        cell_list.extend(self._original_cell_list)

    def test_generate_stack_info(self):
        global cell_list
        cell_list[:] = ["Cell.net1.Class1.forward.0", "Cell.net1.sub.forward.0"]
        with patch("os.path.exists", return_value=True), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.save_json") as mock_save, \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.remove_path"):
            generate_stack_info("/fake/path")
        mock_save.assert_called_once()

    def test_generate_stack_info_statistics_removes_csv(self):
        global cell_list
        cell_list[:] = ["Cell.net1.Class1.forward.0"]
        with patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.dump_task", CoreConst.STATISTICS), \
             patch("os.path.exists", return_value=True), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.save_json"), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.remove_path") as mock_remove:
            generate_stack_info("/fake/path.csv")
        mock_remove.assert_called_once_with("/fake/path.csv")

    def test_generate_stack_info_path_not_exist(self):
        with patch("os.path.exists", return_value=False), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.logger") as mock_logger:
            generate_stack_info("/nonexistent/path")
        mock_logger.error.assert_called()


class TestIsDownloadFinished(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("time.sleep", return_value=None)
    def test_found_flag_file(self, mock_sleep):
        open(os.path.join(self.temp_dir, "step_0"), 'w').close()
        result = is_download_finished(self.temp_dir, "step_0")
        self.assertTrue(result)

    @patch("time.sleep", return_value=None)
    def test_not_found_flag_file(self, mock_sleep):
        result = is_download_finished(self.temp_dir, "step_99")
        self.assertFalse(result)

    @patch("time.sleep", return_value=None)
    def test_directory_not_exist(self, mock_sleep):
        result = is_download_finished("/nonexistent/path", "step_0")
        self.assertFalse(result)


class TestProcessStep(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_process_step_skip(self):
        with patch("os.path.exists") as mock_exists:
            process_step("/fake/dump", "/fake/flag", 1, [2, 3])
        mock_exists.assert_not_called()

    def test_process_step_path_not_exist(self):
        with patch("os.path.exists", return_value=False), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.create_directory") as mock_create, \
             patch("os.environ.get", return_value=None):
            process_step("/fake/dump", "/fake/flag", 1, None)
        mock_create.assert_called_once()

    def test_process_step_full_flow(self):
        dump_path = os.path.join(self.temp_dir, "dump")
        flag_path = os.path.join(self.temp_dir, "flag")
        os.makedirs(dump_path)
        os.makedirs(flag_path)
        step_dir = os.path.join(dump_path, "step1")
        rank_dir = os.path.join(step_dir, "rank0")
        npy_dir = os.path.join(rank_dir, CoreConst.DUMP_TENSOR_DATA)
        os.makedirs(npy_dir)
        open(os.path.join(flag_path, "step_1"), 'w').close()

        npy_file = os.path.join(npy_dir, "Cell.net1.Class1.forward.0.input.0_float32_1.npy")
        np.save(npy_file, np.array([1.0]))

        with patch("os.environ.get", return_value=None), \
             patch("os.listdir", wraps=os.listdir), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.Pool") as mock_pool, \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.save_json"), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.move_directory") as mock_move_dir:
            from multiprocessing.pool import ThreadPool
            mock_pool.return_value.__enter__.return_value.starmap = MagicMock(return_value=[])
            process_step(dump_path, flag_path, 1, None)
        mock_move_dir.assert_called_once()

    def test_process_step_with_rank_id(self):
        dump_path = os.path.join(self.temp_dir, "dump")
        flag_path = os.path.join(self.temp_dir, "flag")
        os.makedirs(dump_path)
        os.makedirs(flag_path)
        step_dir = os.path.join(dump_path, "step1")
        rank_dir = os.path.join(step_dir, "rank3")
        npy_dir = os.path.join(rank_dir, CoreConst.DUMP_TENSOR_DATA)
        os.makedirs(npy_dir)
        open(os.path.join(flag_path, "step_1"), 'w').close()

        with patch("os.environ.get", return_value="3"), \
             patch("os.listdir", wraps=os.listdir), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.Pool") as mock_pool, \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.save_json") as mock_save, \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.move_directory") as mock_move_dir:
            from multiprocessing.pool import ThreadPool
            mock_pool.return_value.__enter__.return_value.starmap = MagicMock(return_value=[])
            process_step(dump_path, flag_path, 1, None)
        mock_move_dir.assert_not_called()
        mock_save.assert_called()

    def test_process_step_download_in_progress(self):
        dump_path = os.path.join(self.temp_dir, "dump")
        flag_path = os.path.join(self.temp_dir, "flag")
        os.makedirs(dump_path)
        os.makedirs(flag_path)
        step_dir = os.path.join(dump_path, "step1")
        rank_dir = os.path.join(step_dir, "rank0")
        npy_dir = os.path.join(rank_dir, CoreConst.DUMP_TENSOR_DATA)
        os.makedirs(npy_dir)
        open(os.path.join(flag_path, "step_1"), 'w').close()

        with patch("os.environ.get", return_value=None), \
             patch("os.listdir", return_value=[]), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.is_download_finished",
                   side_effect=[False, True]), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.Pool") as mock_pool, \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.save_json"), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.move_directory") as mock_move_dir:
            from multiprocessing.pool import ThreadPool
            mock_pool.return_value.__enter__.return_value.starmap = MagicMock(return_value=[])
            process_step(dump_path, flag_path, 1, None)
        mock_move_dir.assert_called_once()

    def test_process_step_timeout(self):
        dump_path = os.path.join(self.temp_dir, "dump")
        flag_path = os.path.join(self.temp_dir, "flag")
        os.makedirs(dump_path)
        os.makedirs(flag_path)

        with patch("os.environ.get", return_value=None), \
             patch("os.path.exists", return_value=True), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.is_download_finished",
                   return_value=False), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.time") as mock_time, \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.Pool") as mock_pool, \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.rename_filename"), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.generate_construct"), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.generate_dump_info"), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.generate_stack_info"), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.move_directory"):
            mock_time.time.side_effect = [0, 601]
            process_step(dump_path, flag_path, 1, None)
        mock_pool.assert_not_called()


class TestRemoveTrailingCommas(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.temp_dir, "test.csv")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_remove_trailing_commas(self):
        csv_data = [
            ["col1", "col2", "col3"],
            ["a", "b", ""],
            ["c", "d", "e"],
        ]
        with patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.read_csv",
                   return_value=csv_data), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.write_csv") as mock_write:
            remove_trailing_commas(self.csv_path)
        mock_write.assert_called_once()
        self.assertEqual(csv_data[1], ["a", "b"])


class TestMergeFile(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_merge_file(self):
        csv1 = os.path.join(self.temp_dir, "file1.csv")
        csv2 = os.path.join(self.temp_dir, "file2.csv")
        data1 = {
            "Op Name": ["dump_tensor_data/Cell-net1-Class1-forward-0-input-0"],
            "Timestamp": [100],
            "Slot": [1],
        }
        data2 = {
            "Op Name": ["dump_tensor_data/Cell-net1-Class1-forward-0-output-0"],
            "Timestamp": [200],
            "Slot": [1],
        }
        pd.DataFrame(data1).to_csv(csv1, index=False)
        pd.DataFrame(data2).to_csv(csv2, index=False)

        file_dict = {"1": [csv1, csv2]}
        with patch("os.listdir", return_value=[]), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.create_directory"), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.write_df_to_csv") as mock_write:
            merge_file(self.temp_dir, "rank_0", file_dict)
        mock_write.assert_called_once()


class TestProcessStatisticsStep(unittest.TestCase):
    def setUp(self):
        global dump_task
        self._original_dump_task = dump_task

    def tearDown(self):
        global dump_task
        dump_task = self._original_dump_task

    def test_process_statistics_step_skip(self):
        with patch("os.path.exists") as mock_exists:
            process_statistics_step("/fake/dump", 1, [2, 3])
        mock_exists.assert_not_called()

    def test_process_statistics_step_path_not_exist(self):
        with patch("os.path.exists", return_value=False), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.create_directory"), \
             patch("os.environ.get", return_value=None):
            process_statistics_step("/fake/dump", 1, None)

    def test_process_statistics_step_no_net_dir(self):
        dump_path = os.path.join(tempfile.mkdtemp(), "dump")
        os.makedirs(dump_path)
        with patch("os.path.exists", return_value=True), \
             patch("os.environ.get", return_value=None), \
             patch("os.path.isdir", return_value=False), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.logger") as mock_logger:
            process_statistics_step(dump_path, 1, None)
        import shutil
        shutil.rmtree(os.path.dirname(dump_path), ignore_errors=True)

    def test_process_statistics_step_full_flow(self):
        dump_path = os.path.join(tempfile.mkdtemp(), "dump")
        os.makedirs(dump_path)
        rank_dir_kbk = os.path.join(dump_path, "rank_0")
        net_dir = os.path.join(rank_dir_kbk, "Net")
        step_subdir = os.path.join(net_dir, "step1")
        os.makedirs(step_subdir)

        csv_data = pd.DataFrame({
            "Op Name": ["dump_tensor_data/Cell-net1-Class1-forward-0-input-0"],
            "Timestamp": [100],
            "Slot": [1],
        })
        csv_data.to_csv(os.path.join(step_subdir, "statistic.csv"), index=False)

        with patch("os.path.exists", return_value=True), \
             patch("os.environ.get", return_value=None), \
             patch("os.path.isdir", return_value=True), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.save_json"), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.remove_path"), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.move_directory"), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.create_directory"), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.write_df_to_csv"), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.process_csv"), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.rename_filename"), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.remove_trailing_commas"), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.merge_file"), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.generate_construct"), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.generate_dump_info"), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.generate_stack_info"):
            process_statistics_step(dump_path, 1, None)

        import shutil
        shutil.rmtree(os.path.dirname(dump_path), ignore_errors=True)


class TestGetTensorDumpMode(unittest.TestCase):
    def test_valid_format(self):
        first, second = get_tensordump_mode("([in-out],[out-in])")
        self.assertEqual(first, "[in-out]")
        self.assertEqual(second, "[out-in]")

    def test_no_parentheses(self):
        first, second = get_tensordump_mode("no_parens")
        self.assertIsNone(first)
        self.assertIsNone(second)

    def test_single_element(self):
        first, second = get_tensordump_mode("(a)")
        self.assertIsNone(first)
        self.assertIsNone(second)

    def test_empty_string(self):
        first, second = get_tensordump_mode("")
        self.assertIsNone(first)
        self.assertIsNone(second)


class TestSetTensorDumpMode(unittest.TestCase):
    def test_set_tensordump_mode(self):
        cell = MagicMock()
        set_tensordump_mode(cell, "([in-out],[out])")
        self.assertEqual(cell.input_dump_mode, ["in-out"])
        self.assertEqual(cell.output_dump_mode, ["out"])

    def test_set_tensordump_mode_invalid(self):
        cell = MagicMock()
        with patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.str_to_list") as mock_str:
            set_tensordump_mode(cell, "invalid")
        mock_str.assert_not_called()


class TestCreateKbykJson(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_kbyk_json_statistics(self):
        with patch("os.environ.get", return_value=None), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.save_json") as mock_save:
            result = create_kbyk_json(self.temp_dir, "statistics", None)
        self.assertIn("kernel_kbyk_dump.json", result)
        mock_save.assert_called_once()
        config = mock_save.call_args[0][1]
        self.assertEqual(config["common_dump_settings"]["saved_data"], "statistic")
        self.assertEqual(config["common_dump_settings"]["statistic_category"],
                         ["max", "min", "avg", "l2norm"])

    def test_create_kbyk_json_with_mean(self):
        with patch("os.environ.get", return_value=None), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.save_json") as mock_save:
            create_kbyk_json(self.temp_dir, ["max", "mean", "l2norm"], None)
        config = mock_save.call_args[0][1]
        self.assertEqual(config["common_dump_settings"]["statistic_category"],
                         ["max", "avg", "l2norm"])

    def test_create_kbyk_json_with_step(self):
        with patch("os.environ.get", return_value=None), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.save_json") as mock_save:
            create_kbyk_json(self.temp_dir, "statistics", [1, 2, 3])
        config = mock_save.call_args[0][1]
        self.assertEqual(config["common_dump_settings"]["iteration"], "1|2|3")

    def test_create_kbyk_json_with_rank_id(self):
        with patch("os.environ.get", return_value="2"), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.save_json") as mock_save:
            result = create_kbyk_json(self.temp_dir, "statistics", None)
        self.assertIn("2kernel_kbyk_dump.json", result)

    def test_create_kbyk_json_other_category(self):
        with patch("os.environ.get", return_value=None), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.save_json") as mock_save:
            create_kbyk_json(self.temp_dir, ["max", "l2norm"], None)
        config = mock_save.call_args[0][1]
        self.assertEqual(config["common_dump_settings"]["statistic_category"],
                         ["max", "l2norm"])


class TestStart(unittest.TestCase):
    def setUp(self):
        global dump_task, parent_cell_types
        self._original_dump_task = dump_task
        parent_cell_types.clear()

    def tearDown(self):
        global dump_task, parent_cell_types
        dump_task = self._original_dump_task
        parent_cell_types.clear()

    @patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.dump_gradient_op_existed", True)
    def test_start_statistics(self):
        config = CellDumpConfig(
            net=MagicMock(),
            dump_path="/tmp/dump",
            data_mode="all",
            task=CoreConst.STATISTICS,
            step=None
        )
        with patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.create_kbyk_json",
                   return_value="/tmp/config.json"), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process._set_init_iter"), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.remove_path"), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.graph_step_flag", True):
            try:
                start(config)
            except Exception:
                pass
        self.assertEqual(os.environ.get("MS_KERNEL_LAUNCH_SKIP"), "TensorDump")

    @patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.dump_gradient_op_existed", False)
    def test_start_no_dump_gradient(self):
        config = CellDumpConfig(
            net=MagicMock(),
            dump_path="/tmp/dump",
            data_mode="all",
            task=CoreConst.STATISTICS,
        )
        with patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.create_kbyk_json",
                   return_value="/tmp/config.json"), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process._set_init_iter"), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.remove_path"), \
             patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.graph_step_flag", True):
            start(config)

    def test_start_with_none_net(self):
        config = CellDumpConfig(
            net=None,
            dump_path="/tmp/dump",
            data_mode="all",
            task=CoreConst.TENSOR,
        )
        with patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.dump_gradient_op_existed", True):
            start(config)


class TestCellConstructWrapper(unittest.TestCase):
    def test_cell_construct_wrapper_exists(self):
        from msprobe.mindspore.dump.dump_processor.cell_dump_process import cell_construct_wrapper
        self.assertTrue(callable(cell_construct_wrapper))


if __name__ == '__main__':
    unittest.main()
