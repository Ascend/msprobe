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
from unittest.mock import MagicMock, patch

import mindspore as ms
import pandas as pd
import numpy as np
from packaging.version import parse as version_parse

from msprobe.mindspore.dump.dump_processor.cell_dump_process import cell_construct_wrapper
from msprobe.mindspore.dump.dump_processor.cell_dump_process import check_relation
from msprobe.mindspore.dump.dump_processor.cell_dump_process import convert_special_values, sort_filenames
from msprobe.mindspore.dump.dump_processor.cell_dump_process import create_kbyk_json
from msprobe.mindspore.dump.dump_processor.cell_dump_process import (
    process_csv, np_ms_dtype_dict, need_tensordump_in, get_cell_name, get_data_mode,
    get_parent_cell_name, generate_construct, process_file, is_download_finished,
    merge_file, process_statistics_step, get_yaml_keys, get_tensordump_mode, str_to_list,
    set_tensordump_mode, generate_stack_info, parent_cell_types, construct, free_cells, get_construct, dump_task,
    start, KEY_TOPLAYER, CellDumpConfig, process_step, TIMEOUT
)

from msprobe.core.common.const import Const as CoreConst
from msprobe.mindspore.dump.dump_processor import cell_dump_process
from msprobe.core.common.log import logger


class TestCellDumpStart(unittest.TestCase):
    _set_init_iter = MagicMock()

    def setUp(self):
        self.original_dump_gradient = cell_dump_process.dump_gradient_op_existed
        self.original_graph_step = cell_dump_process.graph_step_flag

        self.dump_gradient_op_existed = True
        self.graph_step_flag = True
        cell_dump_process.dump_gradient_op_existed = self.dump_gradient_op_existed
        cell_dump_process.graph_step_flag = self.graph_step_flag
        cell_dump_process._set_init_iter = self._set_init_iter

        self.patch_list = [
            patch('os.path.exists', return_value=False),
            patch('msprobe.core.common.file_utils.remove_path', MagicMock()),
            patch('msprobe.core.common.file_utils.load_yaml', return_value={KEY_TOPLAYER: {"cell1": "(in,out)"}}),
            patch.dict(os.environ, clear=True),
            patch('msprobe.mindspore.dump.dump_processor.cell_dump_process.create_kbyk_json',
                  return_value="/fake/config.json"),
            patch('msprobe.mindspore.dump.dump_processor.cell_dump_process.get_yaml_keys', return_value=["Linear"]),
            patch('msprobe.mindspore.dump.dump_processor.cell_dump_process.set_tensordump_mode', MagicMock()),
            patch('msprobe.mindspore.dump.dump_processor.cell_dump_process.cell_construct_wrapper',
                  return_value=MagicMock()),
            patch.object(logger, 'info', MagicMock()),
            patch.object(logger, 'warning', MagicMock()),
            patch.object(logger, 'error', MagicMock()),
        ]

        self.mocks = []
        for p in self.patch_list:
            self.mocks.append(p.start())
        self.mock_config = MagicMock(spec=CellDumpConfig)
        self.mock_config.task = CoreConst.STATISTICS
        self.mock_config.net = (('', MagicMock(), MagicMock()),)
        self.mock_config.dump_path = "/fake/dump"
        self.mock_config.data_mode = "all"
        self.mock_config.summary_mode = "statistics"
        self.mock_config.step = 0

    def tearDown(self):
        cell_dump_process.dump_gradient_op_existed = self.original_dump_gradient
        cell_dump_process.graph_step_flag = self.original_graph_step

        for p in self.patch_list:
            p.stop()

    def test_start_statistics_main(self):
        start(self.mock_config)
        self.mocks[4].assert_called()
        self._set_init_iter.assert_called()

    def test_start_all_edge_cases(self):
        self.dump_gradient_op_existed = False
        cell_dump_process.dump_gradient_op_existed = self.dump_gradient_op_existed
        start(self.mock_config)

        self.dump_gradient_op_existed = True
        cell_dump_process.dump_gradient_op_existed = self.dump_gradient_op_existed
        self._set_init_iter.assert_called()

        self.mock_config.net = None
        start(self.mock_config)
        self.mock_config.net = (('', MagicMock(), MagicMock()),)

        self.graph_step_flag = False
        cell_dump_process.graph_step_flag = self.graph_step_flag
        with self.assertRaises(Exception) as ctx:
            start(self.mock_config)
        self.assertIn("Importing _set_init_iter failed", str(ctx.exception))

        self.graph_step_flag = True
        cell_dump_process.graph_step_flag = self.graph_step_flag

        self.mock_config.task = CoreConst.TENSOR
        start(self.mock_config)
        self.mocks[4].assert_called()

        mock_cell = MagicMock()
        mock_cell.__class__.__name__ = f"{CoreConst.REPLACEMENT_CHARACTER}Linear"
        mock_cell.cells_and_names.return_value = [("grad_reducer", mock_cell)]
        self.mock_config.net = (('', mock_cell, MagicMock()),)
        start(self.mock_config)
        logger.info.assert_not_called()


class TestProcessStepCoverage(unittest.TestCase):
    mock_create_directory = MagicMock()
    mock_rename_filename = MagicMock()
    mock_generate_construct = MagicMock()
    mock_generate_dump_info = MagicMock()
    mock_generate_stack_info = MagicMock()
    mock_move_directory = MagicMock()

    @classmethod
    def setUpClass(cls):
        cls.mock_create_directory.reset_mock()
        cls.mock_rename_filename.reset_mock()
        cls.mock_generate_construct.reset_mock()
        cls.mock_generate_dump_info.reset_mock()
        cls.mock_generate_stack_info.reset_mock()
        cls.mock_move_directory.reset_mock()

    def setUp(self):
        self.mock_create_directory.reset_mock()
        self.mock_rename_filename.reset_mock()
        self.mock_generate_construct.reset_mock()
        self.mock_generate_dump_info.reset_mock()
        self.mock_generate_stack_info.reset_mock()
        self.mock_move_directory.reset_mock()

    @patch('os.path.exists', return_value=True)
    @patch('os.environ.get', return_value=None)
    @patch('msprobe.mindspore.dump.dump_processor.cell_dump_process.is_download_finished', return_value=True)
    @patch('msprobe.mindspore.dump.dump_processor.cell_dump_process.create_directory',
           new_callable=lambda: TestProcessStepCoverage.mock_create_directory)
    @patch('msprobe.mindspore.dump.dump_processor.cell_dump_process.rename_filename',
           new_callable=lambda: TestProcessStepCoverage.mock_rename_filename)
    @patch('msprobe.mindspore.dump.dump_processor.cell_dump_process.generate_construct',
           new_callable=lambda: TestProcessStepCoverage.mock_generate_construct)
    @patch('msprobe.mindspore.dump.dump_processor.cell_dump_process.generate_dump_info',
           new_callable=lambda: TestProcessStepCoverage.mock_generate_dump_info)
    @patch('msprobe.mindspore.dump.dump_processor.cell_dump_process.generate_stack_info',
           new_callable=lambda: TestProcessStepCoverage.mock_generate_stack_info)
    @patch('msprobe.mindspore.dump.dump_processor.cell_dump_process.move_directory',
           new_callable=lambda: TestProcessStepCoverage.mock_move_directory)
    def test_process_step_core_branch(self, mock_move, mock_stack, mock_dump_info, mock_construct,
                                      mock_rename, mock_create, mock_is_finished, mock_env_get, mock_exists):
        dump_path = "/fake/dump"
        flag_path = "/fake/flag"
        step = 100
        step_list = [100, 200, 300]

        process_step(dump_path, flag_path, step, step_list)

        self.mock_rename_filename.assert_called()
        self.mock_generate_construct.assert_called()
        self.mock_move_directory.assert_called()

    @patch('os.path.exists', side_effect=[False, True])
    @patch('os.environ.get', return_value="0")
    @patch('msprobe.mindspore.dump.dump_processor.cell_dump_process.is_download_finished',
           side_effect=[False, False, True])
    @patch('msprobe.mindspore.dump.dump_processor.cell_dump_process.create_directory',
           new_callable=lambda: TestProcessStepCoverage.mock_create_directory)
    @patch('time.time', side_effect=[0, TIMEOUT + 1])
    def test_process_step_edge_branch(self, mock_time, mock_create, mock_is_finished, mock_env_get, mock_exists):
        process_step("/fake/dump", "/fake/flag", 400, [100, 200, 300])

        process_step("/fake/dump", "/fake/flag", 100, [100])
        self.mock_create_directory.assert_called_once_with("/fake/dump")

        mock_exists.reset_mock()
        process_step("/fake/dump", "/fake/flag", 100, [100])
        self.mock_move_directory.assert_not_called()


class TestGetConstruct(unittest.TestCase):

    def setUp(self):
        self.original_construct = construct.copy()
        self.original_free_cells = free_cells.copy()
        self.original_parent_cell_types = parent_cell_types.copy()

        construct.clear()
        free_cells.clear()
        parent_cell_types.clear()

        self.patcher_get_cell_name = patch('msprobe.mindspore.dump.dump_processor.cell_dump_process.get_cell_name')
        self.patcher_get_data_mode = patch('msprobe.mindspore.dump.dump_processor.cell_dump_process.get_data_mode')
        self.patcher_check_relation = patch('msprobe.mindspore.dump.dump_processor.cell_dump_process.check_relation')
        self.patcher_get_parent_cell_name = patch(
            'msprobe.mindspore.dump.dump_processor.cell_dump_process.get_parent_cell_name')

        self.mock_get_cell_name = self.patcher_get_cell_name.start()
        self.mock_get_data_mode = self.patcher_get_data_mode.start()
        self.mock_check_relation = self.patcher_check_relation.start()
        self.mock_get_parent_cell_name = self.patcher_get_parent_cell_name.start()

    def tearDown(self):
        construct.clear()
        construct.update(self.original_construct)
        free_cells.clear()
        free_cells.update(self.original_free_cells)
        parent_cell_types.clear()
        parent_cell_types.update(self.original_parent_cell_types)

        self.patcher_get_cell_name.stop()
        self.patcher_get_data_mode.stop()
        self.patcher_check_relation.stop()
        self.patcher_get_parent_cell_name.stop()

    def test_get_construct_found_parent_cell(self):
        cell_list_input = ["cell1", "cell2"]
        self.mock_get_cell_name.side_effect = lambda x: "cell1_name" if x == "cell1" else "cell2_name"
        self.mock_get_data_mode.side_effect = lambda x: "forward" if x == "cell1" else "forward"
        self.mock_check_relation.return_value = True

        get_construct(cell_list_input)

        self.assertEqual(construct["cell1"], "cell1")
        self.assertEqual(construct["cell2"], "cell1")

    def test_get_construct_free_cells_hit(self):
        cell_list_input = ["cell1"]
        self.mock_get_cell_name.return_value = "cell1_name"
        self.mock_get_data_mode.return_value = "forward"
        self.mock_check_relation.return_value = False

        cell_name_with_mode = f"cell1_name{CoreConst.SEP}forward"
        free_cells[cell_name_with_mode] = "cached_parent_cell"

        get_construct(cell_list_input)

        self.assertEqual(construct["cell1"], "cached_parent_cell")
        self.mock_get_parent_cell_name.assert_not_called()

    def test_get_construct_generate_parent_cell(self):
        cell_list_input = ["cell1"]
        cell1_full = "Cell.cell1_name.forward.0.input.0"
        self.mock_get_cell_name.return_value = "cell1_name"
        self.mock_get_data_mode.return_value = "forward"
        self.mock_check_relation.return_value = False
        self.mock_get_parent_cell_name.return_value = "parent_cell_name"

        parent_cell_types["cell1_name"] = "ParentCellType"

        get_construct([cell1_full])

        expected_parent_cell = f"{CoreConst.CELL}{CoreConst.SEP}parent_cell_name{CoreConst.SEP}ParentCellType.input.0"
        self.assertEqual(construct[cell1_full], expected_parent_cell)
        cell_name_with_mode = f"cell1_name{CoreConst.SEP}forward"
        self.assertEqual(free_cells[cell_name_with_mode], expected_parent_cell)

    def test_get_construct_no_parent_cell(self):
        cell_list_input = ["cell1"]
        self.mock_get_cell_name.return_value = "cell1_name"
        self.mock_get_data_mode.return_value = "forward"
        self.mock_check_relation.return_value = False
        self.mock_get_parent_cell_name.return_value = ""
        parent_cell_types.clear()

        get_construct(cell_list_input)

        self.assertEqual(construct["cell1"], None)
        self.mock_get_parent_cell_name.assert_called_with("cell1_name")

    def test_get_construct_cell_name_empty(self):
        cell_list_input = ["cell1"]
        self.mock_get_cell_name.return_value = ""
        self.mock_get_data_mode.return_value = "forward"
        self.mock_check_relation.return_value = False

        get_construct(cell_list_input)

        self.assertEqual(construct["cell1"], None)
        self.mock_get_parent_cell_name.assert_called_once()

    def test_get_construct_parent_cell_name_none(self):
        cell_list_input = ["cell1"]
        self.mock_get_cell_name.return_value = "cell1_name"
        self.mock_get_data_mode.return_value = "forward"
        self.mock_check_relation.return_value = False
        self.mock_get_parent_cell_name.return_value = None
        parent_cell_types["cell1_name"] = "ParentCellType"

        get_construct(cell_list_input)

        self.assertEqual(construct["cell1"], None)
        self.mock_get_parent_cell_name.assert_called_with("cell1_name")


class TestGenerateConstruct(unittest.TestCase):
    """测试generate_construct - 解决NotADirectoryError"""

    def setUp(self):
        global dump_task
        self.original_dump_task = dump_task
        self._logger_patcher = patch.object(logger, 'info', MagicMock())
        self._logger_patcher.start()

    def tearDown(self):
        global dump_task
        dump_task = self.original_dump_task
        self._logger_patcher.stop()

    @patch('msprobe.mindspore.dump.dump_processor.cell_dump_process.read_csv')
    @patch('msprobe.mindspore.dump.dump_processor.cell_dump_process.get_construct')
    @patch('msprobe.mindspore.dump.dump_processor.cell_dump_process.save_json')
    def test_generate_construct_statistics(self, mock_save_json, mock_get_construct, mock_read_csv):
        global dump_task
        dump_task = CoreConst.STATISTICS
        mock_read_csv.return_value = pd.DataFrame({
            CoreConst.OP_NAME: ["Cell.test.forward.0.input.0"]
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "statistic.csv")
            pd.DataFrame({CoreConst.OP_NAME: ["Cell.test.forward.0.input.0"]}).to_csv(csv_path)

            generate_construct(tmpdir)

            mock_get_construct.assert_called()
            mock_save_json.assert_called()


class TestNeedTensorDumpIn(unittest.TestCase):
    """测试need_tensordump_in函数"""

    def test_need_tensordump_in_has_attr(self):
        """测试存在属性且索引有效"""
        mock_cell = MagicMock()
        mock_cell.input_dump_mode = ["in", "out"]
        self.assertTrue(need_tensordump_in(mock_cell, "input_dump_mode", 0))
        self.assertFalse(need_tensordump_in(mock_cell, "input_dump_mode", 1))

    def test_need_tensordump_in_no_attr(self):
        """测试不存在属性"""
        mock_cell = MagicMock()
        del mock_cell.input_dump_mode
        self.assertFalse(need_tensordump_in(mock_cell, "input_dump_mode", 0))

    def test_need_tensordump_in_index_out_of_range(self):
        """测试索引超出范围"""
        mock_cell = MagicMock()
        mock_cell.output_dump_mode = ["in"]
        self.assertFalse(need_tensordump_in(mock_cell, "output_dump_mode", 1))


class TestSortFilenamesExtended(unittest.TestCase):
    """补充sort_filenames的测试场景"""

    @patch('os.listdir')
    @patch('msprobe.mindspore.dump.dump_processor.cell_dump_process.logger')
    def test_sort_filenames_invalid_files(self, mock_logger, mock_listdir):
        """测试包含无效格式文件的场景"""
        mock_listdir.return_value = [
            'valid_123.npy',
            'invalid_file.txt',
            'no_number_.npy',
            'invalid_abc.npy'
        ]

        sorted_files = sort_filenames('/mock/path')
        self.assertEqual(sorted_files, ['valid_123.npy'])
        mock_logger.warning.assert_called()
        self.assertEqual(mock_logger.warning.call_count, 3)

    @patch('os.listdir')
    def test_sort_filenames_empty_list(self, mock_listdir):
        """测试空文件列表"""
        mock_listdir.return_value = []
        sorted_files = sort_filenames('/mock/path')
        self.assertEqual(sorted_files, [])

    @patch('os.listdir')
    def test_sort_filenames_large_numbers(self, mock_listdir):
        """测试大数字id排序"""
        mock_listdir.return_value = [
            'file_1000.npy',
            'file_999.npy',
            'file_100.npy'
        ]
        sorted_files = sort_filenames('/mock/path')
        self.assertEqual(sorted_files, ['file_100.npy', 'file_999.npy', 'file_1000.npy'])


class TestGetCellName(unittest.TestCase):
    """测试get_cell_name函数"""

    def test_get_cell_name_valid(self):
        """测试有效cell字符串"""
        cell_str = "Cell.network.layer.forward.0.input.0"
        self.assertEqual(get_cell_name(cell_str), "network.layer.forward")

    def test_get_cell_name_short(self):
        """测试过短的cell字符串"""
        self.assertIsNone(get_cell_name("Cell.test"))
        self.assertIsNone(get_cell_name("Cell.test.forward"))


class TestGetDataMode(unittest.TestCase):
    """测试get_data_mode函数"""

    def test_get_data_mode_forward(self):
        """测试forward模式"""
        cell_str = "Cell.test.forward.0.input.0"
        self.assertEqual(get_data_mode(cell_str), "input")

    def test_get_data_mode_backward(self):
        """测试backward模式"""
        cell_str = "Cell.test.backward.1.output.1"
        self.assertEqual(get_data_mode(cell_str), "output")


class TestGetParentCellName(unittest.TestCase):
    """测试get_parent_cell_name函数"""

    def setUp(self):
        global KEY_LAYERS
        KEY_LAYERS = "layers"

    def test_get_parent_cell_name_layers_pattern(self):
        """测试layers模式"""
        cell_name = "network.model.layers.0"
        self.assertEqual(get_parent_cell_name(cell_name), "network.model")

    def test_get_parent_cell_name_normal(self):
        """测试普通模式"""
        cell_name = "network.model.layer1"
        self.assertEqual(get_parent_cell_name(cell_name), "network.model")

    def test_get_parent_cell_name_no_dot(self):
        """测试无分隔符"""
        self.assertEqual(get_parent_cell_name("network"), "")


class TestProcessFile(unittest.TestCase):
    """测试process_file函数"""

    @patch('msprobe.mindspore.dump.dump_processor.cell_dump_process.load_npy')
    @patch('msprobe.mindspore.dump.dump_processor.cell_dump_process.move_file')
    @patch('msprobe.mindspore.dump.dump_processor.cell_dump_process.logger')
    def test_process_file_valid(self, mock_logger, mock_move_file, mock_load_npy):
        """测试有效文件处理"""
        mock_npy = MagicMock()
        mock_npy.shape = (2, 3)
        mock_npy.dtype = np.float32
        mock_npy.max.return_value.item.return_value = 1.0
        mock_npy.min.return_value.item.return_value = 0.0
        mock_npy.mean.return_value.item.return_value = 0.5
        np.linalg.norm = MagicMock(return_value=MagicMock(item=lambda: 2.0))
        mock_load_npy.return_value = mock_npy

        with tempfile.NamedTemporaryFile(suffix='.npy') as tmpfile:
            filename = "Cell.test.forward.0.input.0_float32_123.npy"
            file_path = os.path.join(tempfile.gettempdir(), filename)
            result = process_file(file_path)

            self.assertEqual(result[0], "Cell.test.forward.0")
            self.assertEqual(result[1], CoreConst.INPUT_ARGS)
            self.assertEqual(result[2][CoreConst.SHAPE], [2, 3])

    @patch('msprobe.mindspore.dump.dump_processor.cell_dump_process.logger')
    def test_process_file_exception(self, mock_logger):
        """测试文件处理异常"""
        result = process_file("/invalid/path.npy")
        self.assertEqual(result, (None, None, None))
        mock_logger.error.assert_called()


class TestConvertSpecialValuesExtended(unittest.TestCase):
    """补充convert_special_values测试"""

    def test_convert_special_values_inf_nan(self):
        """测试inf和nan"""
        self.assertEqual(convert_special_values(float('nan')), None)

    def test_convert_special_values_non_string(self):
        """测试非字符串类型"""
        self.assertEqual(convert_special_values(True), True)
        self.assertEqual(convert_special_values(None), None)
        self.assertEqual(convert_special_values([]), [])

    def test_convert_special_values_invalid_string(self):
        """测试无效数字字符串"""
        self.assertEqual(convert_special_values("abc"), "abc")


class TestIsDownloadFinished(unittest.TestCase):
    """测试is_download_finished函数"""

    @patch('os.listdir')
    def test_is_download_finished_not_exists(self, mock_listdir):
        """测试标志文件不存在"""
        mock_listdir.return_value = ["other_file"]
        self.assertFalse(is_download_finished("/mock/path", "step_0"))

    @patch('os.path.exists')
    def test_is_download_finished_dir_not_exists(self, mock_exists):
        """测试目录不存在"""
        mock_exists.return_value = False
        self.assertFalse(is_download_finished("/invalid/path", "step_0"))


class TestMergeFile(unittest.TestCase):
    """测试merge_file函数"""

    @patch('msprobe.mindspore.dump.dump_processor.cell_dump_process.remove_trailing_commas')
    @patch('msprobe.mindspore.dump.dump_processor.cell_dump_process.read_csv')
    @patch('msprobe.mindspore.dump.dump_processor.cell_dump_process.rename_filename')
    @patch('msprobe.mindspore.dump.dump_processor.cell_dump_process.write_df_to_csv')
    def test_merge_file(self, mock_write, mock_rename, mock_read, mock_remove):
        """测试文件合并"""
        if (version_parse(ms.__version__) > version_parse("2.7.0")):
            KEY_DUMP_TENSOR_DATA = "dump_tensor_data/"
        else:
            KEY_DUMP_TENSOR_DATA = "dump_tensor_data_"
        op_name = f"{KEY_DUMP_TENSOR_DATA}Cell-test-forward-input-0|123"
        mock_read.return_value = pd.DataFrame({
            CoreConst.OP_NAME: [op_name],
            "Timestamp": [1],
            "Slot": [1]
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            file_dict = {"0": [os.path.join(tmpdir, "statistic.csv")]}
            merge_file(tmpdir, "rank_0", file_dict)
            mock_rename.assert_called()
            mock_write.assert_called()


class TestProcessStatisticsStep(unittest.TestCase):
    """测试process_statistics_step函数"""

    @patch('os.walk')
    @patch('msprobe.mindspore.dump.dump_processor.cell_dump_process.merge_file')
    def test_process_statistics_step_no_files(self, mock_merge, mock_walk):
        """测试无csv文件场景"""
        mock_walk.return_value = [("/mock/path", [], [])]
        with tempfile.TemporaryDirectory() as tmpdir:
            process_statistics_step(tmpdir, 0, [])
            mock_merge.assert_not_called()

    @patch('os.walk')
    @patch('msprobe.mindspore.dump.dump_processor.cell_dump_process.merge_file')
    @patch('msprobe.mindspore.dump.dump_processor.cell_dump_process.generate_construct')
    def test_process_statistics_step_valid(self, mock_construct, mock_merge, mock_walk):
        """测试有效场景"""
        mock_walk.return_value = [("/mock/path/Net/step0", [], ["statistic.csv"])]
        with tempfile.TemporaryDirectory() as tmpdir:
            rank_dir = os.path.join(tmpdir, "rank_0", "Net")
            os.makedirs(rank_dir)
            process_statistics_step(tmpdir, 0, [])
            mock_merge.assert_called()


class TestHelperFunctions(unittest.TestCase):
    """测试辅助函数"""

    def test_get_yaml_keys(self):
        """测试get_yaml_keys"""
        self.assertEqual(get_yaml_keys({"key1": 1, "key2": 2}), ["key1", "key2"])

    def test_get_tensordump_mode(self):
        """测试get_tensordump_mode"""
        self.assertEqual(get_tensordump_mode("([in, out], [in])"), ('[in', 'out]'))
        self.assertEqual(get_tensordump_mode("invalid"), (None, None))

    def test_str_to_list(self):
        """测试str_to_list"""
        self.assertEqual(str_to_list("[in, out]"), ["in", "out"])
        self.assertEqual(str_to_list(""), [''])

    def test_set_tensordump_mode(self):
        """测试set_tensordump_mode"""
        mock_cell = MagicMock()
        set_tensordump_mode(mock_cell, "([in], [out])")
        self.assertEqual(mock_cell.input_dump_mode, ["in"])
        self.assertEqual(mock_cell.output_dump_mode, ["out"])


class TestCreateKbykJsonExtended(unittest.TestCase):
    """补充create_kbyk_json测试"""

    @patch('msprobe.mindspore.dump.dump_processor.cell_dump_process.create_directory')
    @patch('msprobe.mindspore.dump.dump_processor.cell_dump_process.save_json')
    def test_create_kbyk_json_summary_mode_mean(self, mock_save, mock_create):
        """测试summary_mode包含mean"""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_kbyk_json(tmpdir, ["mean", "max"], [0, 1])
            # 验证mean被替换为avg
            args = mock_save.call_args[0][1]
            self.assertEqual(args["common_dump_settings"]["statistic_category"], ["avg", "max"])

    @patch('msprobe.mindspore.dump.dump_processor.cell_dump_process.create_directory')
    @patch('msprobe.mindspore.dump.dump_processor.cell_dump_process.save_json')
    def test_create_kbyk_json_statistics(self, mock_save, mock_create):
        """测试summary_mode为statistics"""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_kbyk_json(tmpdir, "statistics", [])
            args = mock_save.call_args[0][1]
            self.assertEqual(args["common_dump_settings"]["statistic_category"], ["max", "min", "avg", "l2norm"])


class TestGenerateStackInfo(unittest.TestCase):
    """测试generate_stack_info函数"""

    def test_generate_stack_info_path_not_exists(self):
        """测试路径不存在"""
        with patch('msprobe.mindspore.dump.dump_processor.cell_dump_process.logger') as mock_logger:
            generate_stack_info("/invalid/path")
            mock_logger.error.assert_called()


class TestCellWrapperProcess(unittest.TestCase):

    @patch('msprobe.mindspore.dump.dump_processor.cell_dump_process.ops.is_tensor')
    @patch('msprobe.mindspore.dump.dump_processor.cell_dump_process.td')
    @patch('msprobe.mindspore.dump.dump_processor.cell_dump_process.td_in')
    def test_cell_construct_wrapper(self, mock_td_in, mock_td, mock_istensor):

        # Mock the TensorDump operations
        mock_td.return_value = MagicMock()
        mock_td_in.return_value = MagicMock()
        mock_istensor.return_value = False

        # Create a mock cell with necessary attributes
        mock_cell = MagicMock()
        mock_cell.data_mode = "all"
        mock_cell.dump_path = "mock_dump_path"
        mock_cell.cell_prefix = "mock_cell_prefix"

        # Define a mock function to wrap
        def mock_func(*args, **kwargs):
            return args

        # Wrap the mock function using cell_construct_wrapper
        wrapped_func = cell_construct_wrapper(mock_func, mock_cell)

        # Create mock inputs
        mock_input = ms.Tensor([1, 2, 3])
        mock_args = (mock_input,)

        # Call the wrapped function
        wrapped_func(mock_cell, *mock_args)

        # Verify that the TensorDump operations were not called
        mock_td_in.assert_not_called()
        mock_td.assert_not_called()

    @patch('msprobe.mindspore.dump.dump_processor.cell_dump_process.ops.is_tensor')
    @patch('msprobe.mindspore.dump.dump_processor.cell_dump_process.td')
    @patch('msprobe.mindspore.dump.dump_processor.cell_dump_process.td_in')
    def test_cell_construct_wrapper_not_tuple(self, mock_td_in, mock_td, mock_istensor):
        mock_td.return_value = MagicMock()
        mock_td_in.return_value = MagicMock()
        mock_istensor.return_value = False

        mock_cell = MagicMock()
        mock_cell.data_mode = "all"
        mock_cell.dump_path = "mock_dump_path"
        mock_cell.cell_prefix = "mock_cell_prefix"

        def mock_func(*args, **kwargs):
            return []

        wrapped_func = cell_construct_wrapper(mock_func, mock_cell)

        mock_input = ms.Tensor([1, 2, 3])
        mock_args = (mock_input,)

        wrapped_func(mock_cell, *mock_args)

        mock_td_in.assert_not_called()
        mock_td.assert_not_called()


class TestSortFilenames(unittest.TestCase):

    @patch('os.listdir')
    def test_sort_filenames(self, mock_listdir):
        # Mock the list of filenames returned by os.listdir
        mock_listdir.return_value = [
            'Cell.network._backbone.model.LlamaModel.backward.0.input.0_float16_177.npy',
            'Cell.network._backbone.model.LlamaModel.forward.0.input.0_in_int32_1.npy',
            'Cell.network._backbone.model.LlamaModel.forward.0.output.10_float16_165.npy',
            'Cell.network._backbone.model.norm_out.LlamaRMSNorm.backward.0.input.0_float16_178.npy'
        ]

        # Mock the CoreConst values
        CoreConst.REPLACEMENT_CHARACTER = '_'
        CoreConst.NUMPY_SUFFIX = '.npy'

        # Expected sorted filenames
        expected_sorted_filenames = [
            'Cell.network._backbone.model.LlamaModel.forward.0.input.0_in_int32_1.npy',
            'Cell.network._backbone.model.LlamaModel.forward.0.output.10_float16_165.npy',
            'Cell.network._backbone.model.LlamaModel.backward.0.input.0_float16_177.npy',
            'Cell.network._backbone.model.norm_out.LlamaRMSNorm.backward.0.input.0_float16_178.npy'
        ]

        # Call the function
        sorted_filenames = sort_filenames('/mock/path')

        # Assert the filenames are sorted correctly
        self.assertEqual(sorted_filenames, expected_sorted_filenames)


class TestCheckRelation(unittest.TestCase):

    def setUp(self):
        CoreConst.SEP = '.'
        global KEY_LAYERS
        KEY_LAYERS = "layers"

    def test_direct_parent_child_relation(self):
        self.assertTrue(check_relation("network._backbone", "network"))
        self.assertTrue(check_relation("network._backbone.model", "network._backbone"))

    def test_no_relation(self):
        self.assertFalse(check_relation("network._backbone", "network.loss"))
        self.assertFalse(check_relation("network._backbone.model", "network.loss"))

    def test_layer_pattern_relation(self):
        self.assertTrue(check_relation("network.model.layers.0", "network.model"))
        self.assertTrue(check_relation("network._backbone.model.layers.1", "network._backbone.model"))

    def test_edge_cases(self):
        self.assertFalse(check_relation("", "network"))
        self.assertFalse(check_relation("network.layer1", ""))
        self.assertFalse(check_relation("", ""))


class TestRenameFilename(unittest.TestCase):
    def setUp(self):
        self.logger_patcher = patch.object(cell_dump_process, "logger", MagicMock())
        self.logger_patcher.start()

    def tearDown(self):
        self.logger_patcher.stop()

    @patch.object(cell_dump_process, "sort_filenames")
    @patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.move_file")
    def test_rename_filename_tensor(self, mock_move_file, mock_sort_filenames):
        cell_dump_process.dump_task = CoreConst.TENSOR

        with tempfile.TemporaryDirectory() as tmpdir:
            filenames = [
                "Cell.a.b.c.X.forward.input.0_float32_1.npy",
                "Cell.a.b.c.X.forward.input.1_float32_2.npy",
                "Cell.a.b.c.X.forward.output.0_float32_3.npy",
                "Cell.a.b.c.X.forward.input.0_float32_11.npy",
                "Cell.a.b.c.X.forward.input.1_float32_12.npy",
                "Cell.a.b.c.X.forward.output.0_float32_13.npy",
                "Cell.a.b.c.X.backward.input.0_float32_30.npy"
            ]
            for fname in filenames:
                with open(os.path.join(tmpdir, fname), "wb") as f:
                    f.write(b"dummy")

            mock_sort_filenames.return_value = filenames

            rename_calls = []
            def fake_rename(src, dst):
                rename_calls.append((os.path.basename(src), os.path.basename(dst)))
            mock_move_file.side_effect = fake_rename

            cell_dump_process.rename_filename(path=tmpdir)

            expected = [
                ("Cell.a.b.c.X.forward.input.0_float32_1.npy", "Cell.a.b.c.X.forward.0.input.0_float32_1.npy"),
                ("Cell.a.b.c.X.forward.input.1_float32_2.npy", "Cell.a.b.c.X.forward.0.input.1_float32_2.npy"),
                ("Cell.a.b.c.X.forward.output.0_float32_3.npy", "Cell.a.b.c.X.forward.0.output.0_float32_3.npy"),
                ("Cell.a.b.c.X.forward.input.0_float32_11.npy", "Cell.a.b.c.X.forward.1.input.0_float32_11.npy"),
                ("Cell.a.b.c.X.forward.input.1_float32_12.npy", "Cell.a.b.c.X.forward.1.input.1_float32_12.npy"),
                ("Cell.a.b.c.X.forward.output.0_float32_13.npy", "Cell.a.b.c.X.forward.1.output.0_float32_13.npy"),
                ("Cell.a.b.c.X.backward.input.0_float32_30.npy", "Cell.a.b.c.X.backward.0.input.0_float32_30.npy")
            ]
            self.assertEqual(rename_calls, expected)

    @patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.move_file")
    def test_rename_filename_statistics(self, mock_move_file):
        cell_dump_process.dump_task = CoreConst.STATISTICS

        data = {
            'Op Name': [
                "Cell.a.b.c.X.forward.input.0",
                "Cell.a.b.c.X.forward.input.1",
                "Cell.a.b.c.X.forward.output.0",
                "Cell.a.b.c.X.forward.input.0",
                "Cell.a.b.c.X.forward.input.1",
                "Cell.a.b.c.X.forward.output.0",
                "Cell.a.b.c.X.backward.input.0"
            ]
        }
        df = pd.DataFrame(data)

        cell_dump_process.rename_filename(data_df=df)

        self.assertEqual(df['Op Name'].iloc[0], "Cell.a.b.c.X.forward.0.input.0")
        self.assertEqual(df['Op Name'].iloc[1], "Cell.a.b.c.X.forward.0.input.1")
        self.assertEqual(df['Op Name'].iloc[2], "Cell.a.b.c.X.forward.0.output.0")
        self.assertEqual(df['Op Name'].iloc[3], "Cell.a.b.c.X.forward.1.input.0")
        self.assertEqual(df['Op Name'].iloc[4], "Cell.a.b.c.X.forward.1.input.1")
        self.assertEqual(df['Op Name'].iloc[5], "Cell.a.b.c.X.forward.1.output.0")
        self.assertEqual(df['Op Name'].iloc[6], "Cell.a.b.c.X.backward.0.input.0")


class TestConvertSpecialValues(unittest.TestCase):
    TEST_CASES = [
        ("true", True),
        ("True", True),
        ("false", False),
        ("False", False),
        ("1.23", 1.23),
        ("0", 0.0),
        ("-5.6", -5.6),
        (42, 42),
        (3.14, 3.14),
        (pd.NA, None)
    ]

    def test_convert_special_values(self):
        for input_value, expected in self.TEST_CASES:
            result = convert_special_values(input_value)
            self.assertEqual(result, expected)


class TestProcessCsv(unittest.TestCase):

    @staticmethod
    def make_df(rows):
        import pandas as pd
        return pd.DataFrame(rows)

    @patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.read_csv")
    def test_process_csv_input_and_output(self, mock_read_csv):
        rows = [
            {
                'Op Name': 'Cell.net.layer.forward.0.input.0',
                'Shape': '(2,3)',
                'Data Type': 'float32',
                'Max Value': 1.0,
                'Min Value': 0.0,
                'Avg Value': 0.5,
                'L2Norm Value': 2.0
            },
            {
                'Op Name': 'Cell.net.layer.forward.0.output.0',
                'Shape': '(2,3)',
                'Data Type': 'float32',
                'Max Value': 2.0,
                'Min Value': -1.0,
                'Avg Value': 0.0,
                'L2Norm Value': 3.0
            }
        ]
        df = self.make_df(rows)
        mock_read_csv.return_value = df

        result = process_csv("dummy_path")
        self.assertEqual(len(result), 2)

        op_name, key, tensor_json = result[0]
        self.assertEqual(op_name, 'Cell.net.layer.forward.0')
        self.assertEqual(key, CoreConst.INPUT_ARGS)
        self.assertEqual(tensor_json[CoreConst.TYPE], 'mindspore.Tensor')
        self.assertEqual(tensor_json[CoreConst.DTYPE], str(np_ms_dtype_dict['float32']))
        self.assertEqual(tensor_json[CoreConst.SHAPE], [2, 3])
        self.assertEqual(tensor_json[CoreConst.MAX], 1.0)
        self.assertEqual(tensor_json[CoreConst.MIN], 0.0)
        self.assertEqual(tensor_json[CoreConst.MEAN], 0.5)
        self.assertEqual(tensor_json[CoreConst.NORM], 2.0)

        op_name, key, tensor_json = result[1]
        self.assertEqual(op_name, 'Cell.net.layer.forward.0')
        self.assertEqual(key, CoreConst.OUTPUT)
        self.assertEqual(tensor_json[CoreConst.MAX], 2.0)
        self.assertEqual(tensor_json[CoreConst.MIN], -1.0)
        self.assertEqual(tensor_json[CoreConst.MEAN], 0.0)
        self.assertEqual(tensor_json[CoreConst.NORM], 3.0)

    @patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.read_csv")
    def test_process_csv_handles_missing_columns(self, mock_read_csv):
        rows = [
            {
                'Op Name': 'Cell.net.layer.forward.0.input.0',
                'Shape': '(1,)',
                'Data Type': 'int32'
            }
        ]
        df = self.make_df(rows)
        mock_read_csv.return_value = df

        result = process_csv("dummy_path")
        self.assertEqual(len(result), 1)
        op_name, key, tensor_json = result[0]
        self.assertEqual(tensor_json[CoreConst.DTYPE], str(np_ms_dtype_dict['int32']))
        self.assertEqual(tensor_json[CoreConst.SHAPE], [1])

    @patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.read_csv")
    def test_process_csv_handles_unknown_io_key(self, mock_read_csv):
        rows = [
            {
                'Op Name': 'Cell.net.layer.forward.0.unknown.0',
                'Shape': '(1,2)',
                'Data Type': 'float16'
            }
        ]
        df = self.make_df(rows)
        mock_read_csv.return_value = df

        result = process_csv("dummy_path")
        self.assertEqual(len(result), 1)
        op_name, key, tensor_json = result[0]
        self.assertIsNone(op_name)
        self.assertIsNone(key)
        self.assertIsNone(tensor_json)

    @patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.read_csv")
    def test_process_csv_shape_parsing(self, mock_read_csv):
        rows = [
            {
                'Op Name': 'Cell.net.layer.forward.0.input.0',
                'Shape': '(4, 5, 6)',
                'Data Type': 'float64'
            }
        ]
        df = self.make_df(rows)
        mock_read_csv.return_value = df

        result = process_csv("dummy_path")
        self.assertEqual(result[0][2][CoreConst.SHAPE], [4, 5, 6])

    @patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.read_csv")
    def test_process_csv_convert_special_values_bool_and_nan(self, mock_read_csv):
        rows = [
            {
                'Op Name': 'Cell.net.layer.forward.0.input.0',
                'Shape': '(1,)',
                'Data Type': 'float32',
                'Max Value': 'True',
                'Min Value': 'False',
                'Avg Value': float('nan'),
                'L2Norm Value': 1.23
            }
        ]
        df = self.make_df(rows)
        mock_read_csv.return_value = df

        result = process_csv("dummy_path")
        tensor_json = result[0][2]
        self.assertIs(tensor_json[CoreConst.MAX], True)
        self.assertIs(tensor_json[CoreConst.MIN], False)
        self.assertIsNone(tensor_json[CoreConst.MEAN])
        self.assertEqual(tensor_json[CoreConst.NORM], 1.23)


class TestCreateKbykJsonMultiRank(unittest.TestCase):
    @patch("msprobe.mindspore.dump.dump_processor.cell_dump_process.create_directory", lambda path: None)
    @patch(
        "msprobe.mindspore.dump.dump_processor.cell_dump_process.save_json",
        lambda path, data, indent=4: open(path, "w").write("test")
    )
    def test_create_kbyk_json_multi_rank(self):
        
        test_cases = [
            (None, "0kernel_kbyk_dump.json"),
            ("1", "1kernel_kbyk_dump.json"),
            ("3", "3kernel_kbyk_dump.json"),
        ]

        for rank_id_env, expected_prefix in test_cases:
            with tempfile.TemporaryDirectory() as dump_path:
                summary_mode = ["max"]
                step = 0
                # Patch environment variable
                if rank_id_env is not None:
                    with patch.dict(os.environ, {"RANK_ID": rank_id_env}):
                        config_json_path = create_kbyk_json(dump_path, summary_mode, step)
                else:
                    with patch.dict(os.environ, {}, clear=True):
                        config_json_path = create_kbyk_json(dump_path, summary_mode, step)
                self.assertEqual(os.path.basename(config_json_path), expected_prefix)
                self.assertTrue(config_json_path.startswith(dump_path))


if __name__ == '__main__':
    unittest.main()
