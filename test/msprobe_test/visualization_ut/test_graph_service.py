# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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
import re
import signal
import unittest
import shutil
import argparse
from dataclasses import dataclass

from unittest.mock import patch
from msprobe.visualization.graph_service import (
    _compare_graph_result,
    _build_graph_result,
    _compare_graph_ranks,
    _compare_graph_steps,
    _build_graph_ranks,
    _build_graph_steps,
    _graph_service_command,
    _graph_service_parser,
)
from msprobe.core.common.utils import CompareException
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from msprobe.core.common.const import Const, FileCheckConst
from msprobe.core.common.utils import CompareException
from msprobe.visualization import graph_service
from msprobe.visualization.graph_service import (
    ProgressConfig,
    _build_graph_result,
    _build_graph_ranks_parallel,
    _export_build_graph_result,
    _graph_service_command,
    _mp_compare_and_export,
    _run_build_graph_and_export,
    _run_build_graph_compare,
    _run_build_graph_single,
    _run_graph_compare,
    _run_with_progress,
)
from msprobe.visualization.utils import GraphConst, ParallelParam


@dataclass
class Args:
    target_path: str = None
    golden_path: str = None
    output_path: str = None
    layer_mapping: str = None
    overflow_check: bool = False
    fuzzy_match: bool = False
    is_print_compare_log: bool = True
    parallel_merge: bool = False
    parallel_params: tuple = None
    is_print_progress_log: bool = False
    file_type: str = 'db'

    rank_size: list = None
    tp: list = None
    pp: list = None
    vpp: list = (1,)
    order: list = ('tp-cp-ep-dp-pp',)


class TestGraphService(unittest.TestCase):
    def setUp(self):
        self.current_path = os.path.dirname(os.path.realpath(__file__))
        self.input = os.path.join(self.current_path, "input_format_correct")
        self.output = os.path.join(self.current_path, 'output')
        self.input_param = {
            'npu_path': os.path.join(self.input, 'step0', 'rank0'),
            'bench_path': os.path.join(self.input, 'step0', 'rank0'),
            'is_print_compare_log': True,
        }
        self.layer_mapping = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'layer_mapping.yaml')
        self.pattern = r'\b\w+\.vis.db\b'
        self.pattern_rank = r'[\w_]+\.vis.db\b'
        self.output_json = []
        for i in range(7):
            self.output_json.append(os.path.join(self.current_path, f"compare{i}.json"))
        self.original_sigpipe = signal.getsignal(signal.SIGPIPE)
        signal.signal(signal.SIGPIPE, signal.SIG_IGN)

    def assert_log_info(
        self, mock_log_info, log_info='Model graphs compared successfully, the result file is saved in'
    ):
        last_call_args = mock_log_info.call_args[0][0]
        self.assertIn(log_info, last_call_args)
        matches = re.findall(self.pattern, last_call_args)
        if matches:
            self.assertTrue(os.path.exists(os.path.join(self.output, matches[0])))

    @patch('msprobe.core.common.log.logger.info')
    def test_compare_graph_result(self, mock_log_info):
        args = Args(output_path=self.output)
        result = _compare_graph_result(self.input_param, args)
        self.assertEqual(mock_log_info.call_count, 2)
        self.assertIsNotNone(result)

        args = Args(output_path=self.output)
        result = _compare_graph_result(self.input_param, args)
        self.assertIsNotNone(result)

        args = Args(output_path=self.output, layer_mapping=self.layer_mapping)
        result = _compare_graph_result(self.input_param, args)
        self.assertIsNotNone(result)

        args = Args(output_path=self.output, overflow_check=True)
        result = _compare_graph_result(self.input_param, args)
        self.assertIsNotNone(result)

    @patch('msprobe.core.common.log.logger.info')
    def test_build_graph_result(self, mock_log_info):
        result = _build_graph_result(os.path.join(self.input, 'step0', 'rank0'), Args(overflow_check=True))
        self.assertEqual(mock_log_info.call_count, 1)
        self.assertIsNotNone(result)

    @patch('msprobe.core.common.log.logger.info')
    def test_compare_graph_ranks(self, mock_log_info):
        input_param = {
            'npu_path': os.path.join(self.input, 'step0'),
            'bench_path': os.path.join(self.input, 'step0'),
            'is_print_compare_log': True,
        }
        args = Args(output_path=self.output)
        _compare_graph_ranks(input_param, args)
        self.assert_log_info(mock_log_info, 'Successfully exported compare graph results.')

    @patch('msprobe.core.common.log.logger.info')
    def test_compare_graph_steps(self, mock_log_info):
        input_param = {'npu_path': self.input, 'bench_path': self.input, 'is_print_compare_log': True}
        args = Args(output_path=self.output)
        _compare_graph_steps(input_param, args)
        self.assert_log_info(mock_log_info, 'Successfully exported compare graph results.')

        input_param1 = {
            'npu_path': self.input,
            'bench_path': os.path.join(self.current_path, "input"),
            'is_print_compare_log': True,
        }
        args = Args(output_path=self.output)
        with self.assertRaises(CompareException):
            _compare_graph_steps(input_param1, args)

    @patch('msprobe.core.common.log.logger.info')
    def test_build_graph_ranks(self, mock_log_info):
        _build_graph_ranks(Args(target_path=self.input, output_path=self.output), 'step0')
        self.assert_log_info(mock_log_info, "Successfully exported build graph results.")

    @patch('msprobe.core.common.log.logger.info')
    def test_build_graph_steps(self, mock_log_info):
        _build_graph_steps(Args(target_path=self.input, output_path=self.output))
        self.assert_log_info(mock_log_info, "Successfully exported build graph results.")

    @patch('msprobe.core.common.log.logger.info')
    def test_graph_service_command(self, mock_log_info):
        args = Args(
            target_path=self.input_param.get('npu_path'),
            golden_path=self.input_param.get('bench_path'),
            output_path=self.output,
        )
        _graph_service_command(args)
        self.assert_log_info(mock_log_info, 'Adding index to db file completed.')

    @patch('msprobe.core.common.log.logger.info')
    def test_graph_service_command1(self, mock_log_info):
        args = Args(target_path=os.path.join(self.input, 'step0', 'rank0'), output_path=self.output)
        _graph_service_command(args)
        self.assert_log_info(mock_log_info, "Adding index to db file completed.")

    @patch('msprobe.core.common.log.logger.info')
    def test_graph_service_command2(self, mock_log_info):
        args = Args(
            target_path=os.path.join(self.input, 'step0'),
            golden_path=os.path.join(self.input, 'step0'),
            output_path=self.output,
        )
        _graph_service_command(args)
        self.assert_log_info(mock_log_info, 'Adding index to db file completed.')

    @patch('msprobe.core.common.log.logger.info')
    def test_graph_service_command3(self, mock_log_info):
        args = Args(target_path=self.input, golden_path=self.input, output_path=self.output)
        _graph_service_command(args)
        self.assert_log_info(mock_log_info, 'Adding index to db file completed.')

    @patch('msprobe.core.common.log.logger.info')
    def test_graph_service_command4(self, mock_log_info):
        args = Args(target_path=os.path.join(self.input, 'step0'), output_path=self.output)
        _graph_service_command(args)
        self.assert_log_info(mock_log_info, "Adding index to db file completed.")

    @patch('msprobe.core.common.log.logger.info')
    def test_graph_service_command5(self, mock_log_info):
        args = Args(target_path=self.input, output_path=self.output)
        _graph_service_command(args)
        self.assert_log_info(mock_log_info, "Adding index to db file completed.")

    @patch('msprobe.core.common.log.logger.info')
    def test_graph_service_command6(self, mock_log_info):
        args = Args(target_path=self.input, golden_path=os.path.join(self.input, 'step0'), output_path=self.output)
        with self.assertRaises(ValueError):
            _graph_service_command(args)

    @patch('msprobe.core.common.log.logger.info')
    def test_graph_service_command7(self, mock_log_info):
        args = Args(
            target_path=os.path.join(self.input, 'step0'),
            golden_path=os.path.join(self.input, 'step0'),
            output_path=self.output,
            rank_size=[2, 2],
            tp=[2, 2],
            pp=[1, 1],
        )
        _graph_service_command(args)
        self.assert_log_info(mock_log_info, 'Adding index to db file completed.')

    @patch('msprobe.core.common.log.logger.info')
    def test_graph_service_command8(self, mock_log_info):
        args = Args(
            target_path=os.path.join(self.input, 'step0'),
            golden_path=os.path.join(self.input, 'step1'),
            output_path=self.output,
        )
        _graph_service_command(args)
        self.assert_log_info(mock_log_info, 'Adding index to db file completed.')

    def test_graph_service_parser(self):
        parser = argparse.ArgumentParser()
        _graph_service_parser(parser)
        args = parser.parse_args(['-tp', 'input.json', '-o', 'output.json'])
        self.assertEqual(args.target_path, 'input.json')
        self.assertEqual(args.output_path, 'output.json')

        args = parser.parse_args(['-tp', 'input.json', '-o', 'output.json', '-lm', 'mapping.json'])
        self.assertEqual(args.layer_mapping, 'mapping.json')

        args = parser.parse_args(['-tp', 'input.json', '-o', 'output.json', '-oc'])
        self.assertTrue(args.overflow_check)

        args = parser.parse_args(['-tp', 'input.json', '-o', 'output.json'])
        self.assertFalse(args.overflow_check)

        actions = {action.dest: action for action in parser._actions}
        self.assertEqual(actions['overflow_check'].help, 'Whether to enable overflow check for the graph.')
        self.assertEqual(
            actions['file_type'].help,
            'Output file type; only supported in graph merge build scenarios.',
        )

    def tearDown(self):
        signal.signal(signal.SIGPIPE, self.original_sigpipe)
        if os.path.exists(self.output):
            shutil.rmtree(self.output)
        for json_data in self.output_json:
            if os.path.exists(json_data):
                os.remove(json_data)



class TestGraphServiceExtra(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_build_graph_result(self):
        graph = MagicMock()
        graph.paging_by_micro_step.return_value = ["m1"]
        graph_info = SimpleNamespace(graph=graph)
        args = SimpleNamespace(overflow_check=True)
        with patch("msprobe.visualization.graph_service._build_graph_info", return_value=graph_info):
            result = _build_graph_result("path", args)
        self.assertIs(result.graph, graph)
        self.assertEqual(result.micro_steps, ["m1"])
        graph.overflow_check.assert_called_once()

    def test_run_build_graph_compare(self):
        graph_n = MagicMock()
        graph_b = MagicMock()
        args = MagicMock()
        with patch("msprobe.visualization.graph_service._build_graph_info",
                   side_effect=[graph_n, graph_b]):
            result = _run_build_graph_compare({"npu_path": "n", "bench_path": "b"}, args, "rank0", "rank0")
        self.assertIs(result.graph_info_n, graph_n)
        self.assertIs(result.graph_info_b, graph_b)
        self.assertEqual(result.npu_rank, "rank0")

    def test_run_build_graph_single(self):
        result = SimpleNamespace(rank=0, step=0)
        args = MagicMock()
        with patch("msprobe.visualization.graph_service._build_graph_result", return_value=result):
            _run_build_graph_single("/data", f"{Const.RANK}2", "step1", args)
        self.assertEqual(result.rank, 2)

        with patch("msprobe.visualization.graph_service._build_graph_result", return_value=result):
            _run_build_graph_single("/data", Const.RANK, "step1", args)

    def test_run_build_graph_and_export(self):
        result = SimpleNamespace(step=0, rank=0)
        args = SimpleNamespace(output_path=self.base)
        with patch("msprobe.visualization.graph_service._run_build_graph_single", return_value=result), \
                patch("msprobe.visualization.graph_service.create_directory") as create, \
                patch("msprobe.visualization.graph_service._export_build_graph_result",
                      return_value="ok") as export:
            output = _run_build_graph_and_export("/data", "rank0", "step3", args)
        self.assertEqual(output, "ok")
        self.assertEqual(result.step, 3)
        create.assert_called_once_with(self.base)
        export.assert_called_once()

    def test_run_graph_compare(self):
        graph_n = MagicMock()
        graph_b = MagicMock()
        graph_task_info = SimpleNamespace(
            graph_info_n=graph_n, graph_info_b=graph_b, npu_rank=f"{Const.RANK}3"
        )
        comparator = MagicMock()
        args = SimpleNamespace(overflow_check=True)
        with patch("msprobe.visualization.graph_service._compare_graph", return_value=comparator):
            result = _run_graph_compare(graph_task_info, {}, args)
        self.assertIs(result.graph_n, graph_n.graph)
        self.assertIs(result.graph_b, graph_b.graph)
        self.assertEqual(result.rank, 3)
        graph_n.graph.overflow_check.assert_called_once()

    def test_export_build_graph_result_json(self):
        args = SimpleNamespace(output_path=self.base, overflow_check=False, file_type="json")
        result = SimpleNamespace(graph=MagicMock(), micro_steps=1, rank=0, step=0)
        with patch("msprobe.visualization.graph_service.GraphBuilder.to_json") as to_json:
            self.assertIsNone(_export_build_graph_result(args, result))
        to_json.assert_called_once()

        with patch("msprobe.visualization.graph_service.GraphBuilder.to_json",
                   side_effect=RuntimeError("boom")):
            self.assertEqual(_export_build_graph_result(args, result), "json_export_error")

    def test_export_build_graph_result_db(self):
        args = SimpleNamespace(
            output_path=self.base, overflow_check=False, file_type="db",
            output_db_name="out.db", rank_list=[0], step_list=[0],
        )
        result = SimpleNamespace(graph=MagicMock(), micro_steps=1, rank=0, step=0)
        with patch("msprobe.visualization.graph_service.GraphBuilder.to_db") as to_db:
            self.assertIsNone(_export_build_graph_result(args, result))
        to_db.assert_called_once()

        with patch("msprobe.visualization.graph_service.GraphBuilder.to_db",
                   side_effect=RuntimeError("boom")):
            self.assertEqual(_export_build_graph_result(args, result), "out.db")

    def test_mp_compare_and_export(self):
        graph_result = SimpleNamespace(step=0, rank=0)
        args = SimpleNamespace(output_path=self.base)
        with patch("msprobe.visualization.graph_service._mp_compare", return_value=graph_result), \
                patch("msprobe.visualization.graph_service.create_directory"), \
                patch("msprobe.visualization.graph_service._export_compare_graph_result",
                      return_value="ok") as export:
            output = _mp_compare_and_export({}, args, "rank0", "step2")
        self.assertEqual(output, "ok")
        self.assertEqual(graph_result.step, 2)

    def test_build_graph_ranks_parallel(self):
        args = SimpleNamespace(
            target_path=os.path.join(self.base, "step0"),
            output_path=self.base,
            parallel_params=[ParallelParam(2, 1, 2)],
            rank_list=None,
        )
        build_result = SimpleNamespace(rank=0)
        with patch("msprobe.visualization.graph_service.check_and_return_dir_contents",
                   return_value=["rank0", "rank1"]), \
                patch("msprobe.visualization.graph_service._run_build_graph_single",
                      return_value=build_result), \
                patch("msprobe.visualization.graph_service._export_build_graph_result",
                      return_value=None), \
                patch("msprobe.visualization.graph_service.GraphMerger") as merger, \
                patch("msprobe.visualization.graph_service.validate_parallel_param"), \
                patch("msprobe.visualization.graph_service.create_directory"), \
                patch("msprobe.visualization.graph_service.Pool") as pool_cls, \
                patch("msprobe.visualization.graph_service.PbarInfo") as pbar_info_cls:
            pool = pool_cls.return_value.__enter__.return_value
            task = MagicMock()
            task.get.return_value = build_result
            pool.apply_async.return_value = task
            merger.return_value.merge_graph.return_value = [build_result]
            pbar_info_cls.update_task_id.return_value = MagicMock()
            _build_graph_ranks_parallel(args, step="step0", pbar_info=MagicMock(step=1))
        merger.assert_called_once()

    def test_build_graph_ranks_parallel_no_ranks_no_merge(self):
        args = SimpleNamespace(
            target_path=self.base, output_path=self.base, parallel_params=None, rank_list=None
        )
        with patch("msprobe.visualization.graph_service.check_and_return_dir_contents", return_value=[]), \
                patch("msprobe.visualization.graph_service.find_proc_dir", return_value="/x/proc0"), \
                patch("msprobe.visualization.graph_service._run_build_graph_single",
                      return_value=SimpleNamespace(rank=0, step=0)), \
                patch("msprobe.visualization.graph_service._export_build_graph_result",
                      return_value="fail"), \
                patch("msprobe.visualization.graph_service.create_directory"), \
                patch("msprobe.visualization.graph_service.Pool") as pool_cls, \
                patch("msprobe.visualization.graph_service.PbarInfo") as pbar_info_cls:
            pool = pool_cls.return_value.__enter__.return_value
            task = MagicMock()
            task.get.return_value = SimpleNamespace(rank=0, step=0)
            pool.apply_async.return_value = task
            pbar_info_cls.update_task_id.return_value = MagicMock()
            _build_graph_ranks_parallel(args)

    def _args(self, **kwargs):
        defaults = dict(
            target_path=self.base,
            golden_path=None,
            rank_size=None,
            tp=None,
            pp=None,
            vpp=None,
            order=None,
            is_print_progress_log=False,
            is_print_compare_log=False,
            file_type="db",
            output_path=self.base,
            overflow_check=False,
            parallel_merge=False,
            parallel_params=None,
        )
        defaults.update(kwargs)
        return SimpleNamespace(**defaults)

    def test_graph_service_command_json_with_bench(self):
        args = self._args(file_type="json", golden_path="bench")
        with patch("msprobe.visualization.graph_service.ProgressInfo.update_process_running"):
            with self.assertRaises(CompareException):
                _graph_service_command(args)

    def test_graph_service_command_build_single(self):
        args = self._args(file_type="json")
        with patch("msprobe.visualization.graph_service.check_file_or_directory_path"), \
                patch("msprobe.visualization.graph_service.check_file_type",
                      return_value=FileCheckConst.DIR), \
                patch("msprobe.visualization.graph_service.check_directory_content",
                      return_value=GraphConst.FILES), \
                patch("msprobe.visualization.graph_service._build_graph_with_pbar") as build, \
                patch("msprobe.visualization.graph_service.ProgressInfo.update_process_running"):
            _graph_service_command(args)
        build.assert_called_once()
        self.assertEqual(args.file_type, "db")

    def test_graph_service_command_build_ranks_and_steps(self):
        for content, target in ((GraphConst.RANKS, "_build_graph_ranks_with_pbar"),
                                (GraphConst.STEPS, "_build_graph_steps_with_pbar")):
            args = self._args()
            with self.subTest(content=content), \
                    patch("msprobe.visualization.graph_service.check_file_or_directory_path"), \
                    patch("msprobe.visualization.graph_service.check_file_type",
                          return_value=FileCheckConst.DIR), \
                    patch("msprobe.visualization.graph_service.check_directory_content",
                          return_value=content), \
                    patch(f"msprobe.visualization.graph_service.{target}") as build, \
                    patch("msprobe.visualization.graph_service.ProgressInfo.update_process_running"):
                _graph_service_command(args)
            build.assert_called_once()

    def test_graph_service_command_parallel_json_files_error(self):
        args = self._args(file_type="json", rank_size=[2], parallel_merge=True)
        with patch("msprobe.visualization.graph_service.check_file_or_directory_path"), \
                patch("msprobe.visualization.graph_service.check_file_type",
                      return_value=FileCheckConst.DIR), \
                patch("msprobe.visualization.graph_service.check_directory_content",
                      return_value=GraphConst.FILES), \
                patch("msprobe.visualization.graph_service.load_parallel_param",
                      return_value=[ParallelParam(2, 1, 2)]), \
                patch("msprobe.visualization.graph_service.ProgressInfo.update_process_running"):
            with self.assertRaises(CompareException):
                _graph_service_command(args)

    def test_graph_service_command_compare(self):
        cases = [
            (GraphConst.RANKS, "_compare_graph_ranks_with_pbar"),
            (GraphConst.STEPS, "_compare_graph_steps_with_pbar"),
            (GraphConst.FILES, "_compare_graph_with_pbar"),
        ]
        for content, target in cases:
            args = self._args(golden_path="bench")
            with self.subTest(content=content), \
                    patch("msprobe.visualization.graph_service.check_file_or_directory_path"), \
                    patch("msprobe.visualization.graph_service.check_file_type",
                          return_value=FileCheckConst.DIR), \
                    patch("msprobe.visualization.graph_service.check_directory_content",
                          return_value=content), \
                    patch(f"msprobe.visualization.graph_service.{target}") as compare, \
                    patch("msprobe.visualization.graph_service.ProgressInfo.update_process_running"):
                _graph_service_command(args)
            compare.assert_called_once()

    def test_graph_service_command_mismatched_content(self):
        args = self._args(golden_path="bench")
        with patch("msprobe.visualization.graph_service.check_file_or_directory_path"), \
                patch("msprobe.visualization.graph_service.check_file_type",
                      return_value=FileCheckConst.DIR), \
                patch("msprobe.visualization.graph_service.check_directory_content",
                      side_effect=[GraphConst.RANKS, GraphConst.STEPS]), \
                patch("msprobe.visualization.graph_service.ProgressInfo.update_process_running"):
            with self.assertRaises(ValueError):
                _graph_service_command(args)

    def test_graph_service_command_invalid_path_type(self):
        args = self._args()
        with patch("msprobe.visualization.graph_service.check_file_or_directory_path"), \
                patch("msprobe.visualization.graph_service.check_file_type",
                      return_value=FileCheckConst.FILE), \
                patch("msprobe.visualization.graph_service.ProgressInfo.update_process_running"):
            with self.assertRaises(CompareException):
                _graph_service_command(args)

    def test_run_with_progress_json(self):
        args = SimpleNamespace(output_path=self.base, file_type="json", parallel_merge=False)
        core_func = MagicMock()
        config = ProgressConfig(
            core_func=core_func,
            get_ranks=MagicMock(),
            db_name="x.db",
            pbar_info_kwargs={"stage_total": 2},
            use_monitor_thread=False,
            tqdm_total=10,
        )
        with patch("msprobe.visualization.graph_service.PbarInfo") as pbar_info_cls, \
                patch("msprobe.visualization.graph_service.tqdm") as tqdm_cls, \
                patch("msprobe.visualization.graph_service.post_process_db_pbar") as post:
            pbar_info_cls.return_value = MagicMock(total=10)
            tqdm_cls.return_value.__enter__.return_value = MagicMock()
            _run_with_progress(None, args, config)
        core_func.assert_called_once()
        post.assert_not_called()

    def test_run_with_progress_db(self):
        args = SimpleNamespace(output_path=self.base, file_type="db", parallel_merge=False, overflow_check=False)
        config = ProgressConfig(
            core_func=MagicMock(),
            get_ranks=MagicMock(),
            db_name="x.db",
            pbar_info_kwargs={"stage_total": 2},
            use_monitor_thread=False,
        )
        with patch("msprobe.visualization.graph_service.PbarInfo") as pbar_info_cls, \
                patch("msprobe.visualization.graph_service.tqdm") as tqdm_cls, \
                patch("msprobe.visualization.graph_service.post_process_db") as post, \
                patch("msprobe.visualization.graph_service.distributed_analyse") as analyse:
            pbar_info_cls.return_value = MagicMock(total=10)
            tqdm_cls.return_value.__enter__.return_value = MagicMock()
            _run_with_progress({}, args, config)
        post.assert_called_once()
        analyse.assert_not_called()

if __name__ == '__main__':
    unittest.main()
