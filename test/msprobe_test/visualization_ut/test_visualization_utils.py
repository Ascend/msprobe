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
import unittest
from msprobe.visualization.utils import (load_json_file, load_data_json_file, str2float, check_directory_content,
                                         GraphConst, SerializableArgs)
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from msprobe.core.common.const import Const
from msprobe.core.common.exceptions import MsprobeException
from msprobe.core.common.utils import CompareException
from msprobe.visualization.utils import (
    GraphConst,
    ParallelParam,
    ProgressInfo,
    calculate_list,
    check_directory_content,
    extract_rank_number,
    get_log_msg_wrapper,
    get_step_or_rank_int,
    is_serializable,
    load_json_file,
    load_parallel_param,
    monitor_progress,
    sort_rank_number_strings,
    update_pbar_info,
    update_progress_info,
    update_shared_dict,
    validate_parallel_param,
)


class TestMappingConfig(unittest.TestCase):

    def setUp(self):
        self.yaml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "mapping.yaml")
        self.input = os.path.join(os.path.dirname(os.path.realpath(__file__)), "input")

    def test_load_json_file(self):
        result = load_json_file(self.yaml_path)
        self.assertEqual(result, {})

    def test_load_data_json_file(self):
        result = load_data_json_file(self.yaml_path)
        self.assertEqual(result, {})

    def test_str2float(self):
        result = str2float('23.4%')
        self.assertAlmostEqual(result, 0.234)
        result = str2float('2.3.4%')
        self.assertAlmostEqual(result, 0)

    def test_check_directory_content(self):
        input_type = check_directory_content(self.input)
        self.assertEqual(input_type, GraphConst.STEPS)

        input_type = check_directory_content(os.path.join(self.input, "step0"))
        self.assertEqual(input_type, GraphConst.RANKS)

        with self.assertRaises(ValueError):
            check_directory_content(os.path.join(self.input, "step1"))

        input_type = check_directory_content(os.path.join(self.input, "step0", "rank0"))
        self.assertEqual(input_type, GraphConst.FILES)

    def test_serializable_args(self):
        class TmpArgs:
            def __init__(self, a, b, c):
                self.a = a
                self.b = b
                self.c = c
        input_args1 = TmpArgs('a', 123, [1, 2, 3])
        serializable_args1 = SerializableArgs(input_args1)
        self.assertEqual(serializable_args1.__dict__, input_args1.__dict__)
        input_args2 = TmpArgs('a', 123, lambda x: print(x))
        serializable_args2 = SerializableArgs(input_args2)
        self.assertNotEqual(serializable_args2.__dict__, input_args2.__dict__)





class TestUtilsExtra(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def _make_ranks(self, base, count):
        for i in range(count):
            os.makedirs(os.path.join(base, f"{Const.RANK}{i}"))

    def test_load_json_file(self):
        path = os.path.join(self.base, "a.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"a": 1}, f)
        self.assertEqual(load_json_file(path), {"a": 1})

        with open(path, "w", encoding="utf-8") as f:
            f.write("[1, 2]")
        self.assertEqual(load_json_file(path), {})

        with open(path, "w", encoding="utf-8") as f:
            f.write("{invalid")
        self.assertEqual(load_json_file(path), {})

    def test_get_step_or_rank_int(self):
        self.assertEqual(get_step_or_rank_int(Const.RANK), 0)
        self.assertEqual(get_step_or_rank_int(None), 0)
        self.assertEqual(get_step_or_rank_int(f"{Const.PROC}123"), 0)
        self.assertEqual(get_step_or_rank_int(f"{Const.RANK}12", is_rank=True), 12)
        self.assertEqual(get_step_or_rank_int(f"{Const.STEP}3"), 3)
        with self.assertRaises(RuntimeError):
            get_step_or_rank_int(f"{Const.RANK}x", is_rank=True)

    def test_check_directory_content(self):
        empty = os.path.join(self.base, "empty")
        os.makedirs(empty)
        with self.assertRaises(ValueError):
            check_directory_content(empty)

        real_data = os.path.join(self.base, "real")
        os.makedirs(os.path.join(real_data, Const.DUMP_TENSOR_DATA))
        self.assertEqual(check_directory_content(real_data), GraphConst.FILES)

        files_dir = os.path.join(self.base, "files")
        os.makedirs(files_dir)
        with open(os.path.join(files_dir, "a.bin"), "w", encoding="utf-8") as f:
            f.write("x")
        self.assertEqual(check_directory_content(files_dir), GraphConst.FILES)

        ranks = os.path.join(self.base, "ranks")
        self._make_ranks(ranks, 2)
        self.assertEqual(check_directory_content(ranks), GraphConst.RANKS)

        single_rank = os.path.join(self.base, "single")
        os.makedirs(os.path.join(single_rank, Const.RANK))
        self.assertEqual(check_directory_content(single_rank), GraphConst.RANKS)

        proc_dir = os.path.join(self.base, "proc")
        os.makedirs(os.path.join(proc_dir, f"{Const.PROC}123"))
        self.assertEqual(check_directory_content(proc_dir), GraphConst.RANKS)

        steps = os.path.join(self.base, "steps")
        os.makedirs(os.path.join(steps, f"{Const.STEP}0"))
        os.makedirs(os.path.join(steps, f"{Const.STEP}1"))
        self.assertEqual(check_directory_content(steps), GraphConst.STEPS)

        mixed = os.path.join(self.base, "mixed")
        os.makedirs(os.path.join(mixed, f"{Const.RANK}0"))
        os.makedirs(os.path.join(mixed, "other"))
        with self.assertRaises(ValueError):
            check_directory_content(mixed)

    def test_extract_and_sort_rank(self):
        self.assertEqual(extract_rank_number("rank10"), 10)
        self.assertEqual(extract_rank_number("rankx"), 0)
        self.assertEqual(
            sort_rank_number_strings(["rank10", "rank2", "rank1"]),
            ["rank1", "rank2", "rank10"],
        )

    def test_validate_parallel_param_valid(self):
        dump_path = os.path.join(self.base, "validate")
        self._make_ranks(dump_path, 4)
        validate_parallel_param(ParallelParam(4, 2, 2), dump_path)

        dump_path_vpp = os.path.join(self.base, "validate_vpp")
        self._make_ranks(dump_path_vpp, 4)
        validate_parallel_param(ParallelParam(4, 2, 2, vpp=2), dump_path_vpp)

    def test_validate_parallel_param_invalid(self):
        dump_path = os.path.join(self.base, "validate_invalid")
        self._make_ranks(dump_path, 12)
        cases = [
            ParallelParam(3, 2, 2),
            ParallelParam(4, None, 2),
            ParallelParam(4, True, 2),
            ParallelParam(4, 2.0, 2),
            ParallelParam(4, 0, 2),
            ParallelParam(4, 5, 1),
            ParallelParam(4, 1, 5),
            ParallelParam(4, 3, 1),
            ParallelParam(4, 1, 3),
            ParallelParam(12, 4, 4),
            ParallelParam(4, 2, 1, vpp=2),
            ParallelParam(4, 2, 2, order=1),
            ParallelParam(4, 2, 2, order="ABC"),
        ]
        for param in cases:
            with self.subTest(param=(param.rank_size, param.tp, param.pp, param.vpp, param.order)):
                with self.assertRaises(MsprobeException):
                    validate_parallel_param(param, dump_path)

    def test_load_parallel_param(self):
        args = SimpleNamespace(
            rank_size=[4], tp=[2], pp=[2], vpp=[], order=[], golden_path=None
        )
        self.assertEqual(load_parallel_param(args)[0].rank_size, 4)

        args = SimpleNamespace(
            rank_size=[4], tp=[2], pp=[2], vpp=[2], order=["tp-pp"], golden_path=None
        )
        params = load_parallel_param(args)
        self.assertEqual(params[0].vpp, 2)
        self.assertEqual(params[0].order, "tp-pp")

        args = SimpleNamespace(
            rank_size=[4, 4], tp=[2, 2], pp=[2, 2], vpp=[1, 1],
            order=["tp-pp", "pp-tp"], golden_path="golden",
        )
        params = load_parallel_param(args)
        self.assertEqual(len(params), 2)
        self.assertEqual(params[1].order, "pp-tp")

    def test_load_parallel_param_invalid(self):
        invalid_args = [
            SimpleNamespace(rank_size=[4], tp=[], pp=[2], vpp=[], order=[], golden_path=None),
            SimpleNamespace(rank_size=[4, 4], tp=[2, 2], pp=[2, 2], vpp=[], order=[], golden_path=None),
            SimpleNamespace(rank_size=[4], tp=[2, 2], pp=[2, 2], vpp=[], order=[], golden_path="g"),
            SimpleNamespace(rank_size=[4], tp=[2, 2], pp=[2], vpp=[], order=[], golden_path=None),
            SimpleNamespace(rank_size=[4, 4], tp=[2], pp=[2, 2], vpp=[], order=[], golden_path="g"),
        ]
        for args in invalid_args:
            with self.subTest(args=vars(args)):
                with self.assertRaises(MsprobeException):
                    load_parallel_param(args)

    def test_calculate_list(self):
        npu = os.path.join(self.base, "npu")
        bench = os.path.join(self.base, "bench")
        self._make_ranks(npu, 2)
        for name in ("rank1", "rank2"):
            os.makedirs(os.path.join(bench, name))
        self.assertEqual(
            calculate_list(npu, bench, mode=GraphConst.INTERSECTION),
            ["rank1"],
        )
        self.assertEqual(
            calculate_list(npu, bench, mode=GraphConst.UNION),
            ["rank0", "rank1", "rank2"],
        )
        same = os.path.join(self.base, "same")
        self._make_ranks(same, 2)
        self.assertEqual(calculate_list(npu, same), ["rank0", "rank1"])

    def test_calculate_list_invalid(self):
        npu = os.path.join(self.base, "npu2")
        bench = os.path.join(self.base, "bench2")
        self._make_ranks(npu, 1)
        self._make_ranks(bench, 1)
        with self.assertRaises(MsprobeException):
            calculate_list(npu, bench, prefix_str="bad")
        with self.assertRaises(MsprobeException):
            calculate_list(npu, bench, mode="bad")

        empty = os.path.join(self.base, "empty_calc")
        os.makedirs(empty)
        with patch("msprobe.visualization.utils.find_proc_dir", return_value=None):
            with self.assertRaises(CompareException):
                calculate_list(empty, bench)
        with patch("msprobe.visualization.utils.find_proc_dir", return_value="/x/proc0"):
            self.assertEqual(calculate_list(empty, bench), ["proc0"])

        no_overlap = os.path.join(self.base, "no_overlap")
        self._make_ranks(no_overlap, 1)
        os.rename(os.path.join(no_overlap, "rank0"), os.path.join(no_overlap, "rank9"))
        with self.assertRaises(CompareException):
            calculate_list(npu, no_overlap, mode=GraphConst.INTERSECTION)

    def test_is_serializable(self):
        self.assertTrue(is_serializable(1))
        self.assertFalse(is_serializable(lambda x: x))

        class Bad:
            def __reduce__(self):
                raise ValueError("bad")

        with self.assertRaises(RuntimeError):
            is_serializable(Bad())

    def test_progress_info(self):
        ProgressInfo.reset()
        ProgressInfo.info_dict = {}
        ProgressInfo.print_progress_log = True
        with patch("msprobe.visualization.utils.logger.info") as log:
            ProgressInfo.update_current_progress(10)
            ProgressInfo.update_process_running(False)
            ProgressInfo.update_error_msg("error")
        self.assertEqual(ProgressInfo.current_progress, 10)
        self.assertFalse(ProgressInfo.process_running)
        self.assertEqual(ProgressInfo.error_msg, ["error"])
        self.assertEqual(log.call_count, 3)

    def test_get_log_msg_wrapper(self):
        ProgressInfo.reset()
        ProgressInfo.print_progress_log = False

        class Handler:
            @get_log_msg_wrapper
            def show(self, msg):
                return msg

        self.assertEqual(Handler().show("hello"), "hello")
        self.assertEqual(ProgressInfo.error_msg, ["hello"])

    def test_update_shared_dict(self):
        shared = {"a": 1, "b": 2}
        update_shared_dict(shared, "a", 3)
        self.assertEqual(shared["a"], 4)
        update_shared_dict(shared, "a", 3, update_all=True)
        self.assertEqual(shared, {"a": 7, "b": 5})

    def test_update_pbar_info(self):
        pbar_info = SimpleNamespace(
            total=100,
            stage_total=2,
            stage_progress=10,
            current_stage_dict={"task": 2},
            task_id="task",
            pbar=None,
            progress_dict={"task": 0},
        )
        update_pbar_info(pbar_info, 5, 50)
        self.assertEqual(pbar_info.progress_dict["task"], 15)
        update_pbar_info(pbar_info, 5, 50, update_all=True)

        pbar = MagicMock()
        pbar_info.pbar = pbar
        update_pbar_info(pbar_info, 100, 100, step_ratio=1)
        pbar.refresh.assert_called_once()

    def test_update_progress_info(self):
        proc = MagicMock()
        proc.stdout.readline.side_effect = [
            'PROGRESS: {"current_progress": 10, "process_running": true, "error_msg": []}\n',
            "",
            "",
        ]
        proc.poll.side_effect = [0, 1, 1]
        progress_info = MagicMock()
        with patch("msprobe.visualization.utils.time.sleep"):
            update_progress_info(proc, progress_info)
        self.assertEqual(progress_info.current_progress, 10)
        self.assertTrue(progress_info.process_running)
        proc.stdout.close.assert_called_once()

    def test_monitor_progress_stops_immediately(self):
        pbar_info = SimpleNamespace(
            stage_total=2,
            progress_dict={},
            total=100,
            step_total=1,
            stop_monitor=True,
            wait_monitor=False,
            continue_monitor=False,
        )
        pbar = MagicMock()
        monitor_progress(pbar_info, pbar, ["a", "b"])
        pbar.set_description.assert_called_once()

if __name__ == '__main__':
    unittest.main()
