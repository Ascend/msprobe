import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from msprobe.core.common.const import CompareConst
from msprobe.pytorch.api_accuracy_checker.compare.compare import (
    Comparator,
    ResultInfo,
    _aggregate_status,
    _ensure_result_info,
    _format_subject_value,
)
from msprobe.pytorch.api_accuracy_checker.compare.compare_column import CompareColumn


def _make_comparator(tmpdir, is_continue=True):
    """Build a Comparator pointing at temporary CSV files.

    By default ``is_continue=True`` is used so tests don't have to clean up
    header rows they don't care about. Tests that explicitly want to inspect
    the headers should pass ``is_continue=False``.
    """
    summary_path = os.path.join(tmpdir, "summary.csv")
    detail_path = os.path.join(tmpdir, "detail.csv")
    return Comparator(summary_path, detail_path, is_continue_acc_check=is_continue)


class TestEnsureResultInfo(unittest.TestCase):
    def test_passes_namedtuple_through(self):
        ri = ResultInfo("torch.add.0", "pass", "pass", [], [], 0)
        self.assertIs(_ensure_result_info(ri), ri)

    def test_builds_namedtuple_from_tuple(self):
        ri = _ensure_result_info(("torch.add.0", "pass", "pass", [], [], 1))
        self.assertIsInstance(ri, ResultInfo)
        self.assertEqual(ri.full_api_name, "torch.add.0")
        self.assertEqual(ri.rank, 1)
        self.assertEqual(ri.fwd_success_status, "pass")


class TestFormatSubjectValue(unittest.TestCase):
    def test_passes_int_through(self):
        self.assertEqual(_format_subject_value(42), 42)

    def test_passes_str_through(self):
        self.assertEqual(_format_subject_value("hello"), "hello")

    def test_formats_float_to_string(self):
        result = _format_subject_value(1.23456789)
        self.assertIsInstance(result, str)
        self.assertIn("1.23", result)


class TestAggregateStatus(unittest.TestCase):
    def test_passes_single_through(self):
        self.assertEqual(_aggregate_status("pass"), "pass")

    def test_error_wins_over_warning_and_pass(self):
        self.assertEqual(_aggregate_status(["pass", "warning", "error"]), CompareConst.ERROR)

    def test_warning_wins_over_pass(self):
        self.assertEqual(_aggregate_status(["pass", "warning"]), CompareConst.WARNING)

    def test_pass_when_all_pass(self):
        self.assertEqual(_aggregate_status(["pass", "pass"]), CompareConst.PASS)


class TestComparatorInit(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_writes_csv_headers_when_not_continuing(self):
        summary = os.path.join(self.tmpdir, "summary.csv")
        detail = os.path.join(self.tmpdir, "detail.csv")
        Comparator(summary, detail, is_continue_acc_check=False)
        self.assertTrue(os.path.exists(summary))
        self.assertTrue(os.path.exists(detail))
        with open(summary) as f:
            header = f.readline()
        self.assertIn(Comparator.COLUMN_API_NAME, header)
        self.assertIn(Comparator.COLUMN_FORWARD_SUCCESS, header)
        self.assertIn(Comparator.COLUMN_BACKWARD_SUCCESS, header)
        self.assertIn(Comparator.COLUMN_MESSAGE, header)

    def test_skips_csv_headers_when_continuing(self):
        summary = os.path.join(self.tmpdir, "summary.csv")
        detail = os.path.join(self.tmpdir, "detail.csv")
        Comparator(summary, detail, is_continue_acc_check=True)
        self.assertFalse(os.path.exists(summary))
        self.assertFalse(os.path.exists(detail))

    def test_loads_stack_info_when_path_provided(self):
        stack_json = os.path.join(self.tmpdir, "stack.json")
        with open(stack_json, "w") as f:
            json.dump({"torch.add.0": ["frame 1", "frame 2"]}, f)
        c = _make_comparator(self.tmpdir, is_continue=True)
        c2 = Comparator(
            os.path.join(self.tmpdir, "s2.csv"),
            os.path.join(self.tmpdir, "d2.csv"),
            is_continue_acc_check=True,
            stack_info_json_path=stack_json,
        )
        self.assertIsNone(c.stack_info)
        self.assertIn("torch.add.0", c2.stack_info)

    def test_stack_info_is_none_when_path_omitted(self):
        c = _make_comparator(self.tmpdir)
        self.assertIsNone(c.stack_info)


class TestGetPathFromRank(unittest.TestCase):
    def test_single_path_returns_last(self):
        self.assertEqual(
            Comparator.get_path_from_rank(0, ["/a/b.csv"], "/x/rank{rank}.csv"),
            "/a/b.csv",
        )


class TestIsDropoutApi(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.c = _make_comparator(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_top_level_dropout(self):
        self.assertTrue(self.c._is_dropout_api("dropout.0"))

    def test_nested_dropout(self):
        self.assertTrue(self.c._is_dropout_api("torch.nn.functional.dropout.0"))

    def test_unrelated_api(self):
        self.assertFalse(self.c._is_dropout_api("torch.matmul.0"))


class TestCompareDropout(unittest.TestCase):
    """Regression tests for the dropout branch: small-tensor shortcut, zero-count
    tolerance, and (previously broken) device-mismatch migration."""

    def test_small_tensor_passes_without_comparison(self):
        bench = torch.zeros(50)
        device = torch.ones(50)
        status, _ = Comparator._compare_dropout(bench, device)
        self.assertEqual(status, CompareConst.PASS)

    def test_similar_zeros_pass(self):
        bench = torch.zeros(1000)
        bench[100:150] = 1.0
        device = torch.zeros(1000)
        device[100:160] = 1.0
        status, _ = Comparator._compare_dropout(bench, device)
        self.assertEqual(status, CompareConst.PASS)

    def test_disagree_zeros_error(self):
        bench = torch.zeros(1000)
        device = torch.ones(1000)
        status, _ = Comparator._compare_dropout(bench, device)
        self.assertEqual(status, CompareConst.ERROR)


class TestCompareBuiltinType(unittest.TestCase):
    def test_equal_ints_pass(self):
        col = CompareColumn()
        status, _, _ = Comparator._compare_builtin_type(1, 1, col)
        self.assertEqual(status, CompareConst.PASS)
        self.assertEqual(col.error_rate, 0)

    def test_unequal_ints_error(self):
        col = CompareColumn()
        status, _, _ = Comparator._compare_builtin_type(1, 2, col)
        self.assertEqual(status, CompareConst.ERROR)

    def test_equal_floats_pass(self):
        col = CompareColumn()
        status, _, _ = Comparator._compare_builtin_type(1.0, 1.0, col)
        self.assertEqual(status, CompareConst.PASS)


class TestExtractSkipMessage(unittest.TestCase):
    def test_fwd_skip_returns_fwd_message(self):
        col = CompareColumn()
        fwd_row = col.to_column_value(CompareConst.SKIP, "fwd skipped")
        result = ResultInfo("torch.add.0", CompareConst.SKIP, "pass", [fwd_row], [], 0)
        self.assertEqual(Comparator._extract_skip_message(result), "fwd skipped")

    def test_bwd_skip_returns_bwd_message(self):
        col = CompareColumn()
        bwd_row = col.to_column_value(CompareConst.SKIP, "bwd skipped")
        result = ResultInfo("torch.add.0", "pass", CompareConst.SKIP, [], [bwd_row], 0)
        self.assertEqual(Comparator._extract_skip_message(result), "bwd skipped")

    def test_neither_skip_returns_none(self):
        result = ResultInfo("torch.add.0", "pass", "pass", [], [], 0)
        self.assertIsNone(Comparator._extract_skip_message(result))

    def test_fwd_skip_with_empty_results_returns_none(self):
        result = ResultInfo("torch.add.0", CompareConst.SKIP, "pass", [], [], 0)
        self.assertIsNone(Comparator._extract_skip_message(result))


class TestCheckCosine(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.c = _make_comparator(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_cosine_pass_returns_none(self):
        bench = np.array([1.0, 2.0, 3.0])
        device = np.array([1.0, 2.0, 3.0])
        col = CompareColumn()
        notes = []
        result = self.c._check_cosine(bench, device, col, notes)
        self.assertIsNone(result)
        self.assertEqual(col.cosine_sim, 1.0)
        self.assertEqual(notes, [])

    def test_cosine_fail_returns_error(self):
        bench = np.array([1.0, 2.0, 3.0])
        device = np.array([-1.0, -2.0, -3.0])
        col = CompareColumn()
        notes = []
        result = self.c._check_cosine(bench, device, col, notes)
        self.assertEqual(result, CompareConst.ERROR)
        self.assertTrue(any("Cosine similarity is less than 0.99" in n for n in notes))


class TestCheckMaxAbsErr(unittest.TestCase):
    def test_max_abs_below_threshold_returns_pass(self):
        abs_err = np.array([0.0, 0.0005, 0.0001])
        col = CompareColumn()
        notes = []
        result = Comparator._check_max_abs_err(abs_err, col, notes)
        self.assertEqual(result, CompareConst.PASS)
        self.assertEqual(col.max_abs_err, 0.0005)

    def test_max_abs_above_threshold_returns_none(self):
        abs_err = np.array([0.0, 0.5, 0.1])
        col = CompareColumn()
        notes = []
        result = Comparator._check_max_abs_err(abs_err, col, notes)
        self.assertIsNone(result)
        self.assertEqual(col.max_abs_err, 0.5)


class TestCheckLowPrecision(unittest.TestCase):
    """The low-precision gate: hundredth → thousandth. Targets float16/bfloat16."""

    def test_hundred_fail_returns_error(self):
        rel_err = np.full(100, 0.1)
        col = CompareColumn()
        notes = []
        status, _, msg = Comparator._check_low_precision(rel_err, col, notes)
        self.assertEqual(status, CompareConst.ERROR)
        self.assertIn("0.01", msg)
        self.assertEqual(col.rel_err_hundredth, 0.0)

    def test_hundred_pass_thousand_pass_returns_pass(self):
        rel_err = np.full(1000, 0.0001)
        col = CompareColumn()
        notes = []
        status, _, _ = Comparator._check_low_precision(rel_err, col, notes)
        self.assertEqual(status, CompareConst.PASS)

    def test_hundred_pass_thousand_fail_returns_warning(self):
        # 5 elements < 0.001, 995 elements between 0.001 and 0.01
        rel_err = np.concatenate([np.full(5, 0.0001), np.full(995, 0.005)])
        col = CompareColumn()
        notes = []
        status, _, msg = Comparator._check_low_precision(rel_err, col, notes)
        self.assertEqual(status, CompareConst.WARNING)
        self.assertIn("0.001", msg)


class TestCheckHighPrecision(unittest.TestCase):
    """The high-precision gate: thousandth → ten-thousandth. Targets float32/float64."""

    def test_thousand_fail_returns_error(self):
        rel_err = np.full(100, 0.1)
        col = CompareColumn()
        notes = []
        status, _, msg = Comparator._check_high_precision(rel_err, col, notes)
        self.assertEqual(status, CompareConst.ERROR)
        self.assertIn("0.001", msg)

    def test_thousand_pass_ten_thousand_fail_returns_warning(self):
        rel_err = np.concatenate([np.full(5, 0.00001), np.full(995, 0.0005)])
        col = CompareColumn()
        notes = []
        status, _, msg = Comparator._check_high_precision(rel_err, col, notes)
        self.assertEqual(status, CompareConst.WARNING)
        self.assertIn("0.0001", msg)

    def test_thousand_pass_ten_thousand_pass_returns_pass(self):
        rel_err = np.full(1000, 0.00001)
        col = CompareColumn()
        notes = []
        status, _, _ = Comparator._check_high_precision(rel_err, col, notes)
        self.assertEqual(status, CompareConst.PASS)


class TestRunStandardCompare(unittest.TestCase):
    """Verify the dispatcher instantiates the right standard compare class."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.c = _make_comparator(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_routes_to_correct_class(self):
        mock_cls = MagicMock()
        with patch.dict(self.c._STANDARD_COMPARE_CLASSES, {CompareConst.BENCHMARK: mock_cls}):
            input_data = MagicMock()
            self.c._run_standard_compare(CompareConst.BENCHMARK, input_data)
        mock_cls.assert_called_once_with(input_data)
        mock_cls.return_value.compare.assert_called_once_with()

    def test_make_standard_runner_returns_ignoring_callable(self):
        runner = self.c._make_standard_runner(CompareConst.ABSOLUTE_THRESHOLD)
        mock_cls = MagicMock()
        with patch.dict(self.c._STANDARD_COMPARE_CLASSES, {CompareConst.ABSOLUTE_THRESHOLD: mock_cls}):
            runner(MagicMock())
        mock_cls.assert_called_once()


class TestCompareOneDirection(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.c = _make_comparator(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_no_grad_returns_space(self):
        status, results = self.c._compare_one_direction(
            "api", "api", None, None, False, has_grad=False
        )
        self.assertEqual(status, CompareConst.SPACE)
        self.assertEqual(results, [])

    def test_dropout_api_bypasses_core(self):
        with patch.object(Comparator, "_compare_dropout", return_value=("pass", 1)) as m:
            status, results = self.c._compare_one_direction(
                "dropout.0", "dropout", torch.zeros(1000), torch.zeros(1000), False
            )
        m.assert_called_once()
        self.assertEqual(status, CompareConst.PASS)
        self.assertEqual(results, [])

    def test_regular_api_dispatches_to_core_wrapper(self):
        with patch.object(self.c, "_compare_core_wrapper", return_value=("pass", [])) as m:
            self.c._compare_one_direction("torch.matmul.0", "torch.matmul", 1, 1, False)
        m.assert_called_once()


class TestGetAccCheckDetail(unittest.TestCase):
    def test_formats_per_direction_rows(self):
        fwd_row = [1.23456789, 2, 3.0, "final_pass", "msg"]
        bwd_row = [9.87654321, 8, 7.0, "final_pass", "msg2"]
        result = ResultInfo(
            "torch.add.0", "pass", "pass", [fwd_row], [bwd_row], 0
        )
        rows = Comparator._get_acc_check_detail(result)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][0], "torch.add.0.forward.output.0")
        self.assertEqual(rows[1][0], "torch.add.0.backward.output.0")
        # Float in first cell is formatted to string
        self.assertIsInstance(rows[0][1], str)

    def test_skips_non_list_payload(self):
        result = ResultInfo("torch.add.0", "pass", "pass", None, None, 0)
        rows = Comparator._get_acc_check_detail(result)
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
