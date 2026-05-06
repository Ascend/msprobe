#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from msprobe.core.common.const import FileCheckConst, Const
from msprobe.core.common.utils import CompareException
from msprobe.core.compare.auto_compare import compare_auto_mode


def _build_args(
    target_path="/npu",
    golden_path="/bench",
    output_path="/out",
    xlsx=False,
    fuzzy_match=False,
    data_mapping=None,
    diff_analyze=False,
    cell_mapping=None,
    api_mapping=None,
    layer_mapping=None,
    rank=None,
    is_print_compare_log=False,
    consistent_check=False,
    backend=''
):
    return SimpleNamespace(
        target_path=target_path,
        golden_path=golden_path,
        output_path=output_path,
        xlsx=xlsx,
        fuzzy_match=fuzzy_match,
        data_mapping=data_mapping,
        diff_analyze=diff_analyze,
        cell_mapping=cell_mapping,
        api_mapping=api_mapping,
        layer_mapping=layer_mapping,
        rank=rank,
        is_print_compare_log=is_print_compare_log,
        consistent_check=consistent_check,
        backend=backend
    )


class TestCompareAutoMode(unittest.TestCase):
    def test_compare_auto_mode_when_depth_exceeds_limit_then_raise(self):
        args = _build_args()
        with self.assertRaises(CompareException) as cm:
            compare_auto_mode(args, depth=3)
        self.assertEqual(cm.exception.code, CompareException.RECURSION_LIMIT_ERROR)

    @patch("msprobe.pytorch.compare.pt_compare.pt_compare")
    @patch("msprobe.core.compare.auto_compare.check_file_or_directory_path")
    @patch("msprobe.core.compare.auto_compare.check_file_type")
    @patch("msprobe.core.compare.auto_compare.get_compare_framework")
    def test_compare_auto_mode_file_pt_without_mapping_then_call_pt_compare(
        self, mock_get_framework, mock_check_type, mock_check_path, mock_pt_compare
    ):

        args = _build_args()
        mock_check_type.return_value = FileCheckConst.FILE
        mock_get_framework.return_value = Const.PT_FRAMEWORK

        compare_auto_mode(args)

        mock_check_type.assert_any_call(args.target_path)
        mock_check_type.assert_any_call(args.golden_path)
        mock_check_path.assert_any_call(args.target_path)
        mock_check_path.assert_any_call(args.golden_path)
        mock_get_framework.assert_called_once()
        mock_pt_compare.assert_called_once()
        call_args, call_kwargs = mock_pt_compare.call_args
        input_param_arg, output_path_arg = call_args
        self.assertEqual(input_param_arg["npu_path"], args.target_path)
        self.assertEqual(input_param_arg["bench_path"], args.golden_path)
        self.assertEqual(output_path_arg, args.output_path)
        self.assertIn("fuzzy_match", call_kwargs)
        self.assertIn("data_mapping", call_kwargs)
        self.assertIn("diff_analyze", call_kwargs)

    @patch("msprobe.core.compare.auto_compare.logger")
    @patch("msprobe.core.compare.auto_compare.check_file_or_directory_path")
    @patch("msprobe.core.compare.auto_compare.check_file_type")
    @patch("msprobe.core.compare.auto_compare.get_compare_framework")
    def test_compare_auto_mode_file_pt_with_mapping_then_raise_exception(
        self,
        mock_get_framework,
        mock_check_type,
        mock_check_path,
        mock_logger,
    ):

        args = _build_args(api_mapping={"a": "b"})
        mock_check_type.return_value = FileCheckConst.FILE
        mock_get_framework.return_value = Const.PT_FRAMEWORK

        with self.assertRaises(CompareException) as cm:
            compare_auto_mode(args)

        mock_logger.error.assert_called_once()
        mock_check_path.assert_any_call(args.target_path)
        mock_check_path.assert_any_call(args.golden_path)

    @patch("msprobe.core.compare.auto_compare.check_file_or_directory_path")
    @patch("msprobe.core.compare.auto_compare.check_file_type")
    @patch("msprobe.core.compare.auto_compare.get_compare_framework")
    @patch("msprobe.mindspore.compare.ms_compare.ms_compare")
    def test_compare_auto_mode_file_ms_then_call_ms_compare(
        self, mock_ms_compare, mock_get_framework, mock_check_type, mock_check_path
    ):
        args = _build_args(
            cell_mapping={"c": "m"},
            api_mapping={"a": "m"},
            layer_mapping={"l": "m"},
        )
        mock_check_type.return_value = FileCheckConst.FILE
        mock_get_framework.return_value = Const.MS_FRAMEWORK

        compare_auto_mode(args)

        mock_check_type.assert_any_call(args.target_path)
        mock_check_type.assert_any_call(args.golden_path)
        mock_check_path.assert_any_call(args.target_path)
        mock_check_path.assert_any_call(args.golden_path)
        mock_get_framework.assert_called_once()
        mock_ms_compare.assert_called_once()
        call_args, call_kwargs = mock_ms_compare.call_args
        input_param_arg, output_path_arg = call_args
        self.assertEqual(input_param_arg["npu_path"], args.target_path)
        self.assertEqual(input_param_arg["bench_path"], args.golden_path)
        self.assertEqual(output_path_arg, args.output_path)
        self.assertEqual(call_kwargs["cell_mapping"], args.cell_mapping)
        self.assertEqual(call_kwargs["api_mapping"], args.api_mapping)
        self.assertEqual(call_kwargs["layer_mapping"], args.layer_mapping)

    @patch("msprobe.core.compare.auto_compare.compare_distributed_inner")
    @patch("msprobe.core.compare.auto_compare.mix_compare", return_value=True)
    @patch("msprobe.core.compare.auto_compare.check_file_or_directory_path")
    @patch("msprobe.core.compare.auto_compare.check_file_type")
    def test_compare_auto_mode_dir_when_mix_compare_success_then_return_early(
        self, mock_check_type, mock_check_path, mock_mix_compare, mock_compare_inner
    ):
        args = _build_args()
        mock_check_type.return_value = FileCheckConst.DIR

        compare_auto_mode(args)

        mock_check_type.assert_any_call(args.target_path)
        mock_check_type.assert_any_call(args.golden_path)
        mock_check_path.assert_any_call(args.target_path, isdir=True)
        mock_check_path.assert_any_call(args.golden_path, isdir=True)
        mock_mix_compare.assert_called_once_with(args, 1)
        mock_compare_inner.assert_not_called()

    @patch("msprobe.core.compare.auto_compare.compare_distributed_inner")
    @patch("msprobe.mindspore.compare.distributed_compare.ms_graph_compare")
    @patch("msprobe.core.compare.auto_compare.mix_compare", return_value=False)
    @patch("msprobe.core.compare.auto_compare.check_file_or_directory_path")
    @patch("msprobe.core.compare.auto_compare.check_file_type")
    def test_compare_auto_mode_dir_with_rank_then_call_ms_graph_compare(
        self,
        mock_check_type,
        mock_check_path,
        mock_mix_compare,
        mock_ms_graph_compare,
        mock_compare_inner,
    ):
        args = _build_args(rank="0")
        mock_check_type.return_value = FileCheckConst.DIR

        compare_auto_mode(args)

        mock_check_type.assert_any_call(args.target_path)
        mock_check_type.assert_any_call(args.golden_path)
        mock_check_path.assert_any_call(args.target_path, isdir=True)
        mock_check_path.assert_any_call(args.golden_path, isdir=True)
        mock_mix_compare.assert_called_once_with(args, 1)
        mock_ms_graph_compare.assert_called_once_with(args)
        mock_compare_inner.assert_not_called()

    @patch("msprobe.core.compare.auto_compare.compare_distributed_inner")
    @patch("msprobe.core.compare.find_first.analyzer.DiffAnalyzer")
    @patch("msprobe.core.compare.auto_compare.mix_compare", return_value=False)
    @patch("msprobe.core.compare.auto_compare.check_file_or_directory_path")
    @patch("msprobe.core.compare.auto_compare.check_file_type")
    def test_compare_auto_mode_dir_with_diff_analyze_then_call_diff_analyzer(
        self,
        mock_check_type,
        mock_check_path,
        mock_mix_compare,
        mock_diff_analyzer_cls,
        mock_compare_inner,
    ):
        args = _build_args(diff_analyze=True)
        mock_check_type.return_value = FileCheckConst.DIR
        mock_instance = MagicMock()
        mock_diff_analyzer_cls.return_value = mock_instance

        compare_auto_mode(args)

        mock_check_type.assert_any_call(args.target_path)
        mock_check_type.assert_any_call(args.golden_path)
        mock_check_path.assert_any_call(args.target_path, isdir=True)
        mock_check_path.assert_any_call(args.golden_path, isdir=True)
        mock_mix_compare.assert_called_once_with(args, 1)
        mock_diff_analyzer_cls.assert_called_once_with(args.target_path, args.golden_path, args.output_path)
        mock_instance.analyze.assert_called_once()
        mock_compare_inner.assert_not_called()

    @patch("msprobe.core.compare.auto_compare.compare_distributed_inner")
    @patch("msprobe.core.compare.auto_compare.mix_compare", return_value=False)
    @patch("msprobe.core.compare.auto_compare.check_file_or_directory_path")
    @patch("msprobe.core.compare.auto_compare.check_file_type")
    def test_compare_auto_mode_dir_without_rank_and_diff_analyze_then_call_compare_distributed(
        self, mock_check_type, mock_check_path, mock_mix_compare, mock_compare_inner
    ):
        args = _build_args(cell_mapping={"c": "m"}, api_mapping={"a": "m"}, layer_mapping={"l": "m"})
        mock_check_type.return_value = FileCheckConst.DIR

        compare_auto_mode(args)

        mock_check_type.assert_any_call(args.target_path)
        mock_check_type.assert_any_call(args.golden_path)
        mock_check_path.assert_any_call(args.target_path, isdir=True)
        mock_check_path.assert_any_call(args.golden_path, isdir=True)
        mock_mix_compare.assert_called_once_with(args, 1)
        mock_compare_inner.assert_called_once()
        call_args, call_kwargs = mock_compare_inner.call_args
        self.assertEqual(call_args[0], args.target_path)
        self.assertEqual(call_args[1], args.golden_path)
        self.assertEqual(call_args[2], args.output_path)
        self.assertFalse(call_kwargs.get("is_print_compare_log"))
        self.assertEqual(call_kwargs["cell_mapping"], args.cell_mapping)
        self.assertEqual(call_kwargs["api_mapping"], args.api_mapping)
        self.assertEqual(call_kwargs["layer_mapping"], args.layer_mapping)

    @patch("msprobe.core.compare.auto_compare.compare_distributed_inner")
    @patch("msprobe.core.compare.auto_compare.mix_compare")
    @patch("msprobe.core.compare.auto_compare.check_file_or_directory_path")
    @patch("msprobe.core.compare.auto_compare.check_file_type")
    def test_compare_auto_mode_dir_when_depth_not_one_then_skip_mix_compare(
        self, mock_check_type, mock_check_path, mock_mix_compare, mock_compare_inner
    ):
        args = _build_args()
        mock_check_type.return_value = FileCheckConst.DIR

        compare_auto_mode(args, depth=2)

        mock_mix_compare.assert_not_called()
        mock_compare_inner.assert_called_once()

    @patch("msprobe.core.compare.auto_compare.logger")
    @patch("msprobe.core.compare.auto_compare.check_file_type")
    def test_compare_auto_mode_when_path_types_mismatch_then_raise_invalid_mode(
        self, mock_check_type, mock_logger
    ):
        args = _build_args(target_path="/path/target", golden_path="/path/golden")

        def fake_check_type(path):
            if "target" in path:
                return FileCheckConst.FILE
            return FileCheckConst.DIR

        mock_check_type.side_effect = fake_check_type

        with self.assertRaises(CompareException) as cm:
            compare_auto_mode(args)

        self.assertEqual(cm.exception.code, CompareException.INVALID_COMPARE_MODE)
        mock_logger.error.assert_called_once()


if __name__ == "__main__":
    unittest.main()
