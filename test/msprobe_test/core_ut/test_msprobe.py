#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UTs for python/msprobe/msprobe.py
"""

from unittest import TestCase
from unittest.mock import patch, MagicMock

from msprobe.msprobe import main


class TestMsprobeMain(TestCase):
    @patch("msprobe.msprobe.argparse.ArgumentParser")
    def test_main_when_no_args_then_pass(self, mock_arg_parser):
        parser_instance = MagicMock()
        subparsers_instance = MagicMock()
        mock_arg_parser.return_value = parser_instance
        parser_instance.add_subparsers.return_value = subparsers_instance

        with patch("msprobe.msprobe.sys") as mock_sys:
            mock_sys.argv = ["msprobe"]
            mock_sys.exit.side_effect = SystemExit(0)

            with self.assertRaises(SystemExit) as cm:
                main()

            self.assertEqual(cm.exception.code, 0)
            parser_instance.print_help.assert_called_once()

    @patch("msprobe.msprobe.acc_check_cli")
    def test_main_when_acc_check_subcommand_then_pass(self, mock_acc_check_cli):
        with patch("msprobe.msprobe.sys.argv", ["msprobe", "acc_check", "--arg1", "value1"]):
            main()
        mock_acc_check_cli.assert_called_once_with(["--arg1", "value1"])

    @patch("msprobe.msprobe.multi_acc_check_cli")
    def test_main_when_multi_acc_check_subcommand_then_pass(self, mock_multi_acc_check_cli):
        with patch("msprobe.msprobe.sys.argv", ["msprobe", "multi_acc_check", "--arg1", "value1"]):
            main()
        mock_multi_acc_check_cli.assert_called_once_with(["--arg1", "value1"])

    @patch("msprobe.msprobe.compare_cli")
    @patch("msprobe.msprobe.argparse.ArgumentParser.parse_args")
    def test_main_when_compare_subcommand_then_pass(self, mock_parse_args, mock_compare_cli):
        args = MagicMock()
        mock_parse_args.return_value = args
        with patch("msprobe.msprobe.sys.argv", ["msprobe", "compare"]):
            main()
        mock_parse_args.assert_called_once_with(["compare"])
        mock_compare_cli.assert_called_once_with(args, ["compare"])

    @patch("msprobe.msprobe.merge_result_cli")
    @patch("msprobe.msprobe.argparse.ArgumentParser.parse_args")
    def test_main_when_merge_result_subcommand_then_pass(self, mock_parse_args, mock_merge_result_cli):
        args = MagicMock()
        mock_parse_args.return_value = args
        with patch("msprobe.msprobe.sys.argv", ["msprobe", "merge_result"]):
            main()
        mock_parse_args.assert_called_once_with(["merge_result"])
        mock_merge_result_cli.assert_called_once_with(args)

    @patch("msprobe.msprobe._run_overflow_check")
    @patch("msprobe.msprobe.argparse.ArgumentParser.parse_args")
    def test_main_when_overflow_check_subcommand_then_pass(self, mock_parse_args, mock_run_overflow_check):
        args = MagicMock()
        mock_parse_args.return_value = args
        with patch("msprobe.msprobe.sys.argv", ["msprobe", "overflow_check"]):
            main()
        mock_parse_args.assert_called_once_with(["overflow_check"])
        mock_run_overflow_check.assert_called_once_with(args)

    @patch("msprobe.msprobe._graph_service_command")
    @patch("msprobe.msprobe.argparse.ArgumentParser.parse_args")
    def test_main_when_graph_visualize_subcommand_then_pass(self, mock_parse_args, mock_graph_service_cmd):
        args = MagicMock()
        mock_parse_args.return_value = args
        with patch("msprobe.msprobe.sys.argv", ["msprobe", "graph_visualize"]):
            main()
        mock_parse_args.assert_called_once_with(["graph_visualize"])
        mock_graph_service_cmd.assert_called_once_with(args)

    @patch("msprobe.pytorch.api_accuracy_checker.compare.api_precision_compare._api_precision_compare_command")
    @patch("msprobe.msprobe.argparse.ArgumentParser.parse_args")
    def test_main_when_api_precision_compare_subcommand_then_pass(self, mock_parse_args, mock_api_precision_cmd):
        args = MagicMock()
        mock_parse_args.return_value = args
        with patch("msprobe.msprobe.sys.argv", ["msprobe", "api_precision_compare"]):
            main()
        mock_parse_args.assert_called_once_with(["api_precision_compare"])
        mock_api_precision_cmd.assert_called_once_with(args)

    @patch("msprobe.msprobe._run_config_checking_command")
    @patch("msprobe.msprobe.argparse.ArgumentParser.parse_args")
    def test_main_when_config_check_subcommand_then_pass(self, mock_parse_args, mock_run_config):
        args = MagicMock()
        mock_parse_args.return_value = args
        with patch("msprobe.msprobe.sys.argv", ["msprobe", "config_check"]):
            main()
        mock_parse_args.assert_called_once_with(["config_check"])
        mock_run_config.assert_called_once_with(args)

    @patch("msprobe.msprobe._data2db_command")
    @patch("msprobe.msprobe.argparse.ArgumentParser.parse_args")
    def test_main_when_data2db_subcommand_then_pass(self, mock_parse_args, mock_data2db_cmd):
        args = MagicMock()
        mock_parse_args.return_value = args
        with patch("msprobe.msprobe.sys.argv", ["msprobe", "data2db"]):
            main()
        mock_parse_args.assert_called_once_with(["data2db"])
        mock_data2db_cmd.assert_called_once_with(args)

    @patch("msprobe.msprobe.offline_dump_cli")
    @patch("msprobe.msprobe.argparse.ArgumentParser.parse_args")
    def test_main_when_offline_dump_subcommand_then_pass(self, mock_parse_args, mock_offline_dump_cli):
        args = MagicMock()
        mock_parse_args.return_value = args
        with patch("msprobe.msprobe.sys.argv", ["msprobe", "offline_dump"]):
            main()
        mock_parse_args.assert_called_once_with(["offline_dump"])
        mock_offline_dump_cli.assert_called_once_with(args)

    @patch("msprobe.msprobe.install_deps_cli")
    @patch("msprobe.msprobe.argparse.ArgumentParser.parse_args")
    def test_main_when_install_deps_subcommand_then_pass(self, mock_parse_args, mock_install_deps_cli):
        args = MagicMock()
        mock_parse_args.return_value = args
        with patch("msprobe.msprobe.sys.argv", ["msprobe", "install_deps"]):
            main()
        mock_parse_args.assert_called_once_with(["install_deps"])
        mock_install_deps_cli.assert_called_once_with(args)

