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

import unittest
from unittest.mock import patch
import argparse

from msprobe.core.common.cli_help import MindStudioArgumentParser
from msprobe.core.compare.merge_result.merge_result_cli import _merge_result_parser, merge_result_cli


class TestMergeResultCLI(unittest.TestCase):
    def test_help_lists_alias_and_output_file(self):
        parser = MindStudioArgumentParser(prog="msprobe merge_result")
        _merge_result_parser(parser)

        help_text = parser.format_help()
        normalized_help = " ".join(help_text.split())

        self.assertIn("-config, --config-path <FILE>", help_text)
        self.assertIn("YAML path containing distributed APIs", normalized_help)
        self.assertIn("The compare result path, a directory.", normalized_help)
        self.assertIn("The result merge output path, a directory.", normalized_help)
        self.assertIn("<DIR>/multi_ranks_compare_merge_{timestamp}.xlsx", help_text)
        self.assertNotIn("<output_dir>/", help_text)

    @patch('msprobe.core.compare.merge_result.merge_result_cli.merge_result')
    def test_merge_result_cli_success(self, mock_merge_result):
        args = [
            '-i', '/path/to/input',
            '-o', '/path/to/output',
            '-config', '/path/to/config.yaml'
        ]

        parser = argparse.ArgumentParser()
        _merge_result_parser(parser)
        parsed_args = parser.parse_args(args)

        merge_result_cli(parsed_args)

        mock_merge_result.assert_called_once_with(
            '/path/to/input', '/path/to/output', '/path/to/config.yaml'
        )
