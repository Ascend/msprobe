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

    def tearDown(self):
        signal.signal(signal.SIGPIPE, self.original_sigpipe)
        if os.path.exists(self.output):
            shutil.rmtree(self.output)
        for json_data in self.output_json:
            if os.path.exists(json_data):
                os.remove(json_data)


if __name__ == '__main__':
    unittest.main()
