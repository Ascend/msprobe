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

import unittest
from unittest.mock import patch
import argparse
from msprobe.core.common.cli_help import MindStudioArgumentParser
from msprobe.core.config_check.config_check_cli import (
    pack,
    compare,
    _config_checking_parser,
    _run_config_checking_command,
    _get_verl_verify_error_message,
)


class TestConfigCheckCli(unittest.TestCase):
    
    @patch('msprobe.core.config_check.config_check_cli.ConfigChecker')
    def test_pack(self, mock_config_checker):
        """Test pack function"""
        shell_path = ['test.sh']
        output_path = 'test_output.zip'
        
        pack(shell_path, output_path)
        
        mock_config_checker.assert_called_once_with(shell_path=shell_path, output_zip_path=output_path)
    
    @patch('msprobe.core.config_check.config_check_cli.ConfigChecker')
    def test_compare(self, mock_config_checker):
        """Test compare function"""
        bench_zip_path = 'bench.zip'
        cmp_zip_path = 'cmp.zip'
        output_path = 'output_dir'
        
        compare(bench_zip_path, cmp_zip_path, output_path)
        
        mock_config_checker.compare.assert_called_once_with(bench_zip_path, cmp_zip_path, output_path)
    
    def test_config_checking_parser(self):
        """Test _config_checking_parser function"""
        parser = argparse.ArgumentParser()
        _config_checking_parser(parser)
        
        # Check if the arguments are properly configured
        args = parser.parse_args(['-d', 'test.sh', '-o', 'output.zip'])
        self.assertEqual(args.dump, ['test.sh'])
        self.assertEqual(args.output, 'output.zip')

        args = parser.parse_args(['-d'])
        self.assertEqual(args.dump, [])

        args = parser.parse_args([])
        self.assertIsNone(args.dump)
        self.assertIsNone(args.compare)
        self.assertIsNone(args.verl_compare)
        self.assertIsNone(args.verl_verify)
        self.assertIsNone(args.slime_compare)
        
        args = parser.parse_args(['-c', 'file1', 'file2', '-o', 'result'])
        self.assertEqual(args.compare, ['file1', 'file2'])
        self.assertEqual(args.output, 'result')

        args = parser.parse_args(['-vc', 'NPU_log', 'bench_log', '-o', 'verl_compare_result'])
        self.assertEqual(args.verl_compare, ['NPU_log', 'bench_log'])
        self.assertEqual(args.output, 'verl_compare_result')

        args = parser.parse_args(['-sc', 'NPU_log', 'bench_log', '-o', 'slime_compare_result'])
        self.assertEqual(args.slime_compare, ['NPU_log', 'bench_log'])
        self.assertEqual(args.output, 'slime_compare_result')

        args = parser.parse_args(['-vv', 'bench_config', 'tgt_config', '-o', 'verl_verify_result'])
        self.assertEqual(args.verl_verify, ['bench_config', 'tgt_config'])
        self.assertEqual(args.output, 'verl_verify_result')

        args = parser.parse_args(['-vv', 'tgt_config', '-o', 'verl_verify_result'])
        self.assertEqual(args.verl_verify, ['tgt_config'])
        self.assertEqual(args.output, 'verl_verify_result')

    def test_config_checking_help(self):
        parser = MindStudioArgumentParser(prog='msprobe config_check')
        _config_checking_parser(parser)

        help_text = parser.format_help()
        normalized_help_text = ' '.join(help_text.split())

        self.assertNotIn('Required arguments:', help_text)
        self.assertIn('Optional arguments:', help_text)
        self.assertIn('-d,  --dump [<FILE> ...]', help_text)
        self.assertIn('-vc, --verl-compare <NPU_LOG> <BENCH_LOG>', help_text)
        self.assertIn('-vv, --verl-verify [<BENCH_CONFIG>] <TGT_LOG>', help_text)
        self.assertIn('-o, --output <FILE_OR_DIR>', normalized_help_text)
        self.assertNotIn('<operation>', help_text)
        self.assertNotIn('(eg:', help_text)
        for output_name in (
            'config_check_pack.zip',
            'result.xlsx',
            'NPU_config.json',
            'bench_config.json',
            'hyper_params_compare.csv',
            'tgt_config.json',
            'hyper_params_verify.csv',
        ):
            self.assertIn(output_name, help_text)

    @patch('msprobe.core.config_check.config_check_cli.pack')
    def test_run_config_checking_command_dump(self, mock_pack):
        """Test _run_config_checking_command with dump parameter"""
        args = argparse.Namespace()
        args.dump = ['test.sh']
        args.output = 'output.zip'
        args.compare = None
        args.verl_compare = None
        args.verl_verify = None
        args.slime_compare = None
        
        _run_config_checking_command(args)
        
        mock_pack.assert_called_once_with(['test.sh'], 'output.zip')
        
        # Test with default output path
        args.output = None
        _run_config_checking_command(args)
        
        mock_pack.assert_called_with(['test.sh'], './config_check_pack.zip')
    
    @patch('msprobe.core.config_check.config_check_cli.compare')
    @patch('msprobe.core.config_check.config_check_cli.logger')
    def test_run_config_checking_command_compare_zip(self, mock_logger, mock_compare):
        """Test _run_config_checking_command with compare parameter for zip files"""
        args = argparse.Namespace()
        args.dump = None
        args.verl_compare = None
        args.verl_verify = None
        args.slime_compare = None
        args.compare = ['file1.zip', 'file2.zip']
        args.output = 'result_dir'
        
        _run_config_checking_command(args)
        
        mock_logger.info.assert_called_once_with('The input paths is zip files, comparing packed config.')
        mock_compare.assert_called_once_with('file1.zip', 'file2.zip', 'result_dir')
        
        # Test with default output path
        args.output = None
        _run_config_checking_command(args)
        
        mock_compare.assert_called_with('file1.zip', 'file2.zip', './config_check_result')
    
    @patch('msprobe.core.config_check.config_check_cli.compare_checkpoints')
    @patch('msprobe.core.config_check.config_check_cli.logger')
    def test_run_config_checking_command_compare_checkpoint(self, mock_logger, mock_compare_checkpoints):
        """Test _run_config_checking_command with compare parameter for checkpoint files"""
        args = argparse.Namespace()
        args.dump = None
        args.verl_compare = None
        args.verl_verify = None
        args.slime_compare = None
        args.compare = ['file1.ckpt', 'file2.ckpt']
        args.output = 'result.json'
        
        _run_config_checking_command(args)
        
        mock_logger.info.assert_called_once_with('Comparing model checkpoint.')
        mock_compare_checkpoints.assert_called_once_with('file1.ckpt', 'file2.ckpt', 'result.json')
        
        # Test with default output path
        args.output = None
        _run_config_checking_command(args)
        
        mock_compare_checkpoints.assert_called_with('file1.ckpt', 'file2.ckpt', './ckpt_similarity.json')

    @patch('msprobe.core.config_check.config_check_cli.verl_compare_hyper_params')  
    @patch('msprobe.core.config_check.config_check_cli.verl_filter_config_info')
    @patch('msprobe.core.config_check.config_check_cli.verl_get_config_file_path')
    def test_run_config_checking_command_verl_hyper_params_compare(self, mock_verl_get_config_file_path,
                                                                   mock_verl_filter_config_info,
                                                                   mock_verl_compare_hyper_params):
        """Test _run_config_checking_command with verl hyper params compare"""
        args = argparse.Namespace()
        args.dump = None
        args.compare = None
        args.verl_compare = ['NPU_verl.log', 'bench_verl.log']
        args.output = 'verl_compare_result'
        mock_verl_get_config_file_path.return_value = ("verl_compare_result/NPU_config.json",
                                                       "verl_compare_result/bench_config.json")        
        _run_config_checking_command(args)
        mock_verl_get_config_file_path.assert_called_once_with('verl_compare_result')
        mock_verl_filter_config_info.assert_any_call('NPU_verl.log', 'verl_compare_result/NPU_config.json')
        mock_verl_filter_config_info.assert_any_call('bench_verl.log', 'verl_compare_result/bench_config.json')
        mock_verl_compare_hyper_params.assert_called_once_with('verl_compare_result/NPU_config.json',
            'verl_compare_result/bench_config.json', 'verl_compare_result')
        
        # Test with default output path
        args.output = None
        mock_verl_get_config_file_path.return_value = ("./verl_param_compare_result/NPU_config.json",
                                                       "./verl_param_compare_result/bench_config.json")    
        _run_config_checking_command(args)
        mock_verl_get_config_file_path.assert_called_with('./verl_param_compare_result')
        mock_verl_compare_hyper_params.assert_called_with('./verl_param_compare_result/NPU_config.json',
            './verl_param_compare_result/bench_config.json', './verl_param_compare_result')

    @patch('msprobe.core.config_check.config_check_cli.slime_compare_hyper_params')
    @patch('msprobe.core.config_check.config_check_cli.slime_filter_config_info')
    @patch('msprobe.core.config_check.config_check_cli.slime_get_config_file_path')
    def test_run_config_checking_command_slime_hyper_params_compare(self, mock_slime_get_config_file_path,
                                                                   mock_slime_filter_config_info,
                                                                   mock_slime_compare_hyper_params):
        """Test _run_config_checking_command with slime hyper params compare"""
        args = argparse.Namespace()
        args.dump = None
        args.compare = None
        args.verl_compare = None
        args.verl_verify = None
        args.slime_compare = ['NPU_slime.log', 'bench_slime.log']
        args.output = 'slime_compare_result'
        mock_slime_get_config_file_path.return_value = ("slime_compare_result/NPU_config.json",
                                                       "slime_compare_result/bench_config.json")        
        _run_config_checking_command(args)
        mock_slime_get_config_file_path.assert_called_once_with('slime_compare_result')
        mock_slime_filter_config_info.assert_any_call('NPU_slime.log', 'slime_compare_result/NPU_config.json')
        mock_slime_filter_config_info.assert_any_call('bench_slime.log', 'slime_compare_result/bench_config.json')
        mock_slime_compare_hyper_params.assert_called_once_with('slime_compare_result/NPU_config.json',
            'slime_compare_result/bench_config.json', 'slime_compare_result')
        
        # Test with default output path
        args.output = None
        mock_slime_get_config_file_path.return_value = ("./slime_param_compare_result/NPU_config.json",
                                                       "./slime_param_compare_result/bench_config.json")    
        _run_config_checking_command(args)
        mock_slime_get_config_file_path.assert_called_with('./slime_param_compare_result')
        mock_slime_compare_hyper_params.assert_called_with('./slime_param_compare_result/NPU_config.json',
            './slime_param_compare_result/bench_config.json', './slime_param_compare_result')

    @patch('msprobe.core.config_check.config_check_cli.verl_filter_config_info')
    @patch('msprobe.core.config_check.config_check_cli.verl_verify_hyper_params')  
    def test_run_config_checking_command_verl_hyper_params_verify(self, mock_verl_verify_hyper_params,
                                                                mock_verl_filter_config_info):
        """Test _run_config_checking_command with verl hyper params verify"""
        args = argparse.Namespace()
        args.dump = None
        args.compare = None
        args.verl_compare = None
        args.slime_compare = None
        args.verl_verify = ['bench_config.yaml', 'tgt_config.log']
        args.output = 'verl_verify_result'
   
        _run_config_checking_command(args)

        mock_verl_filter_config_info.assert_any_call('tgt_config.log', 'verl_verify_result/tgt_config.json')
        mock_verl_verify_hyper_params.assert_called_once_with('bench_config.yaml',
            'verl_verify_result/tgt_config.json', 'verl_verify_result')
        
        # Test with default output path
        args.output = None

        _run_config_checking_command(args)
        
        mock_verl_verify_hyper_params.assert_called_with('bench_config.yaml',
            './verl_param_verify_result/tgt_config.json', './verl_param_verify_result')

    @patch('msprobe.core.config_check.config_check_cli.verl_compare_hyper_params')  
    @patch('msprobe.core.config_check.config_check_cli.verl_filter_config_info')
    @patch('msprobe.core.config_check.config_check_cli.verl_get_config_file_path')
    @patch('msprobe.core.config_check.config_check_cli.logger')
    def test_run_config_checking_command_verl_hyper_params_compare_invalid_param(
        self, mock_logger, mock_verl_get_config_file_path,
        mock_verl_filter_config_info, mock_verl_compare_hyper_params
    ):
        """Test _run_config_checking_command with verl hyper params compare invalid param"""
        args = argparse.Namespace()
        args.dump = None
        args.compare = None
        args.verl_compare = ['NPU_verl.md', 'bench_verl.log']
        args.output = None

        with self.assertRaises(Exception) as context:
            _run_config_checking_command(args)

        err_msg = ("The param of verl-compare require two log files, "
                    "and the file format just support '.log' or '.txt'.")
        mock_logger.error.assert_called_once_with(err_msg)
        self.assertEqual(str(context.exception), err_msg)
        mock_verl_get_config_file_path.assert_not_called()
        mock_verl_filter_config_info.assert_not_called()
        mock_verl_compare_hyper_params.assert_not_called()

    @patch('msprobe.core.config_check.config_check_cli.slime_compare_hyper_params')  
    @patch('msprobe.core.config_check.config_check_cli.slime_filter_config_info')
    @patch('msprobe.core.config_check.config_check_cli.slime_get_config_file_path')
    @patch('msprobe.core.config_check.config_check_cli.logger')
    def test_run_config_checking_command_slime_hyper_params_compare_invalid_param(
        self, mock_logger, mock_slime_get_config_file_path,
        mock_slime_filter_config_info, mock_slime_compare_hyper_params
    ):
        """Test _run_config_checking_command with slime hyper params compare invalid param"""
        args = argparse.Namespace()
        args.dump = None
        args.compare = None
        args.verl_compare = None
        args.verl_verify = None
        args.slime_compare = ['NPU_slime.md', 'bench_slime.log']
        args.output = None

        with self.assertRaises(Exception) as context:
            _run_config_checking_command(args)

        err_msg = ("The param of slime-compare require two log files, "
                    "and the file format just support '.log' or '.txt'.")
        mock_logger.error.assert_called_once_with(err_msg)
        self.assertEqual(str(context.exception), err_msg)
        mock_slime_get_config_file_path.assert_not_called()
        mock_slime_filter_config_info.assert_not_called()
        mock_slime_compare_hyper_params.assert_not_called()

    @patch('msprobe.core.config_check.config_check_cli.verl_verify_hyper_params')
    @patch('msprobe.core.config_check.config_check_cli.logger')
    def test_run_config_checking_command_verl_hyper_params_verify_invalid_param(
        self, mock_logger, mock_verl_verify_hyper_params
    ):
        """Test _run_config_checking_command with verl hyper params verify invalid param"""
        args = argparse.Namespace()
        args.dump = None
        args.compare = None
        args.verl_compare = None
        args.slime_compare = None
        args.verl_verify = ['bench_config.md', 'tgt_config.json']
        args.output = None
        with self.assertRaises(Exception) as context:
            _run_config_checking_command(args)

        err_msg = (
            "verl-verify requires tgt_log (log or txt file) as mandatory argument. "
            "When providing two files, the first must be bench_config (yaml file) "
            "and the second must be tgt_log (log or txt file)."
        )
        mock_logger.error.assert_called_once_with(err_msg)
        self.assertEqual(str(context.exception), err_msg)
        mock_verl_verify_hyper_params.assert_not_called()

    @patch('msprobe.core.config_check.config_check_cli.verl_verify_hyper_params')
    @patch('msprobe.core.config_check.config_check_cli.logger')
    def test_run_config_checking_command_verl_hyper_params_verify_wrong_order(
        self, mock_logger, mock_verl_verify_hyper_params
    ):
        """Test verl hyper params verify with reversed argument order"""
        args = argparse.Namespace()
        args.dump = None
        args.compare = None
        args.verl_compare = None
        args.slime_compare = None
        args.verl_verify = ['tgt_config.log', 'bench_config.yaml']
        args.output = None
        with self.assertRaises(Exception) as context:
            _run_config_checking_command(args)

        err_msg = (
            "verl-verify parameter order error: the first argument must be bench_config "
            "(yaml file), and the second argument must be tgt_log (log or txt file). "
            "Received 'tgt_config.log' as first and 'bench_config.yaml' as second."
        )
        mock_logger.error.assert_called_once_with(err_msg)
        self.assertEqual(str(context.exception), err_msg)
        mock_verl_verify_hyper_params.assert_not_called()

    def test_get_verl_verify_error_message_valid_cases(self):
        """Test _get_verl_verify_error_message returns None for valid arguments"""
        self.assertIsNone(_get_verl_verify_error_message(['tgt_config.log']))
        self.assertIsNone(_get_verl_verify_error_message(['bench_config.yaml', 'tgt_config.log']))

    def test_get_verl_verify_error_message_too_many_files(self):
        """Test _get_verl_verify_error_message with too many files"""
        err_msg = _get_verl_verify_error_message(['a.yaml', 'b.log', 'c.log'])
        self.assertEqual(err_msg, "verl-verify supports up to two files, but received 3 files")

    @patch('msprobe.core.config_check.config_check_cli.logger')
    def test_run_config_checking_command_invalid_param(self, mock_logger):
        """Test _run_config_checking_command with invalid parameter"""
        args = argparse.Namespace()
        args.dump = None
        args.compare = None
        args.verl_compare = None
        args.verl_verify = None
        args.slime_compare = None
        args.output = None
        
        with self.assertRaises(Exception) as context:
            _run_config_checking_command(args)
        
        self.assertEqual(str(context.exception), "The param is not correct, you need to give '-d' for dump or '-c' for compare "
                    "or '-vc' for verl compare or '-vv' for verl verify or '-sc' for slime compare.")
        mock_logger.error.assert_called_once_with("The param is not correct, you need to give '-d' for dump or '-c' for compare "
                    "or '-vc' for verl compare or '-vv' for verl verify or '-sc' for slime compare.")
