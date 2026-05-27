import os
import glob
import unittest
import logging
from unittest.mock import patch, mock_open, MagicMock
import json
import torch
import numpy as np
import signal
from msprobe.core.common.file_utils import create_directory, save_json, write_csv
from msprobe.core.common.exceptions import FileCheckException
from msprobe.pytorch.api_accuracy_checker.acc_check.multi_acc_check import split_json_file, signal_handler, run_parallel_ut, \
    prepare_config, main, ParallelUTConfig, wait_for_file_write_complete
from msprobe.pytorch.api_accuracy_checker.acc_check.data_generate import (
    gen_data, gen_real_tensor, gen_random_tensor, gen_args,
    gen_kwargs, fp32_to_hf32_to_fp32, gen_common_tensor, gen_bool_tensor,
    gen_list_kwargs, get_output_dtype, gen_api_params
)
from msprobe.pytorch.api_accuracy_checker.common.utils import CompareException
from msprobe.core.common.const import Const


class Args:
    def __init__(self, config_path=None, api_info_path=None, out_path=None, result_csv_path=None):
        self.config_path = config_path
        self.api_info_path = api_info_path
        self.out_path = out_path
        self.result_csv_path = result_csv_path


class TestFileCheck(unittest.TestCase):
    def setUp(self):
        src_path = 'temp_path'
        create_directory(src_path)
        dst_path = 'soft_link'
        os.symlink(src_path, dst_path)
        self.hard_path = os.path.abspath(src_path)
        self.soft_path = os.path.abspath(dst_path)
        json_path = os.path.join(self.hard_path, 'test.json')
        json_data = {'key': 'value'}
        save_json(json_path, json_data)
        self.hard_json_path = json_path
        soft_json_path = 'soft.json'
        os.symlink(json_path, soft_json_path)
        self.soft_json_path = os.path.abspath(soft_json_path)
        csv_path = os.path.join(self.hard_path, 'test.csv')
        csv_data = [['1', '2', '3']]
        write_csv(csv_data, csv_path)
        soft_csv_path = 'soft.csv'
        os.symlink(csv_path, soft_csv_path)
        self.csv_path = os.path.abspath(soft_csv_path)
        self.empty_path = "empty_path"

    def tearDown(self):
        os.unlink(self.soft_json_path)
        os.unlink(self.csv_path)
        os.unlink(self.soft_path)
        for file in os.listdir(self.hard_path):
            os.remove(os.path.join(self.hard_path, file))
        os.rmdir(self.hard_path)

    def test_config_path_soft_link_check(self):
        args = Args(config_path=self.soft_json_path, api_info_path=self.hard_json_path, out_path=self.hard_path)
        
        with self.assertRaises(Exception) as context:
            prepare_config(args)
            self.assertEqual(context.exception.code, FileCheckException.SOFT_LINK_ERROR)

    def test_api_info_path_soft_link_check(self):
        args = Args(config_path=self.hard_json_path, api_info_path=self.soft_json_path, out_path=self.hard_path)
        
        with self.assertRaises(Exception) as context:
            prepare_config(args)
            self.assertEqual(context.exception.code, FileCheckException.SOFT_LINK_ERROR)

    def test_out_path_soft_link_check(self):
        args = Args(config_path=self.hard_json_path, api_info_path=self.hard_json_path, out_path=self.soft_path)
        
        with self.assertRaises(Exception) as context:
            prepare_config(args)
            self.assertEqual(context.exception.code, FileCheckException.SOFT_LINK_ERROR)
    
    def test_result_csv_path_soft_link_check(self):
        args = Args(config_path=self.hard_json_path, api_info_path=self.hard_json_path, out_path=self.hard_path, 
                    result_csv_path=self.csv_path)
        
        with self.assertRaises(Exception) as context:
            prepare_config(args)
            self.assertEqual(context.exception.code, FileCheckException.SOFT_LINK_ERROR)
    
    def test_config_path_empty_check(self):
        args = Args(config_path=self.empty_path, api_info_path=self.hard_json_path, out_path=self.hard_path)
        
        with self.assertRaises(Exception) as context:
            prepare_config(args)
            self.assertEqual(context.exception.code, FileCheckException.ILLEGAL_PATH_ERROR)
    
    def test_api_info_path_empty_check(self):
        args = Args(config_path=self.hard_json_path, api_info_path=self.empty_path, out_path=self.hard_path)
        
        with self.assertRaises(Exception) as context:
            prepare_config(args)
            self.assertEqual(context.exception.code, FileCheckException.ILLEGAL_PATH_ERROR)
    
    def test_out_path_empty_check(self):
        args = Args(config_path=self.hard_json_path, api_info_path=self.hard_json_path, out_path=self.empty_path)
        with self.assertRaises(Exception) as context:
            prepare_config(args)
            self.assertEqual(context.exception.code, FileCheckException.ILLEGAL_PATH_ERROR)
    
    def test_result_csv_path_empty_check(self):
        args = Args(config_path=self.hard_json_path, api_info_path=self.hard_json_path, out_path=self.hard_path, 
                    result_csv_path=self.empty_path)
        with self.assertRaises(Exception) as context:
            prepare_config(args)
            self.assertEqual(context.exception.code, FileCheckException.ILLEGAL_PATH_ERROR)
    
    def test_config_path_invalid_check(self):
        args = Args(config_path=123, api_info_path=self.hard_json_path, out_path=self.hard_path)
        with self.assertRaises(Exception) as context:
            prepare_config(args)
            self.assertEqual(context.exception.code, FileCheckException.ILLEGAL_PATH_ERROR)
    
    def test_api_info_path_invalid_check(self):
        args = Args(config_path=self.hard_json_path, api_info_path="123", out_path=self.hard_path)
        with self.assertRaises(Exception) as context:
            prepare_config(args)
            self.assertEqual(context.exception.code, FileCheckException.ILLEGAL_PATH_ERROR)
    
    def test_out_path_invalid_check(self):
        args = Args(config_path=self.hard_json_path, api_info_path=self.hard_json_path, out_path=123)
        with self.assertRaises(Exception) as context:
            prepare_config(args)
            self.assertEqual(context.exception.code, FileCheckException.ILLEGAL_PATH_ERROR)
    
    def test_result_csv_path_invalid_check(self):
        args = Args(config_path=self.hard_json_path, api_info_path=self.hard_json_path, out_path=self.hard_path, 
                    result_csv_path=123)
        with self.assertRaises(Exception) as context:
            prepare_config(args)
            self.assertEqual(context.exception.code, FileCheckException.ILLEGAL_PATH_ERROR)


class TestMultiRunUT(unittest.TestCase):

    def setUp(self):
        self.test_json_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dump.json")
        self.test_data = {'dump_data_dir': '/test', 'data': {'key1': 'TRUE', 'key2': 'TRUE', 'key3': 'TRUE'}}
        self.test_json_content = json.dumps(self.test_data)
        self.forward_split_files_content = [
            {'key1': 'TRUE', 'key2': 'TRUE'},
            {'key3': 'TRUE', 'key4': 'TRUE'}
        ]
        self.original_sigint = signal.getsignal(signal.SIGINT)
        self.original_sigterm = signal.getsignal(signal.SIGTERM)


    @patch('os.remove')
    @patch('os.path.realpath', side_effect=lambda x: x)
    @patch('msprobe.pytorch.api_accuracy_checker.acc_check.multi_acc_check.check_link')
    @patch('msprobe.pytorch.api_accuracy_checker.acc_check.multi_acc_check.FileChecker')
    @patch('msprobe.pytorch.api_accuracy_checker.acc_check.multi_acc_check.split_json_file',
           return_value=(['forward_split1.json', 'forward_split2.json'], 2))
    def test_prepare_config(self, mock_split_json_file, mock_FileChecker, mock_check_link,
                            mock_realpath, mock_remove):
        mock_FileChecker_instance = MagicMock()
        mock_FileChecker_instance.common_check.return_value = './'
        mock_FileChecker.return_value = mock_FileChecker_instance
        args = MagicMock()
        args.api_info = 'forward.json'
        args.out_path = './'
        args.num_splits = 2
        args.save_error_data = True
        args.jit_compile = False
        args.device_id = [0, 1]
        args.result_csv_path = None
        args.config_path = None

        config = prepare_config(args)

        self.assertEqual(config.num_splits, 2)
        self.assertTrue(config.save_error_data_flag)
        self.assertFalse(config.jit_compile_flag)
        self.assertEqual(config.device_id, [0, 1])
        self.assertEqual(config.total_items, 2)

    @patch('msprobe.pytorch.api_accuracy_checker.acc_check.multi_acc_check.parse_json_info_forward_backward',
           return_value=({"a": 1}, {"b": 2}, "/tmp"))
    @patch('msprobe.pytorch.api_accuracy_checker.acc_check.multi_acc_check.load_json',
           return_value={"dump_data_dir": "/tmp", "data": {"a": 1, "b": 2}})
    @patch('msprobe.pytorch.api_accuracy_checker.acc_check.multi_acc_check.save_json')
    def test_split_json_file(self, mock_save_json, mock_load_json, mock_parse):
        tmp_json = "/tmp/test.json"
        split_files, total_items = split_json_file(tmp_json, 1, False)
        self.assertEqual(total_items, 1)
        self.assertEqual(len(split_files), 1)
        mock_save_json.assert_called_once()

    @patch('os.path.getsize', side_effect=[10, 10])
    def test_wait_for_file_write_complete_success(self, mock_getsize):
        self.assertTrue(wait_for_file_write_complete("/tmp/a.csv", timeout=2))

    @patch('os.path.getsize', side_effect=[1, 2, 3, 3])
    @patch('time.time', side_effect=[0, 1, 2, 100])  # 触发超时
    def test_wait_for_file_write_complete_timeout(self, mock_time, mock_getsize):
        self.assertFalse(wait_for_file_write_complete("/tmp/a.csv", timeout=1))

    @patch('os.path.getsize', side_effect=[10, 10])
    def test_wait_for_file_write_complete_success(self, mock_getsize):
        self.assertTrue(wait_for_file_write_complete("/tmp/a.csv", timeout=2))

    @patch('os.path.getsize', side_effect=[1, 2, 3, 3])
    @patch('time.time', side_effect=[0, 1, 2, 100])  # 触发超时
    def test_wait_for_file_write_complete_timeout(self, mock_time, mock_getsize):
        self.assertFalse(wait_for_file_write_complete("/tmp/a.csv", timeout=1))

    @patch('argparse.ArgumentParser.parse_args')
    @patch('msprobe.pytorch.api_accuracy_checker.acc_check.multi_acc_check.prepare_config')
    @patch('msprobe.pytorch.api_accuracy_checker.acc_check.multi_acc_check.run_parallel_ut')
    def test_main(self, mock_run_parallel_ut, mock_prepare_config, mock_parse_args):
        main()
        mock_parse_args.assert_called()
        mock_prepare_config.assert_called()
        mock_run_parallel_ut.assert_called()

    class TestDataGenerate(unittest.TestCase):

        def test_gen_data_random_tensor_success(self):
            info = {"type": "torch.Tensor", "dtype": "torch.float32", "shape": [2, 3]}
            t = gen_data(info, "any_api", False, None)
            self.assertIsInstance(t, torch.Tensor)
            self.assertEqual(t.shape, torch.Size([2, 3]))

        def test_gen_data_numpy_success(self):
            info = {"type": "numpy.float32", "value": 3.14}
            out = gen_data(info, "a", False, None)
            self.assertTrue(isinstance(out, np.generic))

        def test_gen_data_numpy_unsupported(self):
            info = {"type": "numpy.not_exists", "value": 0.1}
            with self.assertRaises(Exception):
                gen_data(info, "a", False, None)

        def test_gen_data_fp8_on_gpu_raises(self):
            info = {"type": "torch.Tensor", "dtype": "torch.float8_e4m3fn", "shape": [1]}
            with patch("msprobe.pytorch.api_accuracy_checker.acc_check.data_generate.IS_GPU", True):
                with self.assertRaises(CompareException):
                    gen_data(info, "a", False, None)

        @patch("msprobe.pytorch.api_accuracy_checker.acc_check.data_generate.FileChecker")
        @patch("msprobe.pytorch.api_accuracy_checker.acc_check.data_generate.load_pt",
               return_value=torch.zeros(1))
        def test_gen_real_tensor_pt(self, _, mock_checker):
            mock_checker.return_value.common_check.return_value = "data.pt"
            result = gen_real_tensor("data.pt", None)
            self.assertTrue(torch.equal(result, torch.zeros(1)))

        @patch("msprobe.pytorch.api_accuracy_checker.acc_check.data_generate.FileChecker")
        @patch("msprobe.pytorch.api_accuracy_checker.acc_check.data_generate.load_npy",
               return_value=np.ones((2, 2)))
        def test_gen_real_tensor_npy(self, _, mock_checker):
            mock_checker.return_value.common_check.return_value = "data.npy"
            result = gen_real_tensor("data.npy", None)
            self.assertEqual(result.shape, (2, 2))

        def test_gen_real_tensor_invalid_ext(self):
            with patch("msprobe.pytorch.api_accuracy_checker.acc_check.data_generate.FileChecker") as fc:
                fc.return_value.common_check.return_value = "data.txt"
                with self.assertRaises(CompareException):
                    gen_real_tensor("data.txt", None)

        def test_fp32_to_hf32_to_fp32_success(self):
            t = torch.tensor([0.5], dtype=torch.float32)
            o = fp32_to_hf32_to_fp32(t)
            self.assertEqual(o.dtype, torch.float32)

        def test_gen_random_tensor_bool(self):
            info = {"dtype": "torch.bool", "Min": 0, "Max": 1, "shape": [3, 3]}
            t = gen_random_tensor(info, None)
            self.assertEqual(t.dtype, torch.bool)

        def test_gen_random_tensor_invalid_range(self):
            info = {"dtype": "torch.float32", "Min": "a", "Max": 1.0, "shape": [1]}
            with self.assertRaises(CompareException):
                gen_random_tensor(info, None)

        def test_gen_common_tensor_nan_path(self):
            t = gen_common_tensor([0, 0], [float("nan"), float("nan")], [2], "torch.float32", None)
            self.assertTrue(torch.isnan(t).any())

        def test_gen_bool_tensor(self):
            t = gen_bool_tensor(0, 1, [2, 2])
            self.assertEqual(t.dtype, torch.bool)

        def test_gen_args_nested_success(self):
            args_info = [
                [{"type": "torch.Tensor", "dtype": "torch.float32", "shape": [1]}]
            ]
            out = gen_args(args_info, "api", {"need_grad": True})
            self.assertIsInstance(out[0][0], torch.Tensor)

        def test_gen_args_unsupported(self):
            with self.assertRaises(NotImplementedError):
                gen_args([123], "api", {})

        def test_gen_kwargs_tensor_success(self):
            info = {"input_kwargs": {
                "t": {"type": "torch.Tensor", "dtype": "torch.float32", "shape": [1]}
            }}
            out = gen_kwargs(info, "api")
            self.assertTrue(isinstance(out["t"], torch.Tensor))

        def test_get_output_dtype_success(self):
            info = {Const.OUTPUT: [{Const.DTYPE: "torch.float32"}]}
            dtype = get_output_dtype(info)
            self.assertEqual(dtype, torch.float32)

        def test_gen_api_params_invalid_convert_type(self):
            with self.assertRaises(CompareException):
                gen_api_params({}, "api", convert_type="unknown")

        def test_gen_api_params_success(self):
            info = {
                "input_args": [{"type": "torch.Tensor", "dtype": "torch.float32", "shape": [1]}],
                "input_kwargs": {},
                Const.OUTPUT: [{Const.DTYPE: "torch.float32"}]
            }
            args, kwargs, dt = gen_api_params(info, "api")
            self.assertIsInstance(args[0], torch.Tensor)
            self.assertEqual(dt, torch.float32)

    def tearDown(self):
        signal.signal(signal.SIGINT, self.original_sigint)
        signal.signal(signal.SIGTERM, self.original_sigterm)
        current_directory = os.getcwd()
        pattern = os.path.join(current_directory, 'accuracy_checking_*')
        files = glob.glob(pattern)

        for file in files:
            try:
                os.remove(file)
                logging.info(f"Deleted file: {file}")
            except Exception as e:
                logging.error(f"Failed to delete file {file}: {e}")

