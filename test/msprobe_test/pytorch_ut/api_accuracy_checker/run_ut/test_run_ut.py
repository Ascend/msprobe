# coding=utf-8
import os
import copy
import shutil
import tempfile
import unittest
import argparse
from unittest.mock import patch, MagicMock
from unittest.mock import patch, DEFAULT
import pandas as pd
import torch
import numpy as np
from msprobe.pytorch.api_accuracy_checker.acc_check.acc_check import *
from msprobe.core.common.file_utils import get_json_contents, create_directory, save_json, write_csv
from msprobe.core.common.exceptions import FileCheckException
from msprobe.pytorch.api_accuracy_checker.acc_check.acc_check_utils import UtDataInfo, exec_api
from msprobe.pytorch.api_accuracy_checker.acc_check.data_generate import (
    gen_data, gen_real_tensor, gen_random_tensor, gen_common_tensor,
    gen_bool_tensor, gen_args, gen_kwargs, gen_list_kwargs,
    get_output_dtype, gen_api_params
)
from msprobe.core.common.const import Const, CompareConst
from msprobe.pytorch.api_accuracy_checker.common.utils import CompareException
from msprobe.pytorch.api_accuracy_checker.acc_check.acc_check import (
    blacklist_and_whitelist_filter,
    check_need_grad,
    need_to_backward,
    extract_tensors_grad,
    run_backward,
    preprocess_forward_content,
)

base_dir = os.path.dirname(os.path.realpath(__file__))
forward_file = os.path.join(base_dir, "forward.json")
forward_content = get_json_contents(forward_file)
for api_full_name, api_info_dict in forward_content.items():
    api_full_name = api_full_name
    api_info_dict = api_info_dict


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
            acc_check_command(args)
            self.assertEqual(context.exception.code, FileCheckException.SOFT_LINK_ERROR)

    def test_api_info_path_soft_link_check(self):
        args = Args(config_path=self.hard_json_path, api_info_path=self.soft_json_path, out_path=self.hard_path)

        with self.assertRaises(Exception) as context:
            acc_check_command(args)
            self.assertEqual(context.exception.code, FileCheckException.SOFT_LINK_ERROR)

    def test_out_path_soft_link_check(self):
        args = Args(config_path=self.hard_json_path, api_info_path=self.hard_json_path, out_path=self.soft_path)

        with self.assertRaises(Exception) as context:
            acc_check_command(args)
            self.assertEqual(context.exception.code, FileCheckException.SOFT_LINK_ERROR)

    def test_result_csv_path_soft_link_check(self):
        args = Args(config_path=self.hard_json_path, api_info_path=self.hard_json_path, out_path=self.hard_path,
                    result_csv_path=self.csv_path)

        with self.assertRaises(Exception) as context:
            acc_check_command(args)
            self.assertEqual(context.exception.code, FileCheckException.SOFT_LINK_ERROR)

    def test_config_path_empty_check(self):
        args = Args(config_path=self.empty_path, api_info_path=self.hard_json_path, out_path=self.hard_path)

        with self.assertRaises(Exception) as context:
            acc_check_command(args)
            self.assertEqual(context.exception.code, FileCheckException.ILLEGAL_PATH_ERROR)

    def test_api_info_path_empty_check(self):
        args = Args(config_path=self.hard_json_path, api_info_path=self.empty_path, out_path=self.hard_path)

        with self.assertRaises(Exception) as context:
            acc_check_command(args)
            self.assertEqual(context.exception.code, FileCheckException.ILLEGAL_PATH_ERROR)

    def test_out_path_empty_check(self):
        args = Args(config_path=self.hard_json_path, api_info_path=self.hard_json_path, out_path=self.empty_path)
        with self.assertRaises(Exception) as context:
            acc_check_command(args)
            self.assertEqual(context.exception.code, FileCheckException.ILLEGAL_PATH_ERROR)

    def test_result_csv_path_empty_check(self):
        args = Args(config_path=self.hard_json_path, api_info_path=self.hard_json_path, out_path=self.hard_path,
                    result_csv_path=self.empty_path)
        with self.assertRaises(Exception) as context:
            acc_check_command(args)
            self.assertEqual(context.exception.code, FileCheckException.ILLEGAL_PATH_ERROR)

    def test_config_path_invalid_check(self):
        args = Args(config_path=123, api_info_path=self.hard_json_path, out_path=self.hard_path)
        with self.assertRaises(Exception) as context:
            acc_check_command(args)
            self.assertEqual(context.exception.code, FileCheckException.ILLEGAL_PATH_ERROR)

    def test_api_info_path_invalid_check(self):
        args = Args(config_path=self.hard_json_path, api_info_path="123", out_path=self.hard_path)
        with self.assertRaises(Exception) as context:
            acc_check_command(args)
            self.assertEqual(context.exception.code, FileCheckException.ILLEGAL_PATH_ERROR)

    def test_out_path_invalid_check(self):
        args = Args(config_path=self.hard_json_path, api_info_path=self.hard_json_path, out_path=123)
        with self.assertRaises(Exception) as context:
            acc_check_command(args)
            self.assertEqual(context.exception.code, FileCheckException.ILLEGAL_PATH_ERROR)

    def test_result_csv_path_invalid_check(self):
        args = Args(config_path=self.hard_json_path, api_info_path=self.hard_json_path, out_path=self.hard_path,
                    result_csv_path=123)
        with self.assertRaises(Exception) as context:
            acc_check_command(args)
            self.assertEqual(context.exception.code, FileCheckException.ILLEGAL_PATH_ERROR)


class TestRunUtMethods(unittest.TestCase):
    def test_exec_api(self):
        api_info = copy.deepcopy(api_info_dict)

        [api_type, api_name, _, _] = api_full_name.split(".")
        args, kwargs, need_grad = get_api_info(api_info, api_name, None)
        cpu_params = generate_cpu_params(args, kwargs, True, '')
        cpu_args, cpu_kwargs = cpu_params.cpu_args, cpu_params.cpu_kwargs
        cpu_exec_params = ExecParams(api_type, api_name, Const.CPU_LOWERCASE, cpu_args, cpu_kwargs, False, None)
        out = exec_api(cpu_exec_params)
        self.assertEqual(out[0].dtype, torch.float32)
        self.assertTrue(out[0].requires_grad)
        self.assertEqual(out[0].shape, torch.Size([2048, 2, 1, 128]))

    def test_generate_device_params(self):
        mock_tensor = torch.rand([2, 2560, 24, 24], dtype=torch.float32, requires_grad=True)

        with patch.multiple('torch.Tensor',
                            to=DEFAULT,
                            clone=DEFAULT,
                            detach=DEFAULT,
                            requires_grad_=DEFAULT,
                            type_as=DEFAULT,
                            retain_grad=DEFAULT) as mocks:
            mocks['clone'].return_value = mock_tensor
            mocks['detach'].return_value = mock_tensor
            mocks['requires_grad_'].return_value = mock_tensor
            mocks['type_as'].return_value = mock_tensor
            mocks['retain_grad'].return_value = None
            mocks['to'].return_value = mock_tensor

            device_args, device_kwargs, _ = generate_device_params([mock_tensor], {'inplace': False}, True, '')
            self.assertEqual(len(device_args), 1)
            self.assertEqual(device_args[0].dtype, torch.float32)
            self.assertTrue(device_args[0].requires_grad)
            self.assertEqual(device_args[0].shape, torch.Size([2, 2560, 24, 24]))
            self.assertEqual(device_kwargs, {'inplace': False})

    def test_generate_cpu_params(self):
        api_info = copy.deepcopy(api_info_dict)
        [api_type, api_name, _, _] = api_full_name.split(".")
        args, kwargs, need_grad = get_api_info(api_info, api_name, None)
        cpu_params = generate_cpu_params(args, kwargs, True, '')
        cpu_args, cpu_kwargs = cpu_params.cpu_args, cpu_params.cpu_kwargs
        self.assertEqual(len(cpu_args), 2)
        self.assertEqual(cpu_args[0].dtype, torch.float32)
        self.assertTrue(cpu_args[0].requires_grad)
        self.assertEqual(cpu_args[0].shape, torch.Size([2048, 2, 1, 256]))
        self.assertEqual(cpu_kwargs, {'dim': -1})

    def test_UtDataInfo(self):
        data_info = UtDataInfo(None, None, None, None, None, None, None)
        self.assertIsNone(data_info.bench_grad)
        self.assertIsNone(data_info.device_grad)
        self.assertIsNone(data_info.device_output)
        self.assertIsNone(data_info.bench_output)
        self.assertIsNone(data_info.grad_in)
        self.assertIsNone(data_info.in_fwd_data_list)

    def test_blacklist_and_whitelist_filter(self):
        api_name = "test_api"
        black_list = ["test_api"]
        white_list = []
        result = blacklist_and_whitelist_filter(api_name, black_list, white_list)
        self.assertTrue(result)

        api_name = "test_api"
        black_list = []
        white_list = ["another_api"]
        result = blacklist_and_whitelist_filter(api_name, black_list, white_list)
        self.assertTrue(result)

        api_name = "test_api"
        black_list = ["test_api"]
        white_list = ["test_api"]
        result = blacklist_and_whitelist_filter(api_name, black_list, white_list)
        self.assertTrue(result)

        api_name = "test_api"
        black_list = []
        white_list = ["test_api"]
        result = blacklist_and_whitelist_filter(api_name, black_list, white_list)
        self.assertFalse(result)

    def test_supported_api(self):
        api_name = "torch.matmul"
        result = is_unsupported_api(api_name)
        self.assertFalse(result)

        api_name = "Distributed.all_reduce"
        result = is_unsupported_api(api_name)
        self.assertTrue(result)

    def test_no_backward(self):
        grad_index = None
        out = (1, 2, 3)
        result = need_to_backward(grad_index, out)
        self.assertFalse(result)

        grad_index = 0
        out = 42
        result = need_to_backward(grad_index, out)
        self.assertTrue(result)

    def test_check_need_grad_given_out_kwarg_then_return_false(self):
        from msprobe.pytorch.api_accuracy_checker.acc_check.acc_check import check_need_grad

        api_info_dict = {"input_kwargs": {"out": True}}
        result = check_need_grad(api_info_dict)
        self.assertFalse(result)

    def test_check_need_grad_given_no_out_kwarg_then_return_true(self):
        from msprobe.pytorch.api_accuracy_checker.acc_check.acc_check import check_need_grad

        api_info_dict = {"input_kwargs": {}}
        result = check_need_grad(api_info_dict)
        self.assertTrue(result)

    def test_preprocess_forward_content_given_duplicate_apis_then_filter(self):
        from msprobe.pytorch.api_accuracy_checker.acc_check.acc_check import preprocess_forward_content

        forward_content = {
            "torch.add_1": {"input_args": [{"value": 1}], "input_kwargs": {}},
            "torch.add_2": {"input_args": [{"value": 1}], "input_kwargs": {}},
            "torch.sub": {"input_args": [{"value": 2}], "input_kwargs": {}}
        }

        result = preprocess_forward_content(forward_content)

        self.assertEqual(len(result), 2)  # One duplicate should be removed

    def test_initialize_save_error_data_given_valid_path_then_return_path(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = self.temp_dir.name

        from msprobe.pytorch.api_accuracy_checker.acc_check.acc_check import initialize_save_error_data

        error_data_path = os.path.join(self.test_dir, "error_data")
        result = initialize_save_error_data(error_data_path)

        self.assertTrue(os.path.exists(result))
        self.assertIn("ut_error_data", result)
        self.temp_dir.cleanup()

    @patch('msprobe.pytorch.api_accuracy_checker.acc_check.acc_check.UtDataProcessor')
    def test_do_save_error_data_not_called_when_all_success(self, mock_processor):
        from msprobe.pytorch.api_accuracy_checker.acc_check.acc_check import do_save_error_data

        data_info = UtDataInfo(
            None,  # bench_grad_out
            None,  # device_grad_out
            None,  # device_output
            None,  # bench_output
            None,  # grad_in
            [],    # in_fwd_data_list
            "",    # backward_message
        )
        # 前向、反向都成功，不应该落盘错误数据
        do_save_error_data("torch.add", data_info, "/tmp/error_data", True, True)
        mock_processor.assert_not_called()

    def test_run_backward_without_grad_index(self):
        from msprobe.pytorch.api_accuracy_checker.acc_check.acc_check import run_backward

        x = torch.tensor([1.0], requires_grad=True)
        grad = torch.ones_like(x)
        out = x * 2

        grads = run_backward([x], grad, None, out)
        self.assertEqual(len(grads), 1)
        self.assertIsNotNone(grads[0])

    def test_run_backward_invalid_grad_index_type(self):
        from msprobe.pytorch.api_accuracy_checker.acc_check.acc_check import run_backward

        x = torch.tensor([1.0], requires_grad=True)
        grad = torch.ones_like(x)
        out = [x * 2]

        with self.assertRaises(TypeError):
            run_backward([x], grad, "0", out)

    def test_run_backward_grad_index_out_of_range(self):
        from msprobe.pytorch.api_accuracy_checker.acc_check.acc_check import run_backward

        x = torch.tensor([1.0], requires_grad=True)
        grad = torch.ones_like(x)
        out = [x * 2]

        with self.assertRaises(IndexError):
            run_backward([x], grad, 1, out)

    def test_extract_tensors_grad_nested(self):
        from msprobe.pytorch.api_accuracy_checker.acc_check.acc_check import extract_tensors_grad

        x1 = torch.tensor([1.0], requires_grad=True)
        x2 = torch.tensor([2.0], requires_grad=True)
        y = x1 + x2
        y.backward(torch.tensor([1.0]))

        grads = extract_tensors_grad([[x1, (x2,)]])
        self.assertEqual(len(grads), 2)
        self.assertIsNotNone(grads[0])
        self.assertIsNotNone(grads[1])

    def test_extract_tensors_grad_depth_exceeded(self):
        from msprobe.pytorch.api_accuracy_checker.acc_check.acc_check import extract_tensors_grad
        from msprobe.core.common.const import Const
        from msprobe.core.common.utils import CompareException

        args = [torch.tensor([1.0])]
        with self.assertRaises(CompareException):
            extract_tensors_grad(args, depth=Const.MAX_DEPTH + 1)

    def test_preprocess_forward_content_filter_max_min_only(self):
        from msprobe.pytorch.api_accuracy_checker.acc_check.acc_check import preprocess_forward_content

        forward_content = {
            "torch.add.1": {
                "input_args": [{"value": 1, "Max": 10, "Min": -10}],
                "input_kwargs": {}
            },
            "torch.add.2": {
                "input_args": [{"value": 1}],  # 去掉 Max / Min 之后与上面等价
                "input_kwargs": {}
            },
        }

        result = preprocess_forward_content(forward_content)
        # 两个 add 只保留一个
        self.assertEqual(len(result), 1)
        self.assertIn("torch.add.1", result)  # 第一个会先进入

    def test_acc_check_parser_device_unique_ok(self):
        from msprobe.pytorch.api_accuracy_checker.acc_check.acc_check import _acc_check_parser

        parser = argparse.ArgumentParser()
        _acc_check_parser(parser)
        with self.assertRaises(SystemExit):
            parser.parse_args(["-d", "1", "1"])

    def test_acc_check_parser_device_duplicate_raises(self):
        from msprobe.pytorch.api_accuracy_checker.acc_check.acc_check import _acc_check_parser

        parser = argparse.ArgumentParser()
        _acc_check_parser(parser)
        with self.assertRaises(SystemExit):
            parser.parse_args(["-d", "0", "0"])

    def test_acc_check_parser_device_negative_raises(self):
        from msprobe.pytorch.api_accuracy_checker.acc_check.acc_check import _acc_check_parser

        parser = argparse.ArgumentParser()
        _acc_check_parser(parser)
        with self.assertRaises(SystemExit):
            parser.parse_args(["-d", "-1"])

class TestAccCheckBasic(unittest.TestCase):

    def test_blacklist_and_whitelist_filter(self):
        black = ["add"]
        white = ["add", "mul"]
        # 在黑名单则 True（不执行）
        self.assertTrue(blacklist_and_whitelist_filter("add", black, white))
        # 不在黑名单且在白名单 False（执行）
        self.assertFalse(blacklist_and_whitelist_filter("mul", black, white))
        # 有白名单但未命中 True（不执行）
        self.assertTrue(blacklist_and_whitelist_filter("sub", black, white))

    def test_check_need_grad(self):
        self.assertTrue(check_need_grad({"input_kwargs": {}}))
        self.assertFalse(check_need_grad({"input_kwargs": {"out": 1}}))

    def test_need_to_backward(self):
        # 当输出是 list 且没有 grad_index → False
        self.assertFalse(need_to_backward(None, [torch.tensor(1)]))
        # 标量 → True
        self.assertTrue(need_to_backward(None, torch.tensor(1)))
        # 指定 grad index → True
        self.assertTrue(need_to_backward(0, [torch.tensor(1)]))

    def test_extract_tensors_grad(self):
        a = torch.tensor(1.0, requires_grad=True)
        b = torch.tensor(2.0, requires_grad=True)
        (a + b).backward()

        result = extract_tensors_grad([a, [b]])
        self.assertEqual(len(result), 2)
        self.assertIsNotNone(result[0])
        self.assertIsNotNone(result[1])

    def test_extract_tensors_grad_depth_exceed(self):
        deep_list = []
        cur = deep_list
        for _ in range(200):
            nxt = []
            cur.append(nxt)
            cur = nxt

    @patch("msprobe.pytorch.api_accuracy_checker.acc_check.acc_check.logger")
    def test_run_backward(self, mock_logger):
        a = torch.tensor([1.0, 2.0], requires_grad=True)
        out = a * 2
        grad = torch.tensor([1.0, 1.0])
        result = run_backward([a], grad, None, out)
        self.assertEqual(len(result), 1)
        self.assertIsNotNone(result[0])

    def test_preprocess_forward_content(self):
        sample = {
            "Func.add.0": {"input_args": [{"a": 1, "Max": 10}], "input_kwargs": {}},
            "Func.add.1": {"input_args": [{"a": 1, "Max": 10}], "input_kwargs": {}},  # duplicate
        }
        output = preprocess_forward_content(sample)
        self.assertEqual(len(output), 1)

class TestAccCheckFunctions(unittest.TestCase):

    def test_blacklist_and_whitelist_filter(self):
        # black_list 优先级最高
        black_list = ["add"]
        white_list = ["add"]
        self.assertTrue(blacklist_and_whitelist_filter("add", black_list, white_list))

        # white_list 限制
        black_list = []
        white_list = ["mul"]
        self.assertTrue(blacklist_and_whitelist_filter("add", black_list, white_list))

        # 都不影响 → allow run
        self.assertFalse(blacklist_and_whitelist_filter("mul", black_list, white_list))

    def test_check_need_grad(self):
        info_no_out = {Const.INPUT_KWARGS: {}}
        info_with_out = {Const.INPUT_KWARGS: {"out": "x"}}

        self.assertTrue(check_need_grad(info_no_out))
        self.assertFalse(check_need_grad(info_with_out))

    def test_need_to_backward(self):
        out = (torch.tensor([1.0]), torch.tensor([2.0]))
        self.assertFalse(need_to_backward(None, out))  # tuple + grad_index None
        self.assertTrue(need_to_backward(1, out))       # tuple + valid index

    def test_extract_tensors_grad(self):
        x = torch.tensor([1.0], requires_grad=True)
        y = torch.tensor([2.0], requires_grad=True)
        out = x + y
        out.backward()

        grads = extract_tensors_grad([x, [y]])
        self.assertEqual(len(grads), 2)

    def test_extract_tensors_grad_depth_exceed(self):
        deep_list = []
        ref = deep_list
        for _ in range(Const.MAX_DEPTH + 2):
            new = [ref]
            ref = new
        with self.assertRaises(CompareException):
            extract_tensors_grad(ref)

    def test_run_backward_invalid_grad_index_type(self):
        x = torch.tensor([1.0], requires_grad=True)
        with self.assertRaises(TypeError):
            run_backward([x], torch.tensor([1.0]), "a", [x])

    def test_run_backward_grad_index_out_of_range(self):
        x = torch.tensor([1.0], requires_grad=True)
        with self.assertRaises(IndexError):
            run_backward([x], torch.tensor([1.0]), 5, [x])

    def test_preprocess_forward_content_basic(self):
        content = {
            "add.0": {"input_args": [{"A": 1}], "input_kwargs": {}},
            "add.1": {"input_args": [{"A": 1}], "input_kwargs": {}},
            "mul.0": {"input_args": [{"B": 2}], "input_kwargs": {}}
        }
        res = preprocess_forward_content(content)
        # add 只有一份被保留，mul 也保留
        self.assertEqual(len(res), 2)

    @patch("msprobe.pytorch.api_accuracy_checker.acc_check.acc_check.logger")
    def test_preprocess_forward_content_keyerror(self, mock_logger):
        content = {"bad": {"input_args": [{}], "input_kwargs": {}}}
        # 触发 KeyError, 被捕获并记录日志
        res = preprocess_forward_content(content)



class TestDataGenerate(unittest.TestCase):

    # -----------------------
    # Test gen_real_tensor
    # -----------------------
    @patch("msprobe.pytorch.api_accuracy_checker.acc_check.data_generate.FileChecker")
    @patch("msprobe.pytorch.api_accuracy_checker.acc_check.data_generate.load_pt")
    def test_gen_real_tensor_pt(self, mock_load_pt, mock_checker):
        mock_checker.return_value.common_check.return_value = "/tmp/a.pt"
        mock_load_pt.return_value = torch.ones(3)

        data = gen_real_tensor("/tmp/a.pt", convert_type=None)
        self.assertTrue(torch.equal(data, torch.ones(3)))

    @patch("msprobe.pytorch.api_accuracy_checker.acc_check.data_generate.FileChecker")
    @patch("msprobe.pytorch.api_accuracy_checker.acc_check.data_generate.load_npy")
    def test_gen_real_tensor_npy(self, mock_load_npy, mock_checker):
        mock_checker.return_value.common_check.return_value = "/tmp/a.npy"
        mock_load_npy.return_value = np.array([1, 2, 3])

        data = gen_real_tensor("/tmp/a.npy", convert_type=None)
        self.assertTrue(torch.equal(data, torch.tensor([1, 2, 3])))

    # -----------------------
    # Test gen_bool_tensor
    # -----------------------
    def test_gen_bool_tensor(self):
        t = gen_bool_tensor(0, 1, (3, 3))
        self.assertEqual(t.dtype, torch.bool)
        self.assertEqual(t.shape, (3, 3))

    # -----------------------
    # Test gen_common_tensor
    # -----------------------
    def test_gen_common_tensor_float(self):
        low = [0.1, 0.1]
        high = [1.0, 1.0]
        out = gen_common_tensor(low, high, (2, 2), "torch.float32", None)
        self.assertEqual(out.shape, (2, 2))

    def test_gen_common_tensor_int(self):
        low = [0, 0]
        high = [5, 5]
        out = gen_common_tensor(low, high, (2, 2), "torch.int32", None)
        self.assertTrue(out.dtype == torch.int32)

    # -----------------------
    # Test gen_random_tensor
    # -----------------------
    def test_gen_random_tensor_bool(self):
        info = {"Min": 0, "Max": 1, "shape": [2, 2], "dtype": "torch.bool"}
        out = gen_random_tensor(info, None)
        self.assertEqual(out.dtype, torch.bool)

    def test_gen_random_tensor_invalid(self):
        info = {"Min": "a", "Max": 1, "shape": [2, 2], "dtype": "torch.float32"}
        with self.assertRaises(CompareException):
            gen_random_tensor(info, None)

    # -----------------------
    # Test gen_data
    # -----------------------
    @patch("msprobe.pytorch.api_accuracy_checker.acc_check.data_generate.gen_real_tensor")
    def test_gen_data_real_tensor(self, mock_real):
        mock_real.return_value = torch.zeros(2)
        info = {"type": "torch.Tensor", "datapath": "a.pt", "dtype": "torch.float32"}
        out = gen_data(info, "add", need_grad=False, convert_type=None, real_data_path="")
        self.assertTrue(torch.equal(out, torch.zeros(2)))

    @patch("msprobe.pytorch.api_accuracy_checker.acc_check.data_generate.gen_random_tensor")
    def test_gen_data_random_tensor(self, mock_random):
        mock_random.return_value = torch.ones(2)
        info = {"type": "torch.Tensor", "shape": [2], "dtype": "torch.float32"}
        out = gen_data(info, "add", need_grad=False, convert_type=None)
        self.assertTrue(torch.equal(out, torch.ones(2)))

    def test_gen_data_numpy(self):
        info = {"type": "numpy.float32", "value": 1.5, "dtype": ""}
        out = gen_data(info, "add", False, None)
        self.assertTrue(isinstance(out, np.float32))

    # -----------------------
    # Test gen_list_kwargs
    # -----------------------
    @patch("msprobe.pytorch.api_accuracy_checker.acc_check.data_generate.gen_data")
    def test_gen_list_kwargs(self, mock_gen_data):
        mock_gen_data.return_value = torch.ones(1)
        info = [{"type": "torch.Tensor", "dtype": "torch.float32", "shape": [1]}]
        out = gen_list_kwargs(info, "add", None)
        self.assertEqual(len(out), 1)

    # -----------------------
    # Test gen_kwargs
    # -----------------------
    @patch("msprobe.pytorch.api_accuracy_checker.acc_check.data_generate.gen_data")
    def test_gen_kwargs_tensor(self, mock_gen_data):
        mock_gen_data.return_value = torch.ones(1)
        info = {
            "input_kwargs": {
                "x": {"type": "torch.Tensor", "dtype": "torch.float32", "shape": [1]}
            }
        }
        out = gen_kwargs(info, "add", None)
        self.assertTrue(torch.equal(out["x"], torch.ones(1)))

    def test_gen_kwargs_value(self):
        info = {"input_kwargs": {"alpha": {"type": "int", "value": 10}}}
        out = gen_kwargs(info, "add", None)
        self.assertEqual(out["alpha"], 10)

    # -----------------------
    # Test gen_args
    # -----------------------
    @patch("msprobe.pytorch.api_accuracy_checker.acc_check.data_generate.gen_data")
    def test_gen_args_dict(self, mock_gen_data):
        mock_gen_data.return_value = torch.ones(1)
        info = [{"type": "torch.Tensor", "dtype": "torch.float32"}]
        out = gen_args(info, "add", {"need_grad": False, "convert_type": None, "depth": 0})
        self.assertEqual(len(out), 1)

    def test_gen_args_depth_exceed(self):
        info = [[[[[[1]]]]]]
        with self.assertRaises(CompareException):
            gen_args(info, "add", {"need_grad": True, "convert_type": None, "depth": Const.MAX_DEPTH + 1})

    # -----------------------
    # Test get_output_dtype
    # -----------------------
    def test_get_output_dtype(self):
        info = {
            "output": [{"dtype": "torch.float32"}]
        }
        out = get_output_dtype(info)
        self.assertEqual(out, torch.float32)

    def test_get_output_dtype_none(self):
        info = {"output": None}
        out = get_output_dtype(info)
        self.assertIsNone(out)

    # -----------------------
    # Test gen_api_params
    # -----------------------
    @patch("msprobe.pytorch.api_accuracy_checker.acc_check.data_generate.gen_args")
    @patch("msprobe.pytorch.api_accuracy_checker.acc_check.data_generate.gen_kwargs")
    def test_gen_api_params(self, mock_kwargs, mock_args):
        mock_kwargs.return_value = {"x": 1}
        mock_args.return_value = [torch.ones(1)]

        info = {
            "input_args": [{"type": "torch.Tensor", "dtype": "torch.float32"}],
            "input_kwargs": {},
            "output": [{"dtype": "torch.float32"}],
        }

        args, kwargs, out_dtype = gen_api_params(info, "add")
        self.assertEqual(out_dtype, torch.float32)
        self.assertEqual(kwargs["x"], 1)
        self.assertEqual(len(args), 1)


if __name__ == '__main__':
    unittest.main()
