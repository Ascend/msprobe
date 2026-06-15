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

import os
import unittest
import zlib
from unittest.mock import patch, MagicMock

import numpy as np
import torch
from torch import distributed as dist
from torch._subclasses import FakeTensorMode

from msprobe.core.common.const import Const
from msprobe.core.common.exceptions import MsprobeException
from msprobe.core.common.log import logger
from msprobe.core.dump.data_dump.data_processor.pytorch_processor import (
    PytorchDataProcessor,
    TensorDataProcessor,
    TensorStatInfo,
    KernelDumpDataProcessor,
    DiffCheckDataProcessor,
    NanCheckDataProcessor,
)
from msprobe.core.dump.data_dump.data_processor.base import (
    ModuleBackwardInputsOutputs,
    ModuleForwardInputsOutputs,
)


class TestPytorchDataProcessor(unittest.TestCase):
    def setUp(self):
        self.config = MagicMock()
        self.data_writer = MagicMock()
        self.processor = PytorchDataProcessor(self.config, self.data_writer)

    def test_tensor_bytes_view_cpu_and_crc32(self):
        # 构造一个 CPU contiguous 的 tensor
        t = torch.arange(12, dtype=torch.int32).reshape(3, 4)
        t = t.contiguous()

        mv = PytorchDataProcessor.tensor_bytes_view_cpu(t)
        # 可能返回 memoryview / bytes / numpy.ndarray 都要兼容处理
        if isinstance(mv, np.ndarray):
            expected_crc = zlib.crc32(mv.tobytes())
        else:
            expected_crc = zlib.crc32(mv)

        crc_hex = PytorchDataProcessor.compute_crc32_from_tensor(t)
        self.assertEqual(crc_hex, f"{expected_crc:08x}")

    def test_tensor_bytes_view_cpu_empty_tensor(self):
        t = torch.tensor([], dtype=torch.float32)
        mv = PytorchDataProcessor.tensor_bytes_view_cpu(t)
        # 空 tensor 应该返回空的内存视图/bytes
        if isinstance(mv, np.ndarray):
            self.assertEqual(mv.size, 0)
        else:
            self.assertEqual(len(mv), 0)

    def test_dump_async_data(self):
        # 准备缓存两条 tensor
        t1 = torch.tensor([1.0])
        t2 = torch.tensor([2.0])
        self.processor._async_dump_cache = {
            "path1": t1,
            "path2": t2,
        }

        with patch.object(self.processor.tensor_handler, "save_tensor") as mock_save:
            self.processor.dump_async_data()

        self.assertEqual(mock_save.call_count, 2)
        self.assertEqual(self.processor._async_dump_cache, {})

    def test_get_md5_for_tensor(self):
        tensor = torch.tensor([1, 2, 3])
        expected_hash = zlib.crc32(tensor.numpy().tobytes())
        self.assertEqual(
            self.processor.get_md5_for_tensor(tensor), f"{expected_hash:08x}"
        )

    def test_get_md5_for_tensor_bfloat16(self):
        tensor_bfloat16 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
        expected_hash = zlib.crc32(
            tensor_bfloat16.float().cpu().detach().numpy().tobytes()
        )
        result_hash = self.processor.get_md5_for_tensor(tensor_bfloat16)
        self.assertEqual(result_hash, f"{expected_hash:08x}")

    def test_analyze_device_in_kwargs(self):
        device = torch.device("cuda:0")
        result = self.processor.analyze_device_in_kwargs(device)
        expected = {"type": "torch.device", "value": "cuda:0"}
        self.assertEqual(result, expected)

    def test_analyze_dtype_in_kwargs(self):
        dtype = torch.float32
        result = self.processor.analyze_dtype_in_kwargs(dtype)
        expected = {"type": "torch.dtype", "value": "torch.float32"}
        self.assertEqual(result, expected)

    @staticmethod
    def mock_tensor(is_meta):
        tensor = MagicMock()
        tensor.is_meta = is_meta
        return tensor

    def test_get_stat_info_with_meta_tensor(self):
        mock_data = self.mock_tensor(is_meta=True)
        result = self.processor.get_stat_info(mock_data)
        self.assertIsInstance(result, TensorStatInfo)

    def test_get_stat_info_with_fake_tensor(self):
        with FakeTensorMode() as fake_tensor_mode:
            fake_tensor = fake_tensor_mode.from_tensor(torch.randn(1, 2, 3))
        result = self.processor.get_stat_info(fake_tensor)
        self.assertIsNone(result.max)
        self.assertIsNone(result.min)
        self.assertIsNone(result.mean)
        self.assertIsNone(result.norm)

    def test_get_stat_info_float(self):
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = self.processor.get_stat_info(tensor)
        self.assertEqual(result.max, 3.0)
        self.assertEqual(result.min, 1.0)
        self.assertEqual(result.mean, 2.0)
        self.assertEqual(result.norm, torch.norm(tensor).item())

    def test_get_stat_info_int(self):
        tensor = torch.tensor([1, 2, 3], dtype=torch.int32)
        result = self.processor.get_stat_info(tensor)

        self.assertEqual(result.max, 3)
        self.assertEqual(result.min, 1)
        self.assertIsNone(result.mean)
        self.assertIsNone(result.norm)

    def test_get_stat_info_empty(self):
        tensor = torch.tensor([])
        result = self.processor.get_stat_info(tensor)
        self.assertIsNone(result.max)
        self.assertIsNone(result.min)
        self.assertIsNone(result.mean)
        self.assertIsNone(result.norm)

    def test_get_stat_info_bool(self):
        tensor = torch.tensor([True, False, True])
        result = self.processor.get_stat_info(tensor)
        self.assertEqual(result.max, True)
        self.assertEqual(result.min, False)
        self.assertIsNone(result.mean)
        self.assertIsNone(result.norm)

    def test_get_stat_info_with_scalar_tensor(self):
        scalar_tensor = torch.tensor(42.0)
        result = self.processor.get_stat_info(scalar_tensor)
        self.assertIsInstance(result, TensorStatInfo)
        self.assertEqual(result.max, 42.0)
        self.assertEqual(result.min, 42.0)
        self.assertEqual(result.mean, 42.0)
        self.assertEqual(result.norm, 42.0)

    def test_get_stat_info_with_complex_tensor(self):
        complex_tensor = torch.tensor([1 + 2j, 3 + 4j], dtype=torch.complex64)
        result = self.processor.get_stat_info(complex_tensor)
        expected_max = np.abs(np.array([1 + 2j, 3 + 4j])).max().item()
        expected_min = np.abs(np.array([1 + 2j, 3 + 4j])).min().item()
        expected_mean = np.abs(np.array([1 + 2j, 3 + 4j])).mean().item()
        self.assertIsInstance(result, TensorStatInfo)
        self.assertAlmostEqual(result.max, expected_max, places=6)
        self.assertAlmostEqual(result.min, expected_min, places=6)
        self.assertAlmostEqual(result.mean, expected_mean, places=6)

    def test_analyze_builtin(self):
        result = self.processor._analyze_builtin(
            slice(1, torch.tensor(10, dtype=torch.int32), np.int64(2))
        )
        expected = {"type": "slice", "value": [1, 10, 2]}
        self.assertEqual(result, expected)

        result = self.processor._analyze_builtin(
            slice(torch.tensor([1, 2], dtype=torch.int32), None, None)
        )
        expected = {"type": "slice", "value": [None, None, None]}
        self.assertEqual(result, expected)

    def test_process_group_hash(self):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12345"
        if dist.is_initialized():
            dist.destroy_process_group()
        dist.init_process_group(backend="gloo", world_size=1, rank=0)
        process_group_element = dist.group.WORLD
        result = self.processor.process_group_hash(process_group_element)
        expected = f"{zlib.crc32(str([0]).encode('utf-8')):08x}"
        self.assertEqual(result, expected)
        dist.destroy_process_group()

    def test_analyze_torch_size(self):
        size = torch.Size([3, 4, 5])
        result = self.processor._analyze_torch_size(size)
        expected = {"type": "torch.Size", "value": [3, 4, 5]}
        self.assertEqual(result, expected)

    def test_analyze_memory_format(self):
        memory_format_element = torch.contiguous_format
        result = self.processor._analyze_memory_format(memory_format_element)
        expected = {"type": "torch.memory_format", "format": "contiguous_format"}
        self.assertEqual(result, expected)

    def test_analyze_process_group(self):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12345"
        if dist.is_initialized():
            dist.destroy_process_group()
        dist.init_process_group(backend="gloo", world_size=1, rank=0)
        process_group_element = dist.group.WORLD
        result = self.processor._analyze_process_group(process_group_element)
        expected = {
            "type": "torch.ProcessGroup",
            "group_ranks": [0],
            "group_id": f"{zlib.crc32(str([0]).encode('utf-8')):08x}",
        }
        self.assertEqual(result, expected)
        dist.destroy_process_group()

    def test_analyze_reduce_op_successful(self):
        arg = dist.ReduceOp.SUM
        result = self.processor._analyze_reduce_op(arg)
        expected = {"type": "torch.distributed.ReduceOp", "value": "RedOpType.SUM"}
        self.assertEqual(result, expected)

    @patch.object(logger, "warning")
    def test_analyze_reduce_op_failed(self, mock_logger_warning):
        class TestReduceOp:
            def __str__(self):
                raise Exception("failed to convert str type")

        arg = TestReduceOp()
        self.processor._analyze_reduce_op(arg)
        mock_logger_warning.assert_called_with(
            "Failed to get value of torch.distributed.ReduceOp with error info: failed to convert str type."
        )

    def test_get_special_types(self):
        special_types = self.processor.get_special_types()
        self.assertIn(torch.Tensor, special_types)

    def test_analyze_single_element_torch_size(self):
        size_element = torch.Size([2, 3])
        result = self.processor.analyze_single_element(size_element, [])
        self.assertEqual(result, self.processor._analyze_torch_size(size_element))

    def test_analyze_single_element_memory_size(self):
        memory_format_element = torch.contiguous_format
        result = self.processor.analyze_single_element(memory_format_element, [])
        self.assertEqual(
            result, self.processor._analyze_memory_format(memory_format_element)
        )

    def test_analyze_single_element_process_group(self):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12345"
        if dist.is_initialized():
            dist.destroy_process_group()
        dist.init_process_group(backend="gloo", world_size=1, rank=0)
        process_group_element = dist.group.WORLD
        result = self.processor.analyze_single_element(process_group_element, [])
        self.assertEqual(
            result, self.processor._analyze_process_group(process_group_element)
        )
        dist.destroy_process_group()

    def test_analyze_single_element_numpy_conversion(self):
        numpy_element = np.int32(5)
        result = self.processor.analyze_single_element(numpy_element, [])
        expected = {"type": "int32", "value": 5}
        self.assertEqual(result, expected)

        numpy_element = np.float32(3.14)
        result = self.processor.analyze_single_element(numpy_element, [])
        expected = {"type": "float32", "value": 3.140000104904175}
        self.assertEqual(result, expected)

        numpy_element = np.bool_(True)
        result = self.processor.analyze_single_element(numpy_element, [])
        expected = {"type": "bool_", "value": True}
        self.assertEqual(result, expected)

        numpy_element = np.str_("abc")
        result = self.processor.analyze_single_element(numpy_element, [])
        expected = {"type": "str_", "value": "abc"}
        self.assertEqual(result, expected)

        numpy_element = np.byte(1)
        result = self.processor.analyze_single_element(numpy_element, [])
        expected = {"type": "int8", "value": 1}
        self.assertEqual(result, expected)

        numpy_element = np.complex128(1 + 2j)
        result = self.processor.analyze_single_element(numpy_element, [])
        expected = {"type": "complex128", "value": (1 + 2j)}
        self.assertEqual(result, expected)

    def test_analyze_single_element_tensor(self):
        tensor_element = torch.tensor([1, 2, 3])
        result = self.processor.analyze_single_element(tensor_element, ["tensor"])
        expected_result = self.processor._analyze_tensor(tensor_element, "tensor")
        self.assertEqual(result, expected_result, f"{result} {expected_result}")

    def test_analyze_single_element_bool(self):
        bool_element = True
        result = self.processor.analyze_single_element(bool_element, [])
        expected_result = self.processor._analyze_builtin(bool_element)
        self.assertEqual(result, expected_result)

    def test_analyze_single_element_builtin_ellipsis(self):
        result = self.processor.analyze_single_element(Ellipsis, [])
        expected_result = self.processor._analyze_builtin(Ellipsis)
        self.assertEqual(result, expected_result)

    @patch.object(PytorchDataProcessor, "get_md5_for_tensor")
    def test_analyze_tensor(self, get_md5_for_tensor):
        get_md5_for_tensor.return_value = "mocked_md5"
        tensor = torch.tensor([1.0, 2.0, 3.0])
        self.config.summary_mode = "md5"
        self.config.async_dump = False
        result = self.processor._analyze_tensor(tensor, "suffix")
        expected = {
            "type": "torch.Tensor",
            "dtype": str(tensor.dtype),
            "shape": tensor.shape,
            "requires_grad": tensor.requires_grad,
        }
        result.pop("tensor_stat_index", None)
        result.pop("md5_index", None)
        self.assertDictEqual(expected, result)

    def test_analyze_tensor_with_empty_tensor(self):
        tensor = torch.tensor([])
        result = self.processor._analyze_tensor(tensor, "suffix")

        self.assertEqual(result["type"], "torch.Tensor")
        self.assertEqual(result["dtype"], "torch.float32")
        self.assertEqual(result["shape"], torch.Size([0]))
        self.assertEqual(result["requires_grad"], False)


class TestTensorDataProcessor(unittest.TestCase):
    def setUp(self):
        self.config = MagicMock()
        self.data_writer = MagicMock()
        self.processor = TensorDataProcessor(self.config, self.data_writer)
        self.data_writer.dump_tensor_data_dir = "./dump_data"
        self.processor.current_api_or_module_name = "test_api"
        self.processor.api_data_category = "input"

    @patch("torch.save")
    def test_analyze_tensor(self, mock_save):
        self.config.framework = "pytorch"
        self.config.async_dump = False
        tensor = torch.tensor([1.0, 2.0, 3.0])
        suffix = "suffix"
        result = self.processor._analyze_tensor(tensor, suffix)
        mock_save.assert_called_once()
        expected = {
            "type": "torch.Tensor",
            "dtype": "torch.float32",
            "shape": tensor.shape,
            "requires_grad": False,
            "data_name": "test_api.input.suffix.pt",
        }
        result.pop("tensor_stat_index", None)
        self.assertEqual(expected, result)


class TestKernelDumpDataProcessor(unittest.TestCase):
    def setUp(self):
        self.config = MagicMock()
        self.data_writer = MagicMock()
        self.processor = KernelDumpDataProcessor(self.config, self.data_writer)

    @patch.object(logger, "warning")
    def test_print_unsupported_log(self, mock_logger_warning):
        self.processor._print_unsupported_log("test_api_name")
        mock_logger_warning.assert_called_with(
            "The kernel dump does not support the test_api_name API."
        )

    @patch("msprobe.core.dump.data_dump.data_processor.pytorch_processor.is_gpu")
    @patch.object(logger, "warning")
    def test_analyze_pre_forward_with_gpu(self, mock_logger_warning, mock_is_gpu):
        mock_is_gpu.return_value = True
        self.processor.analyze_forward_input("test_api_name", None, None)
        mock_logger_warning.assert_called_with(
            "The current environment is not a complete NPU environment, and kernel dump cannot be used."
        )
        self.assertFalse(self.processor.enable_kernel_dump)

    @patch(
        "msprobe.core.dump.data_dump.data_processor.pytorch_processor.is_gpu", new=False
    )
    @patch(
        "msprobe.core.dump.data_dump.data_processor.pytorch_processor.KernelDumpDataProcessor.analyze_element"
    )
    @patch.object(logger, "warning")
    def test_analyze_pre_forward_with_not_gpu(
        self, mock_logger_warning, mock_analyze_element
    ):
        self.config.is_backward_kernel_dump = True
        mock_module = MagicMock()
        mock_module_input_output = MagicMock()
        self.processor.analyze_forward_input(
            "test_api_name", mock_module, mock_module_input_output
        )
        mock_module.forward.assert_called_once()
        mock_analyze_element.assert_called()
        mock_logger_warning.assert_called_with(
            "The kernel dump does not support the test_api_name API."
        )
        self.assertFalse(self.processor.enable_kernel_dump)

    @patch(
        "msprobe.core.dump.data_dump.data_processor.pytorch_processor.KernelDumpDataProcessor.stop_kernel_dump"
    )
    @patch.object(logger, "info")
    def test_analyze_forward_successfully(
        self, mock_logger_info, mock_stop_kernel_dump
    ):
        self.processor.enable_kernel_dump = True
        self.processor.config.is_backward_kernel_dump = False
        self.processor.analyze_forward_output("test_api_name", None, None)
        self.assertFalse(self.processor.enable_kernel_dump)
        mock_stop_kernel_dump.assert_called_once()
        mock_logger_info.assert_called_with(
            "The kernel data of test_api_name is dumped successfully."
        )

    @patch(
        "msprobe.core.dump.data_dump.data_processor.pytorch_processor.KernelDumpDataProcessor.analyze_element"
    )
    @patch.object(logger, "warning")
    def test_analyze_backward_unsuccessfully(
        self, mock_logger_warning, mock_analyze_element
    ):
        self.processor.enable_kernel_dump = True
        self.processor.is_found_grad_input_tensor = False
        mock_module_input_output = MagicMock()
        self.processor.analyze_backward("test_api_name", None, mock_module_input_output)
        mock_analyze_element.assert_called_once()
        mock_logger_warning.assert_called_with(
            "The kernel dump does not support the test_api_name API."
        )
        self.assertFalse(self.processor.enable_kernel_dump)

    @patch(
        "msprobe.core.dump.data_dump.data_processor.pytorch_processor.KernelDumpDataProcessor.stop_kernel_dump"
    )
    @patch(
        "msprobe.core.dump.data_dump.data_processor.pytorch_processor.KernelDumpDataProcessor.start_kernel_dump"
    )
    @patch(
        "msprobe.core.dump.data_dump.data_processor.pytorch_processor.KernelDumpDataProcessor.analyze_element"
    )
    @patch.object(logger, "info")
    def test_analyze_backward_successfully(
        self, mock_logger_info, mock_analyze_element, mock_start, mock_stop
    ):
        self.processor.enable_kernel_dump = True
        self.processor.is_found_grad_input_tensor = True
        self.processor.forward_output_tensor = MagicMock()
        mock_module_input_output = MagicMock()
        self.processor.analyze_backward("test_api_name", None, mock_module_input_output)
        mock_analyze_element.assert_called_once()
        self.assertFalse(self.processor.enable_kernel_dump)
        self.processor.forward_output_tensor.backward.assert_called_once()
        mock_start.assert_called_once()
        mock_stop.assert_called_once()
        mock_logger_info.assert_called_with(
            "The kernel data of test_api_name is dumped successfully."
        )

    def test_clone_tensor(self):
        tensor = torch.tensor([1.0, 2.0, 3.0])
        clone_tensor = self.processor.clone_and_detach_tensor(tensor)
        self.assertTrue(torch.equal(tensor, clone_tensor))
        self.assertFalse(clone_tensor.requires_grad)

        tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        clone_tensor = self.processor.clone_and_detach_tensor(tensor)
        self.assertTrue(torch.equal(tensor, clone_tensor))
        self.assertTrue(clone_tensor.requires_grad)

        tensor1 = torch.tensor([1.0], requires_grad=True)
        tensor2 = torch.tensor([1.0])
        input_tuple = (tensor1, tensor2)
        clone_tuple = self.processor.clone_and_detach_tensor(input_tuple)
        self.assertEqual(len(input_tuple), len(clone_tuple))
        self.assertTrue(clone_tuple[0].requires_grad)
        self.assertFalse(clone_tuple[1].requires_grad)

        input_list = [tensor1, tensor2]
        clone_list = self.processor.clone_and_detach_tensor(input_list)
        self.assertEqual(len(input_list), len(clone_list))
        self.assertTrue(clone_tuple[0].requires_grad)
        self.assertFalse(clone_tuple[1].requires_grad)

        input_dict = {"tensor1": tensor1, "tensor2": tensor2}
        clone_dict = self.processor.clone_and_detach_tensor(input_dict)
        self.assertEqual(len(clone_dict), len(input_dict))
        self.assertTrue(clone_dict["tensor1"].requires_grad)
        self.assertFalse(clone_dict["tensor2"].requires_grad)

        non_tensor_input = 1
        result = self.processor.clone_and_detach_tensor(non_tensor_input)
        self.assertEqual(result, non_tensor_input)

    def test_analyze_single_element_with_output_grad(self):
        self.processor.is_found_output_tensor = False
        tensor = torch.tensor([1.0], requires_grad=True)
        self.processor.analyze_single_element(tensor, None)
        self.assertTrue(self.processor.is_found_output_tensor)

    def test_analyze_single_element_without_output_grad(self):
        self.processor.is_found_output_tensor = False
        tensor = torch.tensor([1.0])
        self.processor.analyze_single_element(tensor, None)
        self.assertFalse(self.processor.is_found_output_tensor)

    def test_analyze_single_element_with_grad_input(self):
        self.processor.is_found_output_tensor = True
        self.processor.is_found_grad_input_tensor = False
        tensor = torch.tensor([1.0])
        self.processor.analyze_single_element(tensor, None)
        self.assertTrue(self.processor.is_found_grad_input_tensor)

    def test_analyze_single_element_without_grad_input(self):
        self.processor.is_found_output_tensor = True
        self.processor.is_found_grad_input_tensor = True
        tensor = torch.tensor([1.0])
        self.processor.analyze_single_element(tensor, None)
        self.assertTrue(self.processor.is_found_grad_input_tensor)

    def test_reset_status(self):
        self.processor.enable_kernel_dump = False
        self.processor.is_found_output_tensor = True
        self.processor.is_found_grad_input_tensor = True
        self.processor.forward_args = 0
        self.processor.forward_kwargs = 1
        self.processor.forward_output_tensor = 2
        self.processor.grad_input_tensor = 3

        self.processor.reset_status()

        self.assertTrue(self.processor.enable_kernel_dump)
        self.assertFalse(self.processor.is_found_output_tensor)
        self.assertFalse(self.processor.is_found_grad_input_tensor)
        self.assertIsNone(self.processor.forward_args)
        self.assertIsNone(self.processor.forward_kwargs)
        self.assertIsNone(self.processor.forward_output_tensor)
        self.assertIsNone(self.processor.grad_input_tensor)


class TestDiffCheckDataProcessor(unittest.TestCase):
    def setUp(self):
        self.config = MagicMock()
        # diff_nums 用于 is_terminated / handle_diff 分支
        self.config.diff_nums = 2
        self.config.precision = MagicMock()
        self.config.async_dump = False
        self.config.summary_mode = MagicMock()
        self.config.task = MagicMock()
        self.data_writer = MagicMock()
        self.processor = DiffCheckDataProcessor(self.config, self.data_writer)

    def test_is_terminated_property(self):
        # diff_nums = -1 时永不终止
        self.processor.diff_nums = -1
        self.processor.real_diff_nums = 100
        self.assertFalse(self.processor.is_terminated)

        # 正常计数
        self.processor.diff_nums = 2
        self.processor.real_diff_nums = 1
        self.assertFalse(self.processor.is_terminated)
        self.processor.real_diff_nums = 2
        self.assertTrue(self.processor.is_terminated)

    def test_parse_data_name(self):
        # 带 name: 前缀
        name = "name:Functional.relu.2.forward.input.0.pt"
        parsed = DiffCheckDataProcessor._parse_data_name(name)
        self.assertEqual(parsed, ("Functional.relu.2.forward", "input", 0))

        # 不带前缀
        name2 = "MyApi.output.3.pt"
        parsed2 = DiffCheckDataProcessor._parse_data_name(name2)
        self.assertEqual(parsed2, ("MyApi", "output", 3))

        # 非法格式
        self.assertIsNone(DiffCheckDataProcessor._parse_data_name("invalid_name"))

    def test_build_bench_map_from_json_and_expected_counts(self):
        data = {
            "MyApi.forward": {
                "input_args": [
                    {"type": "torch.Tensor", "md5": "aaa", "shape": [1, 2]},
                    {"type": "int", "value": 1},  # 非 tensor，应该被忽略
                ],
                "output": [
                    {"type": "torch.Tensor", "md5": "bbb", "shape": [3, 4]},
                    {"type": "str", "value": "xxx"},  # 非 tensor
                ],
            }
        }

        mp = self.processor._build_bench_map_from_json(data)
        self.processor._bench_map = mp

        # input / output 键存在
        self.assertIn(("MyApi.forward", "input", 0), mp)
        self.assertIn(("MyApi.forward", "output", 0), mp)
        # 非 tensor 的条目不会出现在 map 里
        self.assertNotIn(("MyApi.forward", "input", 1), mp)
        self.assertNotIn(("MyApi.forward", "output", 1), mp)

        n_in, n_out = self.processor._bench_expected_counts_for_api("MyApi.forward")
        self.assertEqual(n_in, 1)
        self.assertEqual(n_out, 1)

    def test_ensure_bench_map_loaded_success(self):
        # 模拟 dump.json 存在且能够被加载
        with (
            patch.object(
                self.processor,
                "_resolve_bench_json_path",
                return_value="/fake/dump.json",
            ),
            patch("os.path.getmtime", return_value=123456),
            patch(
                "msprobe.core.dump.data_dump.data_processor.pytorch_processor.load_json",
                return_value={"data": {"Api": {"input_args": [], "output": []}}},
            ) as mock_load,
            patch.object(
                self.processor,
                "_build_bench_map_from_json",
                return_value={"dummy": {"md5": "xx", "shape": [1]}},
            ) as mock_build,
        ):
            ok = self.processor._ensure_bench_map_loaded()

        self.assertTrue(ok)
        mock_load.assert_called_once()
        mock_build.assert_called_once()
        self.assertEqual(self.processor._bench_ref_path, "/fake/dump.json")
        self.assertEqual(self.processor._bench_ref_mtime, 123456)
        self.assertIn("dummy", self.processor._bench_map)

    def test_ensure_bench_map_loaded_path_none(self):
        # 无 bench 路径时直接 False
        with patch.object(
            self.processor, "_resolve_bench_json_path", return_value=None
        ):
            ok = self.processor._ensure_bench_map_loaded()
        self.assertFalse(ok)

    def test_analyze_maybe_diff_tensor_input_not_in_ref(self):
        # bench map 已加载，但没有对应的 (api, io, idx)
        with patch.object(
            self.processor, "_ensure_bench_map_loaded", return_value=True
        ):
            self.processor._bench_map = {}
            tensor_json = {
                "data_name": "MyApi.forward.input.0.pt",
                "shape": [1, 2],
                "md5": "0011",
            }
            self.processor._analyze_maybe_diff_tensor(tensor_json)

        st = self.processor._bench_state["MyApi.forward"]
        self.assertFalse(st["inputs_equal"])
        self.assertTrue(st["seen_input_not_in_ref"])
        self.assertFalse(self.processor.has_diff)

    def test_analyze_maybe_diff_tensor_input_in_ref_equal(self):
        # bench 有 input，shape/md5 一致
        self.processor._bench_map = {
            ("MyApi.forward", "input", 0): {"md5": "abcd", "shape": [1, 2]},
        }
        with patch.object(
            self.processor, "_ensure_bench_map_loaded", return_value=True
        ):
            tensor_json = {
                "data_name": "MyApi.forward.input.0.pt",
                "shape": [1, 2],
                "md5": "abcd",
            }
            self.processor._analyze_maybe_diff_tensor(tensor_json)

        st = self.processor._bench_state["MyApi.forward"]
        self.assertTrue(st["inputs_equal"])
        self.assertEqual(st["checked_in"], 1)
        self.assertEqual(st["expected_in"], 1)
        self.assertFalse(st["seen_input_not_in_ref"])

    def test_analyze_maybe_diff_tensor_output_diff_when_inputs_ok(self):
        # 手动构造 bench_map 和 bench_state，模拟：输入已经全部一致，输出 md5 不一致 -> has_diff = True
        api = "MyApi.forward"
        self.processor._bench_map = {
            (api, "input", 0): {"md5": "in_md5", "shape": [2, 2]},
            (api, "output", 0): {"md5": "out_ref", "shape": [2, 2]},
        }
        self.processor._bench_state[api] = {
            "expected_in": 1,
            "checked_in": 1,
            "inputs_equal": True,
            "seen_input_not_in_ref": False,
            "any_output_neq": False,
        }

        with patch.object(
            self.processor, "_ensure_bench_map_loaded", return_value=True
        ):
            tensor_json = {
                "data_name": f"{api}.output.0.pt",
                "shape": [2, 2],
                "md5": "out_cur_not_equal",
            }
            self.processor.has_diff = False
            self.processor._analyze_maybe_diff_tensor(tensor_json)

        st = self.processor._bench_state[api]
        self.assertTrue(st["any_output_neq"])
        self.assertTrue(self.processor.has_diff)

    def test_handle_diff_and_clear_bench_state(self):
        # 准备 has_diff = True，且有缓存的 tensor
        fake_tensor = torch.tensor([1.0])
        self.processor.cached_tensors_and_file_paths = {
            "/tmp/a.pt": fake_tensor,
        }
        self.processor.has_diff = True
        self.processor.real_diff_nums = 0
        self.processor.diff_nums = 2
        self.processor.current_api_or_module_name = "TestApi"
        self.processor._bench_state["TestApi"] = {"dummy": 1}

        with patch.object(self.processor.tensor_handler, "save_tensor") as mock_save:
            self.processor.handle_diff()

        # 有 diff 时会调用 save_tensor，并且 real_diff_nums +1，bench_state 对应 api 被清掉
        mock_save.assert_called_once()
        self.assertEqual(self.processor.real_diff_nums, 1)
        self.assertNotIn("TestApi", self.processor._bench_state)
        self.assertEqual(self.processor.cached_tensors_and_file_paths, {})


class TestNanCheckDataProcessor(unittest.TestCase):
    """NanCheckDataProcessor 单元测试"""

    def setUp(self):
        self.config = MagicMock()
        self.config.tensor_list = None
        self.data_writer = MagicMock()
        self.data_writer.cache_data = []
        self.data_writer.cache_debug = None
        self.data_writer.data_updated = False
        self.data_writer.register_pre_flush_callback = MagicMock()
        self.data_writer.write_json = MagicMock()
        self.data_writer._replace_nan_placeholders = MagicMock()
        self.data_writer.append_crc32_to_buffer = MagicMock()

        self._my_ns_patcher = patch("torch.ops.my_ns", create=True)
        self._my_ns_patcher.start()
        self._npu_patcher = patch("torch.npu", create=True)
        self._npu_patcher.start()

        # 部分流水线 torch 版本不支持 torch.uint64，为测试兼容性补上
        if not hasattr(torch, "uint64"):
            self._uint64_patcher = patch.object(torch, "uint64", torch.int64, create=True)
            self._uint64_patcher.start()

    def tearDown(self):
        self._my_ns_patcher.stop()
        self._npu_patcher.stop()
        if hasattr(self, "_uint64_patcher"):
            self._uint64_patcher.stop()

    def _create_processor(self):
        with (
            patch.object(
                NanCheckDataProcessor, "_nan_overflow_ops_available", return_value=True
            ),
            patch("torch.npu.current_device", return_value=0),
        ):
            return NanCheckDataProcessor(self.config, self.data_writer)

    # ---- _nan_overflow_ops_available ----

    def test_ops_available_returns_false_when_gpu(self):
        with patch(
            "msprobe.core.dump.data_dump.data_processor.pytorch_processor.is_gpu", True
        ):
            self.assertFalse(NanCheckDataProcessor._nan_overflow_ops_available())

    def test_ops_available_returns_true_when_import_ok(self):
        with (
            patch(
                "msprobe.core.dump.data_dump.data_processor.pytorch_processor.is_gpu",
                False,
            ),
            patch("importlib.import_module") as mock_import,
        ):
            mock_import.return_value = MagicMock()
            self.assertTrue(NanCheckDataProcessor._nan_overflow_ops_available())
            mock_import.assert_called_once_with("msprobe.pytorch.nan_check")

    def test_ops_available_returns_false_when_import_fails(self):
        with (
            patch(
                "msprobe.core.dump.data_dump.data_processor.pytorch_processor.is_gpu",
                False,
            ),
            patch("importlib.import_module", side_effect=ImportError()),
        ):
            self.assertFalse(NanCheckDataProcessor._nan_overflow_ops_available())

    # ---- __init__ ----

    def test_init_raises_when_ops_unavailable(self):
        with patch.object(
            NanCheckDataProcessor, "_nan_overflow_ops_available", return_value=False
        ):
            with self.assertRaises(MsprobeException) as ctx:
                NanCheckDataProcessor(self.config, self.data_writer)
            self.assertIn("Nan check requires", str(ctx.exception))

    def test_init_sets_default_state(self):
        processor = self._create_processor()
        self.assertIsNone(processor._nan_buffer)
        self.assertEqual(processor._nan_buffer_size, Const.NAN_CHECK_BUFFER_SIZE)
        self.assertEqual(processor._nan_buffer_offset, 0)
        self.assertFalse(processor._nan_overflow_runtime_warned)
        self.assertFalse(processor._nan_collect_runtime_warned)
        self.assertFalse(processor._nan_buffer_full_warned)

    def test_init_registers_flush_callback(self):
        processor = self._create_processor()
        self.data_writer.register_pre_flush_callback.assert_called_once_with(
            processor._flush_nan_buffer
        )

    # ---- prepare_nan_buffer / _ensure_nan_buffer ----

    def test_ensure_nan_buffer_creates_on_first_call(self):
        proc = self._create_processor()
        self.assertIsNone(proc._nan_buffer)
        with patch.object(torch, "empty", return_value=MagicMock()):
            with patch("torch.npu.current_device", return_value=0):
                proc._ensure_nan_buffer()
        self.assertIsNotNone(proc._nan_buffer)

    def test_ensure_nan_buffer_is_idempotent(self):
        proc = self._create_processor()
        proc._nan_buffer = MagicMock()
        proc._ensure_nan_buffer()
        self.assertIsNotNone(proc._nan_buffer)

    def test_prepare_nan_buffer_calls_ensure(self):
        proc = self._create_processor()
        with patch.object(proc, "_ensure_nan_buffer") as mock_ensure:
            proc.prepare_nan_buffer()
            mock_ensure.assert_called_once()

    # ---- _collect_tensors (static) ----

    def test_collect_tensors_single(self):
        tlist, t = [], torch.randn(3, 3)
        NanCheckDataProcessor._collect_tensors(t, tlist)
        self.assertEqual(len(tlist), 1)
        self.assertIs(tlist[0], t)

    def test_collect_tensors_from_list_tuple_dict(self):
        t1, t2 = torch.randn(2), torch.randn(3)
        for container in [[t1, t2], (t1, t2), {"a": t1, "b": t2}]:
            tlist = []
            NanCheckDataProcessor._collect_tensors(container, tlist)
            self.assertEqual(len(tlist), 2)

    def test_collect_tensors_nested(self):
        t1, t2, t3 = torch.randn(1), torch.randn(2), torch.randn(3)
        tlist = []
        NanCheckDataProcessor._collect_tensors(
            {"a": [t1, (t2,)], "b": {"c": t3}}, tlist
        )
        self.assertEqual(len(tlist), 3)

    def test_collect_tensors_skips_non_tensor(self):
        tlist = []
        for elem in [42, "hello", None, 3.14, [], (), {}]:
            NanCheckDataProcessor._collect_tensors(elem, tlist)
        self.assertEqual(len(tlist), 0)

    # ---- _should_collect_nan_tensors ----

    def test_should_collect_false_when_no_tensor_list(self):
        proc = self._create_processor()
        proc.current_api_or_module_name = "conv2d"
        self.assertFalse(proc._should_collect_nan_tensors())
        self.config.tensor_list = []
        proc2 = self._create_processor()
        proc2.current_api_or_module_name = "conv2d"
        self.assertFalse(proc2._should_collect_nan_tensors())

    def test_should_collect_true_only_when_api_matches(self):
        self.config.tensor_list = ["conv2d", "batch_norm"]
        proc = self._create_processor()
        proc.current_api_or_module_name = "linear"
        self.assertFalse(proc._should_collect_nan_tensors())
        proc.current_api_or_module_name = "torch.conv2d.default"
        self.assertTrue(proc._should_collect_nan_tensors())

    def test_should_collect_false_when_api_none(self):
        self.config.tensor_list = ["conv2d"]
        proc = self._create_processor()
        proc.current_api_or_module_name = None
        self.assertFalse(proc._should_collect_nan_tensors())

    # ---- _build_nan_tensor_list ----

    def test_build_tensor_list_from_forward_io(self):
        proc = self._create_processor()
        t_args, t_kwargs, t_out = torch.randn(2), torch.randn(3), torch.randn(4)
        mio = ModuleForwardInputsOutputs(
            args=(t_args,), kwargs={"w": t_kwargs}, output=t_out
        )
        result = proc._build_nan_tensor_list(mio)
        self.assertEqual(len(result), 3)
        self.assertIs(result[0], t_args)
        self.assertIs(result[1], t_kwargs)
        self.assertIs(result[2], t_out)

    def test_build_tensor_list_from_backward_io(self):
        proc = self._create_processor()
        t_gi, t_go = torch.randn(2, 3), torch.randn(4, 5)
        mio = ModuleBackwardInputsOutputs(grad_input=(t_gi,), grad_output=(t_go,))
        result = proc._build_nan_tensor_list(mio)
        self.assertEqual(len(result), 2)
        self.assertIs(result[0], t_gi)
        self.assertIs(result[1], t_go)

    def test_build_tensor_list_empty_when_no_tensors(self):
        proc = self._create_processor()
        mio = ModuleForwardInputsOutputs(args=None, kwargs=None, output=None)
        self.assertEqual(len(proc._build_nan_tensor_list(mio)), 0)

    def test_build_tensor_list_nested_args(self):
        proc = self._create_processor()
        t1, t2 = torch.randn(1), torch.randn(2)
        mio = ModuleForwardInputsOutputs(
            args=([t1, {"inner": t2}],), kwargs=None, output=torch.randn(3)
        )
        self.assertEqual(len(proc._build_nan_tensor_list(mio)), 3)

    # ---- _maybe_collect_overflow_tensors ----

    def test_maybe_collect_skips_when_should_not_collect(self):
        proc = self._create_processor()
        with patch.object(proc, "_should_collect_nan_tensors", return_value=False):
            with patch.object(proc, "_build_nan_tensor_list") as mock_build:
                proc._maybe_collect_overflow_tensors(MagicMock(), MagicMock())
                mock_build.assert_not_called()

    def test_maybe_collect_calls_npu_nan_test(self):
        proc = self._create_processor()
        slot, t1 = torch.zeros(1, dtype=torch.int64), torch.randn(2)
        mio = ModuleForwardInputsOutputs(args=None, kwargs=None, output=(t1,))
        with patch.object(proc, "_should_collect_nan_tensors", return_value=True):
            proc._maybe_collect_overflow_tensors(slot, mio)
            torch.ops.my_ns.npu_nan_test.assert_called_once()
            self.assertIs(torch.ops.my_ns.npu_nan_test.call_args[0][0], slot)
            self.assertIn(t1, torch.ops.my_ns.npu_nan_test.call_args[0][1])

    def test_maybe_collect_sets_warned_flag(self):
        proc = self._create_processor()
        with (
            patch.object(proc, "_should_collect_nan_tensors", return_value=True),
            patch.object(
                proc, "_build_nan_tensor_list", side_effect=RuntimeError("fail")
            ),
        ):
            proc._maybe_collect_overflow_tensors(MagicMock(), MagicMock())
            self.assertTrue(proc._nan_collect_runtime_warned)

    # ---- _check_overflow ----

    def _prepare_real_buffer(self, proc):
        """使用真实 CPU tensor 构造 nan buffer，模拟 _ensure_nan_buffer 创建的 2D tensor。

        注意：本测试套件通过 setUp 中的 patch("torch.ops.my_ns", create=True)
        全局 mock 了 NPU 算子（npu_over_flow / npu_nan_test），因此所有
        _check_overflow 调用实际上不会访问真实 NPU 设备。
        """
        proc._nan_buffer = torch.zeros(
            Const.NAN_CHECK_BUFFER_SIZE, 8, dtype=torch.int64
        )

    def test_check_overflow_returns_slot_index(self):
        proc = self._create_processor()
        self._prepare_real_buffer(proc)
        proc._nan_buffer_offset = 5
        idx = proc._check_overflow()
        self.assertEqual(idx, 5)
        self.assertEqual(proc._nan_buffer_offset, 6)

    def test_check_overflow_returns_none_on_exception(self):
        proc = self._create_processor()
        self._prepare_real_buffer(proc)
        torch.ops.my_ns.npu_over_flow.side_effect = RuntimeError("err")
        self.assertIsNone(proc._check_overflow())
        self.assertTrue(proc._nan_overflow_runtime_warned)

    def test_check_overflow_calls_maybe_collect_with_io(self):
        proc = self._create_processor()
        self._prepare_real_buffer(proc)
        mio = ModuleForwardInputsOutputs(args=None, kwargs=None, output=torch.randn(2))
        with patch.object(proc, "_maybe_collect_overflow_tensors") as mock_collect:
            proc._check_overflow(module_input_output=None)
            mock_collect.assert_not_called()
            proc._check_overflow(module_input_output=mio)
            mock_collect.assert_called_once()

    def test_check_overflow_triggers_flush_when_buffer_full(self):
        proc = self._create_processor()
        self._prepare_real_buffer(proc)
        proc._nan_buffer_offset = Const.NAN_CHECK_BUFFER_SIZE
        proc._nan_buffer_full_warned = False
        # 模拟真实回调行为：write_json 触发 _flush_nan_buffer 重置 offset，
        # 否则 mock 的 write_json 不会触发回调，_nan_buffer_offset 不会被重置，
        # 会导致 _check_overflow 访问 buffer[_nan_buffer_size] 越界。
        mock_write = MagicMock()

        def write_json_side_effect(*args, **kwargs):
            proc._flush_nan_buffer()

        mock_write.side_effect = write_json_side_effect
        orig_write = proc.data_writer.write_json
        proc.data_writer.write_json = mock_write
        try:
            idx = proc._check_overflow()
            # _flush_nan_buffer 将 offset 重置为 0，_check_overflow 将其递增到 1
            self.assertEqual(idx, 0)
            self.assertEqual(proc._nan_buffer_offset, 1)
            mock_write.assert_called_once()
            self.assertFalse(proc._nan_buffer_full_warned)
        finally:
            proc.data_writer.write_json = orig_write

    # ---- _flush_nan_buffer ----

    def test_flush_nan_buffer_noop_when_empty(self):
        proc = self._create_processor()
        proc._nan_buffer_offset = 0
        proc._nan_buffer = MagicMock()
        proc._flush_nan_buffer()
        self.data_writer._replace_nan_placeholders.assert_not_called()

    def test_flush_nan_buffer_flushes_and_resets(self):
        proc = self._create_processor()
        proc._nan_buffer_offset = 3
        proc._nan_buffer_full_warned = True
        # 使用真实 2D tensor 作为 buffer，与 _ensure_nan_buffer 产物一致
        proc._nan_buffer = torch.zeros(
            Const.NAN_CHECK_BUFFER_SIZE, 8, dtype=torch.int64
        )
        # 设置第一列的值以构造可验证的 tolist 输出
        proc._nan_buffer[0, 0] = 0
        proc._nan_buffer[1, 0] = 1
        proc._nan_buffer[2, 0] = 0
        self.data_writer._replace_nan_placeholders.reset_mock()
        proc._flush_nan_buffer()
        self.data_writer._replace_nan_placeholders.assert_called_once()
        # _flush_nan_buffer 调用 _replace_nan_placeholders(cache_data, cpu_vals)
        # 验证第二个参数 cpu_vals 为 [0, 1, 0]
        call_args = self.data_writer._replace_nan_placeholders.call_args[0]
        self.assertEqual(call_args[1], [0, 1, 0])
        self.assertEqual(proc._nan_buffer_offset, 0)
        self.assertFalse(proc._nan_buffer_full_warned)
        self.assertTrue(self.data_writer.data_updated)

    # ---- _analyze_tensor (覆写版，跳过 stat) ----

    def test_analyze_tensor_returns_basic_info(self):
        proc = self._create_processor()
        t = torch.randn(3, 4, requires_grad=True)
        result = proc._analyze_tensor(t, "suffix")
        self.assertIn("type", result)
        self.assertIn("dtype", result)
        self.assertEqual(result["shape"], torch.Size([3, 4]))
        self.assertTrue(result["requires_grad"])

    def test_analyze_tensor_skips_stat_computation(self):
        proc = self._create_processor()
        t = torch.randn(2, 2)
        result = proc._analyze_tensor(t, "suffix")
        for stat_key in ("max", "min", "mean", "norm"):
            self.assertNotIn(stat_key, result)

    # ---- _store_nan_result ----

    def test_store_nan_result_stores_index(self):
        proc = self._create_processor()
        api_info = {"op": {}}
        with patch.object(proc, "_check_overflow", return_value=42):
            proc._store_nan_result(api_info, "op")
            self.assertEqual(api_info["op"][Const.IS_NAN_INDEX], 42)

    def test_store_nan_result_stores_none_on_failure(self):
        proc = self._create_processor()
        api_info = {"op": {}}
        with patch.object(proc, "_check_overflow", return_value=None):
            proc._store_nan_result(api_info, "op")
            self.assertIsNone(api_info["op"][Const.IS_NAN])

    def test_store_nan_result_forwards_module_io(self):
        proc = self._create_processor()
        api_info, mio = (
            {"op": {}},
            ModuleForwardInputsOutputs(args=None, kwargs=None, output=torch.randn(1)),
        )
        with patch.object(proc, "_check_overflow", return_value=0) as mock_check:
            proc._store_nan_result(api_info, "op", mio)
            mock_check.assert_called_once_with(mio)

    # ---- analyze_forward_output ----

    def test_analyze_forward_output_stores_nan(self):
        proc = self._create_processor()
        mio = ModuleForwardInputsOutputs(args=None, kwargs=None, output=torch.randn(2))
        with patch.object(
            PytorchDataProcessor,
            "analyze_forward_output",
            return_value={"conv2d": {"output": [{}]}},
        ):
            with patch.object(proc, "_check_overflow", return_value=7):
                result = proc.analyze_forward_output("conv2d", MagicMock(), mio)
                self.assertEqual(result["conv2d"][Const.IS_NAN_INDEX], 7)

    def test_analyze_forward_output_skips_when_name_absent(self):
        proc = self._create_processor()
        mio = ModuleForwardInputsOutputs(args=None, kwargs=None, output=torch.randn(2))
        with patch.object(
            PytorchDataProcessor, "analyze_forward_output", return_value={}
        ):
            with patch.object(proc, "_store_nan_result") as mock_store:
                proc.analyze_forward_output("conv2d", MagicMock(), mio)
                mock_store.assert_not_called()

    # ---- analyze_backward ----

    def test_analyze_backward_stores_nan(self):
        proc = self._create_processor()
        mio = ModuleBackwardInputsOutputs(
            grad_output=None, grad_input=(torch.randn(3),)
        )
        with patch.object(
            PytorchDataProcessor,
            "analyze_backward",
            return_value={"linear": {"grad_input": [{}]}},
        ):
            with patch.object(proc, "_check_overflow", return_value=3):
                result = proc.analyze_backward("linear", MagicMock(), mio)
                self.assertEqual(result["linear"][Const.IS_NAN_INDEX], 3)

    def test_analyze_backward_skips_when_name_absent(self):
        proc = self._create_processor()
        mio = ModuleBackwardInputsOutputs(
            grad_output=None, grad_input=(torch.randn(1),)
        )
        with patch.object(PytorchDataProcessor, "analyze_backward", return_value={}):
            with patch.object(proc, "_store_nan_result") as mock_store:
                proc.analyze_backward("api", MagicMock(), mio)
                mock_store.assert_not_called()
