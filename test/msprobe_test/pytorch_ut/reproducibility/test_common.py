import os
import random
import re
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

if not hasattr(torch, 'npu'):
    from unittest.mock import MagicMock
    _mock_npu = MagicMock()
    _mock_npu.is_available.return_value = False
    torch.npu = _mock_npu

from msprobe.core.common.exceptions import MsprobeException
from msprobe.pytorch.reproducibility.common import (
    Const,
    _check_torch_gpu_available,
    _check_torch_npu_available,
    gpu_available,
    npu_available,
    get_rank_id,
    create_csv,
    rename_csv,
    check_arguments
)


class TestConst(unittest.TestCase):

    def test_method_list_contains_tensor_random(self):
        self.assertIn("tensor_random", Const.METHOD_LIST)

    def test_api_mapping_keys(self):
        expected_keys = {"python_random", "numpy_random", "torch_random", "tensor_random"}
        self.assertEqual(set(Const.API_MAPPING.keys()), expected_keys)

    def test_api_mapping_python_random(self):
        self.assertIs(Const.API_MAPPING["python_random"], random)

    def test_api_mapping_numpy_random(self):
        self.assertIs(Const.API_MAPPING["numpy_random"], np.random)

    def test_api_mapping_torch_random(self):
        self.assertIs(Const.API_MAPPING["torch_random"], torch)

    def test_api_mapping_tensor_random(self):
        self.assertIs(Const.API_MAPPING["tensor_random"], torch.Tensor)

    def test_csv_header(self):
        self.assertEqual(Const.CSV_HEADER, [['api_name', 'stack']])


class TestCheckTorchGpuAvailable(unittest.TestCase):

    def test_returns_bool(self):
        result = _check_torch_gpu_available()
        self.assertIsInstance(result, bool)

    @patch('msprobe.pytorch.reproducibility.common.torch.cuda.is_available', side_effect=RuntimeError("no cuda"))
    def test_returns_false_on_exception(self, mock_is_available):
        result = _check_torch_gpu_available()
        self.assertFalse(result)

    @patch('msprobe.pytorch.reproducibility.common.torch.cuda.is_available', return_value=True)
    def test_returns_true_when_available(self, mock_is_available):
        result = _check_torch_gpu_available()
        self.assertTrue(result)


class TestCheckTorchNpuAvailable(unittest.TestCase):

    def test_returns_bool(self):
        result = _check_torch_npu_available()
        self.assertIsInstance(result, bool)

    @patch('msprobe.pytorch.reproducibility.common.torch.npu.is_available', side_effect=RuntimeError("no npu"))
    def test_returns_false_on_exception(self, mock_is_available):
        result = _check_torch_npu_available()
        self.assertFalse(result)

    @patch('msprobe.pytorch.reproducibility.common.torch.npu.is_available', return_value=True)
    def test_returns_true_when_available(self, mock_is_available):
        result = _check_torch_npu_available()
        self.assertTrue(result)


class TestGpuAvailable(unittest.TestCase):

    def test_gpu_available_is_bool(self):
        self.assertIsInstance(gpu_available, bool)


class TestNpuAvailable(unittest.TestCase):

    def test_npu_available_is_bool(self):
        self.assertIsInstance(npu_available, bool)


class TestGetRankId(unittest.TestCase):

    @patch('msprobe.pytorch.reproducibility.common.torch.distributed.is_initialized', return_value=False)
    def test_returns_none_when_not_initialized(self, mock_init):
        result = get_rank_id()
        self.assertIsNone(result)

    @patch('msprobe.pytorch.reproducibility.common.torch.distributed.get_rank', return_value=3)
    @patch('msprobe.pytorch.reproducibility.common.torch.distributed.is_initialized', return_value=True)
    def test_returns_rank_when_initialized(self, mock_init, mock_rank):
        result = get_rank_id()
        self.assertEqual(result, 3)


class TestCreateCsv(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_creates_csv_file_with_rank(self):
        csv_path = create_csv(self.temp_dir, rank_id=0)
        self.assertTrue(os.path.isfile(csv_path))
        self.assertIn("rank0", os.path.basename(csv_path))

    def test_creates_csv_file_without_rank(self):
        csv_path = create_csv(self.temp_dir, rank_id=None)
        self.assertTrue(os.path.isfile(csv_path))
        self.assertIn("proc", os.path.basename(csv_path))

    def test_csv_file_has_header(self):
        csv_path = create_csv(self.temp_dir, rank_id=None)
        with open(csv_path, 'r') as f:
            first_line = f.readline()
        self.assertIn("api_name", first_line)
        self.assertIn("stack", first_line)

    def test_csv_name_format_with_rank(self):
        csv_path = create_csv(self.temp_dir, rank_id=7)
        basename = os.path.basename(csv_path)
        self.assertTrue(re.match(r'random_rank7_\d{14}\.csv', basename))

    def test_csv_name_format_without_rank(self):
        csv_path = create_csv(self.temp_dir, rank_id=None)
        basename = os.path.basename(csv_path)
        self.assertTrue(re.match(r'random_proc\d+_\d{14}\.csv', basename))

    def test_creates_output_directory_if_not_exists(self):
        new_dir = os.path.join(self.temp_dir, "subdir", "nested")
        csv_path = create_csv(new_dir, rank_id=None)
        self.assertTrue(os.path.isdir(new_dir))
        self.assertTrue(os.path.isfile(csv_path))


class TestRenameCsv(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.temp_dir, "random_proc1234_20240101120000.csv")
        with open(self.csv_path, 'w') as f:
            f.write("api_name,stack\n")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_renames_proc_to_rank(self):
        new_path = rename_csv(self.csv_path, rank_id=5)
        self.assertFalse(os.path.isfile(self.csv_path))
        self.assertTrue(os.path.isfile(new_path))
        self.assertIn("rank5", os.path.basename(new_path))

    def test_new_path_preserves_directory(self):
        new_path = rename_csv(self.csv_path, rank_id=5)
        self.assertEqual(os.path.dirname(new_path), self.temp_dir)

    def test_new_name_format(self):
        new_path = rename_csv(self.csv_path, rank_id=5)
        basename = os.path.basename(new_path)
        self.assertEqual(basename, "random_rank5_20240101120000.csv")


class TestCheckArguments(unittest.TestCase):

    def test_valid_seed_zero(self):
        check_arguments(0, False, False)

    def test_valid_seed_max(self):
        check_arguments(2 ** 32 - 1, True, True)

    def test_valid_seed_middle(self):
        check_arguments(1234, False, True)

    def test_negative_seed_raises(self):
        with self.assertRaises(MsprobeException) as ctx:
            check_arguments(-1, False, False)
        self.assertEqual(ctx.exception.code, MsprobeException.INVALID_PARAM_ERROR)

    def test_seed_too_large_raises(self):
        with self.assertRaises(MsprobeException) as ctx:
            check_arguments(2 ** 32, False, False)
        self.assertEqual(ctx.exception.code, MsprobeException.INVALID_PARAM_ERROR)

    def test_non_integer_seed_raises(self):
        with self.assertRaises(MsprobeException) as ctx:
            check_arguments(1.5, False, False)
        self.assertEqual(ctx.exception.code, MsprobeException.INVALID_PARAM_ERROR)

    def test_string_seed_raises(self):
        with self.assertRaises(MsprobeException) as ctx:
            check_arguments("1234", False, False)
        self.assertEqual(ctx.exception.code, MsprobeException.INVALID_PARAM_ERROR)

    def test_is_deterministic_not_bool_raises(self):
        with self.assertRaises(MsprobeException) as ctx:
            check_arguments(1234, 1, False)
        self.assertEqual(ctx.exception.code, MsprobeException.INVALID_PARAM_ERROR)

    def test_is_enhanced_not_bool_raises(self):
        with self.assertRaises(MsprobeException) as ctx:
            check_arguments(1234, False, 1)
        self.assertEqual(ctx.exception.code, MsprobeException.INVALID_PARAM_ERROR)

    def test_is_deterministic_none_raises(self):
        with self.assertRaises(MsprobeException) as ctx:
            check_arguments(1234, None, False)
        self.assertEqual(ctx.exception.code, MsprobeException.INVALID_PARAM_ERROR)

    def test_is_enhanced_none_raises(self):
        with self.assertRaises(MsprobeException) as ctx:
            check_arguments(1234, False, None)
        self.assertEqual(ctx.exception.code, MsprobeException.INVALID_PARAM_ERROR)
