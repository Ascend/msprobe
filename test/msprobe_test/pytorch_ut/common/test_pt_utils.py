# Copyright (c) 2024-2025, Huawei Technologies Co., Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist
from msprobe.core.common.exceptions import DistributedNotInitializedError
from msprobe.pytorch.api_accuracy_checker.common.utils import ApiData
from msprobe.pytorch.common.utils import (
    parameter_adapter,
    get_rank_if_initialized,
    get_tensor_rank,
    get_rank_id,
    print_rank_0,
    load_pt,
    save_pt
)


class TestParameterAdapter(unittest.TestCase):

    def setUp(self):
        self.func_mock = MagicMock()
        self.decorated_func = parameter_adapter(self.func_mock)
        self.api_name = "__getitem__"

    def test_handle_masked_select_bfloat16(self):
        input_tensor = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
        indices = torch.tensor([1, 0], dtype=torch.uint8)
        self.func_mock.return_value = input_tensor[indices.bool()]

        result = self.decorated_func(self, input_tensor, indices)
        expected = input_tensor[indices.bool()]
        self.assertTrue(torch.equal(result, expected))

    def test_handle_masked_select_bool(self):
        input_tensor = torch.tensor([1.0, 2.0, 3.0])
        indices = torch.tensor([True, False, True])
        self.func_mock.return_value = input_tensor[indices]
        result = self.decorated_func(self, input_tensor, indices)
        self.assertTrue(torch.equal(result, torch.tensor([1.0, 3.0])))

    def test_handle_nonzero_indices(self):
        input_tensor = torch.tensor([10.0, 20.0, 30.0])
        indices = torch.tensor([1, 2])
        self.func_mock.return_value = input_tensor[indices]
        result = self.decorated_func(self, input_tensor, indices)
        self.assertTrue(torch.equal(result, torch.tensor([20.0, 30.0])))

    def test_op_name_eq_with_none(self):
        self.api_name = "__eq__"
        args = (torch.tensor([1]), None)
        result = self.decorated_func(self, *args)
        self.assertFalse(result)

    def test_two_dimensional_indices(self):
        input_tensor = torch.tensor([10.0, 20.0, 30.0])
        indices = torch.tensor([[0], [2]])
        self.func_mock.side_effect = [torch.tensor(10.0), torch.tensor(30.0)]

        result = self.decorated_func(self, input_tensor, indices)
        expected = torch.stack([torch.tensor(10.0), torch.tensor(30.0)])
        self.assertTrue(torch.equal(result, expected))

        self.func_mock.assert_any_call(self, input_tensor, [0])
        self.func_mock.assert_any_call(self, input_tensor, [2])

    def test_greater_than_two_dimensional_indices(self):
        input_tensor = torch.tensor([10.0, 20.0, 30.0])
        indices = torch.tensor([[[0]], [[2]]])
        self.func_mock.side_effect = [torch.tensor(10.0), torch.tensor(30.0)]

        result = self.decorated_func(self, input_tensor, indices)
        expected = torch.stack([torch.tensor([[10.0]]), torch.tensor([[30.0]])])
        self.assertTrue(torch.equal(result, expected), f"Expected: {expected}, but got: {result}")


class TestGetRankIfInitialized(unittest.TestCase):

    @patch('torch.distributed')
    def test_get_rank_init(self, mock_distributed):
        mock_distributed.is_initialized.return_value = True
        mock_distributed.get_rank.return_value = 2
        rank = get_rank_if_initialized()
        self.assertEqual(rank, 2)

    @patch('torch.distributed')
    def test_get_rank_not_init(self, mock_distributed):
        mock_distributed.is_initialized.return_value = False
        with self.assertRaises(DistributedNotInitializedError):
            get_rank_if_initialized()


class TestGetTensorRank(unittest.TestCase):

    def setUp(self):
        self.ori_dist_is_initialized = dist.is_initialized
        self.ori_dist_get_rank = dist.get_rank
        self.dist_mock = MagicMock()
        dist.is_initialized = self.dist_mock.is_initialized
        dist.get_rank = self.dist_mock.get_rank

    def tearDown(self):
        dist.is_initialized = self.ori_dist_is_initialized
        dist.get_rank = self.ori_dist_get_rank

    def test_get_tensor_rank_with_initialized_dist(self):
        self.dist_mock.is_initialized.return_value = True
        self.dist_mock.get_rank.return_value = 3
        in_feat = torch.tensor([1, 2, 3])
        out_feat = torch.tensor([4, 5, 6])
        result = get_tensor_rank(in_feat, out_feat)
        self.assertEqual(result, 3)

    def test_get_tensor_rank_with_cpu_tensor(self):
        self.dist_mock.is_initialized.return_value = False
        in_feat = torch.tensor([1, 2, 3])
        out_feat = torch.tensor([4, 5, 6])
        result = get_tensor_rank(in_feat, out_feat)
        self.assertIsNone(result)

    def test_get_tensor_rank_with_list_of_tensor(self):
        self.dist_mock.is_initialized.return_value = False
        in_feat = [torch.tensor([1, 2, 3])]
        out_feat = [torch.tensor([4, 5, 6])]
        result = get_tensor_rank(in_feat, out_feat)
        self.assertIsNone(result)

    def test_get_tensor_rank_with_empty_list(self):
        self.dist_mock.is_initialized.return_value = False
        in_feat = []
        out_feat = [torch.tensor([4, 5, 6])]
        result = get_tensor_rank(in_feat, out_feat)
        self.assertIsNone(result)


class TestGetRankId(unittest.TestCase):

    def test_get_rank_id_init(self):
        torch.distributed.is_initialized = MagicMock(return_value=True)
        torch.distributed.get_rank = MagicMock(return_value=3)
        self.assertEqual(get_rank_id(), 3)

    def test_get_rank_id_not_init(self):
        torch.distributed.is_initialized = MagicMock(return_value=False)
        self.assertEqual(get_rank_id(), 0)


class TestPrintRank0(unittest.TestCase):

    @patch('msprobe.core.common.log.logger.info')
    @patch('torch.distributed.is_initialized', return_value=True)
    @patch('torch.distributed.get_rank', return_value=0)
    def test_print_rank_0_initialized_rank_0(self, mock_get_rank, mock_is_initialized, mock_logger_info):
        message = "Test message"
        print_rank_0(message)
        mock_logger_info.assert_called_once_with(message)


class TestLoadPt(unittest.TestCase):

    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pt')
        tensor = torch.tensor([1, 2, 3])
        torch.save(tensor, self.temp_file.name)

    @patch('torch.load')
    def test_load_pt_cpu(self, mock_load):
        mock_load.return_value = torch.tensor([1, 2, 3])
        result = load_pt(self.temp_file.name, to_cpu=True)
        self.assertTrue(torch.equal(result, torch.tensor([1, 2, 3])))
        mock_load.assert_called_once_with(self.temp_file.name, map_location=torch.device("cpu"), weights_only=True)

    @patch('torch.load')
    def test_load_pt_nogpu(self, mock_load):
        mock_load.return_value = torch.tensor([1, 2, 3])
        result = load_pt(self.temp_file.name, to_cpu=False)
        self.assertTrue(torch.equal(result, torch.tensor([1, 2, 3])))
        mock_load.assert_called_once_with(self.temp_file.name, weights_only=True)

    @patch('torch.load')
    def test_load_pt_failure(self, mock_load):
        mock_load.side_effect = RuntimeError("Load failed")
        with self.assertRaises(RuntimeError) as context:
            load_pt(self.temp_file.name)
        self.assertIn("load pt file", str(context.exception))

    def tearDown(self):
        if os.path.isfile(self.temp_file.name):
            os.remove(self.temp_file.name)

class TestSavePT(unittest.TestCase):

    def setUp(self):
        self.tensor = torch.tensor([1, 2, 3])
        self.filepath = 'temp_tensor.pt'

    def tearDown(self):
        try:
            os.remove(self.filepath)
        except FileNotFoundError:
            pass

    @patch('msprobe.pytorch.common.utils.save_pt')
    @patch('os.path.realpath', return_value='temp_tensor.pt')
    @patch('msprobe.core.common.file_utils.check_path_before_create')
    @patch('msprobe.core.common.file_utils.change_mode')
    def test_save_pt_success(self, mock_change_mode, mock_check_path, mock_realpath, mock_torch_save):
        mock_torch_save(self.tensor, self.filepath)
        mock_torch_save.assert_called_once_with(self.tensor, self.filepath)

    @patch('torch.save', side_effect=Exception("Save failed"))
    @patch('os.path.realpath', return_value='temp_tensor.pt')
    @patch('msprobe.core.common.file_utils.check_path_before_create')
    @patch('msprobe.core.common.file_utils.change_mode')
    def test_save_pt_failure(self, mock_change_mode, mock_check_path, mock_realpath, mock_torch_save):
        with self.assertRaises(RuntimeError) as context:
            save_pt(self.tensor, self.filepath)
        self.assertIn("save pt file temp_tensor.pt failed", str(context.exception))
