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
import torch
import pandas as pd
from unittest.mock import patch, MagicMock

from msprobe.core.config_check.checkers.dataset_checker import compare_dataset, \
    compare_dataset_dicts, parse_args_and_kargs, process_obj


class TestTensorProcessing(unittest.TestCase):

    def test_process_obj_tensor(self):
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = process_obj(tensor)
        self.assertEqual(isinstance(result, dict), True)
        self.assertEqual(set(result.keys()), {'max', 'min', 'mean', 'norm'})

    def test_process_obj_list(self):
        obj = [torch.tensor([1.0]), torch.tensor([2.0])]
        result = process_obj(obj)
        self.assertEqual(isinstance(result, dict), True)
        self.assertEqual(set(result.keys()), {0, 1})

    def test_process_obj_dict(self):
        obj = {'a': torch.tensor([1.0]), 'b': torch.tensor([2.0])}
        result = process_obj(obj)
        self.assertEqual(isinstance(result, dict), True)
        self.assertEqual(set(result.keys()), {'a', 'b'})

    def test_process_obj_other(self):
        obj = "test"
        result = process_obj(obj)
        self.assertEqual(result, "")

    def test_parse_args_and_kargs(self):
        args = (torch.tensor([1.0]),)
        kwargs = {'a': torch.tensor([2.0])}
        result = parse_args_and_kargs(args, kwargs)
        self.assertEqual(isinstance(result, dict), True)
        self.assertEqual(set(result.keys()), {'args', 'kwargs'})

    def test_compare_dataset_dicts_equal(self):
        dict1 = {'a': {'max': 1.0, 'min': 0.0, 'mean': 0.5, 'norm': 0.7}}
        dict2 = {'a': {'max': 1.0, 'min': 0.0, 'mean': 0.5, 'norm': 0.7}}
        results = compare_dataset_dicts(dict1, dict2)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['equal'], True)

    def test_compare_dataset_dicts_not_equal(self):
        dict1 = {'a': {'max': 1.0, 'min': 0.0, 'mean': 0.5, 'norm': 0.7}}
        dict2 = {'a': {'max': 2.0, 'min': 0.0, 'mean': 0.5, 'norm': 0.7}}
        results = compare_dataset_dicts(dict1, dict2)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['equal'], False)

    def test_compare_dataset_dicts_nested(self):
        dict1 = {'a': {'b': {'max': 1.0, 'min': 0.0, 'mean': 0.5, 'norm': 0.7}}}
        dict2 = {'a': {'b': {'max': 1.0, 'min': 0.0, 'mean': 0.5, 'norm': 0.7}}}
        results = compare_dataset_dicts(dict1, dict2)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['tag'], 'a.b')

    @patch('os.listdir', side_effect=[["step1"], ["rank1"]])
    @patch('os.path.isdir', return_value=True)
    @patch('os.path.isfile', return_value=True)
    @patch('msprobe.core.config_check.checkers.dataset_checker.load_json')
    def test_compare_dataset(self, mock_load_json, mock_isfile, mock_isdir, mock_listdir):
        mock_load_json.return_value = {'a': {'max': 1.0, 'min': 0.0, 'mean': 0.5, 'norm': 0.7}}
        bench_dir = 'bench'
        cmp_dir = 'cmp'
        result = compare_dataset(bench_dir, cmp_dir)
        self.assertEqual(isinstance(result, pd.DataFrame), True)



    