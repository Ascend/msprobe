import unittest
from unittest.mock import patch
import pandas as pd
import os
import torch

from msprobe.core.config_check.checkers.weights_checker import collect_weights_data, compare_weight, compare_weight_file


class TestWeightComparison(unittest.TestCase):
    @patch('msprobe.core.config_check.utils.utils.get_tensor_features')
    def test_collect_weights_data(self, mock_get_tensor_features):
        mock_model = torch.nn.Linear(1, 1)
        mock_get_tensor_features.return_value = {'max': 1, 'min': 0, 'mean': 0.5, 'norm': 1}
        result = collect_weights_data(mock_model)
        self.assertEqual(isinstance(result, dict), True)

    @patch('msprobe.core.config_check.checkers.weights_checker.load_json')
    def test_compare_weight_file(self, mock_load_json):
        mock_load_json.side_effect = [
            {'weight1': {'max': 1, 'min': 0, 'mean': 0.5, 'norm': 1}},
            {'weight1': {'max': 1, 'min': 0, 'mean': 0.5, 'norm': 1}}
        ]
        result = compare_weight_file('bench.json', 'cmp.json')
        self.assertEqual(isinstance(result, list), True)

    @patch('msprobe.core.config_check.checkers.weights_checker.os_walk_for_files')
    @patch('msprobe.core.config_check.checkers.weights_checker.load_json')
    @patch('os.path.exists')
    def test_compare_weight(self, mock_exists, mock_load_json, mock_os_walk_for_files):
        mock_os_walk_for_files.return_value = [
            {"root": "bench/step1/rank0", "file": "weights.json"}
        ]
        mock_load_json.return_value = {'weight1': {'max': 1, 'min': 0, 'mean': 0.5, 'norm': 1}}
        mock_exists.return_value = True
        result = compare_weight('bench', 'cmp')
        self.assertEqual(isinstance(result, pd.DataFrame), True)

    @patch('msprobe.core.config_check.checkers.weights_checker.load_json')
    def test_compare_weight_file_different_weights(self, mock_load_json):
        mock_load_json.side_effect = [
            {'weight1': {'max': 1, 'min': 0, 'mean': 0.5, 'norm': 1}},
            {'weight1': {'max': 2, 'min': 1, 'mean': 1.5, 'norm': 2}}
        ]
        result = compare_weight_file('bench.json', 'cmp.json')
        self.assertEqual(isinstance(result, list), True)
        for res in result:
            if res["weight_name"] == "weight1":
                self.assertEqual(res["equal"], False)

    @patch('msprobe.core.config_check.checkers.weights_checker.os_walk_for_files')
    @patch('msprobe.core.config_check.checkers.weights_checker.load_json')
    @patch('os.path.exists')
    def test_compare_weight_cmp_file_missing(self, mock_exists, mock_load_json, mock_os_walk_for_files):
        mock_os_walk_for_files.return_value = [
            {"root": "bench/step1/rank0", "file": "weights.json"}
        ]
        mock_load_json.return_value = {'weight1': {'max': 1, 'min': 0, 'mean': 0.5, 'norm': 1}}
        mock_exists.return_value = False
        result = compare_weight('bench', 'cmp')
        self.assertEqual(isinstance(result, pd.DataFrame), True)
        self.assertEqual(len(result[result["equal"] == "only bench have"]), 1)

    @patch('msprobe.core.config_check.checkers.weights_checker.os_walk_for_files')
    @patch('msprobe.core.config_check.checkers.weights_checker.load_json')
    @patch('os.path.exists')
    def test_compare_weight_multiple_files(self, mock_exists, mock_load_json, mock_os_walk_for_files):
        mock_os_walk_for_files.return_value = [
            {"root": "bench/step1/rank0", "file": "weights1.json"},
            {"root": "bench/step1/rank0", "file": "weights2.json"}
        ]
        mock_load_json.return_value = {'weight1': {'max': 1, 'min': 0, 'mean': 0.5, 'norm': 1}}
        mock_exists.return_value = True
        result = compare_weight('bench', 'cmp')
        self.assertEqual(isinstance(result, pd.DataFrame), True)
        self.assertEqual(len(result), 2)
