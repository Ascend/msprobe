import os
import json
import unittest
import tempfile
import shutil
import pandas as pd
from unittest.mock import patch

from msprobe.core.config_check.slime_param_compare.slime_hyper_params_cmp import slime_compare_hyper_params
from msprobe.core.common.const import CompareConst


class TestSlimeHyperParamsCompare(unittest.TestCase):
    """测试 slime 超参对比工具"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.npu_cfg_path = os.path.join(self.tmp_dir, "npu.json")
        self.bench_cfg_path = os.path.join(self.tmp_dir, "bench.json")
        self.output_dir = os.path.join(self.tmp_dir, "out")
        self.column_list = CompareConst.SLIME_HYPER_PARAM_COMPARE_COLUM
        self.col_key = self.column_list[0]
        self.col_npu_val = self.column_list[1]
        self.col_bench_val = self.column_list[2]
        self.col_consistent = self.column_list[3]

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    @staticmethod
    def _write_json(file_path: str, data: dict):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @staticmethod
    def _load_csv_with_bool(csv_path, col_bool):
        df = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str)
        df[col_bool] = df[col_bool].str.strip() == "是"
        return df

    def test_compare_all_consistent(self):
        """测试所有超参数完全一致"""
        npu_data = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "optimizer": "Adam"
        }
        bench_data = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "optimizer": "Adam"
        }
        self._write_json(self.npu_cfg_path, npu_data)
        self._write_json(self.bench_cfg_path, bench_data)

        slime_compare_hyper_params(self.npu_cfg_path, self.bench_cfg_path, self.output_dir)
        csv_path = os.path.join(self.output_dir, "hyper_params_compare.csv")
        self.assertTrue(os.path.exists(csv_path))

        df = self._load_csv_with_bool(csv_path, self.col_consistent)
        self.assertEqual(len(df), 3)
        self.assertTrue(all(df[self.col_consistent]))

    def test_compare_some_different(self):
        """测试部分参数不一致"""
        npu_data = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "optimizer": "Adam"
        }
        bench_data = {
            "learning_rate": 0.01,
            "batch_size": 32,
            "optimizer": "SGD"
        }
        self._write_json(self.npu_cfg_path, npu_data)
        self._write_json(self.bench_cfg_path, bench_data)

        slime_compare_hyper_params(self.npu_cfg_path, self.bench_cfg_path, self.output_dir)
        csv_path = os.path.join(self.output_dir, "hyper_params_compare.csv")
        df = self._load_csv_with_bool(csv_path, self.col_consistent)

        diff_rows = df[~df[self.col_consistent]]
        self.assertEqual(len(diff_rows), 2)
        diff_keys = set(diff_rows[self.col_key].tolist())
        self.assertSetEqual(diff_keys, {"learning_rate", "optimizer"})

    def test_compare_one_side_missing_key(self):
        """一边存在参数，另一边缺失"""
        npu_data = {"batch_size": 32, "lr": 0.01}
        bench_data = {"batch_size": 32}
        self._write_json(self.npu_cfg_path, npu_data)
        self._write_json(self.bench_cfg_path, bench_data)
        slime_compare_hyper_params(self.npu_cfg_path, self.bench_cfg_path, self.output_dir)
        csv_path = os.path.join(self.output_dir, "hyper_params_compare.csv")
        df = self._load_csv_with_bool(csv_path, self.col_consistent)
        diff_rows = df[~df[self.col_consistent]]
        diff_keys = set(diff_rows[self.col_key].tolist())
        self.assertEqual(len(diff_rows), 1)
        self.assertIn("lr", diff_keys)
