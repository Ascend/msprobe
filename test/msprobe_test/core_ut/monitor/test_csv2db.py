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
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import pandas as pd

from msprobe.core.monitor.csv2db import (
    CSV2DBConfig,
    _pre_scan_single_rank,
    _pre_scan,
    process_single_rank,
    import_data
)


class TestPreScanFunctions(unittest.TestCase):
    def setUp(self):
        # 创建临时目录和测试CSV文件
        self.temp_dir = tempfile.mkdtemp()
        self.temp_dir_rank2 = tempfile.mkdtemp()
        self.test_csv_path_actv = os.path.join(self.temp_dir, "actv_0-100.csv")
        self.test_csv_path_rank2_grad = os.path.join(
            self.temp_dir_rank2, "grad_reduced_100-200.csv")
        self.test_csv_path_rank_inv = os.path.join(
            self.temp_dir_rank2, "invalid_metric_100-200.csv")

        # 创建测试CSV数据
        test_data_actv = {
            "name": ["layer1", "layer2"],
            "vpp_stage": [0, 0],
            "micro_step": [0, 1],
            "step": [10, 20],
            "min": [0.1, 0.2],
            "max": [1.0, 2.0]
        }
        test_data_grad = {
            "name": ["layer1_weight", "layer2_weight"],
            "vpp_stage": [0, 0],
            "micro_step": [0, 1],
            "step": [10, 20],
            "min": [0.1, 0.2],
            "max": [1.0, 2.0]
        }
        df = pd.DataFrame(test_data_actv)
        df.to_csv(self.test_csv_path_actv, index=False)
        df = pd.DataFrame(test_data_grad)
        df.to_csv(self.test_csv_path_rank2_grad, index=False)
        df = pd.DataFrame(test_data_grad)
        df.to_csv(self.test_csv_path_rank_inv, index=False)

    def tearDown(self):
        # 清理临时目录
        shutil.rmtree(self.temp_dir)

    def test_pre_scan_single_rank(self):
        """测试单个rank的预扫描"""
        rank = 0
        files = [self.test_csv_path_actv]
        result = _pre_scan_single_rank(rank, files, {}, False)
        self.assertEqual(result["rank"], rank)
        self.assertEqual(result["min_step"], 10)
        self.assertEqual(result["max_step"], 20)
        self.assertEqual(result["metric_stats"], {"actv": {"min", "max"}})
        self.assertEqual(len(result["targets"]["actv"]), 2)

    def test_pre_scan(self):
        """测试完整预扫描流程"""
        # 模拟MonitorDB
        mock_db = MagicMock()

        # 测试数据
        data_dirs = {0: self.temp_dir, 2: self.temp_dir_rank2}
        config = CSV2DBConfig(target_output_dirs=data_dirs, micro_step=False)
        result = _pre_scan(config, mock_db)

        self.assertEqual(sorted([rank for rank, _ in result[0]]), [0, 2])

        mock_db.init_schema.assert_called_once()
        mock_db.insert_dimensions.assert_called_once()
        mock_db.init_global_stats_data.assert_called_once()
        mock_db.create_trend_data.assert_called_once()
        mock_db.extract_tags_from_processed_targets.assert_called_once()
        call_args = mock_db.init_global_stats_data.call_args.args[0]
        expected = {
            "min_step": 10,
            "max_step": 20,
            "max_rank": 2,
            "actv": {"max", "min"},
            "grad_reduced": {"max", "min"}
        }

        self.assertEqual(set(call_args.keys()), set(expected.keys()))
        for key in expected:
            self.assertEqual(call_args[key], expected[key])


class TestProcessSingleRank(unittest.TestCase):
    @patch("msprobe.core.monitor.csv2db.MonitorDB")
    @patch("msprobe.core.monitor.csv2db.read_csv")
    def test_process_single_rank(self, mock_read_csv, mock_db_class):
        """测试处理单个rank的数据"""
        # 模拟数据库和映射
        mock_db = MagicMock()
        mock_db_class.return_value = mock_db
        mock_db.insert_rows.return_value = 2

        # 模拟CSV数据
        mock_result = pd.DataFrame({
            "name": ["layer1", "layer2"],
            "vpp_stage": [0, 0],
            "micro_step": [0, 1],
            "step": [10, 20],
            "norm": [0.1, 0.2],
            "max": [1.0, 2.0]
        })
        mock_read_csv.return_value = mock_result

        # 测试数据
        task = (0, ["actv_10-20.csv"])
        metric_id_dict = {"actv": (1, ["norm", "max"])}
        target_dict = {("layer1", 0, 0): 1, ("layer2", 0, 0): 2}
        db_path = "dummy.db"
        config = CSV2DBConfig(target_output_dirs={}, db_file=db_path, micro_step=True)
        result = process_single_rank(
            task, metric_id_dict, target_dict, {10:0, 20:1}, config)

        self.assertEqual(result, 2)
        mock_db.insert_rows.assert_called_with(
            [(0, 0, 1, 1, 0.1, 1.0), (0, 2, 2, 1, 0.2, 2.0)]
        )


class TestImportData(unittest.TestCase):
    @patch("msprobe.core.monitor.csv2db._pre_scan")
    @patch("msprobe.core.monitor.csv2db.MonitorDB")
    def test_import_data_success(self, mock_monitordb, mock_pre_scan):
        """测试数据导入成功场景"""
        # 模拟预扫描结果
        rank_files = [[0, ["actv_10-20.csv"]], [1, ["actv_10-20.csv"]]]
        metric_mapping = {"actv": (1, ["min", "max"])}
        target_mapping = {("layer1", 0, 0): 1}
        micro_step_prefix = {10: 0, 20: 1}
        mock_pre_scan.return_value = rank_files, metric_mapping, target_mapping, micro_step_prefix

        # 模拟数据库
        mock_db = MagicMock()
        mock_monitordb.return_value = mock_db

        # 测试数据
        data_dirs = {0: "dir0", 1: "dir1"}
        config = CSV2DBConfig(target_output_dirs=data_dirs, db_file="./trend.db")
        import_data(config)

        mock_pre_scan.assert_called_once()
