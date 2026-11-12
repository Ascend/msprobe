# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
#  MindStudio is licensed under Mulan PSL v2.
#  You can use this software according to the terms and conditions of the Mulan PSL v2.
#  You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
#  THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
#  EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
#  MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
#  See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""两份精度数据拼接的端到端验证。

覆盖 dump / monitor 两种数据格式, step / rank 两种拼接维度, 开启 / 关闭微步计数
两种计数方式的组合场景。
"""

import json
import os
import shutil
import sqlite3
import tempfile
import unittest

import pandas as pd

from msprobe.core.compare.data2db import DBImporter

DUMP_DB_NAME = "dump_data.trend.db"
MONITOR_DB_NAME = "monitor_data.trend.db"


def make_dump_dataset(root, steps, ranks, target_name="Module.layer1", value=1.0):
    """构造一份 dump 数据, 每个 step 下每个 rank 一个 dump.json。

    :param root: 数据根目录
    :param steps: step 目录名列表
    :param ranks: rank 目录名列表
    :param target_name: 网络层名前缀
    :param value: 张量统计量的值
    :return: 数据根目录
    """
    for step_dir in steps:
        step_path = f"{root}/{step_dir}"
        for rank_dir in ranks:
            rank_path = f"{step_path}/{rank_dir}"
            os.makedirs(rank_path, exist_ok=True)
            dump_data = {
                "data": {
                    f"{target_name}.forward": {
                        "input_args": [
                            {
                                "type": "torch.Tensor",
                                "dtype": "torch.float16",
                                "Norm": value,
                                "Max": value,
                                "Min": value,
                                "Mean": value,
                            }
                        ]
                    }
                }
            }
            with open(f"{rank_path}/dump.json", "w", encoding="utf-8") as f:
                json.dump(dump_data, f)
            construct_data = {f"{target_name}.forward": None}
            with open(f"{rank_path}/construct.json", "w", encoding="utf-8") as f:
                json.dump(construct_data, f)
    return root


def make_monitor_dataset(root, rank, rows):
    """构造一份 monitor 数据。

    :param root: 数据根目录
    :param rank: rank 编号
    :param rows: 行数据列表, 每项为 dict
    :return: 数据根目录
    """
    rank_dir = f"{root}/Jan01_00-00-00-rank{rank}-"
    os.makedirs(rank_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{rank_dir}/actv_0-100.csv", index=False)
    return root


def query_db(db_file, sql, params=()):
    """查询数据库"""
    conn = sqlite3.connect(db_file)
    try:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(sql, params)
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


class ConcatTestBase(unittest.TestCase):
    """拼接场景测试的公共逻辑"""

    def setUp(self):
        # 工具的路径校验不允许反斜杠, 这里统一使用正斜杠路径
        self.work_dir = tempfile.mkdtemp().replace("\\", "/")
        self.db_dir = f"{self.work_dir}/db"
        self.data_a = f"{self.work_dir}/data_a"
        self.data_b = f"{self.work_dir}/data_b"
        os.makedirs(self.data_a, exist_ok=True)
        os.makedirs(self.data_b, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.work_dir, ignore_errors=True)

    def get_rank_step_pairs(self, db_file):
        """取库内 (rank, step) 的组合"""
        rows = query_db(db_file, "SELECT DISTINCT rank, step FROM trend_data ORDER BY rank, step")
        return [(row["rank"], row["step"]) for row in rows]

    def get_global_stats(self, db_file):
        """取全局统计信息"""
        rows = query_db(db_file, "SELECT min_step, max_step, max_rank FROM global_stats ORDER BY ROWID DESC LIMIT 1")
        return rows[0] if rows else {}

    def get_concat_info(self, db_file):
        """取拼接信息"""
        rows = query_db(db_file, "SELECT * FROM concat_info")
        return rows[0] if rows else {}

    def get_target_ids(self, db_file):
        """取 target 名称到 ID 的映射"""
        rows = query_db(db_file, "SELECT target_id, target_name FROM monitoring_targets")
        return {row["target_name"]: row["target_id"] for row in rows}


class TestDumpConcat(ConcatTestBase):
    """dump 数据的拼接场景"""

    def _import_dump(self, concat_dim, micro_step):
        importer = DBImporter(
            db_path=self.db_dir,
            data_path=self.data_a,
            format="dump",
            micro_step=micro_step,
            compare_data_path=self.data_b,
            concat_dim=concat_dim,
        )
        importer.import_dump_data()
        return os.path.join(self.db_dir, DUMP_DB_NAME)

    def test_compare_data_requires_explicit_format(self):
        """比对模式下必须显式指定 format, 不能使用 auto"""
        make_dump_dataset(self.data_a, ["step0"], ["rank0"])
        make_dump_dataset(self.data_b, ["step0"], ["rank0"], value=2.0)
        with self.assertRaises(ValueError):
            DBImporter(
                db_path=self.db_dir,
                data_path=self.data_a,
                format="auto",
                micro_step="true",
                compare_data_path=self.data_b,
            )

    def test_step_dim_without_micro_step(self):
        """step 维度拼接, 关闭微步计数"""
        make_dump_dataset(self.data_a, ["step0", "step1"], ["rank0"])
        make_dump_dataset(self.data_b, ["step0", "step1"], ["rank0"], value=2.0)
        db_file = self._import_dump("step", "false")

        self.assertEqual(self.get_rank_step_pairs(db_file), [(0, 0), (0, 1), (0, 2), (0, 3)])
        stats = self.get_global_stats(db_file)
        self.assertEqual(stats["min_step"], 0)
        self.assertEqual(stats["max_step"], 3)
        self.assertEqual(stats["max_rank"], 0)

    def test_step_dim_with_micro_step(self):
        """step 维度拼接, 开启微步计数"""
        make_dump_dataset(self.data_a, ["step0", "step1"], ["rank0"])
        make_dump_dataset(self.data_b, ["step0", "step1"], ["rank0"], value=2.0)
        db_file = self._import_dump("step", "true")

        # 每个 step 各含 1 个微步, 两份数据各占 2 个微步
        self.assertEqual(self.get_rank_step_pairs(db_file), [(0, 0), (0, 1), (0, 2), (0, 3)])
        stats = self.get_global_stats(db_file)
        self.assertEqual(stats["min_step"], 0)
        self.assertEqual(stats["max_step"], 3)

    def test_rank_dim_without_micro_step(self):
        """rank 维度拼接, 关闭微步计数"""
        make_dump_dataset(self.data_a, ["step0", "step1"], ["rank0"])
        make_dump_dataset(self.data_b, ["step0", "step1"], ["rank0"], value=2.0)
        db_file = self._import_dump("rank", "false")

        self.assertEqual(self.get_rank_step_pairs(db_file), [(0, 0), (0, 1), (1, 0), (1, 1)])
        stats = self.get_global_stats(db_file)
        self.assertEqual(stats["min_step"], 0)
        self.assertEqual(stats["max_step"], 1)
        self.assertEqual(stats["max_rank"], 1)

    def test_rank_dim_with_micro_step(self):
        """rank 维度拼接, 开启微步计数"""
        make_dump_dataset(self.data_a, ["step0", "step1"], ["rank0"])
        make_dump_dataset(self.data_b, ["step0", "step1"], ["rank0"], value=2.0)
        db_file = self._import_dump("rank", "true")

        self.assertEqual(self.get_rank_step_pairs(db_file), [(0, 0), (0, 1), (1, 0), (1, 1)])
        stats = self.get_global_stats(db_file)
        self.assertEqual(stats["max_rank"], 1)

    def test_two_sources_share_target_id(self):
        """两份数据中同名的张量目标共享同一个 target_id"""
        make_dump_dataset(self.data_a, ["step0"], ["rank0"])
        make_dump_dataset(self.data_b, ["step0"], ["rank0"], value=2.0)
        db_file = self._import_dump("step", "false")

        targets = self.get_target_ids(db_file)
        self.assertEqual(len(targets), 1)
        rows = query_db(db_file, "SELECT DISTINCT target_id FROM trend_data")
        self.assertEqual(len(rows), 1)

    def test_concat_info_recorded(self):
        """拼接信息被记录到 concat_info 表"""
        make_dump_dataset(self.data_a, ["step0", "step1"], ["rank0"])
        make_dump_dataset(self.data_b, ["step0", "step1"], ["rank0"], value=2.0)
        db_file = self._import_dump("step", "false")

        info = self.get_concat_info(db_file)
        self.assertEqual(info["concat_dim"], "step")
        self.assertEqual(info["dataset_count"], 2)
        self.assertEqual(info["boundary"], 2)
        self.assertEqual(info["dataset1_span"], 2)
        self.assertEqual(info["dataset2_span"], 2)

    def test_single_source_keeps_behavior(self):
        """不传对比数据时行为与现状一致"""
        make_dump_dataset(self.data_a, ["step0", "step1"], ["rank0"])
        importer = DBImporter(
            db_path=self.db_dir,
            data_path=self.data_a,
            format="dump",
            micro_step="false",
        )
        importer.import_dump_data()
        db_file = os.path.join(self.db_dir, DUMP_DB_NAME)

        self.assertEqual(self.get_rank_step_pairs(db_file), [(0, 0), (0, 1)])
        stats = self.get_global_stats(db_file)
        self.assertEqual(stats["min_step"], 0)
        self.assertEqual(stats["max_step"], 1)
        info = self.get_concat_info(db_file)
        self.assertEqual(info["concat_dim"], "none")
        self.assertEqual(info["dataset_count"], 1)
        # 单份数据没有分界点
        self.assertEqual(info["boundary"], 0)
        self.assertEqual(info["dataset2_span"], 0)

    def test_single_source_with_micro_step(self):
        """不传对比数据且开启微步计数时, 记录实际的微步总数"""
        make_dump_dataset(self.data_a, ["step0", "step1"], ["rank0"])
        importer = DBImporter(
            db_path=self.db_dir,
            data_path=self.data_a,
            format="dump",
            micro_step="true",
        )
        importer.import_dump_data()
        db_file = os.path.join(self.db_dir, DUMP_DB_NAME)

        self.assertEqual(self.get_rank_step_pairs(db_file), [(0, 0), (0, 1)])
        info = self.get_concat_info(db_file)
        self.assertEqual(info["dataset_count"], 1)
        self.assertEqual(info["boundary"], 0)
        self.assertEqual(info["dataset1_span"], 2)


class TestMonitorConcat(ConcatTestBase):
    """monitor 数据的拼接场景"""

    ROWS = [
        {"name": "layer1", "vpp_stage": 0, "micro_step": 0, "step": 10, "min": 0.1, "max": 1.0},
        {"name": "layer1", "vpp_stage": 0, "micro_step": 1, "step": 20, "min": 0.2, "max": 2.0},
    ]

    def _import_monitor(self, concat_dim, micro_step):
        importer = DBImporter(
            db_path=self.db_dir,
            data_path=self.data_a,
            format="monitor",
            micro_step=micro_step,
            compare_data_path=self.data_b,
            concat_dim=concat_dim,
        )
        importer.import_monitor_data()
        return os.path.join(self.db_dir, MONITOR_DB_NAME)

    def test_step_dim_without_micro_step(self):
        """step 维度拼接, 关闭微步计数"""
        make_monitor_dataset(self.data_a, 0, self.ROWS)
        make_monitor_dataset(self.data_b, 0, self.ROWS)
        db_file = self._import_monitor("step", "false")

        self.assertEqual(self.get_rank_step_pairs(db_file), [(0, 0), (0, 1), (0, 2), (0, 3)])
        stats = self.get_global_stats(db_file)
        self.assertEqual(stats["min_step"], 0)
        self.assertEqual(stats["max_step"], 3)
        self.assertEqual(stats["max_rank"], 0)

    def test_step_dim_with_micro_step(self):
        """step 维度拼接, 开启微步计数"""
        make_monitor_dataset(self.data_a, 0, self.ROWS)
        make_monitor_dataset(self.data_b, 0, self.ROWS)
        db_file = self._import_monitor("step", "true")

        # step10 含 1 个微步, step20 含 2 个微步, 一份数据共 3 个微步
        self.assertEqual(self.get_rank_step_pairs(db_file), [(0, 0), (0, 2), (0, 3), (0, 5)])
        stats = self.get_global_stats(db_file)
        self.assertEqual(stats["min_step"], 0)
        self.assertEqual(stats["max_step"], 5)

    def test_rank_dim_without_micro_step(self):
        """rank 维度拼接, 关闭微步计数"""
        make_monitor_dataset(self.data_a, 0, self.ROWS)
        make_monitor_dataset(self.data_b, 0, self.ROWS)
        db_file = self._import_monitor("rank", "false")

        self.assertEqual(self.get_rank_step_pairs(db_file), [(0, 0), (0, 1), (1, 0), (1, 1)])
        stats = self.get_global_stats(db_file)
        self.assertEqual(stats["min_step"], 0)
        self.assertEqual(stats["max_step"], 1)
        self.assertEqual(stats["max_rank"], 1)

    def test_rank_dim_with_micro_step(self):
        """rank 维度拼接, 开启微步计数"""
        make_monitor_dataset(self.data_a, 0, self.ROWS)
        make_monitor_dataset(self.data_b, 0, self.ROWS)
        db_file = self._import_monitor("rank", "true")

        # 两份数据共用公共微步时间线, 同一迭代步落在同一个库内 step
        self.assertEqual(self.get_rank_step_pairs(db_file), [(0, 0), (0, 2), (1, 0), (1, 2)])
        stats = self.get_global_stats(db_file)
        self.assertEqual(stats["max_rank"], 1)

    def test_two_sources_share_target_id(self):
        """两份数据中同名的监测目标共享同一个 target_id

        关闭微步计数时, 同一名字在不同微步上的监测目标是不同的 target, 数量为迭代
        步数; 开启微步计数后微步号不再参与 target 标识, 两份数据合并为一个 target。
        无论哪种情况, 两份数据都不会因为叠加而产生额外的 target。
        """
        make_monitor_dataset(self.data_a, 0, self.ROWS)
        make_monitor_dataset(self.data_b, 0, self.ROWS)
        db_file = self._import_monitor("step", "false")
        rows = query_db(db_file, "SELECT DISTINCT target_id FROM trend_data")
        self.assertEqual(len(rows), 2)

        db_file = self._import_monitor("step", "true")
        rows = query_db(db_file, "SELECT DISTINCT target_id FROM trend_data")
        self.assertEqual(len(rows), 1)

    def test_concat_info_recorded(self):
        """拼接信息被记录到 concat_info 表"""
        make_monitor_dataset(self.data_a, 0, self.ROWS)
        make_monitor_dataset(self.data_b, 0, self.ROWS)
        db_file = self._import_monitor("rank", "false")

        info = self.get_concat_info(db_file)
        self.assertEqual(info["concat_dim"], "rank")
        self.assertEqual(info["dataset_count"], 2)
        self.assertEqual(info["boundary"], 1)
        self.assertEqual(info["dataset1_span"], 1)
        self.assertEqual(info["dataset2_span"], 1)

    def test_single_source_keeps_original_step_number(self):
        """不传对比数据时迭代步号沿用原始编号"""
        make_monitor_dataset(self.data_a, 0, self.ROWS)
        importer = DBImporter(
            db_path=self.db_dir,
            data_path=self.data_a,
            format="monitor",
            micro_step="false",
        )
        importer.import_monitor_data()
        db_file = os.path.join(self.db_dir, MONITOR_DB_NAME)

        self.assertEqual(self.get_rank_step_pairs(db_file), [(0, 10), (0, 20)])
        stats = self.get_global_stats(db_file)
        self.assertEqual(stats["min_step"], 10)
        self.assertEqual(stats["max_step"], 20)
        info = self.get_concat_info(db_file)
        self.assertEqual(info["concat_dim"], "none")
        self.assertEqual(info["dataset_count"], 1)
        # 单份数据没有分界点
        self.assertEqual(info["boundary"], 0)
        self.assertEqual(info["dataset2_span"], 0)

    def test_compare_data_without_csv_falls_back(self):
        """对比目录存在但没有可用 CSV 时降级为单数据源"""
        make_monitor_dataset(self.data_a, 0, self.ROWS)
        os.makedirs(f"{self.data_b}/Jan01_00-00-00-rank0-", exist_ok=True)
        db_file = self._import_monitor("step", "false")

        self.assertEqual(self.get_rank_step_pairs(db_file), [(0, 10), (0, 20)])
        info = self.get_concat_info(db_file)
        self.assertEqual(info["concat_dim"], "none")
        self.assertEqual(info["dataset_count"], 1)
        self.assertEqual(info["boundary"], 0)
        self.assertIsNone(info["compare_data_path"])

    def test_invalid_data_path_with_valid_compare_fails(self):
        """被测目录没有可用 CSV 时不导入对比数据, 也不产出数据库"""
        rank_dir = f"{self.data_a}/Jan01_00-00-00-rank0-"
        os.makedirs(rank_dir, exist_ok=True)
        # 缺少 vpp_stage 列, 所有行都会被跳过
        pd.DataFrame([{"name": "layer1", "step": 10, "min": 0.1, "max": 1.0}]).to_csv(
            f"{rank_dir}/actv_0-100.csv", index=False
        )
        make_monitor_dataset(self.data_b, 0, self.ROWS)
        importer = DBImporter(
            db_path=self.db_dir,
            data_path=self.data_a,
            format="monitor",
            micro_step="false",
            compare_data_path=self.data_b,
        )
        importer.import_monitor_data()
        self.assertFalse(os.path.exists(os.path.join(self.db_dir, MONITOR_DB_NAME)))


if __name__ == "__main__":
    unittest.main()
