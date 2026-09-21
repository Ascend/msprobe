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

import os
import unittest
from unittest.mock import patch, MagicMock

import torch

from msprobe.pytorch.monitor.module_metric import get_summary_writer_tag_name, squash_param_name, TensorMetrics, \
    Metric, MinMetric, MeanMetric, MaxMetric, NormMetric, ZerosMetric, NaNsMetric, IdentMetric, ShapeMetric, \
    DtypeMetric, get_metrics, get_sr_metric, get_entropy_metric


class TestModuleMetric(unittest.TestCase):
    def test_get_summary_writer_tag_name(self):
        module_or_param_name = "embeddings.weight"
        tag = "vpp:0"
        rank = None
        result = get_summary_writer_tag_name(module_or_param_name, tag, rank)
        self.assertEqual(result, f"{module_or_param_name}/{tag}")
        rank = 1
        result = get_summary_writer_tag_name(module_or_param_name, tag, rank)
        self.assertEqual(result, f"{module_or_param_name}/rank{rank}/{tag}")

    def test_squash_param_name(self):
        param_name = "embeddings.weight"
        enable = False
        result = squash_param_name(param_name, enable)
        self.assertEqual(result, param_name)
        param_name = "Module.embeddings.weight"
        enable = True
        result = squash_param_name(param_name, enable)
        self.assertEqual(result, "embeddings.weight")
        param_name = "embeddings."
        enable = True
        result = squash_param_name(param_name, enable)
        self.assertEqual(result, param_name)

    def test_get_metrics(self):
        tag2tensor = {}
        eps = 0.1
        out_dict = get_metrics(["op1"], tag2tensor, eps, out_dict=None)
        self.assertEqual(out_dict, {})

    def test_get_sr_metric(self):
        tag2tensor = {}
        out_dict = None
        get_sr_metric(tag2tensor, out_dict)
        self.assertEqual(out_dict, None)

        tag2tensor = {
            "0:conv1.weight/sr": torch.randn(3, 4),
            "0:fc1.weight": torch.randn(3, 4),  # 不含 sr，应该跳过
        }
        out_dict = {}
        get_sr_metric(tag2tensor, out_dict)
        # 只对含 'sr' 的 tag 生成结果
        self.assertIn("0:conv1.weight/sr", out_dict)
        self.assertNotIn("0:fc1.weight", out_dict)
        # 检查包含的 key
        self.assertIn("sr", out_dict["0:conv1.weight/sr"])
        self.assertIn("kernel_norm", out_dict["0:conv1.weight/sr"])
        # 检查 sr 和 kernel_norm 值类型
        sr_val = out_dict["0:conv1.weight/sr"]["sr"]
        kn_val = out_dict["0:conv1.weight/sr"]["kernel_norm"]
        self.assertIsInstance(sr_val, torch.Tensor)
        self.assertIsInstance(kn_val, torch.Tensor)
        # 可进一步检查数值合理性
        self.assertGreaterEqual(sr_val.item(), 0)
        self.assertGreaterEqual(kn_val.item(), 0)

    def test_get_entropy_metric(self):
        tag2tensor = {}
        out_dict = None
        get_entropy_metric(tag2tensor, out_dict)
        self.assertEqual(out_dict, None)

        tag2tensor = {
            "0:layer1.output": torch.randn(3, 3),
        }
        out_dict = {}
        get_entropy_metric(tag2tensor, out_dict)
        # 检查每个 tag 是否生成结果
        for tag in tag2tensor:
            self.assertIn(tag, out_dict)
            self.assertIn("entropy", out_dict[tag])
            self.assertIn("softmax_max", out_dict[tag])
            entropy_val = out_dict[tag]["entropy"]
            softmax_max_val = out_dict[tag]["softmax_max"]
            self.assertIsInstance(entropy_val, torch.Tensor)
            self.assertIsInstance(softmax_max_val, torch.Tensor)


class TestTenosrMetric(unittest.TestCase):
    def test_stat_insert(self):
        tm = TensorMetrics()
        tensor = torch.tensor([1.0, 2.0, 3.0])
        tm.stat_insert(
            tensor=tensor,
            stat_ops=["norm", "max", "min", "mean"],
            module_name="layer1",
            tensor_name="weight",
            rank=0,
        )

        # 计算得到的 key
        for op in ["norm", "max", "min", "mean"]:
            key = f"layer1/rank0/weight_{op}"
            self.assertIn(key, tm.metrics)
            self.assertEqual(len(tm.metrics[key]), 1)

    def test_flush(self):
        tm = TensorMetrics()
        tensor = torch.tensor([3.0, 4.0])

        tm.stat_insert(
            tensor=tensor,
            stat_ops=["norm"],
            module_name="layer1",
            tensor_name="bias",
            rank=0,
        )

        # mock 一个简单的 writer（不用 patch，只定义 add_scalar）
        class Writer:
            def __init__(self):
                self.records = []

            def add_scalar(self, tag, value, global_step):
                self.records.append((tag, value, global_step))

        writer = Writer()
        tm.flush(writer)

        # 检查是否写入
        self.assertEqual(len(writer.records), 1)


class TestMinMetric(unittest.TestCase):
    def test_get_metric_value(self):
        metric = MinMetric()
        tensor = torch.tensor([1.0, 2.0, 3.0])
        eps = 1e-6
        result = metric.get_metric_value(tensor, eps)
        self.assertEqual(result, 1.0)


class TestMeanMetric(unittest.TestCase):
    def test_get_metric_value(self):
        metric = MeanMetric()
        tensor = torch.tensor([1.0, 2.0, 3.0])
        eps = 1e-6
        result = metric.get_metric_value(tensor, eps)
        self.assertEqual(result, 2.0)


class TestMaxMetric(unittest.TestCase):
    def test_get_metric_value(self):
        metric = MaxMetric()
        tensor = torch.tensor([1.0, 2.0, 3.0])
        eps = 1e-6
        result = metric.get_metric_value(tensor, eps)
        self.assertEqual(result, 3.0)


class TestNormMetric(unittest.TestCase):
    def test_get_metric_value(self):
        metric = NormMetric()
        tensor = torch.tensor([1.0, 2.0, 3.0])
        eps = 1e-6
        result = metric.get_metric_value(tensor, eps)
        self.assertEqual(result, torch.norm(tensor.to(torch.float64), p=2))


class TestZerosMetric(unittest.TestCase):
    def test_get_metric_value(self):
        metric = ZerosMetric()
        tensor = torch.tensor([1.0, 2.0, 3.0])
        eps = 1e-6
        result = metric.get_metric_value(tensor, eps)
        self.assertEqual(result, 0)


class TestNaNsMetric(unittest.TestCase):
    def test_get_metric_value(self):
        metric = NaNsMetric()
        tensor = torch.tensor([1.0, 2.0, 3.0])
        eps = 1e-6
        result = metric.get_metric_value(tensor, eps)
        self.assertEqual(result, 0)


class TestIdentMetric(unittest.TestCase):
    def test_get_metric_value(self):
        metric = IdentMetric()
        tensor = torch.tensor([1.0, 2.0, 3.0])
        eps = 1e-6
        result = metric.get_metric_value(tensor, eps)
        self.assertEqual(result, None)


class TestShapeMetric(unittest.TestCase):
    def test_get_metric_value(self):
        metric = ShapeMetric()
        tensor = torch.tensor([1.0, 2.0, 3.0])
        eps = 1e-6
        result = metric.get_metric_value(tensor, eps)
        self.assertEqual(result, torch.Size([3]))


class TestDtypeMetric(unittest.TestCase):
    def test_get_metric_value(self):
        metric = DtypeMetric()
        tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
        eps = 1e-6
        result = metric.get_metric_value(tensor, eps)
        self.assertEqual(result, torch.bfloat16)