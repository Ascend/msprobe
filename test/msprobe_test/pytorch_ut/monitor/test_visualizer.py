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
from msprobe.pytorch.monitor.visualizer import HeatmapVisualizer


class TestHeatmapVisualizer(unittest.TestCase):
    def setUp(self):
        self.heatmap_visualizer = HeatmapVisualizer()

    def test_init(self):
        self.assertEqual(self.heatmap_visualizer.histogram_bins_num, 30)
        self.assertEqual(self.heatmap_visualizer.min_val, -1)
        self.assertEqual(self.heatmap_visualizer.max_val, 1)
        self.assertIsNotNone(self.heatmap_visualizer.histogram_edges)
        self.assertIsNone(self.heatmap_visualizer.histogram_sum_data_np)
        self.assertIsNone(self.heatmap_visualizer.cur_step_histogram_data)

    def test_pre_cal(self):
        tensor = torch.tensor([1., 2., 3., 4., 5.])
        self.heatmap_visualizer.pre_cal(tensor)
        expected_histogram = torch.tensor([0. for _ in range(29)] + [1.0])
        res = torch.allclose(self.heatmap_visualizer.cur_step_histogram_data, expected_histogram, atol=1e-1)
        self.assertTrue(res)

    def mock_summary_writer(self):
        class MockSummaryWriter:
            def add_image(self, tag, img, global_step, dataformats):
                self.called_with_tag = tag
                self.called_with_img = img
                self.called_with_global_step = global_step

        return MockSummaryWriter()

    def test_visualize(self):

        self.tag_name = "histogram"
        self.step = 10
        self.summary_writer = self.mock_summary_writer()

        # 准备测试数据
        self.heatmap_visualizer.cur_step_histogram_data = torch.tensor([1., 2., 3., 4., 5.])
        self.heatmap_visualizer.histogram_edges = [0.01, 0.02, 0.03, 0.04, 0.05]
        self.heatmap_visualizer.histogram_bins_num = 5

        # 调用方法
        self.heatmap_visualizer.visualize(self.tag_name, self.step, self.summary_writer)

        # 验证
        self.assertEqual(self.summary_writer.called_with_tag, self.tag_name)
        self.assertEqual(self.summary_writer.called_with_global_step, self.step)
        img = self.summary_writer.called_with_img
        self.assertEqual(list(img.shape), [4, 480, 640])


if __name__ == '__main__':
    unittest.main()
