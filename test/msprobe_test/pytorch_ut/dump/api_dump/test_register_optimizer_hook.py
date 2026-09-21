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
from unittest.mock import patch
import torch

from msprobe.pytorch.dump.api_dump.register_optimizer_hook import register_optimizer_hook


class DataCollector:
    def __init__(self):
        self.optimizer_status = ""


class TestRegisterOptimizerHook(unittest.TestCase):
    def test_register_optimizer_hook(self):
        data_collector = DataCollector()
        with patch("torch.nn.utils.clip_grad_norm_") as clip, \
                patch("torch.nn.utils.clip_grad_value_") as clip_value:
            clip.return_value = None
            clip_value.return_value = None
            register_optimizer_hook(data_collector)

            torch.nn.utils.clip_grad_norm_()
            self.assertEqual(data_collector.optimizer_status, "end_clip_grad")

            data_collector.optimizer_status = ""
            torch.nn.utils.clip_grad_value_()
            self.assertEqual(data_collector.optimizer_status, "end_clip_grad")
