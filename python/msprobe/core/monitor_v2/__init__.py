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

"""
monitor_v2: next-generation monitoring core for unifying
PyTorch / MindSpore monitor logic behind a single, clean abstraction.

Goal:
  - keep production behavior stable while iterating;
  - migrate mature logic from existing monitors in small steps;
  - enforce clearer layering and performance constraints.

Current structure:
  base.py
    - BaseMonitorV2: minimal shared monitor scaffold.
  trainer.py
    - TrainerMonitorV2: orchestrator for step gating and writer integration.
  factory.py
    - MonitorFactory: registry of framework-specific monitor classes.
  writer.py
    - CSVWriterV2: CSV output for per-step monitor data.
"""

from .trainer import TrainerMonitorV2

__all__ = ["TrainerMonitorV2"]
