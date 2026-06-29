#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
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

from unittest import TestCase
from unittest.mock import patch

from msprobe.core.dump.common_config import CommonConfig, BaseConfig
from msprobe.mindspore.dump.debugger.debugger_config import DebuggerConfig
from msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump import KernelKbykDump
from msprobe.mindspore.dump.task_handler_factory import TaskHandlerFactory
from msprobe.mindspore.common.const import Const


class TestTaskHandlerFactory(TestCase):
    @patch("msprobe.mindspore.dump.debugger.debugger_config.create_directory")
    def test_create(self, _):
        class HandlerFactory:
            def create(self):
                return None

        tasks = {"statistics": HandlerFactory}

        json_config = {
            "task": "statistics",
            "dump_path": "/absolute_path",
            "rank": [],
            "step": [],
            "level": "L2"
        }

        common_config = CommonConfig(json_config)
        task_config = BaseConfig(json_config)
        config = DebuggerConfig(common_config, task_config)
        config.execution_mode = Const.GRAPH_GE_MODE

        handler = TaskHandlerFactory.create(config)
        self.assertTrue(isinstance(handler, tuple))
        self.assertTrue(isinstance(handler[0], KernelKbykDump))

        with patch("msprobe.mindspore.dump.task_handler_factory.TaskHandlerFactory.tasks", new=tasks):
            with self.assertRaises(Exception) as context:
                TaskHandlerFactory.create(config)
            self.assertEqual(str(context.exception), "Can not find task handler")

        config.task = "Free_benchmark"
        with self.assertRaises(Exception) as context:
            TaskHandlerFactory.create(config)
        self.assertEqual(str(context.exception), "Valid task is needed.")
