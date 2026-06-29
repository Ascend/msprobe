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

import unittest
from unittest.mock import patch, MagicMock

from msprobe.core.common.const import Const, MsgConst
from msprobe.core.common.runtime import Runtime
from msprobe.core.dump.common_config import CommonConfig
from msprobe.core.dump.debugger.precision_debugger import BasePrecisionDebugger
from msprobe.mindspore.common.const import Const as MsConst
from msprobe.mindspore.dump.cell_processor import CellProcessor
from msprobe.mindspore.dump.debugger.debugger_config import DebuggerConfig
from msprobe.mindspore.dump.debugger.precision_debugger import PrecisionDebugger
from msprobe.mindspore.dump.dump_processor.hook_cell.hook_cell import HOOKCell
from msprobe.mindspore.dump.ms_config import StatisticsConfig


class TestPrecisionDebugger(unittest.TestCase):

    @patch("msprobe.mindspore.dump.debugger.debugger_config.create_directory")
    def test_start(self, _):
        PrecisionDebugger._instance = None

        class Handler:
            called = False

            def handle(self):
                Handler.called = True

        json_config = {
            "task": "statistics",
            "dump_path": "/absolute_path",
            "rank": [],
            "step": [],
            "level": "L1",
            "async_dump": False
        }

        common_config = CommonConfig(json_config)
        task_config = StatisticsConfig(json_config)
        handler = Handler()

        mock_get_mode = MagicMock()
        mock_parse_json_config = MagicMock()
        with patch.object(BasePrecisionDebugger, "_parse_config_path", new=mock_parse_json_config), \
             patch.object(PrecisionDebugger, "_get_execution_mode", new=mock_get_mode), \
             patch("msprobe.mindspore.dump.debugger.precision_debugger.TaskHandlerFactory.create", return_value=handler), \
             patch("msprobe.mindspore.dump.debugger.precision_debugger.set_register_backward_hook_functions"):
            mock_get_mode.return_value = MsConst.GRAPH_GE_MODE
            mock_parse_json_config.return_value = [common_config, task_config]
            debugger = PrecisionDebugger()
            self.assertEqual(Runtime.step_count, 0)
            self.assertFalse(Runtime.is_running)
            debugger.start()
            self.assertTrue(Runtime.is_running)
            self.assertTrue(isinstance(debugger.config, DebuggerConfig))
            self.assertTrue(Handler.called)

            mock_get_mode.return_value = MsConst.PYNATIVE_MODE
            with patch("msprobe.mindspore.dump.debugger.precision_debugger.MindsporeService") as mock_Service, \
                 patch("msprobe.mindspore.dump.debugger.precision_debugger.set_register_backward_hook_functions"):
                debugger = PrecisionDebugger()
                debugger.start()
                service = mock_Service.return_value
                mock_Service.assert_called_with(debugger.config)
                service.start.assert_called_with(None, None, None)

        PrecisionDebugger._instance = None
        with self.assertRaises(Exception) as context:
            debugger.start()
        self.assertEqual(str(context.exception), MsgConst.NOT_CREATED_INSTANCE)

    def test_stop_step(self):
        class MockPrecisionDebugger:
            def __init__(self):
                self.task = Const.TENSOR
                self.service = None
                self.config = MagicMock()
                self.config.level_ori = MagicMock()
                self.config.level_ori.return_value = Const.LEVEL_L1
        PrecisionDebugger._instance = None
        with self.assertRaises(Exception) as context:
            PrecisionDebugger.stop()
        self.assertEqual(str(context.exception), MsgConst.NOT_CREATED_INSTANCE)
        with self.assertRaises(Exception) as context:
            PrecisionDebugger.step()
        self.assertEqual(str(context.exception), MsgConst.NOT_CREATED_INSTANCE)
        PrecisionDebugger._instance = MockPrecisionDebugger()
        Runtime.is_running = True
        PrecisionDebugger.stop()
        self.assertFalse(Runtime.is_running)
        Runtime.step_count = 0
        PrecisionDebugger.step()
        self.assertEqual(Runtime.step_count, 1)
        Runtime.step_count = 0

        HOOKCell.cell_count["api"] = 1
        PrecisionDebugger.step()
        self.assertEqual(HOOKCell.cell_count["api"], 0)

        with patch.object(CellProcessor, "reset_cell_stats") as mock_reset_cell:
            PrecisionDebugger.step()
        mock_reset_cell.assert_called_once()
