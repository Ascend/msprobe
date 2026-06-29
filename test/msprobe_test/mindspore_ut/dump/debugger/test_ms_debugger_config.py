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
from unittest.mock import patch

from msprobe.core.common.const import Const
from msprobe.core.common.log import logger
from msprobe.core.dump.common_config import CommonConfig
from msprobe.mindspore.dump.debugger.debugger_config import DebuggerConfig
from msprobe.mindspore.dump.ms_config import StatisticsConfig


class TestDebuggerConfig(unittest.TestCase):
    @patch.object(logger, "error")
    @patch("msprobe.mindspore.dump.debugger.debugger_config.create_directory")
    def test_init(self, _, mock_logger_error):
        json_config = {
            "dump_path": "/absolute_path",
            "rank": [],
            "step": [],
            "level": "L2"
        }
        common_config = CommonConfig(json_config)
        task_config = StatisticsConfig(json_config)
        debugger_config = DebuggerConfig(common_config, task_config)
        self.assertEqual(debugger_config.task, Const.STATISTICS)
        self.assertEqual(debugger_config.tensor_list, [])
