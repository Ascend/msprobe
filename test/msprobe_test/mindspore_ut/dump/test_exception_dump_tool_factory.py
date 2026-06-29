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

import os
import sys

from unittest import TestCase
from unittest.mock import patch

from msprobe.core.dump.common_config import CommonConfig, BaseConfig
from msprobe.mindspore.dump.debugger.debugger_config import DebuggerConfig
from msprobe.mindspore.dump.exception_dump.kernel_graph_exception_dump import KernelGraphExceptionDump
from msprobe.core.common.file_utils import move_file


class TestKernelGraphExceptionDump(TestCase):
    @patch("msprobe.mindspore.dump.debugger.debugger_config.create_directory")
    def test_handle(self, _):
        json_config = {
            "task": "exception_dump",
            "dump_path": "/absolute_path",
            "rank": [],
            "step": [],
            "level": "L2"
        }

        common_config = CommonConfig(json_config)
        task_config = BaseConfig(json_config)
        config = DebuggerConfig(common_config, task_config)
        checker = KernelGraphExceptionDump(config)
        self.assertEqual(checker.dump_json["common_dump_settings"]["op_debug_mode"], 4)

        with patch("msprobe.mindspore.dump.exception_dump.kernel_graph_exception_dump.create_directory"), \
             patch("msprobe.mindspore.dump.exception_dump.kernel_graph_exception_dump.logger.info"), \
             patch("msprobe.mindspore.dump.exception_dump.kernel_graph_exception_dump.save_json") as mock_save_json:

            checker.handle()
            self.assertIn("kernel_graph_exception_check.json", mock_save_json.call_args_list[0][0][0])
            self.assertIn("kernel_graph_exception_check.json", os.environ.get("MINDSPORE_DUMP_CONFIG"))

        if "MINDSPORE_DUMP_CONFIG" in os.environ:
            del os.environ["MINDSPORE_DUMP_CONFIG"]