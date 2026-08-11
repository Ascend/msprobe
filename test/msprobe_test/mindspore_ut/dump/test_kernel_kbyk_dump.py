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

from packaging.version import parse as version_parse
from unittest import TestCase
from unittest.mock import patch

from msprobe.core.dump.common_config import CommonConfig, BaseConfig
from msprobe.mindspore.dump.debugger.debugger_config import DebuggerConfig
from msprobe.mindspore.dump.ms_config import StatisticsConfig
from msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump import KernelKbykDump

from collections import Counter

import mindspore as ms
ms_version = version_parse(ms.__version__)


class TestKernelKbykDump(TestCase):
    @patch("msprobe.mindspore.dump.debugger.debugger_config.create_directory")
    def test_handle(self, _):
        json_config = {
            "task": "statistics",
            "dump_path": "/absolute_path",
            "rank": [],
            "step": [0, 2],
            "level": "L2"
        }

        common_config = CommonConfig(json_config)
        task_config = StatisticsConfig(json_config)
        config = DebuggerConfig(common_config, task_config)
        dumper = KernelKbykDump(config)
        self.assertEqual(dumper.dump_json["common_dump_settings"]["iteration"], "0|2")

        os.environ["MS_ACL_DUMP_CFG_PATH"] = "path"
        with patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.create_directory"), \
             patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.logger.info") as mock_info, \
             patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.save_json") as mock_save_json:
            dumper.handle()
        self.assertIn("kernel_kbyk_dump.json", mock_save_json.call_args_list[0][0][0])
        mock_info.assert_called_with("/absolute_path/kernel_kbyk_dump.json has been created.")

        self.assertEqual(os.environ.get("MS_ACL_DUMP_CFG_PATH"), None)
        if "MINDSPORE_DUMP_CONFIG" in os.environ:
            del os.environ["MINDSPORE_DUMP_CONFIG"]

    @patch("msprobe.mindspore.dump.debugger.debugger_config.create_directory")
    def test_handle_when_async_dump_then_pass(self, _):
        json_config = {
            "task": "statistics",
            "dump_path": "/absolute_path",
            "rank": [],
            "step": [0, 2],
            "level": "L2",
            "async_dump": True
        }

        common_config = CommonConfig(json_config)
        task_config = StatisticsConfig(json_config)
        config = DebuggerConfig(common_config, task_config)
        dumper = KernelKbykDump(config)
        self.assertEqual(dumper.dump_json["e2e_dump_settings"]["enable"], False)

        os.environ["MS_ACL_DUMP_CFG_PATH"] = "path"
        with patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.create_directory"), \
             patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.logger.info") as mock_info, \
             patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.save_json") as mock_save_json:
            dumper.handle()
        self.assertIn("kernel_kbyk_dump.json", mock_save_json.call_args_list[0][0][0])
        mock_info.assert_called_with("/absolute_path/kernel_kbyk_dump.json has been created.")

        self.assertEqual(os.environ.get("MS_ACL_DUMP_CFG_PATH"), None)
        if "MINDSPORE_DUMP_CONFIG" in os.environ:
            del os.environ["MINDSPORE_DUMP_CONFIG"]

    @patch("msprobe.mindspore.dump.debugger.debugger_config.create_directory")
    def test_handle_when_device_then_pass(self, _):
        json_config = {
            "task": "statistics",
            "dump_path": "/absolute_path",
            "rank": [],
            "step": [0, 2],
            "level": "L2",
            "statistics": {
                "list": [],
                "data_mode": ["all"],
                "device": "device",
                "summary_mode": "statistics"
            }
        }

        common_config = CommonConfig(json_config)
        task_config = StatisticsConfig(json_config["statistics"])
        config = DebuggerConfig(common_config, task_config)
        dumper = KernelKbykDump(config)
        self.assertEqual(dumper.dump_json["e2e_dump_settings"]["stat_calc_mode"], "device")

        os.environ["MS_ACL_DUMP_CFG_PATH"] = "path"
        with patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.create_directory"), \
             patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.logger.info") as mock_info, \
             patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.save_json") as mock_save_json:
            dumper.handle()
        self.assertIn("kernel_kbyk_dump.json", mock_save_json.call_args_list[0][0][0])
        mock_info.assert_called_with("/absolute_path/kernel_kbyk_dump.json has been created.")

        self.assertEqual(os.environ.get("MS_ACL_DUMP_CFG_PATH"), None)
        if "MINDSPORE_DUMP_CONFIG" in os.environ:
            del os.environ["MINDSPORE_DUMP_CONFIG"]

    @patch("msprobe.mindspore.dump.debugger.debugger_config.create_directory")
    def test_handle_when_precision_then_pass(self, _):
        json_config = {
            "task": "statistics",
            "dump_path": "/absolute_path",
            "rank": [],
            "step": [0, 2],
            "level": "L2",
            "statistics": {
                "list": [],
                "data_mode": ["all"],
                "precision": "low",
                "summary_mode": "statistics"
            }
        }

        common_config = CommonConfig(json_config)
        task_config = StatisticsConfig(json_config["statistics"])
        config = DebuggerConfig(common_config, task_config)
        dumper = KernelKbykDump(config)
        self.assertEqual(dumper.dump_json["e2e_dump_settings"]["device_stat_precision_mode"], "low")

        os.environ["MS_ACL_DUMP_CFG_PATH"] = "path"
        with patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.create_directory"), \
             patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.logger.info") as mock_info, \
             patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.save_json") as mock_save_json:
            dumper.handle()
        self.assertIn("kernel_kbyk_dump.json", mock_save_json.call_args_list[0][0][0])
        mock_info.assert_called_with("/absolute_path/kernel_kbyk_dump.json has been created.")

        self.assertEqual(os.environ.get("MS_ACL_DUMP_CFG_PATH"), None)
        if "MINDSPORE_DUMP_CONFIG" in os.environ:
            del os.environ["MINDSPORE_DUMP_CONFIG"]

    @patch("msprobe.mindspore.dump.debugger.debugger_config.create_directory")
    def test_handle_when_default_then_pass(self, _):
        json_config = {
            "task": "statistics",
            "dump_path": "/absolute_path",
            "rank": [],
            "step": [0, 2],
            "level": "L2",
            "statistics": {
                "list": [],
                "data_mode": ["all"],
                "summary_mode": "statistics"
            }
        }

        common_config = CommonConfig(json_config)
        task_config = StatisticsConfig(json_config)
        config = DebuggerConfig(common_config, task_config)
        dumper = KernelKbykDump(config)
        self.assertEqual(dumper.dump_json["e2e_dump_settings"]["device_stat_precision_mode"], "high")
        self.assertEqual(dumper.dump_json["e2e_dump_settings"]["stat_calc_mode"], "host")
        self.assertEqual(dumper.dump_json["e2e_dump_settings"]["enable"], True)

        os.environ["MS_ACL_DUMP_CFG_PATH"] = "path"
        with patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.create_directory"), \
             patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.logger.info") as mock_info, \
             patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.save_json") as mock_save_json:
            dumper.handle()
        self.assertIn("kernel_kbyk_dump.json", mock_save_json.call_args_list[0][0][0])
        mock_info.assert_called_with("/absolute_path/kernel_kbyk_dump.json has been created.")

        self.assertEqual(os.environ.get("MS_ACL_DUMP_CFG_PATH"), None)
        if "MINDSPORE_DUMP_CONFIG" in os.environ:
            del os.environ["MINDSPORE_DUMP_CONFIG"]

    @patch("msprobe.mindspore.dump.debugger.debugger_config.create_directory")
    def test_handle_tensor(self, _):
        json_config = {
            "task": "tensor",
            "dump_path": "/absolute_path",
            "rank": [],
            "step": [0, 2],
            "level": "L2"
        }

        common_config = CommonConfig(json_config)
        task_config = BaseConfig(json_config)
        config = DebuggerConfig(common_config, task_config)
        dumper = KernelKbykDump(config)
        self.assertEqual(dumper.dump_json["common_dump_settings"]["iteration"], "0|2")

        os.environ["MS_ACL_DUMP_CFG_PATH"] = "path"
        with patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.create_directory"), \
             patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.logger.info") as mock_info, \
             patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.save_json") as mock_save_json:
            dumper.handle()
        mock_info.assert_called_with("/absolute_path/kernel_kbyk_dump.json has been created.")
        self.assertEqual(os.environ.get("MS_ACL_DUMP_CFG_PATH"), None)

    @patch("msprobe.mindspore.dump.debugger.debugger_config.create_directory")
    def test_handle_tensor_input(self, _):
        json_config = {
            "task": "tensor",
            "dump_path": "/absolute_path",
            "rank": [],
            "step": [0, 2],
            "data_mode": [
            "input"
            ],
            "level": "L2"

        }

        common_config = CommonConfig(json_config)
        task_config = BaseConfig(json_config)
        config = DebuggerConfig(common_config, task_config)
        dumper = KernelKbykDump(config)
        self.assertEqual(dumper.dump_json["common_dump_settings"]["iteration"], "0|2")

        os.environ["MS_ACL_DUMP_CFG_PATH"] = "path"
        with patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.create_directory"), \
             patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.logger.info") as mock_info, \
             patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.save_json") as mock_save_json:
            dumper.handle()

        mock_info.assert_called_with("/absolute_path/kernel_kbyk_dump.json has been created.")
        self.assertEqual(os.environ.get("MS_ACL_DUMP_CFG_PATH"), None)

    @patch("msprobe.mindspore.dump.debugger.debugger_config.create_directory")
    def test_handle_tensor_output(self, _):
        json_config = {
            "task": "tensor",
            "dump_path": "/absolute_path",
            "rank": [],
            "step": [0, 2],
            "data_mode": [
            "output"
            ],
            "level": "L2"

        }

        common_config = CommonConfig(json_config)
        task_config = BaseConfig(json_config)
        config = DebuggerConfig(common_config, task_config)
        dumper = KernelKbykDump(config)
        self.assertEqual(dumper.dump_json["common_dump_settings"]["iteration"], "0|2")

        os.environ["MS_ACL_DUMP_CFG_PATH"] = "path"
        with patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.create_directory"), \
             patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.logger.info") as mock_info, \
             patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.save_json") as mock_save_json:
            dumper.handle()
        mock_info.assert_called_with("/absolute_path/kernel_kbyk_dump.json has been created.")
        self.assertEqual(os.environ.get("MS_ACL_DUMP_CFG_PATH"), None)

    @patch("msprobe.mindspore.dump.debugger.debugger_config.create_directory")
    def test_handle_tensor_output_rank_0(self, _):
        json_config = {
            "task": "tensor",
            "dump_path": "/absolute_path",
            "rank": [0],
            "step": [0, 2],
            "data_mode": [
            "output"
            ],
            "level": "L2"

        }

        common_config = CommonConfig(json_config)
        task_config = BaseConfig(json_config)
        config = DebuggerConfig(common_config, task_config)
        dumper = KernelKbykDump(config)
        self.assertEqual(dumper.dump_json["common_dump_settings"]["iteration"], "0|2")

        os.environ["MS_ACL_DUMP_CFG_PATH"] = "path"
        with patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.create_directory"), \
             patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.logger.info") as mock_info, \
             patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.save_json") as mock_save_json:
            dumper.handle()
        mock_info.assert_called_with("/absolute_path/kernel_kbyk_dump.json has been created.")
        self.assertEqual(os.environ.get("MS_ACL_DUMP_CFG_PATH"), None)

    @patch("msprobe.mindspore.dump.debugger.debugger_config.create_directory")
    def test_handle_tensor_output_rank_0(self, _):
        json_config = {
            "task": "tensor",
            "dump_path": "/absolute_path",
            "rank": [0],
            "step": [0, 2],
            "data_mode": [
            "output"
            ],
            "level": "L2"

        }

        common_config = CommonConfig(json_config)
        task_config = BaseConfig(json_config)
        config = DebuggerConfig(common_config, task_config)
        dumper = KernelKbykDump(config)
        self.assertEqual(dumper.dump_json["common_dump_settings"]["iteration"], "0|2")

        os.environ["MS_ACL_DUMP_CFG_PATH"] = "path"
        with patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.create_directory"), \
             patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.logger.info") as mock_info, \
             patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.save_json") as mock_save_json:
            dumper.handle()
        mock_info.assert_called_with("/absolute_path/kernel_kbyk_dump.json has been created.")
        self.assertEqual(os.environ.get("MS_ACL_DUMP_CFG_PATH"), None)

    @patch("msprobe.mindspore.dump.debugger.debugger_config.create_directory")
    def test_handle_tensor_list_output(self, _):
        json_config = {
            "task": "tensor",
            "dump_path": "/absolute_path",
            "list": ['add'],
            "rank": [0],
            "step": [0, 2],
            "data_mode": [
            "output"
            ],
            "level": "L2"

        }

        common_config = CommonConfig(json_config)
        task_config = BaseConfig(json_config)
        config = DebuggerConfig(common_config, task_config)
        dumper = KernelKbykDump(config)
        self.assertEqual(dumper.dump_json["common_dump_settings"]["iteration"], "0|2")

        os.environ["MS_ACL_DUMP_CFG_PATH"] = "path"
        with patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.create_directory"), \
             patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.logger.info") as mock_info, \
             patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.save_json") as mock_save_json:
            dumper.handle()
        mock_info.assert_called_with("/absolute_path/kernel_kbyk_dump.json has been created.")
        self.assertEqual(os.environ.get("MS_ACL_DUMP_CFG_PATH"), None)

    @patch("msprobe.mindspore.dump.debugger.debugger_config.create_directory")
    def test_handle_tensor_list_input(self, _):
        json_config = {
            "task": "tensor",
            "dump_path": "/absolute_path",
            "list": ['add'],
            "rank": [0],
            "step": [0, 2],
            "data_mode": [
            "input"
            ],
            "level": "L2"

        }

        common_config = CommonConfig(json_config)
        task_config = BaseConfig(json_config)
        config = DebuggerConfig(common_config, task_config)
        dumper = KernelKbykDump(config)
        self.assertEqual(dumper.dump_json["common_dump_settings"]["iteration"], "0|2")

        os.environ["MS_ACL_DUMP_CFG_PATH"] = "path"
        with patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.create_directory"), \
             patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.logger.info") as mock_info, \
             patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.save_json") as mock_save_json:
            dumper.handle()
        mock_info.assert_called_with("/absolute_path/kernel_kbyk_dump.json has been created.")
        self.assertEqual(os.environ.get("MS_ACL_DUMP_CFG_PATH"), None)

    @patch("msprobe.mindspore.dump.debugger.debugger_config.create_directory")
    def test_handle_statistics(self, _):
        json_config = {
            "task": "statistics",
            "dump_path": "/absolute_path",
            "rank": [],
            "step": [0, 2],
            "level": "L2",
            "statistics": {
                "list": [],
                "data_mode": ["all"],
                "device": "host",
                "summary_mode": ["hash", "md5", "max", "mean"]
            }
        }

        common_config = CommonConfig(json_config)
        task_config = StatisticsConfig(json_config["statistics"])
        config = DebuggerConfig(common_config, task_config)
        dumper = KernelKbykDump(config)
        self.assertEqual(dumper.dump_json["e2e_dump_settings"]["stat_calc_mode"], "host")
        self.assertEqual(dumper.dump_json["common_dump_settings"]["saved_data"], "statistic")
        if ms_version > version_parse("2.7.0"):
            self.assertEqual(Counter(dumper.dump_json["common_dump_settings"]["statistic_category"]), Counter(["max", "hash", "hash:md5", "avg"]))
        else:
            self.assertEqual(Counter(dumper.dump_json["common_dump_settings"]["statistic_category"]), Counter(["max", "md5", "avg"]))
        os.environ["MS_ACL_DUMP_CFG_PATH"] = "path"
        with patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.create_directory"), \
             patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.logger.info") as mock_info, \
             patch("msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump.save_json") as mock_save_json:
            dumper.handle()
        self.assertIn("kernel_kbyk_dump.json", mock_save_json.call_args_list[0][0][0])
        mock_info.assert_called_with("/absolute_path/kernel_kbyk_dump.json has been created.")

        self.assertEqual(os.environ.get("MS_ACL_DUMP_CFG_PATH"), None)
        if "MINDSPORE_DUMP_CONFIG" in os.environ:
            del os.environ["MINDSPORE_DUMP_CONFIG"]