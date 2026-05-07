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

import unittest
from unittest.mock import patch, MagicMock

from msprobe.core.common.utils import Const
from msprobe.core.dump.data_dump.data_collector import DataCollector
from msprobe.pytorch.dump.debugger.debugger_config import DebuggerConfig
from msprobe.pytorch.dump.pt_config import parse_json_config


class TestDataCollector(unittest.TestCase):
    def setUp(self):
        mock_json_data = {
            "dump_path": "./ut_dump",
        }
        with patch("msprobe.pytorch.dump.pt_config.load_json", return_value=mock_json_data):
            common_config, task_config = parse_json_config("./config.json", Const.STATISTICS)
        config = DebuggerConfig(common_config, task_config, Const.STATISTICS, "./ut_dump", "L1")
        self.data_collector = DataCollector(config)

    def test_dump_data_dir(self):
        self.assertEqual(self.data_collector.dump_data_dir, None)

        self.data_collector.data_writer.dump_tensor_data_dir = "./test_dump"
        self.assertEqual(self.data_collector.dump_data_dir, "./test_dump")

    def test_dump_file_path(self):
        self.assertEqual(self.data_collector.dump_file_path, None)

        self.data_collector.data_writer.dump_file_path = "./test_dump/dump.json"
        self.assertEqual(self.data_collector.dump_file_path, "./test_dump/dump.json")

    def test_scope_none_and_pid_match(self):
        mock_name = "test_module"
        current_pid = os.getpid()
        result = self.data_collector.check_scope_and_pid(None, mock_name, current_pid)
        self.assertTrue(result)

    def test_scope_valid_and_pid_match(self):
        mock_scope = MagicMock()
        mock_scope.check.return_value = True
        mock_name = "valid_module"
        current_pid = os.getpid()
        result = self.data_collector.check_scope_and_pid(mock_scope, mock_name, current_pid)
        self.assertTrue(result)
        mock_scope.check.assert_called_once_with(mock_name)

    def test_scope_invalid_and_pid_match(self):
        mock_scope = MagicMock()
        mock_scope.check.return_value = False
        mock_name = "invalid_module"
        current_pid = os.getpid()
        result = self.data_collector.check_scope_and_pid(mock_scope, mock_name, current_pid)
        self.assertFalse(result)

    def test_scope_valid_but_pid_mismatch(self):
        mock_scope = MagicMock()
        mock_scope.check.return_value = True
        mock_name = "valid_module"
        fake_pid = os.getpid() + 1
        result = self.data_collector.check_scope_and_pid(mock_scope, mock_name, fake_pid)
        self.assertFalse(result)

    def test_scope_none_but_pid_mismatch(self):
        mock_name = "test_module"
        fake_pid = os.getpid() + 1
        result = self.data_collector.check_scope_and_pid(None, mock_name, fake_pid)
        self.assertFalse(result)

    def test_reset_status(self):
        self.data_collector.optimizer_status = "test_optimizer_status"
        self.data_collector.reset_status()

        self.assertEqual(self.data_collector.optimizer_status, "")
        self.assertEqual(
            self.data_collector.optimizer_status_first_start,
            {Const.OPTIMIZER: True, Const.CLIP_GRAD: True}
        )
        self.assertEqual(self.data_collector.backward_module_names, {})

    def test_update_api_or_module_name(self):
        self.assertEqual(self.data_collector.data_processor.current_api_or_module_name, None)

        self.data_collector.update_api_or_module_name("test_api_name")
        self.assertEqual(self.data_collector.data_processor.current_api_or_module_name, "test_api_name")

    def test_write_json(self):
        self.data_collector.data_writer = MagicMock()

        self.data_collector.write_json()
        self.data_collector.data_writer.write_json.assert_called_once()

    def test_write_json_at_exit_with_async_dump_tensor(self):
        self.data_collector.data_processor = MagicMock()
        self.data_collector.data_writer = MagicMock()
        self.data_collector.config.async_dump = True
        self.data_collector.config.task = "tensor"

        self.data_collector.write_json_at_exit()

        self.data_collector.data_processor.dump_async_data.assert_called_once()
        self.data_collector.data_writer.write_json.assert_called_once()

    def test_write_json_at_exit_with_no_async_dump(self):
        self.data_collector.data_processor = MagicMock()
        self.data_collector.data_writer = MagicMock()
        self.data_collector.config.async_dump = False
        self.data_collector.config.task = "tensor"

        self.data_collector.write_json_at_exit()

        self.data_collector.data_processor.dump_async_data.assert_not_called()
        self.data_collector.data_writer.write_json.assert_called_once()

    def test_write_json_at_exit_with_statistics(self):
        self.data_collector.data_processor = MagicMock()
        self.data_collector.data_writer = MagicMock()
        self.data_collector.config.async_dump = True
        self.data_collector.config.task = "statistics"

        self.data_collector.write_json_at_exit()

        self.data_collector.data_processor.dump_async_data.assert_not_called()
        self.data_collector.data_writer.write_json.assert_called_once()

    def test_update_construct_without_construct(self):
        self.data_collector.data_writer = MagicMock()

        self.data_collector.config.level = "L1"
        self.data_collector.update_construct("test")
        self.data_collector.data_writer.update_construct.assert_not_called()

    def test_update_construct_with_first_start(self):
        self.data_collector.module_processor = MagicMock()
        self.data_collector.data_writer = MagicMock()
        self.data_collector.config.level = "L0"
        self.data_collector.optimizer_status = "optimizer"
        self.data_collector.optimizer_status_first_start = {"optimizer": True}

        self.data_collector.update_construct("test_name")
        calls = [
            unittest.mock.call({"optimizer": None}),
            unittest.mock.call({"test_name": "optimizer"}),
            unittest.mock.call(self.data_collector.module_processor.module_node)
        ]
        self.data_collector.data_writer.update_construct.assert_has_calls(calls)

    def test_update_construct_with_not_first_start(self):
        self.data_collector.module_processor = MagicMock()
        self.data_collector.data_writer = MagicMock()
        self.data_collector.config.level = "L0"
        self.data_collector.optimizer_status = "clip_grad"
        self.data_collector.optimizer_status_first_start = {"clip_grad": False}

        self.data_collector.update_construct("test_name")
        calls = [
            unittest.mock.call({"test_name": "clip_grad"}),
            unittest.mock.call(self.data_collector.module_processor.module_node)
        ]
        self.data_collector.data_writer.update_construct.assert_has_calls(calls)

    def test_update_construct_with_module_prefix(self):
        self.data_collector.module_processor = MagicMock()
        self.data_collector.data_writer = MagicMock()
        self.data_collector.config.level = "mix"
        self.data_collector.optimizer_status = "other_status"
        test_name = "Module_test_name"

        self.data_collector.update_construct(test_name)
        self.data_collector.data_writer.update_construct.assert_called_with(
            self.data_collector.module_processor.module_node
        )

    def test_handle_data(self):
        with patch.object(DataCollector, "update_data") as mock_update_data, \
                patch.object(DataCollector, "write_json") as mock_write_json, \
                patch("msprobe.core.dump.data_dump.json_writer.DataWriter.flush_data_periodically") as mock_flush:
            self.data_collector.handle_data("Tensor.add", {"min": 0})
            mock_update_data.assert_called_with("Tensor.add", {"min": 0})

            mock_flush.assert_called()
            mock_write_json.assert_not_called()

            mock_update_data.reset_mock()
            mock_flush.reset_mock()
            self.data_collector.handle_data("Tensor.add", {}, flush=True)
            mock_update_data.assert_not_called()
            mock_flush.assert_not_called()
            mock_write_json.assert_called()


class TestForwardDataCollect(unittest.TestCase):
    def setUp(self):
        mock_json_data = {
            "dump_path": "./test_fwd_dump",
        }
        with patch("msprobe.pytorch.dump.pt_config.load_json", return_value=mock_json_data):
            common_config, task_config = parse_json_config("./config.json", Const.STATISTICS)
        config = DebuggerConfig(common_config, task_config, Const.STATISTICS, "./test_fwd_dump", "L1")
        self.data_collector = DataCollector(config)

        self.data_collector.update_construct = MagicMock()
        self.data_collector.config = MagicMock()
        self.data_collector.data_processor = MagicMock()
        self.data_collector.scope = "test_scope"
        self.data_collector.check_scope_and_pid = MagicMock()
        self.data_collector._should_collect_by_risk_level = MagicMock()
        self.data_collector.handle_data = MagicMock()

        self.Const = MagicMock()
        self.Const.FREE_BENCHMARK = "free_benchmark"
        self.Const.TENSOR = "tensor"
        self.Const.FORWARD = "forward"
        self.Const.BACKWARD = "backward"
        self.Const.STRUCTURE = "structure"
        self.Const.LEVEL_L2 = "L2"

    def test_forward_input_with_scope_pid_check_fail(self):
        self.data_collector.config.task = self.Const.TENSOR
        self.data_collector.check_scope_and_pid.return_value = False
        self.data_collector._should_collect_by_risk_level.return_value = False
        self.data_collector.forward_input_data_collect(
            "test", "module1", 123, "input_output"
        )

        self.data_collector.data_processor.analyze_forward_input.assert_not_called()

    def test_forward_input_with_level_l2(self):
        self.data_collector.config.task = self.Const.TENSOR
        self.data_collector.config.level = self.Const.LEVEL_L2
        self.data_collector.check_scope_and_pid.return_value = True

        self.data_collector.forward_input_data_collect(
            "test", "module1", 123, "input_output"
        )

        self.data_collector.handle_data.assert_not_called()

    def test_forward_output_with_scope_check_fail(self):
        self.data_collector.check_scope_and_pid.return_value = False
        self.data_collector._should_collect_by_risk_level.return_value = False
        self.data_collector.forward_output_data_collect("test", "module", 123, "data")
        self.data_collector.data_processor.analyze_forward_output.assert_not_called()

    def test_forward_output_with_structure_task(self):
        self.data_collector.config.task = self.Const.STRUCTURE
        self.data_collector.forward_output_data_collect("test", "module", 123, "data")
        self.data_collector.data_processor.analyze_forward_output.assert_not_called()

    def test_forward_output_with_level_l2(self):
        self.data_collector.config.level = self.Const.LEVEL_L2
        self.data_collector.forward_output_data_collect("test", "module", 123, "data")
        self.data_collector.handle_data.assert_not_called()

    def test_forward_output_normal(self):
        mock_data = {"key": "value"}
        self.data_collector.data_processor.analyze_forward_output.return_value = mock_data
        self.data_collector.forward_output_data_collect("test", "module", 123, "data")
        self.data_collector.handle_data.assert_called_once_with(
            "test",
            mock_data,
            flush=self.data_collector.data_processor.is_terminated
        )

    def test_forward_with_scope_check_fail(self):
        self.data_collector.check_scope_and_pid.return_value = False
        self.data_collector._should_collect_by_risk_level.return_value = False
        self.data_collector.forward_data_collect("test", "module", 123, "data")
        self.data_collector.data_processor.analyze_forward.assert_not_called()

    def test_forward_with_structure_task(self):
        self.data_collector.config.task = self.Const.STRUCTURE
        self.data_collector.forward_data_collect("test", "module", 123, "data")
        self.data_collector.data_processor.analyze_forward.assert_not_called()

    def test_forward_normal(self):
        mock_data = {"key": "value"}
        self.data_collector.data_processor.analyze_forward.return_value = mock_data
        self.data_collector.forward_data_collect("test", "module", 123, "data")
        self.data_collector.handle_data.assert_called_once_with(
            "test",
            mock_data,
            flush=self.data_collector.data_processor.is_terminated
        )


class TestBackwardDataCollector(unittest.TestCase):
    def setUp(self):
        mock_json_data = {
            "dump_path": "./test_bwd_dump",
        }
        with patch("msprobe.pytorch.dump.pt_config.load_json", return_value=mock_json_data):
            common_config, task_config = parse_json_config("./config.json", Const.STATISTICS)
        config = DebuggerConfig(common_config, task_config, Const.STATISTICS, "./test_bwd_dump", "L1")
        self.data_collector = DataCollector(config)

        self.data_collector.config = MagicMock()
        self.data_collector.data_processor = MagicMock()
        self.data_collector.scope = "test_scope"
        self.data_collector.check_scope_and_pid = MagicMock(return_value=True)
        self.data_collector._should_collect_by_risk_level = MagicMock(return_value=True)
        self.data_collector.handle_data = MagicMock()
        self.data_collector.update_construct = MagicMock()
        self.data_collector.backward_module_names = {}

        self.Const = MagicMock()
        self.Const.STRUCTURE = "structure"
        self.Const.TENSOR = "tensor"
        self.Const.LEVEL_L2 = "L2"
        self.Const.SEP = "."
        self.Const.MODULE_PREFIX = ["module"]

    def test_backward_with_scope_check_fail(self):
        self.data_collector.check_scope_and_pid.return_value = False
        self.data_collector._should_collect_by_risk_level.return_value = False
        self.data_collector.backward_data_collect("test", "module", 123, "data")
        self.data_collector.data_processor.analyze_backward.assert_not_called()

    def test_backward_with_level_l2(self):
        self.data_collector.config.level = self.Const.LEVEL_L2
        self.data_collector.backward_data_collect("test", "module", 123, "data")
        self.data_collector.handle_data.assert_not_called()

    def test_backward_data_module_prefix_match(self):
        self.data_collector.check_scope_and_pid.return_value = True
        self.data_collector.config.task = self.Const.TENSOR
        self.data_collector.config.level = "L1"
        mock_data = {"key": "value"}
        self.data_collector.data_processor.analyze_backward.return_value = mock_data
        test_name = "Module.layer1.backward"
        self.data_collector.backward_data_collect(test_name, "module", 123, "data")
        self.assertEqual(self.data_collector.backward_module_names, {"Module": True})

    def test_backward_input_with_structure_task(self):
        self.data_collector.config.task = self.Const.STRUCTURE
        self.data_collector.backward_input_data_collect("test", "module", 123, "data")
        self.data_collector.data_processor.analyze_backward_input.assert_not_called()

    def test_backward_input_with_normal(self):
        mock_data = {"key": "value"}
        self.data_collector.data_processor.analyze_backward_input.return_value = mock_data
        self.data_collector.backward_input_data_collect("test", "module", 123, "data")

    def test_backward_output_with_scope_check_fail(self):
        self.data_collector.check_scope_and_pid.return_value = False
        self.data_collector._should_collect_by_risk_level.return_value = False
        self.data_collector.backward_output_data_collect("test", "module", 123, "data")
        self.data_collector.data_processor.analyze_backward_output.assert_not_called()

    def test_backward_output_with_recompute(self):
        mock_data = {"key": "value"}
        self.data_collector.data_processor.analyze_backward_output.return_value = mock_data
        self.data_collector.backward_output_data_collect("test", "module", 123, "data")


class TestShouldCollectByRiskLevel(unittest.TestCase):
    """测试 _should_collect_by_risk_level 方法"""

    def setUp(self):
        mock_json_data = {
            "dump_path": "./test_risk_dump",
        }
        with patch("msprobe.pytorch.dump.pt_config.load_json", return_value=mock_json_data):
            common_config, task_config = parse_json_config("./config.json", Const.STATISTICS)
        config = DebuggerConfig(common_config, task_config, Const.STATISTICS, "./test_risk_dump", "L1")
        self.data_collector = DataCollector(config)
        self.data_collector.config.framework = Const.PT_FRAMEWORK

    def test_not_l1_level(self):
        """测试非L1级别时，不进行风险等级判断，返回True"""
        self.data_collector.config.level = "L0"
        result = self.data_collector._should_collect_by_risk_level("Functional.conv2d.0.forward")
        self.assertTrue(result)

    def test_no_risk_level_config(self):
        """测试未配置risk_level时，返回True"""
        self.data_collector.config.level = Const.LEVEL_L1
        if hasattr(self.data_collector.config, 'risk_level'):
            delattr(self.data_collector.config, 'risk_level')
        result = self.data_collector._should_collect_by_risk_level("Functional.conv2d.0.forward")
        self.assertTrue(result)

    def test_risk_level_all(self):
        """测试risk_level为ALL时，返回True"""
        self.data_collector.config.level = Const.LEVEL_L1
        self.data_collector.config.risk_level = Const.RISK_LEVEL_ALL
        result = self.data_collector._should_collect_by_risk_level("Functional.conv2d.0.forward")
        self.assertTrue(result)

    def test_risk_level_core_with_core_api(self):
        """测试CORE级别，高风险API返回True"""
        self.data_collector.config.level = Const.LEVEL_L1
        self.data_collector.config.risk_level = Const.RISK_LEVEL_CORE
        # matmul 是高风险API
        with patch('msprobe.pytorch.dump.api_dump.api_risk_level.get_api_risk_level') as mock_get:
            mock_get.return_value = Const.RISK_LEVEL_CORE
            result = self.data_collector._should_collect_by_risk_level("Functional.matmul.0.forward")
            self.assertTrue(result)
            mock_get.assert_called_once_with("matmul")

    def test_risk_level_core_with_low_risk_api(self):
        """测试CORE级别，低风险API返回False"""
        self.data_collector.config.level = Const.LEVEL_L1
        self.data_collector.config.risk_level = Const.RISK_LEVEL_CORE
        # reshape 是低风险API
        with patch('msprobe.pytorch.dump.api_dump.api_risk_level.get_api_risk_level') as mock_get:
            mock_get.return_value = Const.RISK_LEVEL_LOW
            result = self.data_collector._should_collect_by_risk_level("Tensor.reshape.0.forward")
            self.assertFalse(result)
            mock_get.assert_called_once_with("reshape")

    def test_risk_level_focus_with_core_api(self):
        """测试FOCUS级别，高风险API返回True"""
        self.data_collector.config.level = Const.LEVEL_L1
        self.data_collector.config.risk_level = Const.RISK_LEVEL_FOCUS
        with patch('msprobe.pytorch.dump.api_dump.api_risk_level.get_api_risk_level') as mock_get:
            mock_get.return_value = Const.RISK_LEVEL_CORE
            result = self.data_collector._should_collect_by_risk_level("Functional.matmul.0.forward")
            self.assertTrue(result)

    def test_risk_level_focus_with_focus_api(self):
        """测试FOCUS级别，中等风险API返回True"""
        self.data_collector.config.level = Const.LEVEL_L1
        self.data_collector.config.risk_level = Const.RISK_LEVEL_FOCUS
        with patch('msprobe.pytorch.dump.api_dump.api_risk_level.get_api_risk_level') as mock_get:
            mock_get.return_value = Const.RISK_LEVEL_FOCUS
            result = self.data_collector._should_collect_by_risk_level("Functional.unknown_api.0.forward")
            self.assertTrue(result)

    def test_risk_level_focus_with_low_risk_api(self):
        """测试FOCUS级别，低风险API返回False"""
        self.data_collector.config.level = Const.LEVEL_L1
        self.data_collector.config.risk_level = Const.RISK_LEVEL_FOCUS
        with patch('msprobe.pytorch.dump.api_dump.api_risk_level.get_api_risk_level') as mock_get:
            mock_get.return_value = Const.RISK_LEVEL_LOW
            result = self.data_collector._should_collect_by_risk_level("Tensor.reshape.0.forward")
            self.assertFalse(result)

    def test_backward_direction(self):
        """测试backward方向的API"""
        self.data_collector.config.level = Const.LEVEL_L1
        self.data_collector.config.risk_level = Const.RISK_LEVEL_CORE
        with patch('msprobe.pytorch.dump.api_dump.api_risk_level.get_api_risk_level') as mock_get:
            mock_get.return_value = Const.RISK_LEVEL_CORE
            result = self.data_collector._should_collect_by_risk_level("Functional.matmul.0.backward")
            self.assertTrue(result)
            mock_get.assert_called_once_with("matmul")

    def test_api_name_with_dots(self):
        """测试API名称中包含点的情况，如 linalg.matmul"""
        self.data_collector.config.level = Const.LEVEL_L1
        self.data_collector.config.risk_level = Const.RISK_LEVEL_CORE
        with patch('msprobe.pytorch.dump.api_dump.api_risk_level.get_api_risk_level') as mock_get:
            mock_get.return_value = Const.RISK_LEVEL_CORE
            result = self.data_collector._should_collect_by_risk_level("Functional.linalg.matmul.0.forward")
            self.assertTrue(result)
            # 验证提取的api_name是 linalg.matmul（包含点）
            mock_get.assert_called_once_with("linalg.matmul")

    def test_api_name_extraction_simple(self):
        """测试简单API名称的提取：Functional.conv2d.0.forward -> conv2d"""
        self.data_collector.config.level = Const.LEVEL_L1
        self.data_collector.config.risk_level = Const.RISK_LEVEL_CORE
        with patch('msprobe.pytorch.dump.api_dump.api_risk_level.get_api_risk_level') as mock_get:
            mock_get.return_value = Const.RISK_LEVEL_CORE
            result = self.data_collector._should_collect_by_risk_level("Functional.conv2d.0.forward")
            self.assertTrue(result)
            mock_get.assert_called_once_with("conv2d")

    def test_api_name_extraction_tensor_prefix(self):
        """测试Tensor前缀的API名称提取：Tensor.matmul.1.forward -> matmul"""
        self.data_collector.config.level = Const.LEVEL_L1
        self.data_collector.config.risk_level = Const.RISK_LEVEL_CORE
        with patch('msprobe.pytorch.dump.api_dump.api_risk_level.get_api_risk_level') as mock_get:
            mock_get.return_value = Const.RISK_LEVEL_CORE
            result = self.data_collector._should_collect_by_risk_level("Tensor.matmul.1.forward")
            self.assertTrue(result)
            mock_get.assert_called_once_with("matmul")

    def test_api_name_extraction_distributed_prefix(self):
        """测试Distributed前缀的API名称提取：Distributed.all_reduce.0.forward -> all_reduce"""
        self.data_collector.config.level = Const.LEVEL_L1
        self.data_collector.config.risk_level = Const.RISK_LEVEL_CORE
        with patch('msprobe.pytorch.dump.api_dump.api_risk_level.get_api_risk_level') as mock_get:
            mock_get.return_value = Const.RISK_LEVEL_CORE
            result = self.data_collector._should_collect_by_risk_level("Distributed.all_reduce.0.forward")
            self.assertTrue(result)
            mock_get.assert_called_once_with("all_reduce")

    def test_non_pytorch_framework(self):
        """测试非PyTorch框架时，返回True"""
        self.data_collector.config.level = Const.LEVEL_L1
        self.data_collector.config.risk_level = Const.RISK_LEVEL_CORE
        self.data_collector.config.framework = "mindspore"
        result = self.data_collector._should_collect_by_risk_level("Functional.conv2d.0.forward")
        self.assertTrue(result)

    def test_risk_level_core_with_focus_api(self):
        """测试CORE级别，中等风险API返回False"""
        self.data_collector.config.level = Const.LEVEL_L1
        self.data_collector.config.risk_level = Const.RISK_LEVEL_CORE
        with patch('msprobe.pytorch.dump.api_dump.api_risk_level.get_api_risk_level') as mock_get:
            mock_get.return_value = Const.RISK_LEVEL_FOCUS
            result = self.data_collector._should_collect_by_risk_level("Functional.unknown_api.0.forward")
            self.assertFalse(result)
