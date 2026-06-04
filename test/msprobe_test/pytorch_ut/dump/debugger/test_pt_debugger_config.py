import unittest
from unittest.mock import MagicMock

import torch

from msprobe.core.common.const import Const
from msprobe.core.common.exceptions import MsprobeException
from msprobe.pytorch.dump.debugger.debugger_config import DebuggerConfig


class TestDebuggerConfig(unittest.TestCase):
    def setUp(self):
        self.common_config = MagicMock()
        self.task_config = MagicMock()

        self.common_config.dump_path = "./dump_path"
        self.common_config.task = Const.STATISTICS
        self.common_config.level = "L1"
        self.common_config.async_dump = False

    def test_default_init(self):
        debugger = DebuggerConfig(self.common_config, self.task_config, None, None, None)
        self.assertEqual(debugger.dump_path, "./dump_path")
        self.assertEqual(debugger.task, Const.STATISTICS)
        self.assertEqual(debugger.level, "L1")
        self.assertEqual(debugger.custom_op_namespaces, ["_C_ascend"])

    def test_custom_op_namespaces_from_config(self):
        self.common_config.custom_op_namespaces = ["_C_ascend", "my_ns"]
        debugger = DebuggerConfig(self.common_config, self.task_config, None, None, None)
        self.assertEqual(debugger.custom_op_namespaces, ["_C_ascend", "my_ns"])

    def test_check_kwargs_with_invalid_task(self):
        self.common_config.task = "invalid_task"
        with self.assertRaises(MsprobeException) as context:
            DebuggerConfig(self.common_config, self.task_config, None, None, None)
        self.assertIn(f"The task <invalid_task> is not in the {Const.TORCH_TASK_LIST}", str(context.exception))

    def test_check_kwargs_with_invalid_level(self):
        self.common_config.level = "invalid_level"
        with self.assertRaises(MsprobeException) as context:
            DebuggerConfig(self.common_config, self.task_config, None, None, None)
        self.assertIn(f"The level <invalid_level> is not in the {Const.LEVEL_LIST}.", str(context.exception))

    def test_check_kwargs_with_invalid_dump_path(self):
        self.common_config.dump_path = None
        with self.assertRaises(MsprobeException) as context:
            DebuggerConfig(self.common_config, self.task_config, None, None, None)
        self.assertIn(f"The dump_path not found.", str(context.exception))

    def test_check_kwargs_with_invalid_async_dump(self):
        self.common_config.async_dump = 1
        with self.assertRaises(MsprobeException) as context:
            DebuggerConfig(self.common_config, self.task_config, None, None, None)
        self.assertIn(f"The parameters async_dump should be bool.", str(context.exception))

    def test_check_kwargs_with_async_dump_and_debug(self):
        self.common_config.async_dump = True
        self.common_config.task = Const.TENSOR
        self.common_config.level = Const.LEVEL_DEBUG
        self.task_config.list = ["linear"]
        config = DebuggerConfig(self.common_config, self.task_config, None, None, None)
        self.assertEqual(config.list, [])

    def test_check_kwargs_with_async_dump_and_not_debug(self):
        self.common_config.async_dump = True
        self.common_config.task = Const.TENSOR
        self.common_config.level = Const.LEVEL_MIX
        self.task_config.list = []
        self.task_config.summary_mode = Const.SUMMARY_MODE
        with self.assertRaises(MsprobeException) as context:
            DebuggerConfig(self.common_config, self.task_config, None, None, None)
        self.assertIn(f"the parameters list cannot be empty.", str(context.exception))

    def test_check_kwargs_with_structure_task(self):
        self.common_config.task = Const.STRUCTURE
        self.common_config.level = Const.LEVEL_L1
        config = DebuggerConfig(self.common_config, self.task_config, None, None, None)
        self.assertEqual(config.level, Const.LEVEL_MIX)

    def test_check_async_dump_and_md5(self):
        self.common_config.async_dump = True
        self.common_config.task = Const.STATISTICS
        self.common_config.level = Const.LEVEL_L1
        self.task_config.summary_mode = Const.MD5
        with self.assertRaises(MsprobeException) as context:
            DebuggerConfig(self.common_config, self.task_config, None, None, None)
        self.assertIn(f"the parameters summary_mode cannot be md5.", str(context.exception))

    def test_check_model_with_incorrect_model(self):
        self.common_config.level = Const.LEVEL_L0
        model1 = torch.nn.ReLU()
        model2 = [torch.nn.Linear(2, 2), torch.nn.ReLU(), "test_model"]

        instance = MagicMock()
        instance.model = model1
        config = DebuggerConfig(self.common_config, self.task_config, None, None, None)
        with self.assertRaises(MsprobeException) as context:
            config.check_model(instance, None)
        self.assertIn("must be a torch.nn.Module or list[torch.nn.Module]", str(context.exception))

    def test_check_and_adjust_config_with_l2_scope_not_empty(self):
        self.common_config.dump_path = "./dump_path"
        self.common_config.task = Const.TENSOR

        self.task_config.scope = ["test_api_name"]
        debugger = DebuggerConfig(self.common_config, self.task_config, None, None, None)
        with self.assertRaises(MsprobeException) as context:
            debugger._check_and_adjust_config_with_l2()
        self.assertIn("the scope cannot be configured", str(context.exception))

    def test_check_and_adjust_config_with_l2_list_empty(self):
        self.common_config.dump_path = "./dump_path"
        self.common_config.task = Const.TENSOR
        self.common_config.async_dump = False

        self.task_config.scope = []
        self.task_config.list = []
        debugger = DebuggerConfig(self.common_config, self.task_config, None, None, None)
        with self.assertRaises(MsprobeException) as context:
            debugger._check_and_adjust_config_with_l2()
        self.assertIn("the list must be configured", str(context.exception))

    def test_check_and_adjust_config_with_l2_success(self):
        self.common_config.dump_path = "./dump_path"
        self.common_config.task = Const.TENSOR

        self.task_config.scope = []
        self.task_config.list = ["Functional.conv2d.0.backward"]
        debugger = DebuggerConfig(self.common_config, self.task_config, None, None, None)
        debugger._check_and_adjust_config_with_l2()
        self.assertIn("Functional.conv2d.0.forward", self.task_config.list)

    def test_check_and_adjust_config_with_l2_task_not_tensor(self):
        self.common_config.dump_path = "./dump_path"
        self.common_config.task = Const.STATISTICS

        self.task_config.scope = []
        self.task_config.list = ["Functional.conv2d.0.forward"]
        debugger = DebuggerConfig(self.common_config, self.task_config, None, None, None)
        with self.assertRaises(MsprobeException) as context:
            debugger._check_and_adjust_config_with_l2()
        self.assertIn("the task must be set to tensor", str(context.exception))

    def test_check_statistics_config_task_not_statistics(self):
        self.common_config.dump_path = "./dump_path"
        self.common_config.task = Const.TENSOR

        debugger = DebuggerConfig(self.common_config, self.task_config, None, None, None)
        debugger._check_statistics_config(self.task_config)
        self.assertFalse(hasattr(debugger, "tensor_list"))

    def test_check_statistics_config_not_tensor_list(self):
        self.common_config.dump_path = "./dump_path"
        self.common_config.task = Const.STATISTICS
        delattr(self.task_config, "tensor_list")

        debugger = DebuggerConfig(self.common_config, self.task_config, None, None, None)
        debugger._check_statistics_config(self.task_config)
        self.assertEqual(debugger.tensor_list, [])

    def test_check_statistics_config_debug_level(self):
        self.common_config.dump_path = "./dump_path"
        self.common_config.task = Const.STATISTICS
        self.common_config.level = Const.DEBUG

        debugger = DebuggerConfig(self.common_config, self.task_config, None, None, None)
        self.task_config.tensor_list = ["Functional.conv2d"]
        debugger._check_statistics_config(self.task_config)
        self.assertEqual(debugger.tensor_list, [])

    def test_check_statistics_config_success(self):
        self.common_config.dump_path = "./dump_path"
        self.common_config.task = Const.STATISTICS

        self.task_config.tensor_list = ["Functional.conv2d"]
        debugger = DebuggerConfig(self.common_config, self.task_config, None, None, None)
        debugger._check_statistics_config(self.task_config)
        self.assertEqual(debugger.tensor_list, self.task_config.tensor_list)
