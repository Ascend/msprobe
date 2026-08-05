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
        self.task_config.request_id = None

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

    def test_slice_info_valid_with_tensor_l0(self):
        """TC-301: tensor + L0 → slice 有效"""
        self.common_config.task = Const.TENSOR
        self.common_config.level = Const.LEVEL_L0
        self.task_config.slice_info = [{"dim": 0, "size": 100, "begin": 0, "end": 50}]
        debugger = DebuggerConfig(self.common_config, self.task_config, None, None, None)
        self.assertEqual(debugger.slice_info, [{"dim": 0, "size": 100, "begin": 0, "end": 50}])

    def test_slice_info_valid_with_tensor_l1(self):
        """TC-302: tensor + L1 → slice 有效"""
        self.common_config.task = Const.TENSOR
        self.common_config.level = Const.LEVEL_L1
        self.task_config.slice_info = [{"dim": 0, "size": 100, "begin": 0, "end": 50}]
        debugger = DebuggerConfig(self.common_config, self.task_config, None, None, None)
        self.assertEqual(debugger.slice_info, [{"dim": 0, "size": 100, "begin": 0, "end": 50}])

    def test_slice_info_valid_with_tensor_mix(self):
        """TC-303: tensor + MIX → slice 有效"""
        self.common_config.task = Const.TENSOR
        self.common_config.level = Const.LEVEL_MIX
        self.task_config.slice_info = [{"dim": 0, "size": 100, "begin": 0, "end": 50}]
        debugger = DebuggerConfig(self.common_config, self.task_config, None, None, None)
        self.assertEqual(debugger.slice_info, [{"dim": 0, "size": 100, "begin": 0, "end": 50}])

    def test_slice_info_valid_with_statistics_l1(self):
        """TC-304: statistics + L1 → slice 有效"""
        self.common_config.task = Const.STATISTICS
        self.common_config.level = Const.LEVEL_L1
        self.task_config.slice_info = [{"dim": 0, "size": 100, "begin": 0, "end": 50}]
        debugger = DebuggerConfig(self.common_config, self.task_config, None, None, None)
        self.assertEqual(debugger.slice_info, [{"dim": 0, "size": 100, "begin": 0, "end": 50}])

    def test_slice_info_valid_multi_dim_config(self):
        """TC-305: 多维度切片配置有效"""
        self.common_config.task = Const.TENSOR
        self.common_config.level = Const.LEVEL_L1
        self.task_config.slice_info = [
            {"dim": 0, "size": 100, "begin": 0, "end": 50},
            {"dim": 1, "size": 3, "begin": 0, "end": 2},
        ]
        debugger = DebuggerConfig(self.common_config, self.task_config, None, None, None)
        self.assertEqual(debugger.slice_info, [
            {"dim": 0, "size": 100, "begin": 0, "end": 50},
            {"dim": 1, "size": 3, "begin": 0, "end": 2},
        ])

    def test_slice_info_invalid_with_debug_level(self):
        """TC-306: tensor + DEBUG → slice 置空"""
        self.common_config.task = Const.TENSOR
        self.common_config.level = Const.LEVEL_DEBUG
        self.task_config.slice_info = [{"dim": 0, "size": 100, "begin": 0, "end": 50}]
        debugger = DebuggerConfig(self.common_config, self.task_config, None, None, None)
        self.assertEqual(debugger.slice_info, [])

    def test_slice_info_invalid_with_structure_task(self):
        """TC-307: structure + L1 → slice 置空"""
        self.common_config.task = Const.STRUCTURE
        self.common_config.level = Const.LEVEL_L1
        self.task_config.slice_info = [{"dim": 0, "size": 100, "begin": 0, "end": 50}]
        debugger = DebuggerConfig(self.common_config, self.task_config, None, None, None)
        self.assertEqual(debugger.slice_info, [])

    def test_slice_info_invalid_with_acc_check_task(self):
        """TC-308: acc_check + L1 → slice 置空"""
        self.common_config.task = Const.ACC_CHECK
        self.common_config.level = Const.LEVEL_L1
        self.task_config.slice_info = [{"dim": 0, "size": 100, "begin": 0, "end": 50}]
        debugger = DebuggerConfig(self.common_config, self.task_config, None, None, None)
        self.assertEqual(debugger.slice_info, [])

    def test_slice_info_default_empty(self):
        """未配置 slice_info 时默认为空 list"""
        self.task_config.slice_info = []
        debugger = DebuggerConfig(self.common_config, self.task_config, None, None, None)
        self.assertEqual(debugger.slice_info, [])

    # ------------------ request_id 加载与 task/level 约束 ------------------

    def test_request_id_default_none(self):
        """未配置 request_id 时默认为 None"""
        debugger = DebuggerConfig(self.common_config, self.task_config, None, None, None)
        self.assertIsNone(debugger.request_id)

    def test_request_id_valid_with_tensor_l1(self):
        """TC-201: tensor + L1 → request_id 有效"""
        self.common_config.task = Const.TENSOR
        self.common_config.level = Const.LEVEL_L1
        self.task_config.request_id = "req_0"
        debugger = DebuggerConfig(self.common_config, self.task_config, None, None, None)
        self.assertEqual(debugger.request_id, "req_0")

    def test_request_id_valid_with_statistics_l1(self):
        """TC-202: statistics + L1 → request_id 有效"""
        self.common_config.task = Const.STATISTICS
        self.common_config.level = Const.LEVEL_L1
        self.task_config.request_id = "req_0"
        debugger = DebuggerConfig(self.common_config, self.task_config, None, None, None)
        self.assertEqual(debugger.request_id, "req_0")

    def test_request_id_valid_with_tensor_l0(self):
        """TC-203: tensor + L0 → request_id 有效"""
        self.common_config.task = Const.TENSOR
        self.common_config.level = Const.LEVEL_L0
        self.task_config.request_id = "req_0"
        debugger = DebuggerConfig(self.common_config, self.task_config, None, None, None)
        self.assertEqual(debugger.request_id, "req_0")

    def test_request_id_valid_with_tensor_mix(self):
        """tensor + mix → request_id 有效"""
        self.common_config.task = Const.TENSOR
        self.common_config.level = Const.LEVEL_MIX
        self.task_config.request_id = "req_0"
        debugger = DebuggerConfig(self.common_config, self.task_config, None, None, None)
        self.assertEqual(debugger.request_id, "req_0")

    def test_request_id_invalid_with_structure_task(self):
        """EC-201: structure + L1 → request_id 置空"""
        self.common_config.task = Const.STRUCTURE
        self.common_config.level = Const.LEVEL_L1
        self.task_config.request_id = "req_0"
        debugger = DebuggerConfig(self.common_config, self.task_config, None, None, None)
        self.assertIsNone(debugger.request_id)

    def test_request_id_invalid_with_debug_level(self):
        """EC-202: tensor + DEBUG → request_id 置空"""
        self.common_config.task = Const.TENSOR
        self.common_config.level = Const.LEVEL_DEBUG
        self.task_config.request_id = "req_0"
        debugger = DebuggerConfig(self.common_config, self.task_config, None, None, None)
        self.assertIsNone(debugger.request_id)

    def test_request_id_invalid_with_acc_check_task(self):
        """EC-203: acc_check + L1 → request_id 置空"""
        self.common_config.task = Const.ACC_CHECK
        self.common_config.level = Const.LEVEL_L1
        self.task_config.request_id = "req_0"
        debugger = DebuggerConfig(self.common_config, self.task_config, None, None, None)
        self.assertIsNone(debugger.request_id)

    # ------------------ check_scheduled_tokens ------------------

    def _make_debugger_with_request_id(self, request_id="req_0"):
        self.common_config.task = Const.TENSOR
        self.common_config.level = Const.LEVEL_L1
        self.task_config.request_id = request_id
        self.task_config.slice_info = []
        return DebuggerConfig(self.common_config, self.task_config, None, None, None)

    def test_check_tokens_valid(self):
        """TC-301: 合法 request_id + 合法 scheduled_tokens → 返回 True"""
        debugger = self._make_debugger_with_request_id("req_0")
        self.assertTrue(debugger.check_scheduled_tokens({"req_0": 1024, "req_1": 512}))

    def test_check_tokens_request_id_none(self):
        """EC-501: request_id 为 None → 返回 False"""
        self.common_config.task = Const.TENSOR
        self.common_config.level = Const.LEVEL_L1
        # request_id 未配置，默认 None
        debugger = DebuggerConfig(self.common_config, self.task_config, None, None, None)
        self.assertFalse(debugger.check_scheduled_tokens({"req_0": 1024}))

    def test_check_tokens_scheduled_tokens_none(self):
        """EC-402: scheduled_tokens 为 None → 返回 False"""
        debugger = self._make_debugger_with_request_id("req_0")
        self.assertFalse(debugger.check_scheduled_tokens(None))

    def test_check_tokens_scheduled_tokens_empty(self):
        """scheduled_tokens 为空 dict → 返回 False"""
        debugger = self._make_debugger_with_request_id("req_0")
        self.assertFalse(debugger.check_scheduled_tokens({}))

    def test_check_tokens_scheduled_tokens_not_dict(self):
        """EC-302: scheduled_tokens 非 dict → 返回 False"""
        debugger = self._make_debugger_with_request_id("req_0")
        self.assertFalse(debugger.check_scheduled_tokens([("req_0", 1024)]))

    def test_check_tokens_key_not_str(self):
        """EC-303: key 非 str → 返回 False"""
        debugger = self._make_debugger_with_request_id("req_0")
        self.assertFalse(debugger.check_scheduled_tokens({1: 100}))

    def test_check_tokens_value_not_int(self):
        """EC-304: value 非 int → 返回 False"""
        debugger = self._make_debugger_with_request_id("req_0")
        self.assertFalse(debugger.check_scheduled_tokens({"req_0": "100"}))

    def test_check_tokens_value_zero(self):
        """EC-305: value 为 0 → 返回 False"""
        debugger = self._make_debugger_with_request_id("req_0")
        self.assertFalse(debugger.check_scheduled_tokens({"req_0": 0}))

    def test_check_tokens_value_negative(self):
        """value 为负数 → 返回 False"""
        debugger = self._make_debugger_with_request_id("req_0")
        self.assertFalse(debugger.check_scheduled_tokens({"req_0": -1}))

    # ------------------ update_slice_info ------------------

    def test_update_slice_single_request(self):
        """TC-301: 单 request batch → slice_item 正确"""
        debugger = self._make_debugger_with_request_id("req_0")
        debugger.update_slice_info({"req_0": 1024}, True)
        self.assertEqual(debugger.slice_info, [
            {"dim": 0, "size": 1024, "begin": 0, "end": 1024}
        ])
        self.assertTrue(debugger.is_slice_info_modified)

    def test_update_slice_target_in_middle(self):
        """TC-302: 多 request，目标在中间"""
        debugger = self._make_debugger_with_request_id("req_1")
        debugger.update_slice_info({"req_0": 512, "req_1": 1024, "req_2": 256}, True)
        self.assertEqual(debugger.slice_info, [
            {"dim": 0, "size": 1792, "begin": 512, "end": 1536}
        ])

    def test_update_slice_target_at_end(self):
        """TC-303: 多 request，目标在末尾"""
        debugger = self._make_debugger_with_request_id("req_1")
        debugger.update_slice_info({"req_0": 512, "req_1": 1024}, True)
        self.assertEqual(debugger.slice_info, [
            {"dim": 0, "size": 1536, "begin": 512, "end": 1536}
        ])

    def test_update_slice_target_at_start(self):
        """TC-304: 多 request，目标在开头"""
        debugger = self._make_debugger_with_request_id("req_0")
        debugger.update_slice_info({"req_0": 512, "req_1": 1024}, True)
        self.assertEqual(debugger.slice_info, [
            {"dim": 0, "size": 1536, "begin": 0, "end": 512}
        ])

    def test_update_slice_with_existing_slice_first_call(self):
        """TC-401: 已有 slice_info 首次调用 → 追加到末尾"""
        debugger = self._make_debugger_with_request_id("req_0")
        debugger.slice_info = [{"dim": 1, "size": 3, "begin": 0, "end": 2}]
        debugger.update_slice_info({"req_0": 100}, True)
        self.assertEqual(debugger.slice_info, [
            {"dim": 1, "size": 3, "begin": 0, "end": 2},
            {"dim": 0, "size": 100, "begin": 0, "end": 100},
        ])

    def test_update_slice_with_existing_slice_second_call(self):
        """TC-402: 二次调用 → 替换最后一项，不重复追加"""
        debugger = self._make_debugger_with_request_id("req_0")
        debugger.slice_info = [{"dim": 1, "size": 3, "begin": 0, "end": 2}]
        debugger.update_slice_info({"req_0": 100}, True)
        debugger.update_slice_info({"req_0": 200}, True)
        self.assertEqual(debugger.slice_info, [
            {"dim": 1, "size": 3, "begin": 0, "end": 2},
            {"dim": 0, "size": 200, "begin": 0, "end": 200},
        ])

    def test_update_slice_empty_slice_info(self):
        """TC-403: 空 slice_info → 设为 [new_slice_item]"""
        debugger = self._make_debugger_with_request_id("req_0")
        debugger.slice_info = []
        debugger.update_slice_info({"req_0": 100}, True)
        self.assertEqual(debugger.slice_info, [
            {"dim": 0, "size": 100, "begin": 0, "end": 100}
        ])

    def test_update_slice_multiple_start_calls(self):
        """TC-501: 多次 start 调用 → 每次替换最后一项，slice_info 不膨胀"""
        debugger = self._make_debugger_with_request_id("req_0")
        debugger.update_slice_info({"req_0": 100, "req_1": 200}, True)
        debugger.update_slice_info({"req_0": 300, "req_1": 400}, True)
        debugger.update_slice_info({"req_0": 500, "req_1": 600}, True)
        self.assertEqual(len(debugger.slice_info), 1)
        self.assertEqual(debugger.slice_info[-1], {
            "dim": 0, "size": 1100, "begin": 0, "end": 500
        })

    # ------------------ 连续调用：回退路径移除上一轮动态切片 ------------------

    def test_sequence_valid_then_none(self):
        """合法 → None：None 轮 is_add=False 移除上一轮动态切片"""
        debugger = self._make_debugger_with_request_id("req_0")
        debugger.update_slice_info({"req_0": 100}, True)
        self.assertTrue(debugger.is_slice_info_modified)
        self.assertEqual(debugger.slice_info, [
            {"dim": 0, "size": 100, "begin": 0, "end": 100}
        ])
        # None 轮：先移除上一轮动态切片
        debugger.update_slice_info(None, False)
        self.assertFalse(debugger.is_slice_info_modified)
        self.assertEqual(debugger.slice_info, [])

    def test_sequence_valid_then_invalid(self):
        """合法 → 非法：非法轮 is_add=False 移除上一轮动态切片"""
        debugger = self._make_debugger_with_request_id("req_0")
        debugger.update_slice_info({"req_0": 100}, True)
        self.assertEqual(debugger.slice_info, [
            {"dim": 0, "size": 100, "begin": 0, "end": 100}
        ])
        # 非法轮：is_add=False 只做移除，不检查 scheduled_tokens
        debugger.update_slice_info({"req_0": -1}, False)
        self.assertFalse(debugger.is_slice_info_modified)
        self.assertEqual(debugger.slice_info, [])

    def test_sequence_valid_then_missing_then_valid(self):
        """合法 → 目标缺失 → 合法：目标缺失轮回退移除，合法轮重新添加"""
        debugger = self._make_debugger_with_request_id("req_0")
        # 第1轮：合法，添加动态切片
        debugger.update_slice_info({"req_0": 100}, True)
        self.assertEqual(debugger.slice_info, [
            {"dim": 0, "size": 100, "begin": 0, "end": 100}
        ])
        # 第2轮：目标缺失，先移除上一轮，再尝试添加但 request_id 不在 → 不添加
        debugger.update_slice_info({"req_1": 200}, False)
        debugger.update_slice_info({"req_1": 200}, True)
        self.assertFalse(debugger.is_slice_info_modified)
        self.assertEqual(debugger.slice_info, [])
        # 第3轮：合法，重新添加动态切片
        debugger.update_slice_info({"req_0": 300}, True)
        self.assertTrue(debugger.is_slice_info_modified)
        self.assertEqual(debugger.slice_info, [
            {"dim": 0, "size": 300, "begin": 0, "end": 300}
        ])

    def test_sequence_rollback_preserves_static_slice(self):
        """回退只移除动态切片，不影响用户配置的静态切片"""
        debugger = self._make_debugger_with_request_id("req_0")
        debugger.slice_info = [{"dim": 1, "size": 3, "begin": 0, "end": 2}]
        # 添加动态切片
        debugger.update_slice_info({"req_0": 100}, True)
        self.assertEqual(debugger.slice_info, [
            {"dim": 1, "size": 3, "begin": 0, "end": 2},
            {"dim": 0, "size": 100, "begin": 0, "end": 100},
        ])
        # 回退：只移除动态切片，保留静态切片
        debugger.update_slice_info(None, False)
        self.assertEqual(debugger.slice_info, [
            {"dim": 1, "size": 3, "begin": 0, "end": 2}
        ])
        self.assertFalse(debugger.is_slice_info_modified)

    def test_rollback_without_previous_dynamic_slice(self):
        """无上一轮动态切片时 is_add=False 不报错、不影响静态切片"""
        debugger = self._make_debugger_with_request_id("req_0")
        debugger.slice_info = [{"dim": 1, "size": 3, "begin": 0, "end": 2}]
        debugger.update_slice_info(None, False)
        self.assertEqual(debugger.slice_info, [
            {"dim": 1, "size": 3, "begin": 0, "end": 2}
        ])
        self.assertFalse(debugger.is_slice_info_modified)
