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
from unittest.mock import MagicMock, patch

from msprobe.pytorch.dump.pytorch_service import PytorchService
from msprobe.core.common.utils import Const
from msprobe.pytorch.dump.module_dump.module_processor import ModuleProcessor
from msprobe.pytorch.dump.api_dump.hook_module import HOOKModule


class TestPytorchService(unittest.TestCase):
    def setUp(self):
        self.config = MagicMock()
        self.config.step = []
        self.config.rank = []
        self.config.level = Const.LEVEL_MIX
        self.config.task = Const.STATISTICS

        with patch('msprobe.core.dump.service.build_data_collector'):
            self.service = PytorchService(self.config)

        self.service.logger = MagicMock()
        self.service.data_collector = MagicMock()
        self.service.module_processor = MagicMock()
        self.service.api_register = MagicMock()

    def test_framework_type(self):
        self.assertEqual(self.service._get_framework_type, Const.PT_FRAMEWORK)

    @patch('msprobe.pytorch.dump.pytorch_service.get_rank_if_initialized')
    def test_get_current_rank(self, mock_get_rank):
        mock_get_rank.return_value = 5
        self.assertEqual(self.service._get_current_rank(), 5)

    def test_init_specific_components(self):
        with patch('msprobe.core.dump.service.build_data_collector'):
            service = PytorchService(self.config)

        self.assertIsNotNone(service.logger)
        self.assertIsNotNone(service.api_register)
        self.assertIsNotNone(service.module_processor)
        self.assertIsNotNone(service.hook_manager)

    def test_register_hook(self):
        self.service._register_hook()

    @patch('msprobe.pytorch.dump.pytorch_service.register_optimizer_hook')
    def test_register_hook_mix_level(self, mock_register_opt):
        self.service.config.level = Const.LEVEL_MIX
        self.service._register_hook()
        mock_register_opt.assert_called_once_with(self.service.data_collector)

    @patch('msprobe.pytorch.dump.pytorch_service.register_optimizer_hook')
    def test_register_hook_not_mix_level(self, mock_register_opt):
        self.service.config.level = Const.LEVEL_L1
        self.service._register_hook()
        mock_register_opt.assert_not_called()

    @patch('msprobe.pytorch.dump.pytorch_service.wrap_script_func')
    def test_register_api_hook(self, mock_wrap_jit):
        self.service.config.level = Const.LEVEL_L1
        self.service._register_api_hook()
        mock_wrap_jit.assert_called_once()
        self.service.api_register.initialize_hook.assert_called_once()

    def test_register_module_hook_1(self):
        model_mock = MagicMock()
        self.service.model = model_mock
        self.service._register_module_hook()

        self.service.module_processor.register_module_hook.assert_called_once_with(
            model_mock, self.service.build_hook
        )

        self.assertTrue(self.service.module_processor.enable_module_dump)

    @patch.object(HOOKModule, 'reset_module_stats')
    @patch.object(ModuleProcessor, 'reset_module_stats')
    def test_reset_status(self, mock_reset_module_processor, mock_reset_hook_module):
        self.service._reset_status()
        mock_reset_hook_module.assert_called_once()
        mock_reset_module_processor.assert_called_once()
        self.service.data_collector.reset_status.assert_called_once()

    def test_register_module_hook_2(self):
        self.service.model = MagicMock()
        self.service._register_module_hook()
        self.service.module_processor.register_module_hook.assert_called_once()

    # ------------------ _need_dump_data ------------------

    def _make_real_config(self, request_id="req_0"):
        """构造真实 DebuggerConfig 替换 mock config，便于验证 slice_info 真实更新"""
        common_config = MagicMock()
        common_config.dump_path = "./dump_path"
        common_config.task = Const.TENSOR
        common_config.level = Const.LEVEL_L1
        common_config.async_dump = False
        common_config.rank = []
        common_config.step = []
        task_config = MagicMock()
        task_config.request_id = request_id
        task_config.slice_info = []
        from msprobe.pytorch.dump.debugger.debugger_config import DebuggerConfig
        return DebuggerConfig(common_config, task_config, None, None, None)

    def test_need_dump_data_none_returns_true(self):
        """scheduled_tokens=None → 返回 True，回退上一轮动态切片"""
        self.service.config = self._make_real_config("req_0")
        # 先添加一轮动态切片
        self.service.config.update_slice_info({"req_0": 100}, True)
        self.assertEqual(self.service.config.slice_info, [
            {"dim": 0, "size": 100, "begin": 0, "end": 100}
        ])
        # None 轮：回退后返回 True
        result = self.service._need_dump_data(None)
        self.assertTrue(result)
        self.assertEqual(self.service.config.slice_info, [])

    def test_need_dump_data_empty_tokens_returns_true(self):
        """scheduled_tokens 为空 dict → 回退上一轮，返回 True"""
        self.service.config = self._make_real_config("req_0")
        self.service.config.update_slice_info({"req_0": 100}, True)
        result = self.service._need_dump_data({})
        self.assertTrue(result)
        self.assertEqual(self.service.config.slice_info, [])

    def test_need_dump_data_invalid_tokens_returns_true(self):
        """scheduled_tokens 非法 → 回退上一轮，返回 True"""
        self.service.config = self._make_real_config("req_0")
        self.service.config.update_slice_info({"req_0": 100}, True)
        result = self.service._need_dump_data({"req_0": -1})
        self.assertTrue(result)
        self.assertEqual(self.service.config.slice_info, [])

    def test_need_dump_data_request_not_found_returns_false(self):
        """request_id 不在 scheduled_tokens → 返回 False，回退上一轮"""
        self.service.config = self._make_real_config("req_0")
        self.service.config.update_slice_info({"req_0": 100}, True)
        result = self.service._need_dump_data({"req_1": 200})
        self.assertFalse(result)
        self.assertEqual(self.service.config.slice_info, [])

    def test_need_dump_data_request_hit_updates_slice(self):
        """request_id 命中 → update_slice_info 被调用，slice_info 正确更新，返回 True"""
        self.service.config = self._make_real_config("req_1")
        result = self.service._need_dump_data({"req_0": 512, "req_1": 1024})
        self.assertTrue(result)
        self.assertEqual(self.service.config.slice_info, [
            {"dim": 0, "size": 1536, "begin": 512, "end": 1536}
        ])
        self.assertTrue(self.service.config.is_slice_info_modified)

    def test_need_dump_data_consecutive_calls_no_bloat(self):
        """连续多轮调用：每轮先回退再添加，slice_info 不膨胀"""
        self.service.config = self._make_real_config("req_0")
        # 第1轮
        self.service._need_dump_data({"req_0": 100, "req_1": 200})
        self.assertEqual(len(self.service.config.slice_info), 1)
        # 第2轮：先回退上一轮，再添加新一轮
        self.service._need_dump_data({"req_0": 300, "req_1": 400})
        self.assertEqual(len(self.service.config.slice_info), 1)
        self.assertEqual(self.service.config.slice_info[-1], {
            "dim": 0, "size": 700, "begin": 0, "end": 300
        })
        # 第3轮：目标缺失 → 回退，不添加
        result = self.service._need_dump_data({"req_1": 500})
        self.assertFalse(result)
        self.assertEqual(self.service.config.slice_info, [])
