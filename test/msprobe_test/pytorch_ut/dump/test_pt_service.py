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

import tempfile
import unittest
from unittest.mock import MagicMock, patch

from msprobe.pytorch.dump.pytorch_service import PytorchService
from msprobe.core.common.utils import Const
from msprobe.core.common.runtime import Runtime
from msprobe.pytorch.dump.module_dump.module_processor import ModuleProcessor
from msprobe.pytorch.dump.module_load.tensor_loader import TensorLoader
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


class TestSyncTensorLoader(unittest.TestCase):
    """Regression tests: tensor loader state must be synced on every start(),
    including ranks/steps excluded by dump rank/step filters (module hooks with
    load override stay mounted there — see _sync_tensor_loader)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config = MagicMock()
        self.config.step = []
        self.config.rank = [0]  # dump only rank 0
        self.config.level = Const.LEVEL_L0
        self.config.task = Const.TENSOR
        self.config.async_dump = False

        with patch('msprobe.core.dump.service.build_data_collector'):
            self.service = PytorchService(self.config)

        self.service.logger = MagicMock()
        self.service.data_collector = MagicMock()
        self.service.data_collector.data_processor.is_terminated = False
        self.service.api_register = MagicMock()
        self.service.hook_manager = MagicMock()  # real service.step() calls reset_status()

        # re-set attrs on the (deep-copied) service config actually read at runtime
        self.service.config.step = []
        self.service.config.rank = [0]
        self.service.config.level = Const.LEVEL_L0
        self.service.config.task = Const.TENSOR
        self.service.config.async_dump = False
        self.service.config.dump_enable = True
        self.service.config.bench_path = None
        self.service.config.dump_path = self.tmpdir

        # Runtime holds class-level state that persists across tests
        Runtime.current_rank = None
        Runtime.is_running = False
        # so does the module call counter (class-level on ModuleProcessor)
        ModuleProcessor.module_count = {}

        # real TensorLoader behind a mocked module_processor
        self.load_config = MagicMock()
        self.load_config.path = self.tmpdir
        self.load_config.modules = ["Module.blocks.0.attn.qkv.Linear.forward.0"]
        self.load_config.step = []
        self.load_config.rank = []
        self.load_config.dump_after_load = True
        self.load_config.is_enabled = True
        self.service.module_processor = MagicMock()
        self.service.module_processor.tensor_loader = TensorLoader(self.load_config)
        self.service.config.load_config = self.load_config

        self.model = MagicMock()
        self.model.named_modules.return_value = []

        tl_logger_patcher = patch("msprobe.pytorch.dump.module_load.tensor_loader.logger")
        self.mock_tl_logger = tl_logger_patcher.start()
        self.addCleanup(tl_logger_patcher.stop)
        ends_patcher = patch("msprobe.core.dump.service.print_tools_ends_info")
        ends_patcher.start()
        self.addCleanup(ends_patcher.stop)

    def _start(self):
        self.service.start(model=self.model)

    @property
    def _loader(self):
        return self.service.module_processor.tensor_loader

    @patch.object(PytorchService, '_get_current_rank', return_value=1)
    def test_no_dump_rank_synced_every_start(self, mock_rank):
        """rank 1 excluded by dump rank=[0]: loader still gets step/rank each start()."""
        self._start()
        self.assertEqual(self._loader._current_step, 0)
        self.assertEqual(self._loader._current_rank, 1)
        self.assertTrue(self._loader.active)
        self.service.step()  # next step
        self._start()
        self.assertEqual(self._loader._current_step, 1)
        self.assertEqual(self._loader._current_rank, 1)

    @patch.object(PytorchService, '_get_current_rank', return_value=0)
    def test_dump_rank_synced_normal_path(self, mock_rank):
        """Normal path (rank in dump range): loader synced as before, dump flow intact."""
        self.service.config.rank = []
        self._start()
        self.assertEqual(self._loader._current_step, 0)
        self.assertEqual(self._loader._current_rank, 0)
        self.assertTrue(self._loader.active)
        self.assertFalse(self.service.first_start)
        self.service.step()
        self._start()
        self.assertEqual(self._loader._current_step, 1)
        self.assertEqual(self._loader._current_rank, 0)

    @patch.object(PytorchService, '_get_current_rank', return_value=1)
    def test_step_keeps_advancing_after_stop_service(self, mock_rank):
        """After the dump step range is exceeded, step() returns before loop += 1
        (generic behavior), so current_iter freezes; the loader sync must derive
        the step from its own counter to keep loading step-aligned."""
        self.service.config.step = [0]
        self._start()                # step 0: passes filters, loader synced to 0
        self.service.step()          # loop 0 -> 1
        self._start()                # step 1 > max(step): sets should_stop_service
        self.assertTrue(self.service.should_stop_service)
        self.assertEqual(self._loader._current_step, 1)
        self.service.step()          # early return: loop frozen at 1
        self._start()                # real step 2: current_iter frozen at 1
        self.assertEqual(self._loader._current_step, 2)
        self.assertEqual(self._loader._current_rank, 1)
        self.service.step()          # still frozen
        self._start()                # real step 3
        self.assertEqual(self._loader._current_step, 3)

    @patch.object(PytorchService, '_get_current_rank', return_value=1)
    def test_repeated_start_same_step_does_not_advance(self, mock_rank):
        """Repeated start() calls within one training step (multiple forwards,
        multiple models, or unpaired start without step()) must not advance the
        loader step after the dump service has stopped: the load step advances
        only on step(), staying 1:1 with training steps."""
        self.service.config.step = [0]
        self._start()                # step 0
        self.service.step()
        self._start()                # step 1: sets should_stop_service
        self.assertTrue(self.service.should_stop_service)
        self.service.step()          # loop frozen; load counter now at 2
        self._start()                # real step 2
        self._start()                # second start within the same training step
        self._start()                # and a third
        self.assertEqual(self._loader._current_step, 2)  # no over-advance
        self.service.step()          # training advances to step 3
        self._start()
        self.assertEqual(self._loader._current_step, 3)

    @patch.object(PytorchService, '_get_current_rank', return_value=1)
    def test_call_index_stays_aligned_after_stop_service(self, mock_rank):
        """After should_stop_service, step() returns before _reset_status, so the
        module call counter would accumulate across steps and load-side forward
        names (forward.N) would drift from the source dump naming (forward.0 on
        every step). The step boundary must keep resetting it."""
        module_name = 'Module.blocks.0.attn.qkv.Linear'
        self.service.config.step = [0]
        for expected_step in range(4):
            self._start()            # step 0 passes; step 1 sets should_stop_service
            index = ModuleProcessor.set_and_get_calls_number(module_name)
            full_forward_name = f'{module_name}.forward.{index}'
            self.assertEqual(index, 0, f"step {expected_step}: call index drifted")
            self.assertTrue(self._loader.should_override(full_forward_name),
                            f"step {expected_step}: forward name {full_forward_name} missed")
            self.service.step()
        self.assertTrue(self.service.should_stop_service)

    @patch.object(PytorchService, '_get_current_rank', return_value=0)
    def test_load_only_step_gap_reactivates(self, mock_rank):
        """load-only mode with a load.step gap (dump.step=[0,2], load.step=[0,2]):
        step 1 syncs the loader inactive, step 2 must sync it active again."""
        self.service.config.rank = []
        self.service.config.step = [0, 2]
        self.load_config.dump_after_load = False
        self.load_config.step = [0, 2]
        self._loader.src_step = [0, 2]  # loader copies load.step at construction
        self._start()
        self.assertTrue(self._loader.active)    # step 0 in load.step
        self.service.step()
        self._start()                           # step 1: filtered, sync flips active off
        self.assertFalse(self._loader.active)
        self.service.step()
        self._start()                           # step 2: in load.step again
        self.assertTrue(self._loader.active)
        self.assertEqual(self._loader._current_step, 2)

    @patch.object(PytorchService, '_get_current_rank', return_value=0)
    def test_stop_check_unhit_runs_after_stop_service(self, mock_rank):
        """check_unhit must also run on steps where should_stop_service already
        returned early (dump step range exceeded)."""
        self.service.config.rank = []
        self.service.config.step = [0]
        self._start()
        self.service.step()
        self._start()                # sets should_stop_service
        self.service.logger.reset_mock()
        self.service.stop()
        self.mock_tl_logger.warning.assert_called_once()  # module never hit -> unhit warning
        self.assertIn("never matched", self.mock_tl_logger.warning.call_args[0][0])
        self.service.logger.info_on_rank_0.assert_not_called()  # generic stop skipped

    @patch.object(PytorchService, '_get_current_rank', return_value=1)
    def test_synced_when_first_start_step_filtered(self, mock_rank):
        """dump step=[1] filters out step 0: first start() returns before the
        first_start block; loader sync (with its own rank fallback) must still
        happen, without touching generic rank state."""
        self.service.config.step = [1]
        self._start()
        self.assertEqual(self._loader._current_step, 0)
        self.assertEqual(self._loader._current_rank, 1)
        # load sync must not touch generic state; rank resolution stays owned
        # by the (unreached) first_start block
        self.assertIsNone(Runtime.current_rank)
        self.assertIsNone(self.service.current_rank)
        self.assertTrue(self.service.first_start)  # never reached first_start block

    @patch.object(PytorchService, '_get_current_rank', return_value=0)
    def test_load_only_mode_start_survives_without_old_update_block(self, mock_rank):
        """Regression: the load-only branch used to rely on module_processor
        defined by the old in-start loader update block; after that block moved
        into _sync_tensor_loader the branch must define it itself."""
        self.service.config.rank = []
        self.load_config.dump_after_load = False
        self._start()
        self.assertTrue(self._loader.active)
        self.assertEqual(self._loader._current_step, 0)
        self.assertEqual(self._loader._current_rank, 0)
        # load-only mode: dump runtime off, override active
        self.assertFalse(Runtime.is_running)
        self.assertFalse(self.service.primitive_switch)

    @patch.object(PytorchService, '_get_current_rank', return_value=1)
    def test_stop_check_unhit_runs_on_no_dump_rank(self, mock_rank):
        """stop() must report unhit modules on ranks excluded from dump."""
        self._start()
        self.service.logger.reset_mock()  # drop the hook-mount log from start()
        self.service.stop()
        self.mock_tl_logger.warning.assert_called_once()
        self.assertIn("never matched", self.mock_tl_logger.warning.call_args[0][0])
        # rank excluded from dump: no stop-success log
        self.service.logger.info_on_rank_0.assert_not_called()
