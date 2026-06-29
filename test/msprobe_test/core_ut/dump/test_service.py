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
import os
import tempfile

from msprobe.core.dump.service import BaseService
from msprobe.core.common.utils import Const
from msprobe.core.common.runtime import Runtime
from msprobe.core.dump.api_dump.api_registry import ApiRegistry
from msprobe.core.dump.hook_manager import BaseHookManager


class ConcreteBaseService(BaseService):
    def _init_specific_components(self):
        self.logger = MagicMock()
        self.api_register = MagicMock()
        self.hook_manager = MagicMock()
        self.api_template = MagicMock()
    
    def _register_hook(self):
        pass
    
    def _register_module_hook(self):
        pass
    
    def _get_framework_type(self):
        return "TestFramework"
    
    @staticmethod
    def _get_current_rank():
        return 0
    
    def _change_jit_switch(self, status):
        pass

class TestBaseService(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config = MagicMock()
        self.config.level = Const.LEVEL_DEBUG
        self.config.level_ori = self.config.level
        self.config.step = [1, 3]
        self.config.rank = [0, 2]
        self.config.dump_path = self.temp_dir.name
        self.config.task = Const.STATISTICS
        self.config.async_dump = True
        self.config.tensor_list = []
        self.config.framework = "test_framwork"
        with patch('msprobe.core.dump.service.build_data_collector'):
            self.service = ConcreteBaseService(self.config)
            self.service.config.bench_path = None

    def tearDown(self):
        self.temp_dir.cleanup()
    
    def test_initialization(self):
        self.assertEqual(self.service.config.level, Const.LEVEL_DEBUG)
        self.assertIsNone(self.service.model)
        self.assertIsNotNone(self.service.data_collector)
        self.assertEqual(self.service.current_iter, 0)
        self.assertEqual(self.service.loop, 0)
        self.assertTrue(self.service.first_start)
        self.assertFalse(self.service.primitive_switch)
        self.assertIsNone(self.service.current_rank)
        self.assertIsNone(self.service.dump_iter_dir)
        self.assertFalse(self.service.should_stop_service)
        self.assertEqual(self.service.ori_customer_func, {})
    
    def test_properties(self):
        self.service.config.level = Const.LEVEL_DEBUG
        self.assertTrue(self.service._is_debug_level)
        
        self.service.config.level = Const.LEVEL_L2
        self.assertTrue(self.service._is_l2_level)

        self.service.config.level = Const.LEVEL_MIX
        self.assertTrue(self.service._is_mix_level)

        self.service.config.level = Const.LEVEL_MIX
        self.assertTrue(self.service._is_need_module_hook)
 
        self.service.config.level = Const.LEVEL_MIX
        self.assertTrue(self.service._is_need_api_hook)

        self.assertFalse(self.service._need_tensor_data)
        
        self.service.current_iter = 2
        self.assertTrue(self.service._is_no_dump_step)
        
        self.service.current_rank = 1
        self.assertTrue(self.service._is_no_dump_rank)

    @patch.object(BaseService, '_get_current_rank')
    @patch.object(BaseService, '_process_iteration')
    def test_start_debug_level(self, mock_process_iter, mock_get_rank):
        self.service.config.level = Const.LEVEL_DEBUG
        model_mock = MagicMock()
        
        self.service.start(model=model_mock)
    
        mock_get_rank.assert_not_called()
        mock_process_iter.assert_called_once()
        self.service.logger.info.assert_not_called()
        self.assertFalse(Runtime.is_running)
    
      
    @patch.object(ConcreteBaseService, '_register_hook')
    @patch.object(ConcreteBaseService, '_register_module_hook')
    def test_start_normal_level_first_time(self, mock_register_module_hook, mock_register_hook):
        self.service.config.level = Const.LEVEL_MIX
        self.service.config.step = []
        self.service.config.rank = []
        model_mock = MagicMock()
        self.service.data_collector.data_processor.is_terminated = False
        self.service.start(model=model_mock)

        self.assertEqual(self.service.current_rank, 0)
        self.assertEqual(Runtime.current_rank, 0)
        
        mock_register_hook.assert_called_once()
        mock_register_module_hook.assert_called_once()

        self.service.logger.info.assert_called_with(f"Dump data will be saved in {self.service.dump_iter_dir}.")
        self.assertTrue(Runtime.is_running)
        self.assertTrue(self.service.primitive_switch)
        self.assertFalse(self.service.first_start)
    
    @patch.object(ConcreteBaseService, '_register_hook')
    @patch.object(ConcreteBaseService, '_register_module_hook')
    @patch.object(ConcreteBaseService, 'create_dirs')
    def test_start_not_first_calls(self, mock_dirs, mock_register_module_hook, mock_register_hook):
        self.service.config.level = Const.LEVEL_L1
        self.service.config.step = []
        self.service.config.rank = []
        self.service.data_collector.data_processor.is_terminated = False
        self.service.first_start = False
        model_mock = MagicMock()
        
        self.service.start(model=model_mock)
        mock_register_hook.assert_not_called()
        mock_register_module_hook.assert_not_called()
        self.assertTrue(Runtime.is_running)
        self.assertTrue(self.service.primitive_switch)
        mock_dirs.assert_called_once()
        
    def test_start_with_infer_hook(self):
        self.service.config.level = Const.LEVEL_L1
        self.service.config.step = []
        self.service.config.rank = []
        self.service.data_collector.data_processor.is_terminated = False
        model_mock = MagicMock()
        token_range = [10, 20]

        self.service.start(model=model_mock, token_range=token_range)
        model_mock.register_forward_pre_hook.assert_called_once()
        self.assertEqual(self.service.cur_token_id, 0)
    
    def test_stop_debug_level(self):
        self.config.level = Const.LEVEL_DEBUG
        self.service.stop()
        self.service.logger.info.assert_not_called()
    
    @patch.object(BaseService, '_process_async_dump')
    def test_stop_normal_level(self, mock_process_async_dump):
        self.service.config.level = Const.LEVEL_L1
        self.service.current_iter = 1
        self.service.current_rank = 0
        
        self.service.stop()
        self.assertFalse(Runtime.is_running)
        self.assertFalse(self.service.primitive_switch)

        self.service.logger.info.assert_called_with(
            f"{Const.TOOL_NAME}: debugger.stop() is set successfully. "
            "Please set debugger.start() to turn on the dump switch again. "
        )
        mock_process_async_dump.assert_called_once()
        self.service.data_collector.write_json.assert_called_once()

    def test_stop_no_dump_step(self):
        self.config.level = Const.LEVEL_L1
        self.service.current_iter = 2 
        self.service.stop()
        self.service.logger.info.assert_not_called()
    
    def test_stop_no_dump_rank(self):
        self.config.level = Const.LEVEL_L1
        self.service.current_iter = 1
        self.service.current_rank = 1 
        self.service.stop()
        self.service.logger.info.assert_not_called()
    
    @patch.object(BaseService, '_process_async_dump')
    def test_step(self, mock_process_async_dump):
        self.service.step()
        self.assertEqual(self.service.loop, 1)
        mock_process_async_dump.assert_called_once()
        self.service.data_collector.write_json.assert_called_once()
        self.service.data_collector.reset_status.assert_called_once()
    
    @patch.object(BaseService, '_process_async_dump')
    def test_step_should_stop_service(self, mock_process_async_dump):
        self.service.should_stop_service = True
        self.service.step()
        self.assertEqual(self.service.loop, 0)
        mock_process_async_dump.assert_not_called()
    
    def test_save_debug_level(self):
        self.service.loop = 1
        self.service.init_step = 0
        self.service.save("test_var", "test_name", True)
        self.service.data_collector.debug_data_collect_forward.assert_called_with("test_var", "test_name.0")
        self.service.data_collector.debug_data_collect_backward.assert_called_with("test_var", "test_name_grad.0")
    
    def test_save_not_debug_level(self):
        self.service.config.level = Const.LEVEL_L0
        self.service.loop = 1
        self.service.init_step = 0
        self.service.save("test_var", "test_name", True)
        self.service.data_collector.debug_data_collect_forward.assert_not_called()
    
    def test_save_no_dump_step(self):
        self.config.level = Const.LEVEL_DEBUG
        self.service.current_iter = 2 
        self.service.save("test_var", "test_name", True)
        self.service.data_collector.debug_data_collect_forward.assert_not_called()
    
    def test_save_first_time_in_step(self):
        self.service.config.level = Const.LEVEL_DEBUG
        self.service.loop = 1
        self.service.init_step = 0
        
        self.service.save("test_var", "test_name", True)
        
        self.assertEqual(self.service.current_rank, 0)
        self.assertEqual(self.service.debug_variable_counter, {"test_name": 1})
        
        self.assertIsNotNone(self.service.dump_iter_dir)
        self.assertTrue(os.path.exists(self.service.dump_iter_dir))
    
    @patch.object(ApiRegistry, 'register_custom_api')
    def test_register_and_restore_custom_api(self, mock_register_custom_api):
        module_mock = MagicMock()
        api_name = "test_api"
        api_prefix = "test_prefix"
        self.service.register_custom_api(module_mock, api_name, api_prefix)
        key = f"{str(module_mock)}{Const.SEP}{api_name}"
        self.assertIn(key, self.service.ori_customer_func)
        mock_register_custom_api.assert_called_once()
        self.service.restore_custom_api(module_mock, api_name)
        self.assertEqual(module_mock.test_api, self.service.ori_customer_func.get(key))
    
    def test_build_hook(self):
        hook = self.service.build_hook("test_type", "test_name")
        self.service.hook_manager.build_hook.assert_called_with("test_type", "test_name")
    
    def test_create_dirs_pynative_graph(self):
        Runtime.run_mode = Const.PYNATIVE_GRAPH_MODE
        self.service.current_iter = 1
        self.service.current_rank = 0
        
        self.service.create_dirs()
        
        expected_dir = os.path.join(self.config.dump_path, Const.PYNATIVE_MODE, "step1", "rank0")
        self.assertEqual(
            self.service.dump_iter_dir, os.path.join(self.config.dump_path, Const.PYNATIVE_MODE, "step1"))
        self.assertTrue(os.path.exists(expected_dir))

        self.service.data_collector.update_dump_paths.assert_called()
        self.service.data_collector.initialize_json_file.assert_called()

    def test_create_dirs_pynative_mode(self):
        Runtime.run_mode = Const.PYNATIVE_MODE
        self.service.current_iter = 1
        self.service.current_rank = 0
        self.service.create_dirs()
        expected_dir = os.path.join(self.config.dump_path, "step1", "rank0")
        self.assertEqual(self.service.dump_iter_dir, os.path.join(self.config.dump_path, "step1"))
        self.assertTrue(os.path.exists(expected_dir))

    def test_create_dirs_l2_level(self):
        self.service.config.level = Const.LEVEL_L2
        self.service.current_iter = 1
        self.service.current_rank = 0
        self.service.create_dirs()
        expected_dir = os.path.join(self.config.dump_path, "step1")
        self.assertEqual(self.service.dump_iter_dir, expected_dir)
        self.assertTrue(os.path.exists(expected_dir))

        kernel_config_path = os.path.join(expected_dir, f"kernel_config_{os.getpid()}.json")
        self.assertTrue(os.path.exists(kernel_config_path))
        self.assertEqual(self.service.config.kernel_config_path, kernel_config_path)
    
    def test_need_stop_service_conditions(self):
        self.service.current_iter = 4
        self.service.config.step = [1, 2, 3]
        self.assertTrue(self.service._need_stop_service())
        self.assertFalse(Runtime.is_running)
        self.assertFalse(self.service.primitive_switch)

        self.service.current_iter = 1
        self.service.data_collector.data_processor.is_terminated = True
        self.assertTrue(self.service._need_stop_service())
  
        self.service.data_collector.data_processor.is_terminated = False
        self.service.should_stop_service = False
        self.service.current_iter = 1
        self.service.config.step = [1, 2, 3]
        self.assertFalse(self.service._need_stop_service())
    
    def test_register_api_hook(self):
        self.service.config.level = Const.LEVEL_MIX
        self.service._register_api_hook()
        self.service.api_register.initialize_hook.assert_called()
        self.service.api_register.register_all_api.assert_called()
        self.service.logger.info.assert_called_with(
            f"The api {self.config.task} hook function is successfully mounted to the model."
        )
    
    def test_register_infer_count_hook(self):
        model_mock = MagicMock()
        token_range = [5, 10]
        
        self.service._register_infer_count_hook(model_mock, token_range)
        
        model_mock.register_forward_pre_hook.assert_called_once()
        
        hook = model_mock.register_forward_pre_hook.call_args[0][0]
      
        self.service.cur_token_id = 4
        hook(model_mock, None)
        self.assertFalse(Runtime.is_running)
   
        self.service.cur_token_id = 5
        hook(model_mock, None)
        self.assertTrue(Runtime.is_running)

        self.service.cur_token_id = 7
        hook(model_mock, None)
        self.assertTrue(Runtime.is_running)

        self.service.cur_token_id = 11
        hook(model_mock, None)
        self.assertFalse(Runtime.is_running)
    
    def test_process_iteration(self):
        self.service.loop = 5
        self.service.init_step = 10
        self.service._process_iteration()
        
        self.assertEqual(self.service.current_iter, 15)
        self.assertEqual(Runtime.current_iter, 15)
        self.service.data_collector.update_iter.assert_called_with(15)
    
    def test_process_async_dump(self):
        self.service.config.async_dump = True
        self.service.config.task = Const.STATISTICS
        self.service._process_async_dump()
        
        self.service.data_collector.data_processor.dump_async_data.assert_called_once()
    
    def test_process_async_dump_not_needed(self):
        self.service.config.async_dump = False
        self.service._process_async_dump()
        self.service.data_collector.data_processor.dump_async_data.assert_not_called()

    def test_reset_status(self):
        self.service._reset_status()
        self.service.data_collector.reset_status.assert_called_once()
        self.assertEqual(BaseHookManager.params_grad_info, {})
