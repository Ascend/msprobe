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

import os.path
import shutil
import unittest
from unittest.mock import patch, MagicMock

import pandas as pd
import sys
import types
import torch
from msprobe.core.common.const import MonitorConst, Const
from torch import distributed as dist

from msprobe.pytorch import TrainerMon
from msprobe.pytorch.dump.api_dump.api_register import get_api_register
from msprobe.pytorch.monitor.module_hook import FeatureHookContext, OptimizerContext, CommunicationContext, \
    GradContext, ModuleHookContext, param_is_not_tensor_parallel_duplicate, param_is_data_parallel_duplicate
from demo_model import monitor_demo

get_api_register().restore_all_api()

base_dir = os.path.dirname(os.path.realpath(__file__))


class TestTrainerMon(unittest.TestCase):

    def setUp(self):
        self.mon = TrainerMon(config_file_path=base_dir + "/config/xy_config.json")

    def test_init_with_dynamic_env(self):
        os.environ["DYNAMIC_MONITOR"] = "True"
        mon = TrainerMon(config_file_path=base_dir + "/config/xy_config.json")
        self.assertEqual(mon.monitoring, False)
        os.environ["DYNAMIC_MONITOR"] = "False"

    def test_has_register_backward_hook(self):
        # 1. 覆盖返回True分支（所有条件满足，会触发日志但不mock）
        module1 = MagicMock()
        module1._backward_hooks = {1: "hook"}  # 非空
        module1._is_full_backward_hook = False
        self.assertEqual(TrainerMon.has_register_backward_hook("mod1", module1), True)
        # 2. 覆盖返回False分支（_backward_hooks不存在）
        module2 = MagicMock()
        del module2._backward_hooks  # 删除该属性
        self.assertEqual(TrainerMon.has_register_backward_hook("mod2", module2), False)
        # 3. 覆盖返回False分支（_backward_hooks为空）
        module3 = MagicMock()
        module3._backward_hooks = {}  # 空字典
        module3._is_full_backward_hook = False
        self.assertEqual(TrainerMon.has_register_backward_hook("mod3", module3), False)
        # 4. 覆盖返回False分支（_is_full_backward_hook为True）
        module4 = MagicMock()
        module4._backward_hooks = {1: "hook"}
        module4._is_full_backward_hook = True
        self.assertEqual(TrainerMon.has_register_backward_hook("mod4", module4), False)

    def test_get_linear_hook_target(self):
        # 1. 覆盖分支：Embedding模块 → 返回空字符串
        embedding_module = torch.nn.Embedding(10, 5)
        self.assertEqual(TrainerMon.get_linear_hook_target(embedding_module), '')
        # 2. 覆盖分支：有num_embeddings属性 → 返回空字符串
        module_with_num_emb = MagicMock()
        module_with_num_emb.num_embeddings = 10
        self.assertEqual(TrainerMon.get_linear_hook_target(module_with_num_emb), '')
        # 3. 覆盖分支：有vocab_start_index属性 → 返回空字符串
        module_with_vocab = MagicMock()
        module_with_vocab.vocab_start_index = 0
        del module_with_vocab.num_embeddings  # 确保只走vocab分支
        self.assertEqual(TrainerMon.get_linear_hook_target(module_with_vocab), '')
        # 4. 覆盖分支：有weight属性且是2维Tensor → 返回'weight'
        module_with_2d_weight = MagicMock()
        del module_with_2d_weight.num_embeddings, module_with_2d_weight.vocab_start_index
        module_with_2d_weight.weight = torch.randn(10, 5)  # 2维Tensor
        self.assertEqual(TrainerMon.get_linear_hook_target(module_with_2d_weight), 'weight')
        # 5. 覆盖分支：weight是1维Tensor → 继续检查wg
        module_with_1d_weight = MagicMock()
        del module_with_1d_weight.num_embeddings, module_with_1d_weight.vocab_start_index
        module_with_1d_weight.weight = torch.randn(10)  # 1维Tensor
        module_with_1d_weight.wg = torch.randn(8, 4)  # 2维wg
        self.assertEqual(TrainerMon.get_linear_hook_target(module_with_1d_weight), 'wg')
        # 6. 覆盖分支：weight/wg都不满足 → 返回空字符串
        module_no_valid_weight = MagicMock()
        del module_no_valid_weight.num_embeddings, module_no_valid_weight.vocab_start_index
        module_no_valid_weight.weight = "not tensor"  # 非Tensor
        module_no_valid_weight.wg = torch.randn(5)  # 1维Tensor
        self.assertEqual(TrainerMon.get_linear_hook_target(module_no_valid_weight), '')

    def test_monitor_gnorm_with_ad(self):
        # 初始化监测器实例（Mock核心依赖）
        self.mon.set_monitor = MagicMock()  # Mock set_monitor避免执行真实逻辑
        self.mon.logger = MagicMock()  # Mock logger避免日志输出

        # 1. 测试set_wrapped_optimizer：覆盖赋值逻辑
        mock_optimizer = MagicMock()
        self.mon.set_wrapped_optimizer(mock_optimizer)
        self.assertEqual(self.mon.optimizer_trans, mock_optimizer)

        # 2. 测试monitor_gnorm_with_ad：传入optimizer → 调用set_monitor
        mock_model = MagicMock()
        self.mon.monitor_gnorm_with_ad(model=mock_model, optimizer=mock_optimizer)
        self.mon.set_monitor.assert_called_once_with(mock_model, mock_optimizer, 1, None, None, 0)

        # 3. 测试monitor_gnorm_with_ad：optimizer=None但已通过set_wrapped_optimizer设置 → 调用set_monitor
        self.mon.set_monitor.reset_mock()
        self.mon.monitor_gnorm_with_ad(model=mock_model, optimizer=None)
        self.mon.set_monitor.assert_called_once_with(mock_model, mock_optimizer, 1, None, None, 0)

        # 4. 测试monitor_gnorm_with_ad：optimizer=None且未设置optimizer_trans → 输出错误日志并返回
        self.mon.optimizer_trans = None  # 清空已设置的optimizer
        self.mon.set_monitor.reset_mock()
        self.mon.monitor_gnorm_with_ad(model=mock_model, optimizer=None)
        # 验证未调用set_monitor
        self.mon.set_monitor.assert_not_called()

    def test_write_metrics_if_not_empty(self):
        self.mon.summary_writer.write_metrics = MagicMock()
        # 1. 覆盖分支：features为空 → 直接return（不调用write_metrics）
        empty_features = {}
        self.mon.write_metrics_if_not_empty(empty_features, ["entropy"], 0, "attention_hook")
        self.mon.summary_writer.write_metrics.assert_not_called()  # 验证未调用
        self.assertEqual(len(empty_features), 0)

        # 2. 覆盖分支：features为None → 直接return
        self.mon.write_metrics_if_not_empty(None, ["entropy"], 0, "attention_hook")
        self.mon.summary_writer.write_metrics.assert_not_called()

        # 3. 覆盖分支：features非空 + hook_name≠linear_hook → 调用write_metrics+clear
        non_empty_features = {"tag1": "value1"}
        self.mon.write_metrics_if_not_empty(non_empty_features, ["entropy"], 1, "attention_hook")
        # 验证write_metrics被正确调用（参数完全匹配）
        self.mon.summary_writer.write_metrics.assert_called_once_with(
            ["entropy"], non_empty_features, 1, "attention_hook", use_micro_step=True
        )
        # 验证features被清空（调用clear方法）
        self.assertEqual(non_empty_features, {})

        # 4. 覆盖分支：features非空 + hook_name=linear_hook → 调用write_metrics(use_micro_step=False)+clear
        self.mon.summary_writer.write_metrics.reset_mock()  # 重置mock计数
        linear_features = {"tag2": "value2"}
        self.mon.write_metrics_if_not_empty(linear_features, ["sr"], 2, "linear_hook")
        # 验证参数（重点：use_micro_step=False）
        self.mon.summary_writer.write_metrics.assert_called_once_with(
            ["sr"], linear_features, 2, "linear_hook", use_micro_step=False
        )
        self.assertEqual(linear_features, {})

    def test_write_features_tb(self):
        # Mock核心属性/方法
        self.mon.write_metrics_if_not_empty = MagicMock()

        # 1. 覆盖分支：recording_l2_features=False → 直接return
        self.mon.recording_l2_features = False
        self.mon.write_features_tb(step=0)
        self.mon.write_metrics_if_not_empty.assert_not_called()  # 验证无后续调用

        # 2. 覆盖分支：recording_l2_features=True，但context无特征 → 跳过
        self.mon.recording_l2_features = True
        # 模拟空的feature_hook_context_by_module
        empty_context = MagicMock()
        empty_context.attention_feature = {}
        empty_context.linear_feature = {}
        self.mon.feature_hook_context_by_module = {"ctx1": empty_context}

        self.mon.write_features_tb(step=1)
        self.mon.write_metrics_if_not_empty.assert_not_called()  # 验证无调用

        # 3. 覆盖分支：recording_l2_features=True + context有特征 → 调用write_metrics_if_not_empty
        self.mon.write_metrics_if_not_empty.reset_mock()  # 重置mock
        # 模拟有特征的context
        valid_context = MagicMock()
        valid_context.attention_feature = {"tag1": "val1"}  # 非空
        valid_context.linear_feature = {"tag2": "val2"}  # 非空
        self.mon.feature_hook_context_by_module = {"ctx2": valid_context}
        self.mon.write_features_tb(step=2)

        # 4. 覆盖分支：仅attention_feature有值 → 仅调用attention分支
        self.mon.write_metrics_if_not_empty.reset_mock()
        attention_only_context = MagicMock()
        attention_only_context.attention_feature = {"tag3": "val3"}
        attention_only_context.linear_feature = {}  # 空
        self.mon.feature_hook_context_by_module = {"ctx3": attention_only_context}
        self.mon.write_features_tb(step=3)

        # 5. 覆盖分支：仅linear_feature有值 → 仅调用linear分支
        self.mon.write_metrics_if_not_empty.reset_mock()
        linear_only_context = MagicMock()
        linear_only_context.attention_feature = {}
        linear_only_context.linear_feature = {"tag4": "val4"}
        self.mon.feature_hook_context_by_module = {"ctx4": linear_only_context}
        self.mon.write_features_tb(step=4)

    def test_set_wrapped_optimizer(self):
        mock_opt = MagicMock()
        self.mon.set_wrapped_optimizer(mock_opt)
        self.assertEqual(self.mon.optimizer_trans, mock_opt)

        # 传入None → 赋值为None（边界场景）
        self.mon.set_wrapped_optimizer(None)
        self.assertEqual(self.mon.optimizer_trans, None)

    @patch("os.path.getmtime", return_value=123456)
    @patch("json.load", return_value={})
    def test_dynamic_monitor_when_config_updated(self, mock_load, mock_mtime):
        self.mon.dynamic_enable = True
        self.mon.config_timestamp = 0
        self.mon.monitoring = False
        optimizer = MagicMock()
        self.mon.optimizer_context[optimizer] = OptimizerContext()
        self.mon.dynamic_monitor(optimizer)
        self.assertEqual(self.mon.config_timestamp, 123456)

    @patch("msprobe.pytorch.monitor.module_hook.is_recomputation", return_value=False)
    @patch("msprobe.pytorch.monitor.module_hook.get_entropy_metric")
    @patch("msprobe.pytorch.monitor.module_hook.cal_qkt")
    def test_extract_attention_feature_hook(self, mock_cal_qkt, mock_get_entropy_metric, mock_is_recompute):
        self.mon.recording_l2_features = True
        self.mon.print_struct = False
        self.mon.micro_batch_number = 1
        self.mon.build_tbtag_tensor_map = MagicMock(return_value={"tag": torch.tensor(1.0)})

        module = MagicMock()
        submodule = MagicMock()
        module.__dict__['_modules'] = {'sub': submodule}
        module.named_modules.return_value = [("sub", submodule)]
        submodule.register_forward_hook = MagicMock()

        l2_targets = {
            "attention_hook": ["0:sub"],
            "linear_hook": []
        }

        self.mon.feature_hook_context_by_module.clear()
        hooked_count = self.mon._hook_module([], l2_targets, module, vpp_stage="0:")
        self.assertEqual(hooked_count, 3)
        # 查找真正的 attention 特征 hook（排除普通 fwd_hook_fun）
        attention_hook = None
        for call in submodule.register_forward_hook.call_args_list:
            hook = call.args[0]
            base = getattr(hook, "func", hook)
            if getattr(base, "__name__", "") == "extract_attention_feature_hook":
                attention_hook = hook
                break
        self.assertIsNotNone(attention_hook)

        submodule.training = True
        # 覆盖 len(module_input) < 2 分支
        attention_hook(submodule, [torch.randn(2, 2)], None)
        mock_cal_qkt.assert_not_called()

        mock_cal_qkt.return_value = torch.randn(2, 2)
        attention_hook(submodule, [torch.randn(2, 2), torch.randn(2, 2)], None)
        context = self.mon.feature_hook_context_by_module[submodule]
        self.assertEqual(context.step, 1)
        mock_get_entropy_metric.assert_called_once()

    @patch("msprobe.pytorch.monitor.module_hook.is_recomputation", return_value=False)
    @patch("msprobe.pytorch.monitor.module_hook.get_sr_metric")
    def test_extract_linear_sr_hook(self, mock_get_sr_metric, mock_is_recompute):
        self.mon.recording_l2_features = True
        self.mon.print_struct = False
        self.mon.micro_batch_number = 2
        self.mon.build_tbtag_tensor_map = MagicMock(return_value={"tag": torch.tensor(1.0)})

        module = MagicMock()
        submodule = MagicMock()
        module.__dict__['_modules'] = {'sub': submodule}
        module.named_modules.return_value = [("sub", submodule)]
        submodule.register_forward_hook = MagicMock()

        l2_targets = {
            "attention_hook": [],
            "linear_hook": ["0:sub"]
        }

        self.mon.get_linear_hook_target = MagicMock(return_value="weight")
        submodule.weight = torch.randn(2, 2)

        self.mon.feature_hook_context_by_module.clear()
        hooked_count = self.mon._hook_module([], l2_targets, module, vpp_stage="0:")
        self.assertEqual(hooked_count, 2)
        # 查找真正的 linear_sr 特征 hook
        linear_hook = None
        for call in submodule.register_forward_hook.call_args_list:
            hook = call.args[0]
            base = getattr(hook, "func", hook)
            if getattr(base, "__name__", "") == "extract_linear_sr_hook":
                linear_hook = hook
                break
        self.assertIsNotNone(linear_hook)

        submodule.training = True
        # 第一次调用：micro_step 从 0 增加到 1，不触发 sr 计算
        linear_hook(submodule, [torch.randn(2, 2)], None)
        context = self.mon.feature_hook_context_by_module[submodule]
        self.assertEqual(context.micro_step, 1)
        mock_get_sr_metric.assert_not_called()

        # 第二次调用：命中 micro_step == micro_batch_number - 1，触发 sr 计算
        linear_hook(submodule, [torch.randn(2, 2)], None)
        self.assertEqual(context.step, 1)
        mock_get_sr_metric.assert_called_once()

    def test_hook_module_registers_l2_feature_hooks(self):
        self.mon.recording_l2_features = True
        self.mon.print_struct = False

        module = MagicMock()
        submodule = MagicMock()
        module.__dict__['_modules'] = {'sub': submodule}
        module.named_modules.return_value = [("sub", submodule)]
        submodule.register_forward_hook = MagicMock(return_value="handle")

        l2_targets = {
            "attention_hook": ["0:sub"],
            "linear_hook": ["0:sub"]
        }

        hooked_count = self.mon._hook_module([], l2_targets, module, vpp_stage="0:")
        self.assertEqual(hooked_count, 3)
        self.assertEqual(len(self.mon.handles["L2_features"]), 2)

    @patch("torch.distributed.fsdp._runtime_utils._post_backward_hook")
    @patch("importlib.reload")
    @patch("msprobe.pytorch.monitor.module_hook.logger.info")
    @patch("msprobe.pytorch.monitor.module_hook.api_register.restore_api")
    def test_remove_all_hooks(self, mock_restore_api, mock_log_info, mock_reload,
                              mock_fsdp_post_hook):
        # 初始化所有属性
        self.mon.optimizer = MagicMock()

        # 初始化handles（模拟有hook handle）
        mock_handle = MagicMock()
        self.mon.handles = {
            'xy': [mock_handle],
            'L2_features': [mock_handle],
            'wgrads': [mock_handle],
            'cc': [mock_handle]
        }
        # 初始化context（模拟有值）
        mock_fwd_context = MagicMock()
        mock_bwd_context = MagicMock()
        mock_opt_context = MagicMock()
        mock_cc_context = MagicMock()
        self.mon.module_fwd_hook_context_by_module = {"ctx1": mock_fwd_context}
        self.mon.module_bwd_hook_context_by_module = {"ctx2": mock_bwd_context}
        self.mon.optimizer_context = {"opt1": mock_opt_context}
        self.mon.cc_context = {"cc1": mock_cc_context}
        self.mon.grad_context = MagicMock()
        # 初始化FSDP相关属性（覆盖FSDP分支）
        self.mon.fsdp_post_backward_hook = MagicMock()
        self.mon.fsdp2_foreach_reduce = None
        # 初始化优化器相关属性
        self.mon.optimizer_hooked = True
        self.mon.optimizer_mon = MagicMock()
        self.mon.pre_step_hooks = MagicMock()
        self.mon.post_step_hooks = MagicMock()
        # 初始化节点缓存属性（模拟有值）
        self.mon.param2name = MagicMock()
        self.mon.name2indices = MagicMock()
        self.mon.name2param = MagicMock()
        self.mon.duplicate_param = MagicMock()
        self.mon.name2tag = MagicMock()
        self.mon.module_struct = MagicMock()
        self.mon.grad_accs = MagicMock()
        # 初始化采集状态
        self.mon.monitoring = True
        # 执行方法
        self.mon._remove_all_hooks(self.mon.optimizer)
        # 验证节点缓存清空
        self.mon.param2name.clear.assert_called_once()
        self.mon.name2indices.clear.assert_called_once()
        self.mon.name2param.clear.assert_called_once()
        self.mon.duplicate_param.clear.assert_called_once()
        self.mon.name2tag.clear.assert_called_once()
        self.mon.module_struct.clear.assert_called_once()
        self.mon.grad_accs.clear.assert_called_once()
        # 验证采集状态关闭
        self.assertEqual(self.mon.monitoring, False)
        # ------------------------------
        # 覆盖optimizer_hooked=False分支
        # ------------------------------
        self.mon.optimizer_hooked = False
        self.mon.pre_step_hooks.clear.reset_mock()
        self.mon.post_step_hooks.clear.reset_mock()
        self.mon._remove_all_hooks(self.mon.optimizer)
        # 验证pre/post_step_hooks未被清空
        self.mon.pre_step_hooks.clear.assert_not_called()
        self.mon.post_step_hooks.clear.assert_not_called()

    @patch("msprobe.pytorch.monitor.module_hook.load_json")
    @patch("msprobe.pytorch.monitor.module_hook.save_json")
    @patch("os.path.getmtime")
    def test_remove_all_hooks_final_all_lines(self, mock_getmtime,
                                               mock_save_json, mock_load_json):
        # 1. 初始化属性
        self.mon.optimizer = MagicMock()
        self.mon._remove_all_hooks = MagicMock()  # Mock内部调用的_remove_all_hooks
        # ------------------------------
        # 场景1：dynamic_enable=True + 正常执行（无异常）
        # ------------------------------
        self.mon.dynamic_enable = True
        self.mon.config_file_path = "test_config.json"
        mock_load_json.return_value = {"dynamic_on": True}  # 模拟配置文件内容
        mock_getmtime.return_value = 123456789  # 模拟文件时间戳
        # 执行方法
        self.mon._remove_all_hooks_final(self.mon.optimizer)
        # 验证核心逻辑
        # 验证配置加载/修改/保存
        mock_load_json.assert_called_once_with("test_config.json")
        mock_save_json.assert_called_once_with("test_config.json", {"dynamic_on": False}, indent=2)
        # 验证时间戳更新
        mock_getmtime.assert_called_once_with("test_config.json")
        self.assertEqual(self.mon.config_timestamp, 123456789)

        # 验证调用_remove_all_hooks
        self.mon._remove_all_hooks.assert_called_once_with(self.mon.optimizer)
        # ------------------------------
        # 场景2：dynamic_enable=True + 执行异常（触发except）
        # ------------------------------
        # 重置所有Mock状态
        mock_load_json.reset_mock()
        mock_save_json.reset_mock()
        mock_getmtime.reset_mock()
        # 模拟加载配置时抛出异常
        mock_load_json.side_effect = Exception("File not found")
        # 执行方法
        self.mon._remove_all_hooks_final(self.mon.optimizer)
        # 验证_save_json/getmtime未调用（异常中断）
        mock_save_json.assert_not_called()
        mock_getmtime.assert_not_called()
        # ------------------------------
        # 场景3：dynamic_enable=False（跳过配置修改逻辑）
        # ------------------------------
        # 重置所有Mock状态
        mock_load_json.reset_mock()
        mock_save_json.reset_mock()
        mock_getmtime.reset_mock()
        self.mon.dynamic_enable = False
        # 执行方法
        self.mon._remove_all_hooks_final(self.mon.optimizer)
        # 验证核心逻辑
        # 验证配置相关方法未调用
        mock_load_json.assert_not_called()
        mock_save_json.assert_not_called()
        mock_getmtime.assert_not_called()

    @patch("msprobe.pytorch.monitor.module_hook.get_sign_matches")
    @patch("msprobe.pytorch.monitor.module_hook.get_metrics")
    def test_hook_optimizer_mg_direction_branch(self, mock_get_metrics, mock_get_sign_matches):
        class DummyOptimizer:
            def step(self):
                return None

        optimizer = DummyOptimizer()
        self.mon.params_have_main_grad = False
        self.mon.wg_distribution = False
        self.mon.mv_distribution = False
        self.mon.ur_distribution = False
        self.mon.mg_direction = True

        param = MagicMock()
        param.grad = torch.tensor([1.0, -1.0])
        self.mon.param2name = {param: "p1"}

        mv_result = MagicMock()
        mv_result.exp_avg = {"p1": torch.ones_like(param.grad)}
        mv_result.exp_avg_sq = {}
        mv_result.update = {}
        mv_result.ratio = {}

        self.mon.optimizer_mon = MagicMock()
        self.mon.optimizer_mon.fetch_mv.return_value = mv_result
        self.mon.generate_wgrad_metrics = MagicMock(return_value=({}, {}))
        self.mon.generate_mv_metrics = MagicMock()
        self.mon.generate_param_metrics = MagicMock()
        self.mon.generate_param_map = MagicMock(return_value={})

        context = self.mon.optimizer_context[optimizer]
        context.step = 0

        self.mon.hook_optimizer(optimizer)
        self.assertTrue(self.mon.optimizer_hooked)
        pre_hook = self.mon.pre_step_hooks[-1]

        mock_get_sign_matches.return_value = torch.tensor(0.5)
        pre_hook(optimizer, (), {})
        self.assertIn("p1", context.param_mg_direction)
        self.assertTrue(torch.equal(context.param_mg_direction["p1"], torch.tensor(1.0)))

        context.step = 1
        context.param_mg_direction.clear()
        pre_hook(optimizer, (), {})
        mock_get_sign_matches.assert_called_once()
        self.assertTrue(torch.equal(context.param_mg_direction["p1"], mock_get_sign_matches.return_value))

    def test_hook_optimizer_patches_instance_step_only(self):
        class DummyOptimizer:
            def step(self):
                return "original_step"

        optimizer1 = DummyOptimizer()
        
        self.mon.params_have_main_grad = False
        self.mon.wg_distribution = False
        self.mon.mv_distribution = False
        self.mon.ur_distribution = False
        self.mon.mg_direction = False
        
        original_step_method = DummyOptimizer.step
        self.mon.hook_step_final(optimizer1)
        
        self.assertNotEqual(optimizer1.step, original_step_method)
        result = optimizer1.step()
        self.assertEqual(result, "original_step")

    @patch("msprobe.pytorch.monitor.module_hook.get_metrics")
    @patch("torch.distributed.fsdp._runtime_utils._post_backward_hook")
    def test_patch_fsdp_post_backward_hook(self, mock_post_hook, mock_get_metrics):
        from msprobe.core.common.const import MonitorConst as MC

        class DummyFlatParam:
            def __init__(self, param):
                self._fqns = ["param"]
                self._shapes = [(2, 2)]
                self.grad = torch.arange(4.0)
                self._params = [param]

        class DummyHandle:
            def __init__(self, param):
                self.flat_param = DummyFlatParam(param)

            def _get_flat_param_offsets(self):
                return [(0, 3)]

        class DummyModel:
            def __init__(self, handle):
                self._all_handles = [handle]

        flat_prefix = "0:"
        full_name = f"{flat_prefix}{MC.FSDP_FLAT_SEP}param"
        self.mon.origin2squash = {full_name: "sq_param"}
        param = [torch.nn.Parameter(torch.tensor(1.0))]
        self.mon.fsdp_param_name_map = {id(param): full_name}
        self.mon.name2tag = {"sq_param": {MC.PRE_GRAD: "pre_grad_tag"}}
        self.mon.flat_prefix_reverse_iter = iter([flat_prefix])
        self.mon.ops = []
        self.mon.eps = 1e-8

        self.mon._patch_fsdp_post_backward_hook()

        from torch.distributed.fsdp import _runtime_utils
        wrapped = _runtime_utils._post_backward_hook
        handle = DummyHandle(param)
        self.mon.model = [DummyModel(handle)]
        wrapped(None, handle)

        mock_post_hook.assert_called_once()
        args, _ = mock_get_metrics.call_args
        grad_dict = args[1]
        self.assertIn("pre_grad_tag", grad_dict)

    @patch("msprobe.pytorch.monitor.module_hook.importlib.reload")
    @patch("msprobe.pytorch.monitor.module_hook.get_metrics")
    def test_patch_fsdp2_foreach_reduce(self, mock_get_metrics, mock_reload):
        from msprobe.core.common.const import MonitorConst as MC

        # 构造假的 fully_shard 子模块，放入 sys.modules，避免环境缺少该模块时报错
        fake_collectives = types.SimpleNamespace()
        original_called = {"flag": False}

        def original_foreach_reduce(fsdp_params, unsharded_grads, *unused):
            original_called["flag"] = True

        fake_collectives.foreach_reduce = MagicMock(side_effect=original_foreach_reduce)
        fake_param_group = types.SimpleNamespace()
        fake_fully_shard = types.SimpleNamespace(
            _fsdp_collectives=fake_collectives,
            _fsdp_param_group=fake_param_group,
        )

        with patch.dict(sys.modules, {
            "torch.distributed.fsdp._fully_shard": fake_fully_shard,
            "torch.distributed.fsdp._fully_shard._fsdp_collectives": fake_collectives,
            "torch.distributed.fsdp._fully_shard._fsdp_param_group": fake_param_group,
        }):
            import torch.distributed.fsdp as real_fsdp
            real_fsdp._fully_shard = fake_fully_shard
            self.mon.origin2squash = {"param_fqn": "sq_param"}
            self.mon.name2tag = {"sq_param": {MC.PRE_GRAD: "pre_grad_tag"}}
            self.mon.ops = []
            self.mon.eps = 1e-8
            self.mon.monitor_mbs_grad = True
            self.mon._patch_fsdp2_foreach_reduce()

            # patch 后的 foreach_reduce
            wrapped = fake_collectives.foreach_reduce

            param = MagicMock()
            param._param_fqn = "param_fqn"
            grad = torch.tensor([1.0, 2.0])

            wrapped([param], [grad])

        args, _ = mock_get_metrics.call_args
        grad_dict = args[1]
        self.assertIn("pre_grad_tag", grad_dict)
        self.assertTrue(original_called["flag"])

    def test_is_recording_module(self):
        # 1. 初始化核心属性
        self.mon.squash_name = True  # 模拟squash_name配置
        # ------------------------------
        # 场景1：l2_targets非空 + 第一个pattern匹配
        # ------------------------------
        l2_targets = ["stage1mod1"]
        result = self.mon._is_recording_module(
            module_name="mod1",
            l2_targets=l2_targets,
            vpp_stage="stage1",
            hook_name="any_hook"
        )
        # 验证返回匹配的pattern
        self.assertEqual(result, "stage1mod1")

        # ------------------------------
        # 场景3：l2_targets非空 + 无匹配pattern
        # ------------------------------
        l2_targets = ["stage1mod2"]
        result = self.mon._is_recording_module(
            module_name="mod1",
            l2_targets=l2_targets,
            vpp_stage="stage1",
            hook_name="any_hook"
        )
        self.assertEqual(result, "")

        # ------------------------------
        # 场景4：l2_targets为空 + hook_name=linear_hook
        # ------------------------------
        l2_targets = []
        result = self.mon._is_recording_module(
            module_name="mod1",
            l2_targets=l2_targets,
            vpp_stage="stage1",
            hook_name="linear_hook"
        )
        self.assertEqual(result, "stage1mod1")

        # ------------------------------
        # 场景5：l2_targets为空 + hook_name≠linear_hook
        # ------------------------------
        l2_targets = []
        result = self.mon._is_recording_module(
            module_name="mod1",
            l2_targets=l2_targets,
            vpp_stage="stage1",
            hook_name="attention_hook"
        )
        self.assertEqual(result, "")


def clean_output(path):
    if os.path.exists(path):
        shutil.rmtree(path)


class TestModuleHook(unittest.TestCase):
    monitor_output = "./monitor_output"
    ori_dist_is_initialized = dist.is_initialized
    ori_dist_get_rank = dist.get_rank
    ori_dist_get_process_group_ranks = dist.get_process_group_ranks

    @classmethod
    def tearDownClass(cls):
        dist.is_initialized = cls.ori_dist_is_initialized
        dist.get_rank = cls.ori_dist_get_rank
        dist.get_process_group_ranks = cls.ori_dist_get_process_group_ranks

    @staticmethod
    def get_dist_mock(initialized=False):
        dist_mock = MagicMock()
        dist_mock.is_initialized.return_value = initialized
        dist_mock.get_rank.return_value = 0
        dist_mock.get_process_group_ranks.return_value = [0]

        dist.is_initialized = dist_mock.is_initialized
        dist.get_rank = dist_mock.get_rank
        dist.get_process_group_ranks = dist_mock.get_process_group_ranks

    def test_smallest_rank_print(self):
        xy_config = os.path.join(base_dir, "config/xy_config.json")
        hooker = TrainerMon(
            xy_config,
            params_have_main_grad=False
        )
        self.get_dist_mock(True)

        hooker._smallest_rank_print("test print")

        hooker.module_rank_list = [0]
        hooker._smallest_rank_print("test print")
        self.assertIsNotNone(hooker)

    def test_print_struct(self):
        print_struct_config = os.path.join(base_dir, "config/struct_config.json")
        self.get_dist_mock(False)

        with self.assertRaises(Exception) as context:
            monitor_demo(print_struct_config)
        self.assertEqual(str(context.exception), "exit after first monitor step when print model struct")

    def test_xy_distribution(self):
        xy_monitor_output = "./test_xy_distribution"
        clean_output(xy_monitor_output)
        os.environ[MonitorConst.MONITOR_OUTPUT_DIR] = xy_monitor_output
        xy_config = os.path.join(base_dir, "config/xy_config.json")
        monitor_demo(xy_config)
        # validate output file
        output_dir_list = os.listdir(xy_monitor_output)
        self.assertEqual(len(output_dir_list), 1)
        actv_0_csv = os.path.join(xy_monitor_output, output_dir_list[0], "actv_0-0.csv")
        actv_grad_0_csv = os.path.join(xy_monitor_output, output_dir_list[0], "actv_grad_0-0.csv")
        self.assertTrue(os.path.exists(actv_0_csv))
        self.assertTrue(os.path.exists(actv_grad_0_csv))
        # validate columns and lines
        actv_0 = pd.read_csv(actv_0_csv)
        expect_columns = ['vpp_stage', 'name', 'step', 'micro_step', 'norm', 'nans', "shape", "dtype"]
        self.assertListEqual(list(actv_0.columns), expect_columns)
        self.assertEqual(actv_0.shape, tuple([6, 8]))
        actv_grad_0 = pd.read_csv(actv_grad_0_csv)
        expect_columns = ['vpp_stage', 'name', 'step', 'micro_step', 'norm', 'nans', "shape", "dtype"]
        self.assertListEqual(list(actv_grad_0.columns), expect_columns)
        self.assertEqual(actv_0.shape, tuple([6, 8]))

    def test_wg_distribution(self):
        self.get_dist_mock(False)
        wg_monitor_output = "./test_wg_distribution"
        clean_output(wg_monitor_output)
        os.environ[MonitorConst.MONITOR_OUTPUT_DIR] = wg_monitor_output
        mv_config = os.path.join(base_dir, "config/wg_config.json")
        monitor_demo(mv_config)
        # validate output file
        output_dir_list = os.listdir(wg_monitor_output)
        self.assertEqual(len(output_dir_list), 1)
        grad_reduced_0_csv = os.path.join(wg_monitor_output, output_dir_list[0], "grad_reduced_0-0.csv")
        grad_unreduced_0_csv = os.path.join(wg_monitor_output, output_dir_list[0], "grad_unreduced_0-0.csv")
        self.assertTrue(os.path.exists(grad_reduced_0_csv))
        self.assertTrue(os.path.exists(grad_unreduced_0_csv))
        # validate columns and lines
        expect_columns = ["vpp_stage", "name", "step", "norm", "shape", "dtype"]
        grad_reduced_0 = pd.read_csv(grad_reduced_0_csv)
        self.assertListEqual(list(grad_reduced_0.columns), expect_columns)
        self.assertEqual(grad_reduced_0.shape, tuple([2, 6]))
        grad_unreduced_0 = pd.read_csv(grad_unreduced_0_csv)
        self.assertListEqual(list(grad_unreduced_0.columns), expect_columns)
        self.assertEqual(grad_unreduced_0.shape, tuple([2, 6]))

    def test_mv_distribution(self):
        self.get_dist_mock(False)
        mv_monitor_output = "./test_mv_distribution"
        clean_output(mv_monitor_output)
        os.environ[MonitorConst.MONITOR_OUTPUT_DIR] = mv_monitor_output
        mv_config = os.path.join(base_dir, "config/mv_config.json")
        monitor_demo(mv_config)
        # validate output file
        output_dir_list = os.listdir(mv_monitor_output)
        self.assertEqual(len(output_dir_list), 1)
        exp_avg_1_csv = os.path.join(mv_monitor_output, output_dir_list[0], "exp_avg_1-1.csv")
        exp_avg_sq_1_csv = os.path.join(mv_monitor_output, output_dir_list[0], "exp_avg_sq_1-1.csv")
        self.assertTrue(os.path.exists(exp_avg_1_csv))
        self.assertTrue(os.path.exists(exp_avg_sq_1_csv))
        # validate columns and lines
        expect_columns = ["vpp_stage", "name", "step", "norm", "shape", "dtype"]
        exp_avg_1 = pd.read_csv(exp_avg_1_csv)
        self.assertListEqual(list(exp_avg_1.columns), expect_columns)
        self.assertEqual(exp_avg_1.shape, tuple([2, 6]))
        exp_avg_sq_1 = pd.read_csv(exp_avg_sq_1_csv)
        self.assertListEqual(list(exp_avg_sq_1.columns), expect_columns)
        self.assertEqual(exp_avg_sq_1.shape, tuple([2, 6]))

    def test_ur_distribution(self):
        self.get_dist_mock(False)
        ur_monitor_output = "./test_ur_distribution"
        clean_output(ur_monitor_output)
        os.environ[MonitorConst.MONITOR_OUTPUT_DIR] = ur_monitor_output
        ur_config = os.path.join(base_dir, "config/ur_config.json")
        monitor_demo(ur_config)
        # ur_distribution is cancelled with CSV format, validate CSV output from xy_distribution instead
        output_dir_list = os.listdir(ur_monitor_output)
        self.assertEqual(len(output_dir_list), 1)
        actv_0_csv = os.path.join(ur_monitor_output, output_dir_list[0], "actv_0-0.csv")
        self.assertTrue(os.path.exists(actv_0_csv))

    def test_cc_distribution(self):
        cc_config = os.path.join(base_dir, "config/cc_config.json")
        self.get_dist_mock(True)
        hooker = TrainerMon(
            cc_config,
            params_have_main_grad=False
        )
        self.assertIsNotNone(hooker)

    def test_stack_collect(self):
        self.get_dist_mock(False)
        stack_monitor_output = "./test_stack_info"
        clean_output(stack_monitor_output)
        os.environ[MonitorConst.MONITOR_OUTPUT_DIR] = stack_monitor_output
        stack_config = os.path.join(base_dir, "config/stack_config.json")
        monitor_demo(stack_config)
        output_dir_list = os.listdir(stack_monitor_output)
        self.assertEqual(len(output_dir_list), 1)
        stack_csv_path = os.path.join(stack_monitor_output, output_dir_list[0], "stack_info.csv")
        self.assertTrue(os.path.exists(stack_csv_path))

    def test_adhoc_check(self):
        # mock dist
        self.get_dist_mock(True)
        target_tensor = torch.randn(10)
        module_name = 'test_module'
        tensor_name = 'test_tensor'
        rank_list = [1, 2]
        ops_list = ['max', 'min']
        cc_config = os.path.join(base_dir, "config/cc_config.json")
        hooker = TrainerMon(cc_config, params_have_main_grad=False)
        hooker.adhoc_check(target_tensor, module_name, tensor_name, rank_list, ops_list)

    def test_generate_cc_metrics(self):
        self.get_dist_mock(True)

        cc_name = 'test_cc'
        cc_tensor = CommunicationContext()
        cc_tensor.data = {
            'min': {
                'tag1': 'tensor1',
                'tag2': 'tensor2'
            },
            'max': {
                'tag3': 'tensor3',
                'tag4': 'tensor4'
            }
        }
        expected_metrics = {'min': {'test_cc/rank0/tag1': 'tensor1', 'test_cc/rank0/tag2': 'tensor2'},
                            'max': {'test_cc/rank0/tag3': 'tensor3', 'test_cc/rank0/tag4': 'tensor4'}}
        result = TrainerMon.generate_cc_metrics(cc_name, cc_tensor)
        self.assertDictEqual(result, expected_metrics)

    def test_generate_xy_metrics(self):
        xy_config = os.path.join(base_dir, "config/xy_config.json")
        trainer_mon = TrainerMon(
            xy_config,
            params_have_main_grad=False
        )

        fwd_context = ModuleHookContext("module1")
        fwd_context.actv = {'module1': 'value1'}
        trainer_mon.module_fwd_hook_context_by_module = {'module1': fwd_context}
        trainer_mon.grad_context.actv = {'module2': 'value2'}

        actv, actv_grad = trainer_mon.generate_xy_metrics()
        self.assertEqual(actv, {'module1': 'value1'})
        self.assertEqual(actv_grad, {'module2': 'value2'})

    def test_reload_xy(self):
        xy_config = os.path.join(base_dir, "config/xy_config.json")
        trainer_mon = TrainerMon(
            xy_config,
            params_have_main_grad=False
        )
        trainer_mon.rank = 0
        trainer_mon.module_rank_list = [1, 2]
        trainer_mon.handles = {'xy': []}
        trainer_mon.module_fwd_hook_context_by_module = {"a": ModuleHookContext("test")}
        trainer_mon.hook_modules = MagicMock()

        handle = MagicMock()
        trainer_mon.handles['xy'].append(handle)
        trainer_mon.reload_xy()
        self.assertEqual(trainer_mon.handles['xy'], [])


class TestParamIsNotTensorParallelDuplicate(unittest.TestCase):
    @patch('torch.distributed.get_rank')
    def test_param_is_not_tensor_parallel_duplicate(self, mock_get_rank):
        class MockParam:
            def __init__(self, tensor_model_parallel):
                self.tensor_model_parallel = tensor_model_parallel

        param = MockParam(True)
        tp_group = 'dummy_group'
        self.assertTrue(param_is_not_tensor_parallel_duplicate(param, tp_group))


class TestParamIsDataParallelDuplicate(unittest.TestCase):
    @patch('torch.distributed.get_rank')
    def test_param_is_data_parallel_duplicate_true(self, mock_get_rank):
        mock_get_rank.return_value = 1
        dp_group = 'dp_group'
        result = param_is_data_parallel_duplicate(dp_group)
        self.assertTrue(result)

    @patch('torch.distributed.get_rank')
    def test_param_is_data_parallel_duplicate_false(self, mock_get_rank):
        mock_get_rank.return_value = 0
        dp_group = 'dp_group'
        result = param_is_data_parallel_duplicate(dp_group)
        self.assertFalse(result)


class TestContext(unittest.TestCase):

    def test_module_hook_context(self):
        module_ctx = ModuleHookContext("linear")
        module_ctx.reset()
        self.assertEqual(module_ctx.actv, {})
        self.assertEqual(module_ctx.actvgrad, [])

    def test_feature_context(self):
        feature_ctx = FeatureHookContext("linear")
        feature_ctx.reset()
        self.assertEqual(feature_ctx.attention_feature, {})
        self.assertEqual(feature_ctx.linear_feature, {})

    def test_optimizer_context(self):
        optimizer_ctx = OptimizerContext()
        optimizer_ctx.reset()
        self.assertEqual(optimizer_ctx.param_mg_direction, {})
        self.assertEqual(optimizer_ctx.param_adam_update, {})
        self.assertEqual(optimizer_ctx.param_adam_ratio, {})
        self.assertEqual(optimizer_ctx.param_weight_grad, {})
        self.assertEqual(optimizer_ctx.param_exp_avg, {})
        self.assertEqual(optimizer_ctx.exp_avg_metric, {})
        self.assertEqual(optimizer_ctx.param_exp_avg_sq, {})
        self.assertEqual(optimizer_ctx.exp_avg_sq_metric, {})
        self.assertEqual(optimizer_ctx.metric_dict, {})
        self.assertEqual(optimizer_ctx.param_metric, {})

    def test_communication_context(self):
        cc_ctx = CommunicationContext()
        cc_ctx.reset()
        cc_ctx.data = {'tag1': {'min': [1, 2, 3], 'max': [10, 11, 12]},
                       'tag2': {'min': [16, 17, 18], 'max': [22, 23, 24]}}
        cc_ctx.aggregate()
        expected_aggregated_data = {'tag1': {'max': 12, 'min': 1}, 'tag2': {'max': 24, 'min': 16}}
        self.assertEqual(cc_ctx.data, expected_aggregated_data)

    def test_grad_context(self):
        grad_ctx = GradContext()
        grad_ctx.reset()
        self.assertEqual(grad_ctx.pre, {})
        self.assertEqual(grad_ctx.post, {})


if __name__ == '__main__':
    unittest.main()