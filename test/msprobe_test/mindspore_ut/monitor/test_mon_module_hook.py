import pytest
import os
import json
import numpy as np
from datetime import datetime
import unittest
import inspect
from unittest.mock import MagicMock, patch, mock_open
from collections import defaultdict

import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter
from msprobe.core.common.const import MonitorConst, Const
from msprobe.mindspore.monitor.module_hook import (
    TrainerMon,
    ModuleHookContext,
    OptimizerContext,
    GradContext,
    CommunicationContext
)

class MyMomentum(nn.Optimizer):
    def __init__(self, params, learning_rate, momentum=0.9):
        super(MyMomentum, self).__init__(learning_rate, params)
        self.moments = self.parameters.clone(prefix="exp_avg", init="zeros")
        self.momentum = momentum
        self.opt = ops.ApplyMomentum()

    def construct(self, gradients):
        params = self.parameters
        lr = self.get_lr()
        gradients = self.flatten_gradients(gradients)
        gradients = self.decay_weight(gradients)
        gradients = self.gradients_centralization(gradients)
        gradients = self.scale_grad(gradients)

        success = None
        for param, mom, grad in zip(params, self.moments, gradients):
            success = self.opt(param, mom, lr, grad, self.momentum)
        return success


class TestContext(unittest.TestCase):
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

    def test_module_hook_context_initialization(self):
        """测试 ModuleHookContext 初始化状态"""
        ctx = ModuleHookContext(module_name="test_module")
        
        # 验证基本属性
        self.assertEqual(ctx.step, 0)
        self.assertEqual(ctx.micro_step, 0)
        self.assertEqual(ctx.module_name, "test_module")
        self.assertEqual(ctx.stack, "")
        
        # 验证数据结构类型
        self.assertIsInstance(ctx.actv, defaultdict)
        self.assertEqual(len(ctx.actv), 0)  # 应为空字典
        
        self.assertIsInstance(ctx.actvgrad, list)
        self.assertEqual(len(ctx.actvgrad), 0)  # 应为空列表
        
        self.assertIsInstance(ctx.struct, dict)
        self.assertEqual(len(ctx.struct), 0)  # 应为空字典

    def test_module_hook_context_reset(self):
        """测试 ModuleHookContext 重置功能"""
        ctx = ModuleHookContext(module_name="test")
        
        # 填充测试数据
        ctx.step = 5
        ctx.micro_step = 3
        ctx.actv['layer1']['weight'] = [1.2, 3.4]
        ctx.actvgrad.append('grad_data')
        ctx.stack = "test_stack"
        ctx.struct['meta'] = {'size': 10}
        
        # 执行重置
        ctx.reset()
        
        # 验证重置后状态
        self.assertEqual(ctx.step, 5)          # 不应重置
        self.assertEqual(ctx.micro_step, 3)    # 不应重置
        self.assertEqual(len(ctx.actv), 0)      # 字典应清空
        self.assertEqual(len(ctx.actvgrad), 0)  # 列表应清空
        self.assertEqual(ctx.stack, "test_stack")  # 不应重置
        self.assertEqual(len(ctx.struct), 1)    # 不应重置

    def test_optimizer_context_initialization(self):
        """测试 OptimizerContext 初始化状态"""
        ctx = OptimizerContext()
        
        # 验证基本属性
        self.assertEqual(ctx.step, 0)
        
        # 验证所有字典结构均为空
        self.assertIsInstance(ctx.param_mg_direction, defaultdict)
        self.assertEqual(len(ctx.param_mg_direction), 0)
        
        self.assertIsInstance(ctx.param_adam_update, defaultdict)
        self.assertEqual(len(ctx.param_adam_update), 0)
        
        self.assertIsInstance(ctx.param_adam_ratio, defaultdict)
        self.assertEqual(len(ctx.param_adam_ratio), 0)
        
        self.assertIsInstance(ctx.param_weight_grad, defaultdict)
        self.assertEqual(len(ctx.param_weight_grad), 0)
        
        self.assertIsInstance(ctx.param_exp_avg, defaultdict)
        self.assertEqual(len(ctx.param_exp_avg), 0)
        
        self.assertIsInstance(ctx.param_exp_avg_sq, defaultdict)
        self.assertEqual(len(ctx.param_exp_avg_sq), 0)
        
        self.assertIsInstance(ctx.exp_avg_metric, dict)
        self.assertEqual(len(ctx.exp_avg_metric), 0)
        
        self.assertIsInstance(ctx.exp_avg_sq_metric, dict)
        self.assertEqual(len(ctx.exp_avg_sq_metric), 0)
        
        self.assertIsInstance(ctx.metric_dict, dict)
        self.assertEqual(len(ctx.metric_dict), 0)
        
        self.assertIsInstance(ctx.param_metric, dict)
        self.assertEqual(len(ctx.param_metric), 0)

    def test_optimizer_context_reset(self):
        """测试 OptimizerContext 重置功能"""
        ctx = OptimizerContext()
        
        # 填充测试数据
        ctx.step = 100
        ctx.param_mg_direction['weight'] = 0.5
        ctx.param_adam_update['bias'] = (0.1, 0.2)
        ctx.param_adam_ratio['embed'] = 0.8
        ctx.param_weight_grad['linear'] = [-0.4, 0.6]
        ctx.param_exp_avg['conv'] = [0.9]
        ctx.param_exp_avg_sq['norm'] = [0.99]
        ctx.exp_avg_metric['acc'] = 0.75
        ctx.exp_avg_sq_metric['loss'] = 0.25
        ctx.metric_dict['f1'] = 0.9
        ctx.param_metric['weight_metric'] = 1.0
        
        # 执行重置
        ctx.reset()
        
        # 验证重置后状态
        self.assertEqual(ctx.step, 100)  # 不应重置
        
        # 所有字典/默认字典应为空
        self.assertEqual(len(ctx.param_mg_direction), 0)
        self.assertEqual(len(ctx.param_adam_update), 0)
        self.assertEqual(len(ctx.param_adam_ratio), 0)
        self.assertEqual(len(ctx.param_weight_grad), 0)
        self.assertEqual(len(ctx.param_exp_avg), 0)
        self.assertEqual(len(ctx.param_exp_avg_sq), 0)
        self.assertEqual(len(ctx.exp_avg_metric), 0)
        self.assertEqual(len(ctx.exp_avg_sq_metric), 0)
        self.assertEqual(len(ctx.metric_dict), 0)
        self.assertEqual(len(ctx.param_metric), 0)


class TestTrainerMonWithRealNetwork:
    @classmethod
    def setup_class(cls):
        """Setup once for all tests in this class"""
        if ms.hal.device_count("Ascend") == 0:
            pytest.skip("No Ascend NPU device available")

        cls.mock_config = {
            "start_step": 0,
            "collect_times": 10,
            "step_interval": 1,
            "format": "csv",
            "ops": ["norm"],
            "alert": {"rules": [], "dump": False},
            "xy_distribution": True,
            "mv_distribution": True,
            "forward_only": True
        }
        cls.config_file = "test_config.json"
        with open(cls.config_file, 'w') as f:
            json.dump(cls.mock_config, f)

        # Setup real network components
        cls.net = nn.Dense(2, 3)
        cls.loss_fn = nn.MAELoss()
        cls.opt = MyMomentum(cls.net.trainable_params(), 0.01)

    @classmethod
    def teardown_class(cls):
        """Clean up after all tests"""
        if os.path.exists(cls.config_file):
            os.remove(cls.config_file)

    def setup_method(self):
        """Setup before each test"""
        self.trainer = TrainerMon(self.config_file)
        self.trainer.set_monitor(self.net, self.opt)

    def test_monitor_with_real_training_step_when_valid_then_pass(self):
        
        # Create test data
        data = Tensor(np.random.rand(1, 10, 2), ms.float32)
        label = Tensor(np.random.rand(1, 10, 3), ms.float32)

        # Define forward function
        def forward_fn(data, label):
            logits = self.net(data)
            loss = self.loss_fn(logits, label)
            return loss, logits

        # Define grad function
        grad_fn = ms.value_and_grad(forward_fn, None, self.opt.parameters, has_aux=True)

        # Define training step
        def train_step(data, label):
            (loss, _), grads = grad_fn(data, label)
            self.opt(grads)
            return loss

        # Execute training step
        loss = train_step(data, label)

        # Verify monitoring results
        assert isinstance(loss, Tensor)
        assert len(self.trainer.module_fwd_hook_context_by_module) > 0
        assert len(self.trainer.optimizer_context) > 0

    def test_monitor_with_multiple_training_steps_when_valid_then_pass(self):
        
        # Create test data
        data = Tensor(np.random.rand(1, 10, 2), ms.float32)
        label = Tensor(np.random.rand(1, 10, 3), ms.float32)

        # Define forward function
        def forward_fn(data, label):
            logits = self.net(data)
            loss = self.loss_fn(logits, label)
            return loss, logits

        # Define grad function
        grad_fn = ms.value_and_grad(forward_fn, None, self.opt.parameters, has_aux=True)

        # Define training step
        def train_step(data, label):
            (loss, _), grads = grad_fn(data, label)
            self.opt(grads)
            return loss

        # Execute multiple training steps
        for step in range(3):
            loss = train_step(data, label)
            
            # Verify monitoring results
            assert isinstance(loss, Tensor)
            assert len(self.trainer.module_fwd_hook_context_by_module) > 0
            assert len(self.trainer.optimizer_context) > 0
            assert self.trainer.optimizer_context[self.opt].step == step + 1

    def test_monitor_with_parameter_updates_when_valid_then_pass(self):
        # Get initial parameters
        initial_params = [param.value() for param in self.net.get_parameters()]

        # Create test data
        data = Tensor(np.random.rand(1, 10, 2), ms.float32)
        label = Tensor(np.random.rand(1, 10, 3), ms.float32)

        # Define forward function
        def forward_fn(data, label):
            logits = self.net(data)
            loss = self.loss_fn(logits, label)
            return loss, logits

        # Define grad function
        grad_fn = ms.value_and_grad(forward_fn, None, self.opt.parameters, has_aux=True)

        # Define training step
        def train_step(data, label):
            (loss, _), grads = grad_fn(data, label)
            self.opt(grads)
            return loss

        # Execute training step
        loss = train_step(data, label)
        
        # Get updated parameters
        updated_params = [param.value() for param in self.net.get_parameters()]

        # Verify parameters have changed
        for init_param, updated_param in zip(initial_params, updated_params):
            assert not np.array_equal(init_param.asnumpy(), updated_param.asnumpy())

    def test_monitor_with_gradient_collection_when_valid_then_pass(self):
        # Enable gradient monitoring
        self.trainer.wg_distribution = True
        self.monitor_mbs_grad = True
        self.trainer._hook_weights()

        # Create test data
        data = Tensor(np.random.rand(1, 10, 2), ms.float32)
        label = Tensor(np.random.rand(1, 10, 3), ms.float32)

        # Define forward function
        def forward_fn(data, label):
            logits = self.net(data)
            loss = self.loss_fn(logits, label)
            return loss, logits

        # Define grad function
        grad_fn = ms.value_and_grad(forward_fn, None, self.opt.parameters, has_aux=True)

        # Define training step
        def train_step(data, label):
            (loss, _), grads = grad_fn(data, label)
            # Assign to main_grad
            for param, grad in zip(self.opt.parameters, grads):
                param.main_grad = grad
            self.opt(grads)
            return loss

        # Execute training step
        loss = train_step(data, label)
        
        # Verify gradients were collected
        assert len(self.trainer.grad_context.post) > 0

    def test_monitor_with_momentum_collection_when_valid_then_pass(self):
        # Enable momentum monitoring
        self.trainer.mv_distribution = True

        # Create test data
        data = Tensor(np.random.rand(1, 10, 2), ms.float32)
        label = Tensor(np.random.rand(1, 10, 3), ms.float32)

        # Define forward function
        def forward_fn(data, label):
            logits = self.net(data)
            loss = self.loss_fn(logits, label)
            return loss, logits

        # Define grad function
        grad_fn = ms.value_and_grad(forward_fn, None, self.opt.parameters, has_aux=True)

        # Define training step
        def train_step(data, label):
            (loss, _), grads = grad_fn(data, label)
            self.opt(grads)
            return loss

        # Execute training step
        loss = train_step(data, label)
        
        # Verify momentum was collected
        opt_context = self.trainer.optimizer_context[self.opt]
        assert len(opt_context.exp_avg_metric) > 0

    def test_dynamic_monitor_when_change_then_pass(self):
        self.trainer.dynamic_enable = True
        
        # Create test data
        data = Tensor(np.random.rand(1, 10, 2), ms.float32)
        label = Tensor(np.random.rand(1, 10, 3), ms.float32)

        # Define forward function
        def forward_fn(data, label):
            logits = self.net(data)
            loss = self.loss_fn(logits, label)
            return loss, logits

        # Define grad function
        grad_fn = ms.value_and_grad(forward_fn, None, self.opt.parameters, has_aux=True)

        # Define training step
        def train_step(data, label):
            (loss, _), grads = grad_fn(data, label)
            self.opt(grads)
            return loss

        for step in range(3):
            loss = train_step(data, label)
            if step == 0:
                self.mock_config['start_step'] = 2  # 修改为step2
                self.mock_config["collect_times"] = 1
                self.mock_config['dynamic_on'] = True
                with open(self.config_file, 'w') as f:
                    json.dump(self.mock_config, f)
        assert len(self.trainer.module_fwd_hook_context_by_module) > 0
