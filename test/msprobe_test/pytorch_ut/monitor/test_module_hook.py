# Copyright (c) 2024-2025, Huawei Technologies Co., Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os.path
import shutil
import unittest
from unittest.mock import patch, MagicMock

import pandas as pd
import torch
from msprobe.core.common.const import MonitorConst
from torch import distributed as dist

from msprobe.pytorch import TrainerMon
from msprobe.pytorch.dump.api_dump.api_register import get_api_register
from msprobe.pytorch.monitor.module_hook import FeatureHookContext, OptimizerContext, CommunicationContext, \
    GradContext, ModuleHookContext, param_is_not_tensor_parallel_duplicate, param_is_data_parallel_duplicate
from demo_model import monitor_demo

get_api_register().restore_all_api()

base_dir = os.path.dirname(os.path.realpath(__file__))


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
        # validate output file
        output_dir_list = os.listdir(ur_monitor_output)
        self.assertEqual(len(output_dir_list), 1)
        tb_dir = os.listdir(os.path.join(ur_monitor_output, output_dir_list[0]))
        self.assertEqual(len(tb_dir), 1)
        self.assertTrue(tb_dir[0].startswith("events.out.tfevents."))

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
