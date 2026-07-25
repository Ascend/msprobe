import os
import unittest
from unittest.mock import patch, MagicMock

import torch
from msprobe.core.common.const import MonitorConst

from msprobe.core.monitor.utils import get_output_base_dir, filter_special_chars, MsgConst, validate_ops, \
    validate_ndigits, validate_ranks, validate_targets, validate_print_struct, validate_ur_distribution, \
    validate_xy_distribution, validate_wg_distribution, validate_mg_distribution, validate_param_distribution, \
    validate_cc_distribution, validate_squash_name, validate_alert, validate_step_count_per_record, \
    validate_dynamic_on, validate_monitor_mbs_grad, validate_append_output, validate_config, time_str2time_digit, \
    get_target_output_dir, validate_l2_targets, validate_recording_l2_features, validate_sa_order, \
    validate_set_monitor, validate_int_arg
from msprobe.pytorch.monitor.utils import get_nan_tensor, get_param_struct
from msprobe.pytorch.common.utils import is_recomputation


class TestMonitorUtils(unittest.TestCase):
    @patch.multiple(
        "msprobe.pytorch.monitor.utils", device="cpu", NAN_TENSOR_ON_DEVICE=None
    )
    def test_get_nan_tensor(self):
        result = get_nan_tensor()
        self.assertTrue(torch.isnan(result).item())

    def test_get_param_struct(self):
        param = (torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]))
        res = get_param_struct(param)
        self.assertEqual(res['config'], 'tuple[2]')

        param = (torch.randn(1), 42, "abc")
        result = get_param_struct(param)
        self.assertEqual(result['config'], 'tuple[3]')
        self.assertEqual(result[0], f'size={(1,)}, dtype={param[0].dtype}')
        self.assertEqual(result[1], "<class 'int'>")
        self.assertEqual(result[2], "<class 'str'>")

        param = torch.randn(2, 3)
        result = get_param_struct(param)
        self.assertEqual(result['config'], 'tensor')
        self.assertEqual(result['tensor'], f'size={(2, 3)}, dtype={param.dtype}')

        param = {"a": 1}
        result = get_param_struct(param)
        self.assertEqual(result['config'], "<class 'dict'>")


class TestCoreMonitorUtils(unittest.TestCase):

    def test_get_output_base_dir(self):
        # not set env
        if os.getenv(MonitorConst.MONITOR_OUTPUT_DIR):
            del os.environ[MonitorConst.MONITOR_OUTPUT_DIR]
        output_base_dir = get_output_base_dir()
        expect_output_base_dir = "./monitor_output"
        self.assertEqual(output_base_dir, expect_output_base_dir)

        # set env
        os.environ[MonitorConst.MONITOR_OUTPUT_DIR] = "test123"
        output_base_dir = get_output_base_dir()
        expect_output_base_dir = "test123"
        self.assertEqual(output_base_dir, expect_output_base_dir)

    def test_filter_special_chars(self):
        @filter_special_chars
        def func(msg):
            return msg

        self.assertEqual(func(MsgConst.SPECIAL_CHAR[0]), '_')

    def test_validate_ops(self):
        ops = ['op1', 'op2', 'norm', 'max']
        valid_ops = validate_ops(ops)
        self.assertEqual(valid_ops, ['norm', 'max', "shape", "dtype"])

    def test_no_valid_ops(self):
        ops = ['op1', 'op2']
        valid_ops = validate_ops(ops)
        target_ops = [MonitorConst.OP_LIST[0], "shape", "dtype"]
        self.assertEqual(valid_ops, target_ops)

    def test_validate_ndigits(self):
        validate_ndigits(None)
        validate_ndigits(0)
        validate_ndigits(MonitorConst.MAX_NDIGITS)

        with self.assertRaises(ValueError):
            validate_ndigits(3.5)
        with self.assertRaises(ValueError):
            validate_ndigits("abc")
        with self.assertRaises(ValueError):
            validate_ndigits(True)

        with self.assertRaises(ValueError):
            validate_ndigits(-1)
        with self.assertRaises(ValueError):
            validate_ndigits(MonitorConst.MAX_NDIGITS + 1)

    def test_validate_ranks(self):
        ranks = [0, 1, 2, 3]
        res = validate_ranks(ranks)
        self.assertIsNone(res)

        with self.assertRaises(TypeError):
            ranks = ["xxx", 1, 2, 3]
            validate_ranks(ranks)

    def test_validate_targets(self):
        targets = {'module_name': {'input': 'tensor'}}
        validate_targets(targets)

    def test_validate_print_struct(self):
        print_struct = True
        validate_print_struct(print_struct)

        with self.assertRaises(TypeError):
            print_struct = 2
            validate_print_struct(print_struct)

    def test_validate_ur_distribution(self):
        ur_distribution = True
        validate_ur_distribution(ur_distribution)

        with self.assertRaises(TypeError):
            ur_distribution = 2
            validate_ur_distribution(ur_distribution)

    def test_validate_xy_distribution(self):
        xy_distribution = True
        validate_xy_distribution(xy_distribution)

        with self.assertRaises(TypeError):
            xy_distribution = 2
            validate_xy_distribution(xy_distribution)

    def test_validate_wg_distribution(self):
        wg_distribution = True
        validate_wg_distribution(wg_distribution)

        with self.assertRaises(TypeError):
            wg_distribution = 2
            validate_wg_distribution(wg_distribution)

    def test_validate_mg_distribution(self):
        mg_distribution = True
        validate_mg_distribution(mg_distribution)

        with self.assertRaises(TypeError):
            mg_distribution = 2
            validate_mg_distribution(mg_distribution)

    def test_validate_param_distribution(self):
        param_distribution = True
        validate_param_distribution(param_distribution)

        with self.assertRaises(TypeError):
            param_distribution = 2
            validate_param_distribution(param_distribution)

    def test_validate_cc_distribution(self):
        cc_distribution = {'enable': True, 'cc_codeline': ['line1'], 'cc_pre_hook': False, 'cc_log_only': True}
        validate_cc_distribution(cc_distribution)

    def test_validate_squash_name(self):
        squash_name = True
        validate_squash_name(squash_name)

        with self.assertRaises(TypeError):
            squash_name = 2
            validate_squash_name(squash_name)

    def test_validate_alert(self):
        alert = {'rules': [{'rule_name': 'AnomalyTurbulence', 'args': {'threshold': 10.0}}], 'dump': True}
        validate_alert(alert)

    def test_validate_step_count_per_record(self):
        validate_step_count_per_record(10)
        with self.assertRaises(TypeError):
            validate_step_count_per_record("10")
        with self.assertRaises(ValueError):
            validate_step_count_per_record(0)
        with self.assertRaises(ValueError):
            validate_step_count_per_record(100000000)

    def test_validate_dynamic_on(self):
        dynamic_on = True
        validate_dynamic_on(dynamic_on)

        with self.assertRaises(TypeError):
            dynamic_on = 2
            validate_dynamic_on(dynamic_on)

    def test_validate_monitor_mbs_grad(self):
        monitor_mbs_grad = True
        result = validate_monitor_mbs_grad(monitor_mbs_grad)
        self.assertEqual(result, True)

        monitor_mbs_grad = 2
        result = validate_monitor_mbs_grad(monitor_mbs_grad)
        self.assertEqual(result, False)

    def test_validate_append_output(self):
        append_output = [1, 2]
        validate_append_output(append_output)

        with self.assertRaises(TypeError):
            append_output = 2
            validate_append_output(append_output)

        with self.assertRaises(ValueError):
            append_output = [1, 2, 3]
            validate_append_output(append_output)

    def test_validate_config(self):
        config = {
            'ops': ['op1', 'op2'],
            'eps': 1e-8,
            'module_ranks': [0, 1, 2, 3],
            'targets': {'module_name': {'input': 'tensor'}},
            'print_struct': True,
            'ur_distribution': True,
            'xy_distribution': True,
            'wg_distribution': True,
            'mg_distribution': True,
            'cc_distribution': {'enable': True, 'cc_codeline': ['line1'], 'cc_pre_hook': False, 'cc_log_only': True},
            'alert': {'rules': [{'rule_name': 'AnomalyTurbulence', 'args': {'threshold': 10.0}}], 'dump': True}
        }
        validate_config(config)
        target_ops = [MonitorConst.OP_LIST[0], "shape", "dtype"]
        self.assertEqual(config["ops"], target_ops)
        del config["targets"]
        validate_config(config)
        self.assertEqual(config["targets"], {"": {}})
        self.assertEqual(config["all_xy"], True)

    def test_str2time_digit(self):
        time_str = "Dec03_21-34-40"
        time_str2time_digit(time_str)

        with self.assertRaises(TypeError):
            time_str2time_digit(12345)
        with self.assertRaises(TypeError):
            time_str2time_digit(None)
        with self.assertRaises(TypeError):
            time_str2time_digit(["Dec03_21-34-40"])

        with self.assertRaises(RuntimeError):
            time_str2time_digit("2025-11-28_21:34:40")  # 错误格式的字符串
        with self.assertRaises(RuntimeError):
            time_str2time_digit("Dec03-21-34-40")  # 少了下划线
        with self.assertRaises(RuntimeError):
            time_str2time_digit("Dec32_25-61-61")  # 无效日期时间

    # ===== validate_l2_targets 测试 =====
    def test_validate_l2_targets_valid_input(self):
        """测试合法输入"""
        valid_targets = {
            "attention_hook": ["0:0.self_attention.core_attention.flash_attention"],
            "linear_hook": []
        }
        validate_l2_targets(valid_targets)

    def test_validate_l2_targets_invalid_root_type(self):
        """测试非 dict 输入"""
        with self.assertRaises(TypeError) as cm:
            validate_l2_targets("not_a_dict")
        self.assertEqual(str(cm.exception),
                         'l2_targets in config.json should be a dict')

    def test_validate_l2_targets_invalid_hook_name(self):
        """测试非法 hook_name"""
        with self.assertRaises(TypeError) as cm:
            validate_l2_targets({"invalid_hook": ["module1"]})
        self.assertIn(f'key of l2_targtes must be in {MonitorConst.L2_HOOKS}',
                      str(cm.exception))

    def test_validate_l2_targets_invalid_value_type(self):
        """测试非法 value 类型"""
        with self.assertRaises(TypeError) as cm:
            validate_l2_targets({"linear_hook": "not_a_list"})
        self.assertEqual(str(cm.exception),
                         'values of l2_targets should be a list in config.json')

    def test_validate_l2_targets_invalid_item_type(self):
        """测试非法 list item 类型"""
        with self.assertRaises(TypeError) as cm:
            validate_l2_targets({"linear_hook": [123]})
        self.assertEqual(str(cm.exception),
                         'item of "linear_hook" in l2_targets should be module_name[str] in config.json')

    # ===== validate_recording_l2_features 测试 =====
    def test_validate_recording_l2_features_valid(self):
        """测试合法布尔值输入"""
        validate_recording_l2_features(True)
        validate_recording_l2_features(False)

    def test_validate_recording_l2_features_invalid_type(self):
        """测试非法类型输入"""
        with self.assertRaises(TypeError) as cm:
            validate_recording_l2_features("xx")
            self.assertEqual(str(cm.exception),
                             "recording_l2_features should be a bool")

    def test_valid_orders(self):
        validate_sa_order("b,s,h,d")
        validate_sa_order("s, b,h,  d")

    def test_invalid_orders(self):
        with self.assertRaises(TypeError) as cm:
            validate_sa_order("xx")
            self.assertEqual(str(cm.exception),
                             f'sa_order must be in {MonitorConst.SA_ORDERS}, got xx')

    def test_validate_set_monitor(self):
        grad_acc_steps = 8
        start_iteration = 1
        result = validate_set_monitor(grad_acc_steps, start_iteration)
        self.assertEqual(result[0], grad_acc_steps)
        self.assertEqual(result[1], start_iteration)

    def test_validate_int_arg(self):
        default = 10
        result = validate_int_arg(None, "arg", 0, default)
        self.assertEqual(result, default)
        result = validate_int_arg(5, "arg", 0, 10)
        self.assertEqual(result, 5)
        # 非整数
        result = validate_int_arg("abc", "arg", 0, default)
        self.assertEqual(result, default)
        result = validate_int_arg(3.5, "arg", 0, default)
        self.assertEqual(result, default)
        result = validate_int_arg(-5, "arg", 0, default)
        self.assertEqual(result, default)
        result = validate_int_arg(0, "arg", 0, default)
        self.assertEqual(result, 0)  # 等于 minimum 时应该通过


class TestIsRecomputation(unittest.TestCase):
    @patch('inspect.stack')
    def test_in_recomputation_megatron(self, mock_stack):
        # 模拟megatron框架下的调用栈
        frame1 = MagicMock()
        frame1.function = 'backward'
        frame1.filename = 'megatron/torch/_tensor.py'

        frame2 = MagicMock()
        frame2.function = 'some_function'
        frame2.filename = 'torch/autograd/function.py'

        mock_stack.return_value = [frame1, frame2]

        self.assertTrue(is_recomputation())

    @patch('inspect.stack')
    def test_in_recomputation_mindspeed_L0L1(self, mock_stack):
        # 模拟mindspeed L0&L1场景下的调用栈
        frame1 = MagicMock()
        frame1.function = 'checkpoint_function_backward'
        frame1.filename = 'megatron/some_module.py'

        frame2 = MagicMock()
        frame2.function = 'some_other_function'
        frame2.filename = 'torch/autograd/function.py'

        mock_stack.return_value = [frame1, frame2]

        self.assertTrue(is_recomputation())

    @patch('inspect.stack')
    def test_in_recomputation_mindspeed_L2(self, mock_stack):
        # 模拟mindspeed L2场景下的调用栈
        frame1 = MagicMock()
        frame1.function = 'checkpoint_function_backward'
        frame1.filename = 'megatron/another_module.py'

        frame2 = MagicMock()
        frame2.function = 'yet_another_function'
        frame2.filename = 'some_file.py'

        frame3 = MagicMock()
        frame3.function = 'final_function'
        frame3.filename = 'torch/autograd/function.py'

        mock_stack.return_value = [frame1, frame2, frame3]

        self.assertTrue(is_recomputation())

    @patch('inspect.stack')
    def test_not_in_recomputation(self, mock_stack):
        # 模拟非重计算阶段的调用栈
        frame1 = MagicMock()
        frame1.function = 'forward'
        frame1.filename = 'my_model.py'

        mock_stack.return_value = [frame1]

        self.assertFalse(is_recomputation())


if __name__ == '__main__':
    unittest.main()
