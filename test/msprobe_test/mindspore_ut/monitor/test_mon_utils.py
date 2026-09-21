# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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
import os
import tempfile
import re
from datetime import datetime
from mindspore import dtype as mstype, Tensor
from msprobe.mindspore.monitor.features import FUNC_MAP
from msprobe.core.common.const import MonitorConst
from msprobe.core.common.utils import is_int
from msprobe.core.common.log import logger
from msprobe.core.common.file_utils import check_file_or_directory_path

class TestMonitorUtils(unittest.TestCase):
    def setUp(self):
        # 创建临时目录用于测试
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建符合 MonitorConst.OUTPUT_DIR_PATTERN 的测试目录
        self.valid_dir1 = os.path.join(self.temp_dir, "Dec03_21-34-40_rank0")
        os.makedirs(self.valid_dir1)
        self.valid_dir2 = os.path.join(self.temp_dir, "Dec04_22-35-41_rank1")
        os.makedirs(self.valid_dir2)
        self.invalid_dir = os.path.join(self.temp_dir, "invalid_directory")
        os.makedirs(self.invalid_dir)

    def tearDown(self):
        # 清理临时目录
        for root, dirs, files in os.walk(self.temp_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(self.temp_dir)

    def test_get_summary_writer_tag_name(self):
        from msprobe.mindspore.monitor.utils import get_summary_writer_tag_name
        
        # 测试不带rank的情况
        result = get_summary_writer_tag_name("module1", "tag1", None)
        self.assertEqual(result, "module1/tag1")
        
        # 测试带rank的情况
        result = get_summary_writer_tag_name("module2", "tag2", 1)
        self.assertEqual(result, "module2/rank1/tag2")

    def test_step_accumulates_one(self):
        from msprobe.mindspore.monitor.utils import step_accumulates_one
        
        class MockContext:
            def __init__(self):
                self.micro_step = 0
                self.step = 0
        
        # 测试micro_step未达到micro_batch_number的情况
        context = MockContext()
        step_accumulates_one(context, 3)
        self.assertEqual(context.micro_step, 1)
        self.assertEqual(context.step, 0)
        
        # 测试micro_step达到micro_batch_number的情况
        context.micro_step = 2
        step_accumulates_one(context, 3)
        self.assertEqual(context.micro_step, 0)
        self.assertEqual(context.step, 1)

    def test_is_skip_step(self):
        from msprobe.mindspore.monitor.utils import is_skip_step
        
        # 测试step小于start_step的情况
        self.assertTrue(is_skip_step(5, 10, 1))
        
        # 测试step等于start_step的情况
        self.assertFalse(is_skip_step(10, 10, 1))
        
        # 测试step大于start_step但不满足interval的情况
        self.assertTrue(is_skip_step(11, 10, 2))
        
        # 测试step大于start_step且满足interval的情况
        self.assertFalse(is_skip_step(12, 10, 2))
        
        # 测试has_collect_times大于等于collect_times的情况
        self.assertTrue(is_skip_step(12, 10, 2, has_collect_times=5, collect_times=5))

    def test_validate_ops(self):
        from msprobe.core.monitor.utils import validate_ops
        
        # 测试输入不是list的情况
        with self.assertRaises(TypeError):
            validate_ops("not_a_list")
        
        # 测试包含不支持op的情况
        ops = ["mean", "unsupported_op"]
        result = validate_ops(ops)
        self.assertIn("mean", result)
        self.assertNotIn("unsupported_op", result)
        
        # 测试空列表情况，应该返回默认op
        result = validate_ops([])
        self.assertEqual(len(result), 3)  # 默认op + shape + dtype
        
        # 测试shape和dtype自动添加
        result = validate_ops(["mean"])
        self.assertIn("mean", result)
        self.assertIn("shape", result)
        self.assertIn("dtype", result)

    def test_validate_ranks(self):
        from msprobe.core.monitor.utils import validate_ranks
        
        # 测试输入不是list的情况
        with self.assertRaises(TypeError):
            validate_ranks("not_a_list")
        
        # 测试包含非int元素的情况
        with self.assertRaises(TypeError):
            validate_ranks([1, "not_an_int", 2])
        
        # 测试正常情况
        try:
            validate_ranks([1, 2, 3])
        except Exception as e:
            self.fail(f"validate_ranks raised unexpected exception: {e}")

    def test_validate_targets(self):
        from msprobe.core.monitor.utils import validate_targets
        
        # 测试输入不是dict的情况
        with self.assertRaises(TypeError):
            validate_targets("not_a_dict")
        
        # 测试key不是str的情况
        with self.assertRaises(TypeError):
            validate_targets({1: {"input": "tensor"}})
        
        # 测试value不是dict的情况
        with self.assertRaises(TypeError):
            validate_targets({"module1": "not_a_dict"})
        
        # 测试正常情况
        try:
            validate_targets({"module1": {"input": "tensor"}, "module2": {"output": "tensor"}})
        except Exception as e:
            self.fail(f"validate_targets raised unexpected exception: {e}")

    def test_validate_config(self):
        from msprobe.core.monitor.utils import validate_config
        
        # 测试基本配置验证
        config = {
            "ops": ["mean", "max"],
            "eps": 1e-6,
            "module_ranks": [0, 1],
            "targets": {"module1": {"input": "tensor"}},
            "print_struct": True,
            "ur_distribution": False,
            "xy_distribution": True,
            "wg_distribution": False,
            "mg_distribution": True,
            "param_distribution": False,
            "cc_distribution": {
                "enable": True,
                "cc_codeline": ["line1", "line2"],
                "cc_pre_hook": False,
                "cc_log_only": True
            },
            "alert": {
            },
            "step_count_per_record": 10,
            "start_step": 0,
            "step_interval": 1,
            "collect_times": 100,
            "monitor_mbs_grad": True,
            "dynamic_on": False
        }
        
        try:
            validate_config(config)
        except Exception as e:
            self.fail(f"validate_config raised unexpected exception: {e}")
        
        # 测试无效eps类型
        invalid_config = config.copy()
        invalid_config["eps"] = "not_a_float"
        with self.assertRaises(TypeError):
            validate_config(invalid_config)

    def test_time_str2time_digit(self):
        from msprobe.core.monitor.utils import time_str2time_digit
        
        # 测试有效时间字符串
        time_str = "Dec03_21-34-40"
        result = time_str2time_digit(time_str)
        self.assertIsInstance(result, datetime)
        self.assertEqual(result.month, 12)
        self.assertEqual(result.day, 3)
        self.assertEqual(result.hour, 21)
        
        # 测试无效时间字符串
        invalid_time_str = "InvalidTimeString"
        with self.assertRaises(RuntimeError):
            time_str2time_digit(invalid_time_str)

    def test_get_target_output_dir(self):
        from msprobe.core.monitor.utils import get_target_output_dir
        
        # 测试不带时间范围的情况
        result = get_target_output_dir(self.temp_dir, None, None)
        self.assertEqual(len(result), 0)


if __name__ == '__main__':
    unittest.main()