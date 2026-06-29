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

from msprobe.mindspore.dump.ms_config import (
    parse_task_config,
    TensorConfig,
    StatisticsConfig
)


class TestMsConfig(unittest.TestCase):

    def test_parse_task_config(self):
        mock_json_config = {
            "tensor": None,
            "statistics": None
        }

        task_config = parse_task_config("tensor", mock_json_config)
        self.assertTrue(isinstance(task_config, TensorConfig))

        task_config = parse_task_config("statistics", mock_json_config)
        self.assertTrue(isinstance(task_config, StatisticsConfig))

        with self.assertRaises(Exception) as context:
            parse_task_config("unsupported_task", mock_json_config)
        self.assertEqual(str(context.exception), "task is invalid.")
