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

from unittest import TestCase
from unittest.mock import patch

from msprobe.core.common.log import logger
from msprobe.core.common.const import Const
from msprobe.core.common.exceptions import MsprobeException
from msprobe.core.dump.common_config import CommonConfig, BaseConfig


class TestCommonConfig(TestCase):
    @patch.object(logger, "error_log_with_exp")
    def test_common_config(self, mock_error_log_with_exp):
        json_config = dict()

        common_config = CommonConfig(json_config)
        self.assertIsNone(common_config.task)
        self.assertIsNone(common_config.dump_path)
        self.assertEqual(common_config.rank, [])
        self.assertEqual(common_config.step, [])
        self.assertIsNone(common_config.level)

        json_config.update({"task": "md5"})
        CommonConfig(json_config)
        self.assertEqual(mock_error_log_with_exp.call_args[0][0],
                         "task is invalid, it should be one of {}".format(Const.TASK_LIST))
        self.assertEqual(str(mock_error_log_with_exp.call_args[0][1]),
                         MsprobeException.err_strs.get(MsprobeException.INVALID_PARAM_ERROR))

        json_config.update({"task": Const.TENSOR})
        json_config.update({"rank": [0]})
        json_config.update({"step": [0]})
        json_config.update({"level": "L3"})
        CommonConfig(json_config)
        self.assertEqual(mock_error_log_with_exp.call_args[0][0],
                         "level is invalid, it should be one of {}".format(Const.LEVEL_LIST))
        self.assertEqual(str(mock_error_log_with_exp.call_args[0][1]),
                         MsprobeException.err_strs.get(MsprobeException.INVALID_PARAM_ERROR))

        json_config.update({"task": Const.TENSOR})
        json_config.update({"rank": [0]})
        json_config.update({"step": [0]})
        json_config.update({"level": "L0"})
        CommonConfig(json_config)
        self.assertEqual(mock_error_log_with_exp.call_args[0][0],
                         "level is invalid, it should be one of ['L0', 'L1', 'L2', 'mix', 'debug']")
        self.assertEqual(str(mock_error_log_with_exp.call_args[0][1]),
                         MsprobeException.err_strs.get(MsprobeException.INVALID_PARAM_ERROR))


class TestBaseConfig(TestCase):
    @patch.object(logger, "error_log_with_exp")
    def test_base_config(self, mock_error_log_with_exp):
        json_config = dict()

        base_config = BaseConfig(json_config)
        base_config.check_config()
        self.assertIsNone(base_config.scope)
        self.assertIsNone(base_config.list)
        self.assertIsNone(base_config.summary_mode)

        json_config.update({"scope": "Tensor_Add"})
        base_config = BaseConfig(json_config)
        base_config.check_config()
        self.assertEqual(mock_error_log_with_exp.call_args[0][0],
                         "scope is invalid, it should be a list[str]")
        self.assertEqual(str(mock_error_log_with_exp.call_args[0][1]),
                         MsprobeException.err_strs.get(MsprobeException.INVALID_PARAM_ERROR))

        json_config.update({"scope": ["Tensor_Add"]})
        json_config.update({"list": "Tensor_Add"})
        base_config = BaseConfig(json_config)
        base_config.check_config()
        self.assertEqual(mock_error_log_with_exp.call_args[0][0],
                         "list is invalid, it should be a list[str]")
        self.assertEqual(str(mock_error_log_with_exp.call_args[0][1]),
                         MsprobeException.err_strs.get(MsprobeException.INVALID_PARAM_ERROR))

    @patch.object(logger, "error_log_with_exp")
    def test_check_data_mode(self, mock_error_log_with_exp):
        self.config = BaseConfig({})

        self.config._check_data_mode()
        mock_error_log_with_exp.assert_not_called()

        self.config.data_mode = ["all"]
        self.config._check_data_mode()
        mock_error_log_with_exp.assert_not_called()

        self.config.data_mode = "all"
        self.config._check_data_mode()
        self.assertEqual(
            mock_error_log_with_exp.call_args_list[0][0][0],
            "data_mode is invalid, it should be a list[str]"
        )

        mock_error_log_with_exp.reset_mock()
        self.config.data_mode = ["all", "forward"]
        self.config._check_data_mode()
        self.assertEqual(
            mock_error_log_with_exp.call_args_list[0][0][0],
            "'all' cannot be combined with other options in data_mode."
        )

        mock_error_log_with_exp.reset_mock()
        self.config.data_mode = ["test", "input", "output", "forward", "backward"]
        self.config._check_data_mode()
        self.assertEqual(
            mock_error_log_with_exp.call_args_list[0][0][0],
            f"The number of elements in the data_made cannot exceed {len(Const.DUMP_DATA_MODE_LIST) - 1}."
        )

        mock_error_log_with_exp.reset_mock()
        self.config.data_mode = [123, 'test_case_1']
        self.config._check_data_mode()
        self.assertEqual(
            mock_error_log_with_exp.call_args_list[0][0][0],
            "data_mode is invalid, it should be a list[str]"
        )

        mock_error_log_with_exp.reset_mock()
        self.config.data_mode = ['forward', 'test_case_1']
        self.config._check_data_mode()
        self.assertEqual(
            mock_error_log_with_exp.call_args_list[0][0][0],
            f"The element 'test_case_1' of data_mode {self.config.data_mode} is not in {Const.DUMP_DATA_MODE_LIST}."
        )

