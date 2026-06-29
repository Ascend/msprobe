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

import os

from mindspore import nn

from msprobe.core.common.const import Const
from msprobe.core.common.exceptions import MsprobeException
from msprobe.core.common.file_utils import create_directory
from msprobe.core.common.log import logger
from msprobe.mindspore.common.const import Const as MsConst
from msprobe.mindspore.common.utils import is_mindtorch

if is_mindtorch():
    import torch


class DebuggerConfig:
    def __init__(self, common_config, task_config):
        self.execution_mode = None
        self.dump_path = common_config.dump_path
        self.task = common_config.task
        self.rank = [] if not common_config.rank else common_config.rank
        self.step = [] if not common_config.step else common_config.step
        self.extra_info = getattr(common_config, "extra_info", True)
        common_config.level = Const.LEVEL_L1 if not common_config.level else common_config.level
        self.level = MsConst.TOOL_LEVEL_DICT.get(common_config.level, MsConst.API)
        self.level_ori = common_config.level
        self.list = [] if not task_config.list else task_config.list
        self.scope = [] if not task_config.scope else task_config.scope
        self.data_mode = [Const.ALL] if not task_config.data_mode else task_config.data_mode
        self.framework = Const.MS_FRAMEWORK
        self.summary_mode = task_config.summary_mode
        self.stat_cal_mode = task_config.stat_cal_mode if hasattr(task_config, 'stat_cal_mode') else None
        self.device_stat_precision_mode = (
            task_config.device_stat_precision_mode if hasattr(task_config, 'device_stat_precision_mode') else None
        )
        self.async_dump = common_config.async_dump if common_config.async_dump else False
        self.precision = common_config.precision if common_config.precision else Const.DUMP_PRECISION_LOW
        self.check()
        self._check_statistics_config(task_config)
        create_directory(self.dump_path)

    @staticmethod
    def check_model(models, token_range=None):
        if token_range and not models:
            error_info = "The 'model' parameter must be provided when token_range is not None"
            raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR, error_info)

        target_module_type = (torch.nn.Module, "torch.nn.Module") if is_mindtorch() else (nn.Cell, "mindspore.nn.Cell")
        if models is None or isinstance(models, target_module_type[0]):
            return models
        if isinstance(models, (list, tuple)):
            error_model = None
            for model in models:
                if not isinstance(model, target_module_type[0]):
                    error_model = model
                    break
            if error_model is not None:
                error_info = (
                    f"The 'model' parameter must be a {target_module_type[1]} or list[{target_module_type[1]}] "
                    f"type, currently there is a {type(error_model)} type."
                )
                raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR, error_info)

        else:
            error_info = (
                f"The 'model' parameter must be a {target_module_type[1]} or list[{target_module_type[1]}] "
                f"type, currently there is a {type(models)} type."
            )
            raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR, error_info)
        return models

    def check(self):
        if not self.dump_path:
            raise Exception("Dump path is empty.")  # pylint: disable=W0719
        self.dump_path = os.path.abspath(self.dump_path)
        if not self.task:
            self.task = "statistics"
        if not self.level:
            raise Exception("level must be L0, L1 or L2")  # pylint: disable=W0719
        if not isinstance(self.async_dump, bool):
            raise Exception("The parameters async_dump should be bool.")  # pylint: disable=W0719
        if self.task == Const.STRUCTURE and self.level_ori not in [Const.LEVEL_L0, Const.LEVEL_MIX]:
            logger.warning_on_rank_0(
                f"When the task is set to structure, the level should be one of {[Const.LEVEL_L0, Const.LEVEL_MIX]}. "
                f"If not, the default level is {Const.LEVEL_MIX}."
            )
            self.level_ori = Const.LEVEL_MIX
        if self.async_dump:
            if self.task == Const.TENSOR:
                if self.level_ori == Const.LEVEL_DEBUG:
                    self.list = []  # async_dump + debug level case ignore list
                if not self.list and self.level_ori != Const.LEVEL_DEBUG:
                    raise MsprobeException(
                        MsprobeException.INVALID_PARAM_ERROR,
                        "The parameters async_dump is true in tensor task, the parameters list cannot be empty.",
                    )
            is_unsupported_mode = (
                self.summary_mode == Const.MD5 or isinstance(self.summary_mode, list) and Const.MD5 in self.summary_mode
            )
            if is_unsupported_mode:
                raise MsprobeException(
                    MsprobeException.INVALID_PARAM_ERROR,
                    "The parameters async_dump is true, the parameters summary_mode cannot be/contain md5.",
                )
        return True

    def _check_statistics_config(self, task_config):
        if self.task != Const.STATISTICS:
            return
        self.tensor_list = []
        if not hasattr(task_config, "tensor_list"):
            return
        if self.level_ori == Const.LEVEL_DEBUG and task_config.tensor_list:
            logger.warning_on_rank_0("When level is set to debug, the tensor_list will be invalid.")
            return
        self.tensor_list = task_config.tensor_list
