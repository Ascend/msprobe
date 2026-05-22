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

from msprobe.core.common.const import Const
from msprobe.core.common.exceptions import MsprobeException
from msprobe.pytorch.common.log import logger
from msprobe.pytorch.common.utils import is_torch_nn_module


def _is_torch_npu_importable():
    try:
        import torch_npu  # noqa: F401

        return True
    except ImportError:
        return False


class DebuggerConfig:
    def __init__(self, common_config, task_config, task, dump_path, level):
        self.dump_path = dump_path if dump_path else common_config.dump_path
        self.task = task or common_config.task or Const.STATISTICS
        self.rank = common_config.rank if common_config.rank else []
        self.step = common_config.step if common_config.step else []
        common_dump_enable = getattr(common_config, "dump_enable", None)
        self.dump_enable = common_dump_enable if isinstance(common_dump_enable, bool) else None
        self.extra_info = getattr(common_config, "extra_info", True)
        self.level = level or common_config.level or Const.LEVEL_L1
        self.scope = task_config.scope if task_config.scope else []
        self.list = task_config.list if task_config.list else []
        self.data_mode = task_config.data_mode if task_config.data_mode else ["all"]
        self.summary_mode = task_config.summary_mode if task_config.summary_mode else Const.STATISTICS
        self.framework = Const.PT_FRAMEWORK
        self.async_dump = common_config.async_dump if common_config.async_dump else False
        self.precision = common_config.precision if common_config.precision else Const.DUMP_PRECISION_LOW
        self.diff_nums = task_config.diff_nums if task_config.diff_nums else 1
        self.bench_path = getattr(task_config, "bench_path", None)
        self.risk_level = common_config.risk_level if common_config.risk_level else Const.RISK_LEVEL_ALL
        self.check()
        self._check_statistics_config(task_config)

        if self.level == Const.LEVEL_L2:
            self.is_backward_kernel_dump = False
            self._check_and_adjust_config_with_l2()

    def check(self):
        if self.task and self.task not in Const.TORCH_TASK_LIST:
            raise MsprobeException(
                MsprobeException.INVALID_PARAM_ERROR, f"The task <{self.task}> is not in the {Const.TORCH_TASK_LIST}."
            )
        if self.level and self.level not in Const.LEVEL_LIST:
            raise MsprobeException(
                MsprobeException.INVALID_PARAM_ERROR, f"The level <{self.level}> is not in the {Const.LEVEL_LIST}."
            )
        if not self.dump_path:
            raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR, "The dump_path not found.")
        if not isinstance(self.async_dump, bool):
            raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR, "The parameters async_dump should be bool.")
        if self.task == Const.NAN_CHECK and not _is_torch_npu_importable():
            raise MsprobeException(
                MsprobeException.INVALID_PARAM_ERROR,
                "task nan_check is only supported in NPU environments (torch_npu must be importable).",
            )
        if self.task == Const.NAN_CHECK and self.level != Const.LEVEL_L1:
            raise MsprobeException(
                MsprobeException.INVALID_PARAM_ERROR,
                f"When the task is set to nan_check, the level must be {Const.LEVEL_L1}, but got {self.level}.",
            )
        if self.task == Const.STRUCTURE and self.level not in [Const.LEVEL_L0, Const.LEVEL_MIX]:
            logger.warning_on_rank_0(
                f"When the task is set to structure, the level should be one of {[Const.LEVEL_L0, Const.LEVEL_MIX]}. "
                f"If not, the default level is {Const.LEVEL_MIX}."
            )
            self.level = Const.LEVEL_MIX
        if self.async_dump:
            if self.task == Const.TENSOR:
                if self.level == Const.LEVEL_DEBUG:
                    self.list = []  # async_dump + debug level case ignore list
                if not self.list and self.level != Const.LEVEL_DEBUG:
                    raise MsprobeException(
                        MsprobeException.INVALID_PARAM_ERROR,
                        "The parameters async_dump is true in tensor task, the parameters list cannot be empty.",
                    )
            if self.summary_mode == Const.MD5:
                raise MsprobeException(
                    MsprobeException.INVALID_PARAM_ERROR,
                    "The parameters async_dump is true, the parameters summary_mode cannot be md5.",
                )
        return True

    def check_model(self, models, token_range=None):
        if token_range and not models:
            error_info = "The 'model' parameter must be provided when token_range is not None"
            raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR, error_info)

        if self.level not in [Const.LEVEL_L0, Const.LEVEL_MIX] and token_range is None:
            return

        if models is None:
            logger.error_on_rank_0(
                f"For level {self.level} or non-empty token_range, "
                f"PrecisionDebugger or start interface must receive a 'model' parameter."
            )
            raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR, "missing the parameter 'model'")

        if is_torch_nn_module(models):
            return

        if isinstance(models, (list, tuple)):
            error_model = None
            for model in models:
                if not is_torch_nn_module(model):
                    error_model = model
                    break
            if error_model is not None:
                error_info = (
                    f"The 'model' parameter must be a torch.nn.Module or list[torch.nn.Module] "
                    f"type, currently there is an unsupported {type(error_model)} type."
                )
                raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR, error_info)
        else:
            error_info = (
                f"The 'model' parameter must be a torch.nn.Module or list[torch.nn.Module] "
                f"type, currently there is an unsupported {type(models)} type."
            )
            raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR, error_info)

    def _check_and_adjust_config_with_l2(self):
        if self.scope:
            raise MsprobeException(
                MsprobeException.INVALID_PARAM_ERROR, "When level is set to L2, the scope cannot be configured."
            )
        if not self.list or len(self.list) != 1:
            raise MsprobeException(
                MsprobeException.INVALID_PARAM_ERROR,
                "When level is set to L2, the list must be configured as a list with one api name.",
            )
        if self.task != Const.TENSOR:
            raise MsprobeException(
                MsprobeException.INVALID_PARAM_ERROR, "When level is set to L2, the task must be set to tensor."
            )

        api_name = self.list[0]
        if api_name.endswith(Const.BACKWARD):
            self.is_backward_kernel_dump = True
            api_forward_name = api_name[: -len(Const.BACKWARD)] + Const.FORWARD
            self.list.append(api_forward_name)

    def _check_statistics_config(self, task_config):
        if self.task not in {Const.STATISTICS, Const.NAN_CHECK}:
            return
        self.tensor_list = []
        if not hasattr(task_config, "tensor_list"):
            return
        if self.level == Const.LEVEL_DEBUG and task_config.tensor_list:
            logger.warning_on_rank_0("When level is set to debug, the tensor_list will be invalid.")
            return
        self.tensor_list = task_config.tensor_list
