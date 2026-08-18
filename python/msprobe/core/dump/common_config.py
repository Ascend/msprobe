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
import re

from msprobe.core.common.const import Const
from msprobe.core.common.log import logger
from msprobe.core.common.exceptions import MsprobeException
from msprobe.core.common.utils import get_real_step_or_rank, check_slice_info


class CommonConfig:
    def __init__(self, json_config):
        self.task = json_config.get('task')
        self.dump_path = json_config.get('dump_path')
        self.rank = get_real_step_or_rank(json_config.get('rank'), Const.RANK)
        self.step = get_real_step_or_rank(json_config.get('step'), Const.STEP)
        self.level = json_config.get('level')
        # None means "static mode": keep startup config forever and skip dynamic enable feature.
        self.dump_enable = json_config.get("dump_enable")
        self.extra_info = json_config.get("extra_info", True)
        self.async_dump = json_config.get("async_dump", False)
        self.precision = json_config.get("precision", Const.DUMP_PRECISION_LOW)
        self.risk_level = json_config.get("risk_level", Const.RISK_LEVEL_FOCUS)
        self.custom_op_namespaces = json_config.get("custom_op_namespaces")
        self._check_config()
        self.load_config = LoadConfig(json_config)
        logger.debug(
            f"CommonConfig: task={self.task}, dump_path={self.dump_path}, "
            f"rank={self.rank}, step={self.step}, level={self.level}, "
            f"dump_enable={self.dump_enable}, extra_info={self.extra_info}, "
            f"async_dump={self.async_dump}, precision={self.precision}, "
            f"risk_level={self.risk_level}, custom_op_namespaces={self.custom_op_namespaces}"
        )

    def _check_config(self):
        if self.task and self.task not in Const.TASK_LIST:
            logger.error_log_with_exp(
                "task is invalid, it should be one of {}".format(Const.TASK_LIST),
                MsprobeException(MsprobeException.INVALID_PARAM_ERROR),
            )
        if self.dump_path is not None and not isinstance(self.dump_path, str):
            logger.error_log_with_exp(
                "dump_path is invalid, it should be a string", MsprobeException(MsprobeException.INVALID_PARAM_ERROR)
            )
        if self.level and self.level not in Const.LEVEL_LIST:
            logger.error_log_with_exp(
                "level is invalid, it should be one of {}".format(Const.LEVEL_LIST),
                MsprobeException(MsprobeException.INVALID_PARAM_ERROR),
            )
        if self.dump_enable is not None and not isinstance(self.dump_enable, bool):
            logger.error_log_with_exp(
                "dump_enable is invalid, it should be a boolean", MsprobeException(MsprobeException.INVALID_PARAM_ERROR)
            )
        if not isinstance(self.extra_info, bool):
            logger.error_log_with_exp(
                "extra_info is invalid, it should be a boolean", MsprobeException(MsprobeException.INVALID_PARAM_ERROR)
            )
        if not isinstance(self.async_dump, bool):
            logger.error_log_with_exp(
                "async_dump is invalid, it should be a boolean", MsprobeException(MsprobeException.INVALID_PARAM_ERROR)
            )
        elif self.async_dump:
            logger.warning("async_dump is True, it may cause OOM when dumping large tensor.")

        if self.precision not in Const.DUMP_PRECISION_LIST:
            logger.error_log_with_exp(
                "precision is invalid, it should be one of {}".format(Const.DUMP_PRECISION_LIST),
                MsprobeException(MsprobeException.INVALID_PARAM_ERROR),
            )
        if self.risk_level and self.risk_level not in Const.RISK_LEVEL_LIST:
            logger.error_log_with_exp(
                "risk_level is invalid, it should be one of {}".format(Const.RISK_LEVEL_LIST),
                MsprobeException(MsprobeException.INVALID_PARAM_ERROR),
            )
        if self.custom_op_namespaces is not None:
            if not isinstance(self.custom_op_namespaces, list):
                logger.error_log_with_exp(
                    "custom_op_namespaces is invalid, it should be a list[str]",
                    MsprobeException(MsprobeException.INVALID_PARAM_ERROR),
                )
            for namespace in self.custom_op_namespaces:
                if not isinstance(namespace, str):
                    logger.error_log_with_exp(
                        "custom_op_namespaces is invalid, it should be a list[str]",
                        MsprobeException(MsprobeException.INVALID_PARAM_ERROR),
                    )


class LoadConfig:
    """Configuration for loading previously-dumped module input tensors to override actual inputs.

    Parsed from the "load" section of config.json. When "load" section is absent,
    the feature is disabled and existing behavior is unchanged.
    """

    def __init__(self, json_config):
        load = json_config.get("load") or {}
        self.path = load.get("path")
        self.modules = load.get("modules")
        self.step = load.get("step", [])  # list, [] = auto-align to current step
        self.rank = load.get("rank", [])  # list, [] = auto-align to current rank
        self.dump_after_load = load.get("dump_after_load", False)
        self.is_enabled = "load" in json_config  # "load" section present
        self._check()

    def _check(self):
        if not self.is_enabled:
            return
        if not isinstance(self.modules, list) or not self.modules:
            logger.error_log_with_exp(
                "load.modules is required and must be a non-empty list[str]",
                MsprobeException(
                    MsprobeException.INVALID_PARAM_ERROR,
                    "load.modules is required and must be a non-empty list[str]",
                ),
            )
        if not all(isinstance(m, str) for m in self.modules):
            logger.error_log_with_exp(
                "load.modules must be a list[str]",
                MsprobeException(MsprobeException.INVALID_PARAM_ERROR, "load.modules must be a list[str]"),
            )
        if not self.path:
            logger.error_log_with_exp(
                "load.path is required",
                MsprobeException(MsprobeException.INVALID_PARAM_ERROR, "load.path is required"),
            )
        if not isinstance(self.path, str) or not os.path.isdir(self.path):
            logger.error_log_with_exp(
                f"load.path does not exist or is not a directory: {self.path}",
                MsprobeException(
                    MsprobeException.INVALID_PARAM_ERROR,
                    f"load.path does not exist or is not a directory: {self.path}",
                ),
            )
        if not isinstance(self.dump_after_load, bool):
            logger.error_log_with_exp(
                "load.dump_after_load must be bool",
                MsprobeException(MsprobeException.INVALID_PARAM_ERROR, "load.dump_after_load must be bool"),
            )
        self.step = get_real_step_or_rank(self.step, Const.STEP)
        self.rank = get_real_step_or_rank(self.rank, Const.RANK)
        logger.debug(
            f"LoadConfig: path={self.path}, modules={self.modules}, "
            f"step={self.step}, rank={self.rank}, dump_after_load={self.dump_after_load}"
        )


class BaseConfig:
    def __init__(self, json_config):
        self.scope = json_config.get('scope')
        self.list = json_config.get('list')
        self.data_mode = json_config.get('data_mode')
        self.slice_info = json_config.get('slice')
        self.request_id = json_config.get('request_id')
        self.summary_mode = json_config.get("summary_mode")
        self.diff_nums = json_config.get("diff_nums")
        self.is_regex_valid = True
        logger.debug(
            f"BaseConfig: scope={self.scope}, list={self.list}, "
            f"data_mode={self.data_mode}, summary_mode={self.summary_mode}, "
            f"diff_nums={self.diff_nums}, slice={self.slice_info}, request_id={self.request_id}"
        )

    @staticmethod
    def _check_str_list_config(config_item, config_name):
        if config_item is not None:
            if not isinstance(config_item, list):
                logger.error_log_with_exp(
                    f"{config_name} is invalid, it should be a list[str]",
                    MsprobeException(MsprobeException.INVALID_PARAM_ERROR),
                )
            for name in config_item:
                if not isinstance(name, str):
                    logger.error_log_with_exp(
                        f"{config_name} is invalid, it should be a list[str]",
                        MsprobeException(MsprobeException.INVALID_PARAM_ERROR),
                    )

    def check_config(self):
        self._check_str_list_config(self.scope, "scope")
        self._check_str_list_config(self.list, "list")
        check_slice_info(self.slice_info)
        self._check_data_mode()
        self._check_regex_in_list()
        self._check_request_id()

    def _check_data_mode(self):
        if self.data_mode is not None:
            if not isinstance(self.data_mode, list):
                logger.error_log_with_exp(
                    "data_mode is invalid, it should be a list[str]",
                    MsprobeException(MsprobeException.INVALID_PARAM_ERROR),
                )

            if Const.ALL in self.data_mode and len(self.data_mode) != 1:
                logger.error_log_with_exp(
                    "'all' cannot be combined with other options in data_mode.",
                    MsprobeException(MsprobeException.INVALID_PARAM_ERROR),
                )

            if len(self.data_mode) >= len(Const.DUMP_DATA_MODE_LIST):
                logger.error_log_with_exp(
                    f"The number of elements in the data_made cannot exceed {len(Const.DUMP_DATA_MODE_LIST) - 1}.",
                    MsprobeException(MsprobeException.INVALID_PARAM_ERROR),
                )

            for mode in self.data_mode:
                if not isinstance(mode, str):
                    logger.error_log_with_exp(
                        "data_mode is invalid, it should be a list[str]",
                        MsprobeException(MsprobeException.INVALID_PARAM_ERROR),
                    )
                if mode not in Const.DUMP_DATA_MODE_LIST:
                    logger.error_log_with_exp(
                        f"The element '{mode}' of data_mode {self.data_mode} is not in {Const.DUMP_DATA_MODE_LIST}.",
                        MsprobeException(MsprobeException.INVALID_PARAM_ERROR),
                    )

    def _check_summary_mode(self):
        if self.summary_mode and self.summary_mode not in Const.SUMMARY_MODE:
            logger.error_log_with_exp(
                f"summary_mode is invalid, summary_mode is not in {Const.SUMMARY_MODE}.",
                MsprobeException(MsprobeException.INVALID_PARAM_ERROR),
            )

    def _check_regex_in_list(self):
        if self.list:
            for name in self.list:
                if name.startswith('name-regex(') and name.endswith(')'):
                    try:
                        re.compile(name[len('name-regex(') : -1])
                    except re.error:
                        self.is_regex_valid = False
                        break

    def _check_request_id(self):
        if self.request_id is not None and not isinstance(self.request_id, str):
            logger.error_log_with_exp(
                f"request_id is invalid, it should be str, actual type={type(self.request_id).__name__}.",
                MsprobeException(MsprobeException.INVALID_PARAM_ERROR),
            )
