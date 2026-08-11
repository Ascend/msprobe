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

from packaging.version import parse as version_parse

from msprobe.core.common.const import Const
from msprobe.core.common.file_utils import create_directory, save_json
from msprobe.mindspore.common.log import logger
from msprobe.mindspore.dump.debugger.debugger_config import DebuggerConfig

import mindspore as ms

ms_version = version_parse(ms.__version__)


class KernelKbykDump:
    COMMON_SETTINGS = "common_dump_settings"
    E2E_SETTINGS = "e2e_dump_settings"

    def __init__(self, config: DebuggerConfig):
        self.dump_json = dict()
        common_set = dict()
        e2e_set = dict()

        common_set = dict()
        common_set["dump_mode"] = 0
        common_set["path"] = ""
        common_set["net_name"] = "Net"
        common_set["iteration"] = "all"
        common_set["saved_data"] = "statistic"
        common_set["input_output"] = 0
        common_set["kernels"] = []
        common_set["support_device"] = [0, 1, 2, 3, 4, 5, 6, 7]
        common_set["statistic_category"] = []

        if config.stat_cal_mode and config.device_stat_precision_mode:
            e2e_set = {
                "enable": not config.async_dump,
                "trans_flag": True,
                "stat_calc_mode": config.stat_cal_mode,
                "device_stat_precision_mode": config.device_stat_precision_mode,
            }
        else:
            e2e_set = {"enable": not config.async_dump, "trans_flag": True}

        if config.list:
            common_set["dump_mode"] = 1
            common_set["kernels"] = config.list
        common_set["path"] = config.dump_path
        if config.step:
            step_str = ""
            for s in config.step:
                step_str += str(s) + '|'
            common_set["iteration"] = step_str[:-1]
        if config.rank:
            common_set["support_device"] = config.rank
        if config.task == Const.TENSOR:
            common_set["saved_data"] = Const.TENSOR
        if len(config.data_mode) == 1:
            if config.data_mode[0] == Const.INPUT:
                common_set["input_output"] = 1
            if config.data_mode[0] == Const.OUTPUT:
                common_set["input_output"] = 2
        if config.summary_mode:
            if isinstance(config.summary_mode, str):
                if config.summary_mode == Const.STATISTICS:
                    common_set["statistic_category"] = ["max", "min", "avg", "l2norm"]
                else:
                    mode = self._process_hash(config.summary_mode)
                    common_set["statistic_category"] = [mode]
            elif isinstance(config.summary_mode, list):
                common_set["statistic_category"] = list(
                    {self._process_hash("avg" if mode == "mean" else mode) for mode in config.summary_mode}
                )

        self.dump_json[KernelKbykDump.COMMON_SETTINGS] = common_set
        self.dump_json[KernelKbykDump.E2E_SETTINGS] = e2e_set

    @staticmethod
    def _process_hash(value):
        if ms_version <= version_parse("2.7.0") and value in (Const.HASH, Const.MD5):
            value = "md5"
        elif value == Const.MD5:
            value = "hash:md5"
        return value

    def handle(self):
        json_path = self.dump_json[KernelKbykDump.COMMON_SETTINGS]["path"]
        create_directory(json_path)
        json_path = os.path.join(json_path, "kernel_kbyk_dump.json")
        save_json(json_path, self.dump_json, indent=4)
        logger.info(json_path + " has been created.")

        os.environ["MINDSPORE_DUMP_CONFIG"] = json_path
        if "MS_ACL_DUMP_CFG_PATH" in os.environ:
            del os.environ["MS_ACL_DUMP_CFG_PATH"]
