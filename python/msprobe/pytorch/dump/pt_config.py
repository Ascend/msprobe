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

from msprobe.core.common.const import Const
from msprobe.core.common.file_utils import load_json
from msprobe.core.dump.common_config import BaseConfig, CommonConfig
from msprobe.pytorch.dump.api_dump.utils import get_ops
from msprobe.core.common.utils import is_int


class TensorConfig(BaseConfig):
    def __init__(self, json_config):
        super().__init__(json_config)
        self.check_config()
        self._check_summary_mode()
        self.bench_path = json_config.get("bench_path")


class StatisticsConfig(BaseConfig):
    def __init__(self, json_config):
        super().__init__(json_config)
        self.check_config()
        self._check_summary_mode()

        self.tensor_list = json_config.get("tensor_list", [])
        self._check_str_list_config(self.tensor_list, "tensor_list")


class ACCCheckConfig(BaseConfig):
    WrapApi = get_ops()

    def __init__(self, json_config):
        super().__init__(json_config)
        self.white_list = json_config.get("white_list", Const.DEFAULT_LIST)
        self.black_list = json_config.get("black_list", Const.DEFAULT_LIST)
        self.error_data_path = json_config.get("error_data_path", Const.DEFAULT_PATH)

        self.check_acc_check_config()

    @classmethod
    def check_filter_list_config(cls, key, filter_list):
        if not isinstance(filter_list, list):
            raise Exception("%s must be a list type" % key)  # pylint: disable=W0719
        if not all(isinstance(item, str) for item in filter_list):
            raise Exception("All elements in %s must be string type" % key)  # pylint: disable=W0719
        invalid_api = [item for item in filter_list if item not in cls.WrapApi]
        if invalid_api:
            raise Exception("Invalid api in %s: %s" % (key, invalid_api))  # pylint: disable=W0719

    @classmethod
    def check_error_data_path_config(cls, error_data_path):
        if not os.path.exists(error_data_path):
            raise Exception("error_data_path: %s does not exist" % error_data_path)  # pylint: disable=W0719

    def check_acc_check_config(self):
        ACCCheckConfig.check_filter_list_config(Const.WHITE_LIST, self.white_list)
        ACCCheckConfig.check_filter_list_config(Const.BLACK_LIST, self.black_list)
        ACCCheckConfig.check_error_data_path_config(self.error_data_path)


class DiffCheckConfig(BaseConfig):
    def __init__(self, json_config):
        super().__init__(json_config)
        self.diff_nums = json_config.get("diff_nums")
        self.check_diff_config()

    def check_diff_config(self):
        if self.diff_nums is not None and not is_int(self.diff_nums):
            raise Exception("diff_num is invalid")  # pylint: disable=W0719
        if self.diff_nums is not None and self.diff_nums != -1 and self.diff_nums <= 0:
            raise Exception("diff_nums should be -1 or positive integer")  # pylint: disable=W0719


class StructureConfig(BaseConfig):
    pass


class NanCheckConfig(BaseConfig):
    def __init__(self, json_config):
        super().__init__(json_config)
        self.check_config()
        self.tensor_list = json_config.get("tensor_list", [])
        self._check_str_list_config(self.tensor_list, "tensor_list")


TaskDict = {
    Const.TENSOR: TensorConfig,
    Const.STATISTICS: StatisticsConfig,
    Const.DIFF_CHECK: DiffCheckConfig,
    Const.STRUCTURE: StructureConfig,
    Const.ACC_CHECK: ACCCheckConfig,
    Const.NAN_CHECK: NanCheckConfig,
}


def parse_task_config(task, json_config):
    if task not in Const.TORCH_TASK_LIST:
        raise Exception(f"task <{task}> is invalid, it is not in the {Const.TORCH_TASK_LIST}.")  # pylint: disable=W0719
    task_map = json_config.get(task, dict())
    return TaskDict.get(task)(task_map)


def parse_json_config(json_file_path, task):
    if not json_file_path:
        config_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        json_file_path = os.path.join(config_dir, "../../config.json")
    json_config = load_json(json_file_path)
    common_config = CommonConfig(json_config)
    if task:
        task_config = parse_task_config(task, json_config)
    else:
        task_config = parse_task_config(common_config.task, json_config)
    return common_config, task_config
