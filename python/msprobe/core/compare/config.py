# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025-2026 Huawei Technologies Co.,Ltd.
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

from msprobe.core.common.const import Const, CompareConst
from msprobe.core.common.file_utils import load_yaml


class ModeConfig:
    def __init__(self, **kwargs):
        self.stack_mode = kwargs.get('stack_mode', False)
        self.fuzzy_match = kwargs.get('fuzzy_match', False)
        self.xlsx = kwargs.get('xlsx', False)
        self.dump_mode = kwargs.get('dump_mode', Const.SUMMARY)
        self.first_diff_analyze = kwargs.get('first_diff_analyze', False)
        self.diff_analyze = kwargs.get('diff_analyze', False)
        self.compared_file_type = kwargs.get('compared_file_type', Const.DUMP_JSON_FILE)
        self.consistent_check = kwargs.get('consistent_check', False)
        self.backend = kwargs.get('backend', '')


class MappingConfig:
    def __init__(self, cell_mapping=None, api_mapping=None, data_mapping=None):
        self.cell_mapping = cell_mapping
        self.api_mapping = api_mapping
        self.data_mapping = data_mapping


class MappingDict:
    def __init__(self, mapping_config: MappingConfig):
        self.cell_mapping_dict = self.load_mapping_file(mapping_config.cell_mapping)
        self.api_mapping_dict = self.load_mapping_file(mapping_config.api_mapping)
        if mapping_config.api_mapping is not None:
            self.ms_to_pt_mapping = self.load_internal_api()
        self.data_mapping_dict = self.init_data_mapping(mapping_config.data_mapping)

    @staticmethod
    def load_internal_api():
        cur_path = os.path.dirname(os.path.realpath(__file__))
        yaml_path = os.path.abspath(os.path.join(cur_path, CompareConst.INTERNAL_API_MAPPING_FILE))
        return load_yaml(yaml_path)

    @staticmethod
    def load_mapping_file(mapping_file):
        if isinstance(mapping_file, str):
            mapping_dict = load_yaml(mapping_file)
        else:
            mapping_dict = {}
        return mapping_dict

    def init_data_mapping(self, data_mapping):
        """
        初始化data_mapping_dict
        """
        if isinstance(data_mapping, str) or data_mapping is None:
            data_mapping_dict = self.load_mapping_file(data_mapping)
        elif isinstance(data_mapping, dict):
            data_mapping_dict = data_mapping
        else:
            raise TypeError(f"The type of parameter `data_mapping` must be dict, str or None, but got "
                            f"{type(data_mapping)}")
        return data_mapping_dict
