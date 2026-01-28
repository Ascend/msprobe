# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# ==============================================================================
from .graph_service_db import DbGraphService
from .graph_service_vis import JsonGraphService
from ..utils.global_state import GraphState
from ..utils.constant import DataType


class ServiceFactory:

    def __init__(self):
        self.run = ''
        self.tag = ''
        self.data_type = None
        self.strategy = JsonGraphService('', '')

    def create_strategy(self, data_type, run, tag):
        if not (data_type == self.data_type and run == self.run and tag == self.tag):
            if data_type == DataType.DB.value:
                self.strategy = DbGraphService(run, tag)
            else:
                self.strategy = JsonGraphService(run, tag)
            self.data_type = data_type
            self.run = run
            self.tag = tag
        return self.strategy
    
    def create_strategy_without_tag(self, data_type, run):
        if not (data_type == self.data_type and run == self.run):
            self.data_type = data_type
            self.run = run
            self.tag = GraphState.get_global_value('first_run_tags', {}).get(self.run)
            if data_type == DataType.DB.value:
                self.strategy = DbGraphService(run, self.tag)
            else:
                self.strategy = JsonGraphService(run, self.tag)
        return self.strategy
