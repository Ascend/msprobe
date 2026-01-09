# Copyright (c) 2025, Huawei Technologies.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
            self.data_type = data_type
            self.run = run
            self.tag = tag
            if data_type == DataType.DB.value:
                self.strategy = DbGraphService(run, tag)
            else:
                self.strategy = JsonGraphService(run, tag)
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
