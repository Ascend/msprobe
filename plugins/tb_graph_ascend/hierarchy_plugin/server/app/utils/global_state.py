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

# 全局常量

from .i18n import ZH_CN


class GraphState:
    # 模块级全局变量
    _state = {
        'logdir': '',
        'current_tag': '',
        'current_run': '',
        'current_file_data': {},
        'current_hierarchy': {},
        'config_data': {
            "run": "",
            "npuUnMatchNodes": [],
            "benchUnMatchNodes": [],
            "npuMatchNodes": {},
            "benchMatchNodes": {},
            "manualMatchNodes": {},
            
        },
        'config_info': {},  # 全局的配置信息
        'first_run_tags': {},
        'runs': {},
        'update_precision_cache': {},  # {node_name{precision，...}}，方便查询精度，提高性能
        'all_node_info_cache': {},  # {rank_step_micro_step:{node_name:node_info,...}}，方便查询节点信息，提高性能
        'lang': ZH_CN,
    }

    @staticmethod
    def init_defaults():
        """
        初始化全局变量的默认值
        """
        GraphState._state = {
            'logdir': '',
            'current_tag': '',
            'current_run': '',
            'current_file_data': {},
            'current_hierarchy': {},
            'config_data': {
                "run": "",
                "npuUnMatchNodes": [],
                "benchUnMatchNodes": [],
                "npuMatchNodes": {},
                "benchMatchNodes": {},
                "manualMatchNodes": {},
                
            },
            'config_info': {},  # 全局的配置信息
            'first_run_tags': {},
            'runs': {},
            'update_precision_cache': {},
            'all_node_info_cache': {},
            'lang': ZH_CN,
        }

    @staticmethod
    def set_global_value(key, value):
        """
        设置全局变量的值
        """
        GraphState._state[key] = value

    @staticmethod
    def get_global_value(key, default=None):
        """
        获取全局变量的值
        """
        return GraphState._state.get(key, default)

    @staticmethod
    def reset_global_state():
        """
        重置所有全局变量为默认值
        """
        GraphState.init_defaults()

