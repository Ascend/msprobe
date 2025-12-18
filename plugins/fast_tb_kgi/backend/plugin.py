# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#         http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# --------------------------------------------------------------------------------------------#

from tensorboard.plugins import base_plugin
from .routes import KgiRouter

PLUGIN_NAME = 'kgi'

class KgiPlugin(base_plugin.TBPlugin):

    plugin_name = PLUGIN_NAME

    def __init__(self, context: base_plugin.TBContext):
        super().__init__(context)

    def is_active(self):
        return True

    def get_plugin_apps(self):
        return {
            '/main.js': KgiRouter.static_file_route,
            '/index.html': KgiRouter.static_file_route,
            '/index.js': KgiRouter.static_file_route,
            '/index.css': KgiRouter.static_file_route,
            '/api/get_controls_info': KgiRouter.get_controls_info,
            '/api/get_graph': KgiRouter.get_graph,
            '/api/change_comapre_mode': KgiRouter.change_compare_mode,
            '/api/set_graph': KgiRouter.set_graph,
            '/api/change_to_whole_graph': KgiRouter.change_to_whole_graph,
            '/api/set_anchor_pre_check': KgiRouter.set_anchor_pre_check,
            '/api/set_anchor': KgiRouter.set_anchor,
            '/api/up_compare': KgiRouter.up_compare,
            '/api/down_compare': KgiRouter.down_compare,
            '/api/replace_equal_subgraph_pre_check': KgiRouter.replace_equal_subgraph_pre_check,
            '/api/replace_equal_subgraph': KgiRouter.replace_equal_subgraph,
            '/api/del_nodes_pre_check': KgiRouter.del_nodes_pre_check,
            '/api/del_nodes': KgiRouter.del_nodes,
            '/api/del_edges': KgiRouter.del_edges,
            '/api/set_second_level_anchor': KgiRouter.set_second_level_anchor,
            '/api/del_second_level_anchor_pre_check': KgiRouter.del_second_level_anchor_pre_check,
            '/api/del_second_level_anchor': KgiRouter.del_second_level_anchor,
            '/api/get_node_info': KgiRouter.get_node_info,
        }

    def frontend_metadata(self):
        return base_plugin.FrontendMetadata(
            es_module_path="/main.js",
            disable_reload=True,
        )