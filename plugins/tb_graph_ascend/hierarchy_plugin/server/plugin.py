# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
#
# Copyright (c) 2025, Huawei Technologies.
# Adapt to the model hierarchical visualization data collected by the msprobe tool
# ==============================================================================
"""The TensorBoard Graphs plugin."""

import os
from tensorboard.plugins import base_plugin
from tensorboard.util import tb_logging

from .app.views.graph_views import GraphView
from .app.utils.graph_utils import GraphUtils
from .app.utils.global_state import GraphState
from .app.utils.constant import Extension

logger = tb_logging.get_logger()

PLUGIN_NAME = 'graph_ascend'
PLUGIN_NAME_RUN_METADATA_WITH_GRAPH = 'graph_ascend_run_metadata_graph'
DB_EXT = Extension.DB.value
JSON_EXT = Extension.JSON.value


class GraphsPlugin(base_plugin.TBPlugin):
    """Graphs Plugin for TensorBoard."""

    plugin_name = PLUGIN_NAME

    def __init__(self, context):
        """Instantiates GraphsPlugin via TensorBoard core.

        Args:
          context: A base_plugin.TBContext instance.
        """
        super().__init__(context)
        GraphState.reset_global_state()
        self.logdir = os.path.abspath(os.path.expanduser(context.logdir.rstrip('/')))
        # 将logdir赋值给global_state中的logdir属性,方便其他模块使用
        GraphState.set_global_value('logdir', self.logdir)

    def get_plugin_apps(self):
        return {     
            '/index.js': GraphView.static_file_route,
            '/index.html': GraphView.static_file_route,
            "/load_meta_dir": GraphView.load_meta_dir,
            "/screen": GraphView.search_node,
            '/loadGraphData': GraphView.load_graph_data,
            '/loadGraphConfigInfo': GraphView.load_graph_config_info,
            '/loadGraphAllNodeList': GraphView.load_graph_all_node_list,
            '/changeNodeExpandState': GraphView.change_node_expand_state,
            '/updateHierarchyData': GraphView.update_hierarchy_data,
            '/getNodeInfo': GraphView.get_node_info,
            '/addMatchNodes': GraphView.add_match_nodes,
            '/addMatchNodesByConfig': GraphView.add_match_nodes_by_config,
            '/deleteMatchNodes': GraphView.delete_match_nodes,
            '/saveData': GraphView.save_data,
            '/updateColors': GraphView.update_colors,
            '/saveMatchedRelations': GraphView.save_matched_relations,
            '/updatePrecisionError': GraphView.update_precision_error,
        }

    def is_active(self):
        """The graphs plugin is active if any run has a graph."""

        def _is_vis(path, file_name):
            return os.path.isfile(path) and (file_name.endswith(DB_EXT) or file_name.endswith(JSON_EXT))
        
        _, error = GraphUtils.safe_check_load_file_path(self.logdir, True)
        if error:
            return False
        for content in os.listdir(self.logdir):
            content_path = os.path.join(self.logdir, content)
            if _is_vis(content_path, content):
                return True
            if os.path.isdir(content_path):
                _, error = GraphUtils.safe_check_load_file_path(content_path, True)
                if error:
                    continue
                for file in os.listdir(content_path):
                    if _is_vis(os.path.join(content_path, file), file):
                        return True
        return False

    def data_plugin_names(self):
        return (
            PLUGIN_NAME,
            PLUGIN_NAME_RUN_METADATA_WITH_GRAPH,
        )

    def frontend_metadata(self):
        return base_plugin.FrontendMetadata(
            es_module_path='/index.js',
            disable_reload=True,
        )
