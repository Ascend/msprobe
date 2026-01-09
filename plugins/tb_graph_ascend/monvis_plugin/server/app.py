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

import os
from typing import Dict, Any
from tensorboard.plugins import base_plugin
from .controllers.monvis_controller import MonvisController


class MonVis(base_plugin.TBPlugin):
    """MonVis TensorBoard Plugin for visualizing monitoring data."""

    plugin_name = "mon_vis"

    def __init__(self, context):
        super().__init__(context)
        self._log_dir = context.logdir
        self.db_path = os.path.join(self._log_dir, "monitor_metrics.db")
        self.monvis_controller = MonvisController(self.db_path)
        self.is_db_connected = self.monvis_controller.is_db_connected

    def get_plugin_apps(self) -> Dict[str, Any]:
        """Return all HTTP routes for the plugin."""
 
        return {
            "/metrics": self.monvis_controller.request_metrics,
            "/values": self.monvis_controller.request_values,     
            "/heatmap_data": self.monvis_controller.request_heatmap_data,
            "/trend": self.monvis_controller.request_trend_data,
            '/index.js': self.monvis_controller.static_file_route,
            '/index.html': self.monvis_controller.static_file_route
        }

    def is_active(self) -> bool:
        """Determine if the plugin is active."""
        return self.is_db_connected 

    def frontend_metadata(self):
        """Return frontend metadata."""
        return base_plugin.FrontendMetadata(
            es_module_path="/index.js",
            tab_name="Mon_Vis"
        )
  
