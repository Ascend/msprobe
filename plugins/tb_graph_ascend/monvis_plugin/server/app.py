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
from tensorboard.util import tb_logging
from .controllers.monvis_controller import MonvisController

logger = tb_logging.get_logger()

class TrendVis(base_plugin.TBPlugin):
    """MonVis TensorBoard Plugin for visualizing monitoring data."""

    plugin_name = "TrendVis"

    def __init__(self, context):
        super().__init__(context)
        self.logdir = context.logdir
        # 寻找当前目录下，第一个.trend.db后缀的文件
        for file in os.listdir(self.logdir):
            if file.endswith(".trend.db"):
                self.db_path = os.path.join(self.logdir, file)
                break
        if hasattr(self, 'db_path'):
            self.monvis_controller = MonvisController(self.db_path)
            self.is_db_connected = self.monvis_controller.is_db_connected
        else:
           logger.error("No trend.db file found in logdir")

    def get_plugin_apps(self) -> Dict[str, Any]:
        """Return all HTTP routes for the plugin."""
        if not hasattr(self,'monvis_controller') or not self.monvis_controller:
            return {}
        return {
            "/metrics": self.monvis_controller.request_metrics,
            "/values": self.monvis_controller.request_values,
            "/tags": self.monvis_controller.request_tags,     
            "/heatmap_data": self.monvis_controller.request_heatmap_data,
            "/trend": self.monvis_controller.request_trend_data,
            '/index.js': self.monvis_controller.static_file_route,
            '/index.html': self.monvis_controller.static_file_route,
        }

    def is_active(self) -> bool:
        """Determine if the plugin is active."""
        if not hasattr(self,'is_db_connected') and not self.is_db_connected:
            return False
        # 遍历logdir目录， 如果logdir目录下面存在后缀名为.trend.db文件，则认为插件是活跃的
        for file in os.listdir(self.logdir):
            if file.endswith(".trend.db"):
                return True
        return False    

    def frontend_metadata(self):
        """Return frontend metadata."""
        return base_plugin.FrontendMetadata(
            es_module_path="/index.js",
            tab_name="Trend Analyzer"
        )
  
