# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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
# ==============================================================================

import os
from typing import Dict, Any
from tensorboard.plugins import base_plugin
from tensorboard.util import tb_logging
from .common.utils import Utils, FILE_EXTENSION
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
            if file.endswith(FILE_EXTENSION):
                self.db_path = os.path.join(self.logdir, file)
                break
        if hasattr(self, "db_path"):
            self.monvis_controller = MonvisController(self.db_path)
            self.is_db_connected = self.monvis_controller.is_db_connected
        else:
            logger.error("No trend.db file found in logdir")

    def get_plugin_apps(self) -> Dict[str, Any]:
        """Return all HTTP routes for the plugin."""
        if not hasattr(self, "monvis_controller") or not self.monvis_controller:
            return {}
        return {
            "/metrics": self.monvis_controller.request_metrics,
            "/values": self.monvis_controller.request_values,
            "/tags": self.monvis_controller.request_tags,
            "/heatmap_data": self.monvis_controller.request_heatmap_data,
            "/trend": self.monvis_controller.request_trend_data,
            "/db_files": self.monvis_controller.request_db_files,
            "/switch_db": self.monvis_controller.request_switch_db,
            "/index.js": self.monvis_controller.static_file_route,
            "/index.html": self.monvis_controller.static_file_route,
        }

    def is_active(self) -> bool:
        """Determine if the plugin is active."""
        if not hasattr(self, "is_db_connected") or not self.is_db_connected:
            return False
        success, error = Utils.safe_check_load_file_path(self.logdir, True)
        if not success:
            logger.error(error)
            return False
        # 遍历logdir目录， 如果logdir目录下面存在后缀名为.trend.db文件，则认为插件是活跃的
        for file in os.listdir(self.logdir):
            if file.endswith(FILE_EXTENSION):
                return True
        return False

    def frontend_metadata(self):
        """Return frontend metadata."""
        return base_plugin.FrontendMetadata(es_module_path="/index.js", tab_name="Trend Analyzer")
