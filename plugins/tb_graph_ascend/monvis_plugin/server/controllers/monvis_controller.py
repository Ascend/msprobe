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
import sqlite3

from pathlib import Path
from tensorboard.backend import http_util
from werkzeug import wrappers, Response, exceptions

from ..common.utils import Utils
from ..services.monvis_service import MonvisService


class MonvisController:

    def __init__(self, db_path):
        self.db_path = db_path
        self.monvis_service = MonvisService(self.db_path)
        self.is_db_connected = self.monvis_service.is_db_connected

    @staticmethod
    @wrappers.Request.application
    def static_file_route(request):
        filename = os.path.basename(request.path)
        extension = os.path.splitext(filename)[1]
        if extension == '.html':
            mimetype = 'text/html'
        elif extension == '.js':
            mimetype = 'application/javascript'
        else:
            mimetype = 'application/octet-stream'

        try:
            # 添加白名单校验
            if filename != 'index.html' and filename != 'index.js':
                raise exceptions.NotFound('404 Not Found') from e
            server_dir = Path(__file__).resolve().parent.parent
            filepath = server_dir / "static" / filename 
            with open(filepath, 'rb') as infile:
                contents = infile.read()
        except IOError as e:
            raise exceptions.NotFound('404 Not Found') from e
        return Response(contents, content_type=mimetype, headers={"X-Content-Type-Options": "nosniff"}) 

    @wrappers.Request.application
    def request_metrics(self, request):
        """Return all available metrics and fixed stats."""

        try:
            result = self.monvis_service.get_metrics_stat()
        except sqlite3.Error as e:
            result = {'success': False, 'error': f"sqlite error: {str(e)}"}
        except Exception as e:
            result = {'success': False, 'error': str(e)}
        return http_util.Respond(request, result, "application/json")
    
    @wrappers.Request.application
    def request_values(self, request):
        """Return list of values for specified metric, stat and dimension."""  
        try:
            data = Utils.safe_json_loads(request.get_data().decode('utf-8'), {})
            metric = data.get('metric')
            stat = data.get('stat')
            dimension = data.get('dimension')
            tags = data.get('tags')
            result = self.monvis_service.get_values(metric, stat, dimension,tags)
        except Exception as e:
            result = {'success': False, 'error': str(e)}
        return http_util.Respond(request, result, "application/json")
    
    @wrappers.Request.application
    def request_tags(self, request):
        """Return list of tags for specified metric, stat and dimension."""
        try:
            metric = request.args.get('metric')
            result = self.monvis_service.get_tags(metric)
        except Exception as e:
            result = {'success': False, 'error': str(e)}
        return http_util.Respond(request, result, "application/json")
    
    @wrappers.Request.application
    def request_heatmap_data(self, request):
        """Return heatmap data for specified parameters."""
        try:
            data = Utils.safe_json_loads(request.get_data().decode('utf-8'), {})
            metric = data.get('metric')
            stat = data.get('stat')
            dimension = data.get('dimension')
            value = data.get('value')
            tags = data.get('tags')
            result = self.monvis_service.get_heatmap_data(metric, stat, dimension, value,tags)
        except Exception as e:
            result = {'success': False, 'error': str(e)}
        return http_util.Respond(request, result, "application/json")
         
    @wrappers.Request.application
    def request_trend_data(self, request):
        """Return trend data for specified parameters."""
        try:
            data = Utils.safe_json_loads(request.get_data().decode('utf-8'), {})
            metric =  data.get('metric')
            stat =  data.get('stat')
            dimension =  data.get('dimension')
            dim_x =  data.get('dimX')
            dim_y =  data.get('dimYIdx')
            tags =  data.get('tags')
            result = self.monvis_service.get_trend_data(metric, stat, dimension, dim_x, dim_y,tags)
        except Exception as e:
            result = {'success': False, 'error': str(e)}
        return http_util.Respond(request, result, "application/json")

