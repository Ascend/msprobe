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
from tensorboard.backend import http_util
from werkzeug.wrappers.request import Request

from ..utils.graph_utils import GraphUtils
from ..utils.global_state import GraphState


def check_file_type(func):

    def wrapper(*args, **kwargs):
        try:
            if len(args) <= 0:
                raise RuntimeError('Illegal function call, at least 1 parameter is required but got 0')
            request = args[0]
            if not isinstance(request, Request):
                raise RuntimeError('The request "parameter" is not in a format supported by werkzeug')
            data = GraphUtils.safe_json_loads(request.get_data().decode('utf-8'), {})
            meta_data = GraphUtils.safe_get_meta_data(data)
            result = {'success': False, 'error': ''}
            if meta_data is None or not isinstance(meta_data, dict):
                result['error'] = GraphUtils.t('metaDataError')
                return http_util.Respond(request, result, "application/json")
            # 设置语言
            GraphState.set_global_value('lang', meta_data.get('lang', 'zh-CN'))
        except Exception as e:
            result = {'success': False, 'error': str(e)}
            return http_util.Respond(request, result, "application/json")

        return func(*args, **kwargs)

    return wrapper
