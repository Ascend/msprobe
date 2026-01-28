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
