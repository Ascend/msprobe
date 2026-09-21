# -------------------------------------------------------------------------
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
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

import json

from werkzeug.test import EnvironBuilder
from werkzeug.wrappers import Request

from plugins.tb_graph_ascend.hierarchy_plugin.server.app.utils.file_check_wrapper import check_file_type
from plugins.tb_graph_ascend.hierarchy_plugin.server.app.utils.graph_utils import GraphUtils
from plugins.tb_graph_ascend.hierarchy_plugin.server.app.utils.global_state import GraphState
from plugins.tb_graph_ascend.hierarchy_plugin.server.app.utils.i18n import ZH


def _make_request(payload: dict) -> Request:
    builder = EnvironBuilder(
        method="POST",
        path="/",
        data=json.dumps(payload),
        content_type="application/json",
    )
    env = builder.get_environ()
    return Request(env)


def test_check_file_type_sets_lang_and_calls_func(monkeypatch):
    responded = {}

    def fake_respond(request, result, content_type):
        responded["result"] = result
        responded["content_type"] = content_type
        return {"ok": True, "result": result}

    monkeypatch.setattr(
        "plugins.tb_graph_ascend.hierarchy_plugin.server.app.utils.file_check_wrapper.http_util.Respond",
        fake_respond,
    )

    called = {"v": False}

    @check_file_type
    def handler(request):
        called["v"] = True
        # 返回值不重要，装饰器应该透传
        return {"success": True}

    req = _make_request(
        {
            "metaData": {
                "tag": "t",
                "microStep": -1,
                "run": "r",
                "type": "db",
                "lang": ZH,
            }
        }
    )
    ret = handler(req)

    assert called["v"] is True
    assert ret == {"success": True}
    assert GraphState.get_global_value("lang") == ZH
    assert responded == {}  # 未触发 Respond


def test_check_file_type_returns_error_when_metadata_missing(monkeypatch):
    def fake_respond(request, result, content_type):
        return {"result": result, "content_type": content_type}

    monkeypatch.setattr(
        "plugins.tb_graph_ascend.hierarchy_plugin.server.app.utils.file_check_wrapper.http_util.Respond",
        fake_respond,
    )

    @check_file_type
    def handler(request):
        return {"success": True}

    req = _make_request({"metaData": {"lang": ZH}})
    ret = handler(req)

    assert ret["content_type"] == "application/json"
    assert ret["result"]["success"] is False
    assert ret["result"]["error"] == GraphUtils.t("metaDataError")


def test_check_file_type_returns_error_when_request_type_invalid(monkeypatch):
    def fake_respond(request, result, content_type):
        return {"result": result, "content_type": content_type}

    monkeypatch.setattr(
        "plugins.tb_graph_ascend.hierarchy_plugin.server.app.utils.file_check_wrapper.http_util.Respond",
        fake_respond,
    )

    @check_file_type
    def handler(request):
        return {"success": True}

    ret = handler(object())
    assert ret["result"]["success"] is False
    assert "werkzeug" in ret["result"]["error"]

