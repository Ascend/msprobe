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

from plugins.tb_graph_ascend.hierarchy_plugin.server.app.service.graph_service_factory import ServiceFactory
from plugins.tb_graph_ascend.hierarchy_plugin.server.app.utils.global_state import GraphState


def test_create_strategy_caches_by_type_run_tag(monkeypatch):
    created = {"count": 0, "args": []}

    class DummyDbGraphService:
        def __init__(self, run, tag):
            created["count"] += 1
            created["args"].append((run, tag))

    monkeypatch.setattr(
        "plugins.tb_graph_ascend.hierarchy_plugin.server.app.service.graph_service_factory.DbGraphService",
        DummyDbGraphService,
    )

    f = ServiceFactory()
    s1 = f.create_strategy("db", "r1", "t1")
    s2 = f.create_strategy("db", "r1", "t1")
    assert s1 is s2
    assert created["count"] == 1

    s3 = f.create_strategy("db", "r1", "t2")
    assert s3 is not s1
    assert created["count"] == 2


def test_create_strategy_without_tag_uses_first_run_tags(monkeypatch):
    created = {"args": []}

    class DummyDbGraphService:
        def __init__(self, run, tag):
            created["args"].append((run, tag))

    monkeypatch.setattr(
        "plugins.tb_graph_ascend.hierarchy_plugin.server.app.service.graph_service_factory.DbGraphService",
        DummyDbGraphService,
    )

    GraphState.set_global_value("first_run_tags", {"runA": "tagA"})

    f = ServiceFactory()
    s = f.create_strategy_without_tag("db", "runA")
    assert created["args"][-1] == ("runA", "tagA")
    assert s is f.strategy
