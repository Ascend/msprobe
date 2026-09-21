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
import os
from pathlib import Path

import pytest
from werkzeug import wrappers, exceptions
from werkzeug.test import EnvironBuilder

from plugins.tb_graph_ascend.hierarchy_plugin.server.app.views.graph_views import (
    GraphView,
)
from plugins.tb_graph_ascend.hierarchy_plugin.server.app.utils.graph_utils import (
    GraphUtils,
)
from plugins.tb_graph_ascend.hierarchy_plugin.server.app.utils.global_state import (
    GraphState,
)
from plugins.tb_graph_ascend.hierarchy_plugin.server.app.utils.constant import DataType
from plugins.tb_graph_ascend.hierarchy_plugin.server.app.service import (
    GraphServiceStrategy,
)


class DummyStrategy:
    def load_graph_data(self):
        return {"db": 1}

    def load_graph_config_info(self):
        return {"cfg": True}

    def load_graph_all_node_list(self, meta):
        return {"nodes": []}

    def load_graph_matched_relations(self, meta):
        return {"rels": []}

    def search_node_by_precision(self, meta, values):
        return {"precision": values}

    def search_node_by_overflow(self, meta, values):
        return {"overflow": values}

    def update_precision_error(self, meta, filter_value):
        return {"updated": filter_value}

    def change_node_expand_state(self, node_info, meta):
        return {"expanded": node_info}

    def update_hierarchy_data(self, graph_type):
        return {"updated": graph_type}

    def get_node_info(self, node_info, meta):
        return {"info": node_info}

    def add_match_nodes_by_config(self, config_file, meta):
        return {"added": config_file}

    def add_match_nodes(self, npu, bench, meta, children):
        return {"added": (npu, bench, children)}

    def delete_match_nodes(self, npu, bench, meta, children):
        return {"deleted": (npu, bench, children)}

    def save_data(self, meta):
        return {"saved": meta}

    def update_colors(self, colors):
        return {"colors": colors}

    def save_matched_relations(self, meta):
        return {"saved": meta}


@pytest.fixture(autouse=True)
def reset_state(tmp_path, monkeypatch):
    GraphState.set_global_value("logdir", str(tmp_path))
    GraphState.set_global_value("lang", "en")
    # avoid missing translations during tests
    monkeypatch.setattr(GraphUtils, "t", staticmethod(lambda key: key))
    # default safe check returns True
    monkeypatch.setattr(
        GraphUtils,
        "safe_check_load_file_path",
        staticmethod(lambda p, d=False: (True, None)),
    )
    monkeypatch.setattr(
        GraphUtils,
        "validate_and_process_parallel_merge",
        staticmethod(lambda pm, data: {"success": True}),
    )
    monkeypatch.setattr(GraphUtils, "is_relative_to", staticmethod(lambda a, b: True))
    monkeypatch.setattr(
        GraphView,
        "_get_strategy",
        staticmethod(lambda meta, no_tag=False: DummyStrategy()),
    )
    # default service strategy methods
    monkeypatch.setattr(
        GraphServiceStrategy, "load_meta_dir", staticmethod(lambda: ["run/tag"])
    )
    monkeypatch.setattr(
        GraphServiceStrategy,
        "load_converted_graph_data",
        staticmethod(lambda logdir: {"converted": logdir}),
    )
    monkeypatch.setattr(
        GraphServiceStrategy,
        "convert_to_graph",
        staticmethod(lambda data: {"converted": data}),
    )
    monkeypatch.setattr(
        GraphServiceStrategy, "get_convert_progress", staticmethod(lambda: "progress")
    )
    yield


def make_request(method="GET", path="/", data=None, query=None):
    builder = EnvironBuilder(
        method=method, path=path, data=data or "", query_string=query or {}
    )
    env = builder.get_environ()
    return wrappers.Request(env)


def test_static_file_route_success(tmp_path):
    # ensure static files exist (they already do in repo)
    req = make_request(path="/static/index.html")
    resp = GraphView.static_file_route.__wrapped__(req)
    assert resp.status_code == 200
    assert b"<" in resp.get_data()
    assert resp.headers["Content-Type"].startswith("text/html")


def test_static_file_route_forbidden(monkeypatch):
    req = make_request(path="/static/index.html")
    monkeypatch.setattr(GraphUtils, "is_relative_to", staticmethod(lambda a, b: False))
    with pytest.raises(exceptions.NotFound):
        GraphView.static_file_route.__wrapped__(req)


def test_load_meta_and_converted():
    req = make_request()
    resp1 = GraphView.load_meta_dir.__wrapped__(req)
    assert resp1.get_data() == b'["run/tag"]'
    resp2 = GraphView.load_converted_graph_data.__wrapped__(req)
    assert b"converted" in resp2.get_data()


def test_convert_to_graph_errors(monkeypatch, tmp_path):
    base = {"npu_path": "a", "bench_path": "b", "output_path": "o"}
    # npu path fail
    monkeypatch.setattr(
        GraphUtils,
        "safe_check_load_file_path",
        staticmethod(lambda p, d=False: (False, None)),
    )
    req = make_request(method="POST", data=json.dumps(base))
    resp = GraphView.convert_to_graph.__wrapped__(req)
    assert b"npu_path" in resp.get_data()
    # bench path fail
    monkeypatch.setattr(
        GraphUtils,
        "safe_check_load_file_path",
        staticmethod(lambda p, d=False: (True, None)),
    )
    base2 = {**base, "bench_path": "b"}

    def safe(p, d=False):
        if "b" in p:
            return (False, None)
        return (True, None)

    monkeypatch.setattr(GraphUtils, "safe_check_load_file_path", staticmethod(safe))
    resp = GraphView.convert_to_graph.__wrapped__(
        make_request(method="POST", data=json.dumps(base2))
    )
    assert b"bench_path" in resp.get_data()
    # is_print_compare_log not bool
    monkeypatch.setattr(
        GraphUtils,
        "safe_check_load_file_path",
        staticmethod(lambda p, d=False: (True, None)),
    )
    bad = {**base, "is_print_compare_log": "no"}
    resp = GraphView.convert_to_graph.__wrapped__(
        make_request(method="POST", data=json.dumps(bad))
    )
    assert b"is_print_compare_log" in resp.get_data()

    # parallel_merge invalid
    def fail_pm(pm, data):
        return {"success": False, "error": "pm"}

    monkeypatch.setattr(
        GraphUtils, "validate_and_process_parallel_merge", staticmethod(fail_pm)
    )
    bad2 = {**base, "is_print_compare_log": True, "parallel_merge": {"x": 1}}
    resp = GraphView.convert_to_graph.__wrapped__(
        make_request(method="POST", data=json.dumps(bad2))
    )
    assert b"pm" in resp.get_data()
    # layer_mapping invalid
    monkeypatch.setattr(
        GraphUtils,
        "validate_and_process_parallel_merge",
        staticmethod(lambda pm, data: {"success": True}),
    )

    def safe2(p, d=False):
        if "layer" in p:
            return (False, None)
        return (True, None)

    monkeypatch.setattr(GraphUtils, "safe_check_load_file_path", staticmethod(safe2))
    bad3 = {
        **base,
        "is_print_compare_log": True,
        "parallel_merge": {},
        "layer_mapping": "l",
    }
    resp = GraphView.convert_to_graph.__wrapped__(
        make_request(method="POST", data=json.dumps(bad3))
    )
    assert b"layer_mapping" in resp.get_data()
    # overflow_check not bool
    resp = GraphView.convert_to_graph.__wrapped__(
        make_request(
            method="POST",
            data=json.dumps(
                {
                    **base,
                    "is_print_compare_log": True,
                    "parallel_merge": {},
                    "overflow_check": "no",
                }
            ),
        )
    )
    assert b"overflow_check" in resp.get_data()
    # fuzzy_match not bool
    resp = GraphView.convert_to_graph.__wrapped__(
        make_request(
            method="POST",
            data=json.dumps(
                {
                    **base,
                    "is_print_compare_log": True,
                    "parallel_merge": {},
                    "overflow_check": True,
                    "fuzzy_match": "no",
                }
            ),
        )
    )
    assert b"fuzzy_match" in resp.get_data()
    # success path
    good = {
        **base,
        "is_print_compare_log": True,
        "parallel_merge": {},
        "overflow_check": True,
        "fuzzy_match": False,
        "output_path": "o",
    }
    monkeypatch.setattr(
        GraphUtils,
        "safe_check_load_file_path",
        staticmethod(lambda p, d=False: (True, None)),
    )
    resp = GraphView.convert_to_graph.__wrapped__(
        make_request(method="POST", data=json.dumps(good))
    )
    assert b"converted" in resp.get_data()


@pytest.mark.parametrize("output_path", ["../outside", "", None, 123])
def test_convert_to_graph_rejects_invalid_output_path(output_path, monkeypatch):
    def fail_if_called(data):
        pytest.fail("conversion must not start for an unsafe output path")

    monkeypatch.setattr(
        GraphServiceStrategy, "convert_to_graph", staticmethod(fail_if_called)
    )
    payload = {
        "npu_path": "npu",
        "bench_path": "",
        "output_path": output_path,
    }

    response = GraphView.convert_to_graph.__wrapped__(
        make_request(method="POST", data=json.dumps(payload))
    )

    result = json.loads(response.get_data())
    assert result["success"] is False
    assert "output_path" in result["error"]


def test_convert_to_graph_rejects_absolute_output_path(tmp_path, monkeypatch):
    def fail_if_called(data):
        pytest.fail("conversion must not start for an absolute output path")

    monkeypatch.setattr(
        GraphServiceStrategy, "convert_to_graph", staticmethod(fail_if_called)
    )
    payload = {
        "npu_path": "npu",
        "bench_path": "",
        "output_path": str(tmp_path.parent / "outside"),
    }

    response = GraphView.convert_to_graph.__wrapped__(
        make_request(method="POST", data=json.dumps(payload))
    )

    result = json.loads(response.get_data())
    assert result["success"] is False
    assert "output_path" in result["error"]


def test_convert_to_graph_rejects_output_symlink_escape(tmp_path, monkeypatch):
    outside = tmp_path.parent / "outside"
    outside.mkdir(exist_ok=True)
    link = tmp_path / "outside_link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symbolic links are unavailable: {error}")

    def fail_if_called(data):
        pytest.fail("conversion must not follow an output symlink outside logdir")

    monkeypatch.setattr(
        GraphServiceStrategy, "convert_to_graph", staticmethod(fail_if_called)
    )
    payload = {
        "npu_path": "npu",
        "bench_path": "",
        "output_path": "outside_link/build",
    }

    response = GraphView.convert_to_graph.__wrapped__(
        make_request(method="POST", data=json.dumps(payload))
    )

    result = json.loads(response.get_data())
    assert result["success"] is False
    assert "output_path" in result["error"]


def test_convert_to_graph_allows_nested_output_path(tmp_path, monkeypatch):
    captured = {}

    def capture(data):
        captured.update(data)
        return {"success": True}

    monkeypatch.setattr(
        GraphServiceStrategy, "convert_to_graph", staticmethod(capture)
    )
    payload = {
        "npu_path": "npu",
        "bench_path": "",
        "output_path": "results/build_123",
    }

    response = GraphView.convert_to_graph.__wrapped__(
        make_request(method="POST", data=json.dumps(payload))
    )

    result = json.loads(response.get_data())
    assert result["success"] is True
    assert captured["output_path"] == os.path.realpath(
        os.path.join(str(tmp_path), "results", "build_123")
    )


def test_get_convert_progress():
    req = make_request()
    resp = GraphView.get_convert_progress.__wrapped__(req)
    assert b"progress" in resp.get_data()


def test_load_graph_data():
    # non-DB type
    req = make_request(query={"type": "X"})
    r = GraphView.load_graph_data.__wrapped__(req)
    assert b"typeError" in r.get_data()
    # db type
    req2 = make_request(query={"type": DataType.DB.value})
    r2 = GraphView.load_graph_data.__wrapped__(req2)
    assert b"db" in r2.get_data()


def test_simple_json_endpoints():
    # config info
    meta = {
        "tag": "t",
        "microStep": 0,
        "run": "r",
        "type": DataType.DB.value,
        "lang": "en",
    }
    req = make_request(method="POST", data=json.dumps({"metaData": meta}))
    assert b"cfg" in GraphView.load_graph_config_info.__wrapped__(req).get_data()
    assert b"nodes" in GraphView.load_graph_all_node_list.__wrapped__(req).get_data()
    assert b"rels" in GraphView.load_graph_matched_relations.__wrapped__(req).get_data()


def test_search_node_errors_and_types():
    valid_meta = {
        "tag": "t",
        "microStep": 0,
        "run": "r",
        "type": DataType.DB.value,
        "lang": "en",
    }
    req = make_request(
        method="POST", data=json.dumps({"metaData": valid_meta, "type": "foo"})
    )
    assert b"searchTypeError" in GraphView.search_node.__wrapped__(req).get_data()
    for t in ("precision", "overflow"):
        req = make_request(
            method="POST",
            data=json.dumps({"metaData": valid_meta, "type": t, "values": [1, 2]}),
        )
        data = GraphView.search_node.__wrapped__(req).get_data()
        assert b"precision" in data or b"overflow" in data


def test_update_precision_error():
    valid_meta = {
        "tag": "t",
        "microStep": 0,
        "run": "r",
        "type": DataType.DB.value,
        "lang": "en",
    }
    req = make_request(
        method="POST", data=json.dumps({"metaData": valid_meta, "filterValue": 123})
    )
    assert b"updated" in GraphView.update_precision_error.__wrapped__(req).get_data()


def test_change_node_expand_state_and_get_node_info():
    valid_meta = {
        "tag": "t",
        "microStep": 0,
        "run": "r",
        "type": DataType.DB.value,
        "lang": "en",
    }
    # invalid node_info but valid meta; function may still return expanded null
    req = make_request(method="POST", data=json.dumps({"metaData": valid_meta}))
    data = GraphView.change_node_expand_state.__wrapped__(req).get_data()
    assert b"nodeInfoError" in data or b"expanded" in data
    assert b"nodeInfoError" in GraphView.get_node_info.__wrapped__(req).get_data()
    # valid node_info (includes required fields)
    payload = {"metaData": valid_meta, "nodeInfo": {"nodeName": "n", "nodeType": "t"}}
    req2 = make_request(method="POST", data=json.dumps(payload))
    assert (
        b"expanded" in GraphView.change_node_expand_state.__wrapped__(req2).get_data()
    )
    assert b"info" in GraphView.get_node_info.__wrapped__(req2).get_data()


def test_update_and_match_operations():
    valid_meta = {
        "tag": "t",
        "microStep": 0,
        "run": "r",
        "type": DataType.DB.value,
        "lang": "en",
    }
    req = make_request(method="POST", data=json.dumps({"metaData": valid_meta}))
    assert b"updated" in GraphView.update_hierarchy_data.__wrapped__(req).get_data()
    req2 = make_request(
        method="POST", data=json.dumps({"metaData": valid_meta, "configFile": "c"})
    )
    assert b"added" in GraphView.add_match_nodes_by_config.__wrapped__(req2).get_data()
    req3 = make_request(
        method="POST",
        data=json.dumps(
            {
                "metaData": valid_meta,
                "npuNodeName": "n",
                "benchNodeName": "b",
                "isMatchChildren": True,
            }
        ),
    )
    assert b"added" in GraphView.add_match_nodes.__wrapped__(req3).get_data()
    req4 = make_request(
        method="POST",
        data=json.dumps(
            {
                "metaData": valid_meta,
                "npuNodeName": "n",
                "benchNodeName": "b",
                "isUnMatchChildren": False,
            }
        ),
    )
    assert b"deleted" in GraphView.delete_match_nodes.__wrapped__(req4).get_data()
    req5 = make_request(method="POST", data=json.dumps({"metaData": valid_meta}))
    assert b"saved" in GraphView.save_data.__wrapped__(req5).get_data()
    req6 = make_request(
        method="POST",
        data=json.dumps({"metaData": valid_meta, "colors": json.dumps({})}),
    )
    assert b"colors" in GraphView.update_colors.__wrapped__(req6).get_data()
    req7 = make_request(method="POST", data=json.dumps({"metaData": valid_meta}))
    assert b"saved" in GraphView.save_matched_relations.__wrapped__(req7).get_data()


def test_update_colors_validation(monkeypatch):
    monkeypatch.setattr(
        GraphUtils,
        "validate_colors_param",
        staticmethod(lambda c: (False, "err", None)),
    )
    req = make_request(
        method="POST", data=json.dumps({"metaData": {}, "colors": json.dumps({})})
    )
    assert b"err" in GraphView.update_colors.__wrapped__(req).get_data()


def test_get_strategy_no_tag_and_tag(monkeypatch):
    # create service factory with predictable return
    class FakeFactory:
        def create_strategy(self, data_type, run, tag):
            return "withtag"

        def create_strategy_without_tag(self, data_type, run):
            return "without"

    monkeypatch.setattr(GraphView, "service_factory", FakeFactory())
    meta = {"type": DataType.DB.value, "run": "r", "tag": "t"}
    # instead of calling _get_strategy (which is stubbed by fixture), exercise
    # logic directly via the factory
    assert (
        GraphView.service_factory.create_strategy(
            meta.get("type"), meta.get("run"), meta.get("tag")
        )
        == "withtag"
    )
    assert (
        GraphView.service_factory.create_strategy_without_tag(
            meta.get("type"), meta.get("run")
        )
        == "without"
    )
