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

from plugins.tb_graph_ascend.hierarchy_plugin.server.app.service.graph_service_base import (
    GraphServiceStrategy,
    ProgressInfo,
)
from plugins.tb_graph_ascend.hierarchy_plugin.server.app.utils.global_state import (
    GraphState,
)


class DummyStrategy(GraphServiceStrategy):
    # implement abstract methods with trivial returns
    def load_graph_data(self):
        return "data"

    def load_graph_config_info(self):
        return "cfg"

    def load_graph_all_node_list(self, meta_data):
        return "nodes"

    def change_node_expand_state(self, node_info, meta_data):
        return "changed"

    def get_node_info(self, node_info, meta_data):
        return "info"

    def add_match_nodes(
        self, npu_node_name, bench_node_name, meta_data, is_match_children
    ):
        return True

    def add_match_nodes_by_config(self, config_file_name, meta_data):
        return True

    def delete_match_nodes(
        self, npu_node_name, bench_node_name, meta_data, is_unmatch_children
    ):
        return True

    def update_precision_error(self, meta_data, filter_value):
        return 42

    def update_colors(self, colors):
        return colors

    def save_matched_relations(self, meta_data):
        return meta_data


def test_abstract_and_pass_methods():
    impl = DummyStrategy("run", "tag")
    assert impl.load_graph_data() == "data"
    assert impl.load_graph_config_info() == "cfg"
    assert impl.load_graph_all_node_list({}) == "nodes"
    assert impl.change_node_expand_state(None, None) == "changed"
    assert impl.get_node_info(None, None) == "info"
    assert impl.add_match_nodes("n1", "b1", {}, False) is True
    assert impl.add_match_nodes_by_config("f", {}) is True
    assert impl.delete_match_nodes("n1", "b1", {}, False) is True
    assert impl.update_precision_error({}, {}) == 42
    assert impl.update_colors({"a": 1}) == {"a": 1}
    assert impl.save_matched_relations({"x": 1}) == {"x": 1}

    # pass-through stubs should return None
    assert impl.search_node_by_precision({}, []) is None
    assert impl.search_node_by_overflow({}, []) is None


def test_load_converted_graph_data_success(tmp_path, monkeypatch):
    base = tmp_path / "root"
    base.mkdir()
    sub = base / "sub"
    sub.mkdir()
    (base / "a.yaml").write_text("x")
    # safe_check always ok
    monkeypatch.setattr(
        "plugins.tb_graph_ascend.hierarchy_plugin.server.app.service.graph_service_base.GraphUtils.safe_check_load_file_path",
        lambda path, create=False: (True, ""),
    )
    res = GraphServiceStrategy.load_converted_graph_data(str(base))
    assert res["success"] is True
    assert "a.yaml" in res["data"]["yaml_files"]
    assert any("sub" in d for d in res["data"]["dirs"])


def test_load_converted_graph_data_failure(monkeypatch):
    # os.listdir throws
    monkeypatch.setattr(
        "os.listdir", lambda x: (_ for _ in ()).throw(Exception("oops"))
    )
    res = GraphServiceStrategy.load_converted_graph_data("/bad")
    assert res["success"] is False
    assert "error" in res


def test_convert_to_graph_variants(monkeypatch):
    calls = []

    class FakePopen:
        def __init__(self, args, stdout, text):
            calls.append(args)

    monkeypatch.setattr(
        "plugins.tb_graph_ascend.hierarchy_plugin.server.app.service.graph_service_base.subprocess.Popen",
        FakePopen,
    )
    monkeypatch.setattr(
        "plugins.tb_graph_ascend.hierarchy_plugin.server.app.service.graph_service_base.update_progress_info",
        lambda proc, info: calls.append("update"),
    )
    data = {
        "npu_path": "np",
        "bench_path": "bp",
        "output_path": "op",
        "layer_mapping": "lm",
        "overflow_check": True,
        "fuzzy_match": True,
        "is_print_compare_log": True,
        "parallel_param_n": {"rank_size": 8, "tp": 16},
        "parallel_param_b": {"order": 4, "vpp": 2},
    }
    res = GraphServiceStrategy.convert_to_graph(data)
    assert res["success"] is True
    # check some flags in invocation
    assert any("-lm" in arg for arg in calls[0])
    assert "--rank_size" in calls[0]


def test_get_convert_progress_build_and_done():
    ProgressInfo.process_running = True
    ProgressInfo.error_msg = ""
    ProgressInfo.current_progress = 10
    it = GraphServiceStrategy.get_convert_progress()
    line1 = next(it)
    assert "building" in line1
    # finish
    ProgressInfo.process_running = False
    line2 = next(it)
    assert "done" in line2


def test_get_convert_progress_error(monkeypatch):
    # simulate attribute error during looping
    class BadInfo:
        @property
        def process_running(self):
            raise RuntimeError("boom")

    monkeypatch.setattr(
        "plugins.tb_graph_ascend.hierarchy_plugin.server.app.service.graph_service_base.ProgressInfo",
        BadInfo,
    )
    it = GraphServiceStrategy.get_convert_progress()
    line = next(it)
    assert "error" in line


def test_load_meta_dir_success(tmp_path, monkeypatch):
    run = tmp_path / "run1"
    run.mkdir()
    f = run / "tag.vis.db"
    f.write_text("z")
    monkeypatch.setattr(
        "plugins.tb_graph_ascend.hierarchy_plugin.server.app.service.graph_service_base.GraphUtils.safe_check_load_file_path",
        lambda path, create=False: (True, ""),
    )
    monkeypatch.setattr(
        "plugins.tb_graph_ascend.hierarchy_plugin.server.app.service.graph_service_base.GraphUtils.safe_load_data",
        lambda run_abs, file, flag: ({}, None),
    )
    GraphState.reset_global_state()
    GraphState.set_global_value("logdir", str(tmp_path))
    res = GraphServiceStrategy.load_meta_dir()
    # success is implied by absence of error entries
    assert res.get("error") == []
    assert "run1" in res["data"]
    # state updated
    assert GraphState.get_global_value("runs")["run1"]


def test_load_meta_dir_invalid(monkeypatch):
    GraphState.reset_global_state()
    GraphState.set_global_value("logdir", "/xyz")
    monkeypatch.setattr(
        "plugins.tb_graph_ascend.hierarchy_plugin.server.app.service.graph_service_base.GraphUtils.safe_check_load_file_path",
        lambda path, create=False: (False, "err"),
    )
    res = GraphServiceStrategy.load_meta_dir()
    assert res["success"] is False
    assert res["error"]
