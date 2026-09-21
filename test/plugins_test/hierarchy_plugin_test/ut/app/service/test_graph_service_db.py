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

from plugins.tb_graph_ascend.hierarchy_plugin.server.app.service.graph_service_db import (
    DbGraphService,
)
from plugins.tb_graph_ascend.hierarchy_plugin.server.app.utils.constant import (
    NPU,
    BENCH,
    SINGLE,
    UN_MATCHED_VALUE,
    MAX_RELATIVE_ERR,
    MIN_RELATIVE_ERR,
    NORM_RELATIVE_ERR,
    MEAN_RELATIVE_ERR,
)
from plugins.tb_graph_ascend.hierarchy_plugin.server.app.utils import (
    global_state,
    graph_utils,
)


class DummyService(DbGraphService):
    """子类化以绕过真实仓库初始化逻辑，仅测试与仓库无关的方法。"""

    def __init__(self):
        # 不调用父类 __init__，手工设置必要属性
        self.run = "dummy_run"
        self.tag = "dummy_tag"
        self.repo = None
        self.conn = object()
        self.config_info = {}


def test_update_hierarchy_data_with_valid_types(monkeypatch):
    dummy_service = DummyService()

    updated_result = {"some": "hierarchy"}

    # 模拟 LayoutHierarchyModel.update_hierarchy_data 返回值
    from plugins.tb_graph_ascend.hierarchy_plugin.server.app.model import (
        layout_hierarchy_model,
    )
    from plugins.tb_graph_ascend.hierarchy_plugin.server.app.utils.constant import (
        NPU,
        BENCH,
    )

    def fake_update_hierarchy_data(graph_type):
        # graph_type 只是透传参数，这里简单返回固定结果，方便断言
        return {"graph_type": graph_type, **updated_result}

    monkeypatch.setattr(
        layout_hierarchy_model.LayoutHierarchyModel,
        "update_hierarchy_data",
        staticmethod(fake_update_hierarchy_data),
    )

    npu_result = dummy_service.update_hierarchy_data(NPU)
    bench_result = dummy_service.update_hierarchy_data(BENCH)

    assert npu_result["success"] is True
    assert bench_result["success"] is True
    assert npu_result["data"]["some"] == "hierarchy"
    assert bench_result["data"]["some"] == "hierarchy"


def test_update_hierarchy_data_with_invalid_type():
    dummy_service = DummyService()
    result = dummy_service.update_hierarchy_data("UNKNOWN")
    assert result["success"] is False
    assert "error" in result


def test_load_graph_all_node_list_validates_rank_step_micro_step():
    s = DummyService()
    s.conn = object()
    s.repo = object()
    s.config_info = {"isSingleGraph": True}

    ret = s.load_graph_all_node_list({"rank": None, "step": 1, "microStep": -1})
    assert ret["success"] is False


def test_load_graph_all_node_list_single_graph_uses_query_node_name_list(monkeypatch):
    class Repo:
        def query_node_name_list(self, rank, step, micro_step):
            return ["a", "b"]

        def query_config_info(self):
            return {"isSingleGraph": True}

    s = DummyService()
    s.conn = object()
    s.repo = Repo()
    s.config_info = {"isSingleGraph": True}

    ret = s.load_graph_all_node_list({"rank": 0, "step": 1, "microStep": -1})
    assert ret["success"] is True
    assert ret["data"]["npuNodeList"] == ["a", "b"]


def test_search_node_by_precision_uses_cache_when_present():
    s = DummyService()
    s.conn = object()
    s.repo = object()
    # cache: 只返回 is_leaf_nodes=True 的节点
    from plugins.tb_graph_ascend.hierarchy_plugin.server.app.utils.global_state import (
        GraphState,
    )

    GraphState.set_global_value(
        "update_precision_cache",
        {
            "n1": {
                "is_leaf_nodes": True,
                "precision_index": 0.2,
                "matched_node_link": ["b1"],
            },
            "n2": {
                "is_leaf_nodes": False,
                "precision_index": 0.9,
                "matched_node_link": ["b2"],
            },
            "n3": {
                "is_leaf_nodes": True,
                "precision_index": 0.9,
                "matched_node_link": [],
            },
        },
    )
    ret = s.search_node_by_precision(
        {"rank": 0, "step": 1, "microStep": -1}, ["pass", -1]
    )
    assert ret["success"] is True
    # n1: pass；n3: unmatched
    names = {x["name"] for x in ret["data"]}
    assert "n1" in names
    assert "n3" in names


def test_load_graph_data_edge_cases(monkeypatch):
    s = DummyService()
    # no repo -> error
    s.repo = None
    res = s.load_graph_data()
    assert res["success"] is False

    # repo exists but conn empty
    class Repo2:
        def get_db_connection(self):
            return object()

    s.repo = Repo2()
    s.conn = None
    res2 = s.load_graph_data()
    # with valid get_db_connection the call now succeeds
    assert res2["success"] is True


def test_load_graph_config_info_success(monkeypatch, tmp_path):
    s = DummyService()

    class Repo:
        def query_config_info(self):
            return {"foo": "bar"}

    s.repo = Repo()
    # simulate directory files
    from plugins.tb_graph_ascend.hierarchy_plugin.server.app.utils.graph_utils import (
        GraphUtils,
    )

    monkeypatch.setattr(
        GraphUtils, "find_config_files", staticmethod(lambda run: ["file1"])
    )
    ret = s.load_graph_config_info()
    assert ret["success"] is True
    assert "matchedConfigFiles" in ret["data"]


def test_load_graph_config_info_exception(monkeypatch):
    s = DummyService()

    class Repo:
        def query_config_info(self):
            raise RuntimeError("boom")

    s.repo = Repo()
    ret = s.load_graph_config_info()
    assert ret["success"] is False
    assert "error" in ret


def test_load_graph_all_node_list_double_graph(monkeypatch):
    class Repo:
        def query_config_info(self):
            return {"isSingleGraph": False}

        def query_all_node_info_list(self, rank, step, micro):
            return {"npu_node_list": [1], "bench_node_list": [2]}

    s = DummyService()
    s.conn = object()
    s.repo = Repo()
    s.config_info = {"isSingleGraph": False}
    ret = s.load_graph_all_node_list({"rank": 0, "step": 1, "microStep": 2})
    assert ret["success"] is True
    assert ret["data"]["benchNodeList"] == [2]


def test_load_graph_all_node_list_exception(monkeypatch):
    s = DummyService()
    s.conn = object()

    class Repo:
        def query_config_info(self):
            raise Exception("fail")

    s.repo = Repo()
    res = s.load_graph_all_node_list({"rank": 0, "step": 0, "microStep": 0})
    assert res["success"] is False


def test_load_graph_matched_relations_basic(monkeypatch):
    class Repo:
        def query_config_info(self):
            return {}

        def query_all_matched_relations(self, rank, step, micro):
            return {
                "npu_match_node": [1],
                "bench_match_node": [2],
                "npu_unmatch_node": [],
                "bench_unmatch_node": [],
            }

    s = DummyService()
    s.conn = object()
    s.repo = Repo()
    global_state.GraphState.set_global_value("config_data", {})
    ret = s.load_graph_matched_relations({"rank": 0, "step": 0, "microStep": 0})
    assert ret["success"]
    assert ret["data"]["npuMatchNodes"] == [1]


def test_load_graph_matched_relations_rank_missing():
    s = DummyService()
    s.conn = object()
    s.repo = DummyService()
    res = s.load_graph_matched_relations({})
    assert res["success"] is False


def test_change_node_expand_state_paths(monkeypatch):
    s = DummyService()
    s.conn = object()
    s.repo = DummyService()
    # monkeypatch LayoutHierarchyModel
    from plugins.tb_graph_ascend.hierarchy_plugin.server.app.model import (
        layout_hierarchy_model,
    )

    monkeypatch.setattr(
        layout_hierarchy_model.LayoutHierarchyModel,
        "change_expand_state",
        staticmethod(lambda *a, **k: {"ok": True}),
    )
    s.config_info = {"isSingleGraph": True}
    r = s.change_node_expand_state(
        {"nodeType": NPU, "nodeName": "n"}, {"rank": 0, "step": 1, "microStep": 2}
    )
    assert r["success"]
    # NPU path
    s.config_info = {}
    r2 = s.change_node_expand_state(
        {"nodeType": NPU, "nodeName": "n"}, {"rank": 0, "step": 1, "microStep": 2}
    )
    assert r2["success"]
    # invalid graph_type
    r3 = s.change_node_expand_state(
        {"nodeType": "X", "nodeName": "n"}, {"rank": 0, "step": 1, "microStep": 2}
    )
    assert r3["success"]
    # missing rank
    r4 = s.change_node_expand_state(
        {"nodeType": NPU, "nodeName": "n"}, {"step": 1, "microStep": 2}
    )
    assert r4["success"] is False


def test_search_node_by_precision_db(monkeypatch):
    s = DummyService()
    s.conn = object()

    class Repo:
        def query_node_list_by_precision(self, meta, precision_range, values, filt):
            return ["dbres"]

    s.repo = Repo()
    res = s.search_node_by_precision(
        {"rank": 0, "step": 0, "microStep": 0}, [UN_MATCHED_VALUE]
    )
    assert res["success"]
    assert "dbres" in res["data"]


def test_search_node_by_precision_rank_missing():
    s = DummyService()
    s.conn = object()
    s.repo = object()
    assert s.search_node_by_precision({}, [])["success"] is False


def test_search_node_by_overflow_basic(monkeypatch):
    s = DummyService()
    s.conn = object()

    class Repo:
        def query_node_list_by_overflow(self, step, rank, micro, vals):
            return [1]

    s.repo = Repo()
    res = s.search_node_by_overflow({"rank": 0, "step": 1, "microStep": 2}, ["v"])
    assert res["success"] and res["data"][0] == 1


def test_search_node_by_overflow_errors():
    s = DummyService()
    s.conn = None
    assert (
        s.search_node_by_overflow({"rank": 0, "step": 1, "microStep": 2}, [])["success"]
        is False
    )
    s.conn = object()
    assert s.search_node_by_overflow({"rank": None}, [])["success"] is False


def test_get_node_info_variants(monkeypatch):
    s = DummyService()
    s.conn = object()

    class Repo:
        def query_node_info(self, name, gtype, rank, step):
            return (
                {"matched_node_link": ["m"]}
                if gtype == NPU
                else {"matched_node_link": []}
            )

    s.repo = Repo()
    s.config_info = {"isSingleGraph": False}
    res = s.get_node_info({"nodeType": NPU, "nodeName": "x"}, {"rank": 0, "step": 1})
    assert res["success"]
    res2 = s.get_node_info({"nodeType": BENCH, "nodeName": "x"}, {"rank": 0, "step": 1})
    assert res2["success"]
    res3 = s.get_node_info(
        {"nodeType": SINGLE, "nodeName": "x"}, {"rank": 0, "step": 1}
    )
    assert res3["success"]
    assert s.get_node_info({"nodeType": NPU}, {"rank": None})["success"] is False


def test_add_match_nodes_variants(monkeypatch):
    s = DummyService()
    s.conn = object()

    class Repo:
        def query_node_and_sub_nodes(self, *a, **k):
            return {}

        def query_matched_nodes_info(self, *a, **k):
            return {}

        def update_nodes_info(self, data, rank, step):
            return True

    s.repo = Repo()
    s.config_info = {"task": "md5"}
    from plugins.tb_graph_ascend.hierarchy_plugin.server.app.model import (
        match_nodes_model,
    )

    monkeypatch.setattr(
        match_nodes_model.MatchNodesController,
        "process_task_add",
        lambda *a, **k: [{"success": True, "data": [1]}],
    )
    monkeypatch.setattr(
        match_nodes_model.MatchNodesController,
        "process_task_add_child_layer",
        lambda *a, **k: [{"success": True, "data": [2]}],
    )
    # patch _generate_matched_result to avoid internal complexity
    s._generate_matched_result = lambda mr, r, s: {"success": True}
    res1 = s.add_match_nodes("n", "b", {"rank": 0, "step": 1}, False)
    assert "success" in res1
    res2 = s.add_match_nodes("n", "b", {"rank": 0, "step": 1}, True)
    assert "success" in res2
    res3 = s.add_match_nodes("n", "b", {"rank": None, "step": 1}, True)
    assert res3["success"] is False
    s.config_info = {"task": "other"}
    assert (
        s.add_match_nodes("n", "b", {"rank": 0, "step": 1}, False)["success"] is False
    )


def test_add_match_nodes_by_config(monkeypatch):
    s = DummyService()
    s.conn = object()

    class Repo2:
        def query_matched_nodes_info_by_config(self, links, rank, step):
            return {}

    s.repo = Repo2()
    meta = {"rank": 0, "step": 1, "run": "r"}
    # simulate load error
    from plugins.tb_graph_ascend.hierarchy_plugin.server.app.utils.graph_utils import (
        GraphUtils,
    )

    monkeypatch.setattr(
        GraphUtils, "safe_load_data", staticmethod(lambda run, name: ([], "err"))
    )
    s.config_info = {"task": "md5"}
    assert s.add_match_nodes_by_config("c", meta)["success"] is False
    # simulate empty links
    monkeypatch.setattr(
        GraphUtils, "safe_load_data", staticmethod(lambda run, name: ([], None))
    )
    assert s.add_match_nodes_by_config("c", meta)["success"] is False
    # simulate normal flow
    monkeypatch.setattr(
        GraphUtils, "safe_load_data", staticmethod(lambda run, name: (["x"], None))
    )

    class Repo2:
        def query_matched_nodes_info_by_config(self, links, rank, step):
            return {}

    s.repo = Repo2()
    from plugins.tb_graph_ascend.hierarchy_plugin.server.app.model import (
        match_nodes_model,
    )

    monkeypatch.setattr(
        match_nodes_model.MatchNodesController,
        "process_task_add_child_layer_by_config",
        lambda gd, links, task: [{"success": True, "data": [3]}],
    )
    # patch _generate_matched_result to skip deeper logic
    s._generate_matched_result = lambda mr, r, s: {"success": True}
    res = s.add_match_nodes_by_config("c", meta)
    assert "success" in res


def test_delete_match_nodes_variants(monkeypatch):
    s = DummyService()
    s.conn = object()

    class RepoDummy(DummyService):
        def query_matched_nodes_info(self, *a, **k):
            return {}

        def query_node_and_sub_nodes(self, *a, **k):
            return {}

    s.repo = RepoDummy()
    s.config_info = {"task": "md5"}
    from plugins.tb_graph_ascend.hierarchy_plugin.server.app.model import (
        match_nodes_model,
    )

    monkeypatch.setattr(
        match_nodes_model.MatchNodesController,
        "process_task_delete",
        staticmethod(lambda *a, **k: [{"success": True, "data": [4]}]),
    )
    monkeypatch.setattr(
        match_nodes_model.MatchNodesController,
        "process_task_delete_child_layer",
        staticmethod(lambda *a, **k: [{"success": True, "data": [5]}]),
    )
    # patch _generate_matched_result similarly
    s._generate_matched_result = staticmethod(lambda mr, r, s: {"success": True})
    res1 = s.delete_match_nodes("n", "b", {"rank": 0, "step": 1}, False)
    assert "success" in res1
    res2 = s.delete_match_nodes("n", "b", {"rank": 0, "step": 1}, True)
    assert "success" in res2
    assert s.delete_match_nodes("n", "b", {"rank": None}, False)["success"] is False
    s.config_info = {"task": "x"}
    assert (
        s.delete_match_nodes("n", "b", {"rank": 0, "step": 1}, False)["success"]
        is False
    )


def test_update_precision_error_and_cache(monkeypatch):
    s = DummyService()
    s.conn = object()

    class Repo:
        def query_node_info_by_data_source(self, step, rank, t):
            return {
                "n1": {
                    "node_name": "n1",
                    "data": {"precision_index": 0.5},
                    "matched_node_link": [],
                    "subnodes": [],
                    "output_data": {},
                }
            }

    s.repo = Repo()
    s.config_info = {}
    res = s.update_precision_error({"rank": 0, "step": 1}, [MAX_RELATIVE_ERR])
    assert res["success"]
    # no nodes case
    s.repo.query_node_info_by_data_source = lambda step, rank, t: {}
    assert s.update_precision_error({"rank": 0, "step": 1}, [])["success"] is False
    # missing params
    assert s.update_precision_error({}, [])["success"] is False


def test_update_colors_paths(monkeypatch):
    s = DummyService()
    s.conn = None
    assert s.update_colors([1])["success"] is False
    s.conn = object()

    class Repo:
        def update_config_colors(self, colors):
            return False

    s.repo = Repo()
    assert s.update_colors([1])["success"] is False

    class Repo2:
        def update_config_colors(self, colors):
            return True

    s.repo = Repo2()
    assert s.update_colors([1])["success"] is True


def test_save_matched_relations(monkeypatch):
    s = DummyService()
    s.conn = None
    assert s.save_matched_relations({"rank": 0, "step": 0})["success"] is False
    s.conn = object()
    # missing rank/step
    assert s.save_matched_relations({})["success"] is False

    # with error from safe_save_data
    class Repo:
        def query_modify_matched_nodes_list(self, rank, step):
            return [1]

    s.repo = Repo()
    from plugins.tb_graph_ascend.hierarchy_plugin.server.app.utils.graph_utils import (
        GraphUtils,
    )

    monkeypatch.setattr(
        GraphUtils,
        "safe_save_data",
        staticmethod(lambda data, run, name: (None, "err")),
    )
    assert (
        s.save_matched_relations({"run": "r", "tag": "t", "rank": 0, "step": 0})[
            "success"
        ]
        is False
    )
    # successful save
    monkeypatch.setattr(
        GraphUtils, "safe_save_data", staticmethod(lambda data, run, name: ("f", None))
    )
    res = s.save_matched_relations({"run": "r", "tag": "t", "rank": 0, "step": 0})
    assert res["success"] and res.get("data")


def test_generate_matched_result_variants(monkeypatch):
    s = DummyService()
    s.conn = object()
    # case empty
    out = s._generate_matched_result([], 0, 0)
    assert out["success"] is False

    # case update_db_res false
    class Repo:
        def update_nodes_info(self, data, rank, step):
            return False

    s.repo = Repo()
    from plugins.tb_graph_ascend.hierarchy_plugin.server.app.model import (
        layout_hierarchy_model,
    )

    monkeypatch.setattr(
        layout_hierarchy_model.LayoutHierarchyModel,
        "update_current_hierarchy_data",
        staticmethod(lambda x: None),
    )
    global_state.GraphState.set_global_value("config_data", {})
    out2 = s._generate_matched_result([{"success": True, "data": [{"a": 1}]}], 0, 0)
    assert out2["success"] is False

    # case update_db_res true
    class Repo2:
        def update_nodes_info(self, data, rank, step):
            return True

    s.repo = Repo2()
    global_state.GraphState.set_global_value(
        "config_data",
        {
            "npuMatchNodes": {},
            "benchMatchNodes": {},
            "npuUnMatchNodes": [],
            "benchUnMatchNodes": [],
        },
    )
    out3 = s._generate_matched_result([{"success": True, "data": [{"a": 1}]}], 0, 0)
    assert out3["success"]


def test_load_with_no_connection():
    s = DummyService()
    s.repo = object()
    s.conn = None
    assert (
        s.load_graph_all_node_list({"rank": 0, "step": 0, "microStep": 0})["success"]
        is False
    )
    assert (
        s.load_graph_matched_relations({"rank": 0, "step": 0, "microStep": 0})[
            "success"
        ]
        is False
    )


def test_load_matched_relations_exception(monkeypatch):
    s = DummyService()
    s.conn = object()

    class Repo:
        def query_config_info(self):
            return {}

        def query_all_matched_relations(self, *a, **k):
            raise RuntimeError("boom")

    s.repo = Repo()
    res = s.load_graph_matched_relations({"rank": 0, "step": 0, "microStep": 0})
    assert res["success"] is False
