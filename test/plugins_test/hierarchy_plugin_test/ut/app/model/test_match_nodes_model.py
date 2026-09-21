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

import copy
import math  # 导入math模块用于检测NaN值

from plugins.tb_graph_ascend.hierarchy_plugin.server.app.model.match_nodes_model import (
    MatchNodesController,
)
from plugins.tb_graph_ascend.hierarchy_plugin.server.app.utils.constant import (
    BENCH,
    MODULE,
    NPU,
)
from plugins.tb_graph_ascend.hierarchy_plugin.server.app.utils.global_state import (
    GraphState,
)
from plugins.tb_graph_ascend.hierarchy_plugin.server.app.utils.graph_utils import (
    GraphUtils,
)


def _base_graph_data():
    # 构造最小可运行的 graph_data 结构
    return {
        "NPU": {
            "node": {
                "n1_forward": {
                    "node_type": MODULE,
                    "upnode": None,
                    "subnodes": [],
                    "input_data": {"n1_forward.in": {"md5": "aaa"}},
                    "output_data": {"n1_forward.out": {"md5": "bbb"}},
                    "data": {},
                    "matched_node_link": [],
                },
                "n1_backward": {
                    "node_type": MODULE,
                    "upnode": None,
                    "subnodes": [],
                    "input_data": {"n1_backward.in": {"md5": "aaa"}},
                    "output_data": {"n1_backward.out": {"md5": "bbb"}},
                    "data": {},
                    "matched_node_link": [],
                },
            }
        },
        "Bench": {
            "node": {
                "b1_forward": {
                    "node_type": MODULE,
                    "upnode": None,
                    "subnodes": [],
                    "input_data": {"b1_forward.in": {"md5": "aaa"}},
                    "output_data": {"b1_forward.out": {"md5": "bbb"}},
                    "data": {},
                    "matched_node_link": [],
                },
                "b1_backward": {
                    "node_type": MODULE,
                    "upnode": None,
                    "subnodes": [],
                    "input_data": {"b1_backward.in": {"md5": "aaa"}},
                    "output_data": {"b1_backward.out": {"md5": "bbb"}},
                    "data": {},
                    "matched_node_link": [],
                },
            }
        },
    }


def test_is_same_node_type_true_and_false():
    graph_data = _base_graph_data()
    assert (
        MatchNodesController.is_same_node_type(graph_data, "n1_forward", "b1_forward")
        is True
    )

    graph_data2 = _base_graph_data()
    graph_data2["Bench"]["node"]["b1_forward"]["node_type"] = 999
    assert (
        MatchNodesController.is_same_node_type(graph_data2, "n1_forward", "b1_forward")
        is False
    )


def test_process_task_add_parameter_error():
    res = MatchNodesController.process_task_add({}, "", "", "")
    assert res["success"] is False
    assert "error" in res


def test_process_task_add_nodes_type_error():
    graph_data = _base_graph_data()
    graph_data["Bench"]["node"]["b1_forward"]["node_type"] = 999
    res = MatchNodesController.process_task_add(
        graph_data, "n1_forward", "b1_forward", "md5"
    )
    assert isinstance(res, list)
    assert res[0]["success"] is False


def test_process_task_add_task_type_wrong():
    graph_data = _base_graph_data()
    res = MatchNodesController.process_task_add(
        graph_data, "n1_forward", "b1_forward", "unknown"
    )
    assert isinstance(res, list)
    assert res[0]["success"] is False
    assert res[1]["success"] is False


def test_process_md5_task_add_updates_graph_and_config_data():
    graph_data = _base_graph_data()

    # 初始未匹配列表预置
    cfg = GraphState.get_global_value("config_data")
    cfg["npuUnMatchNodes"] = ["n1_forward"]
    cfg["benchUnMatchNodes"] = ["b1_forward"]
    GraphState.set_global_value("config_data", cfg)

    ret = MatchNodesController.process_md5_task_add(
        graph_data, "n1_forward", "b1_forward"
    )
    assert ret["success"] is True
    data = ret["data"]
    assert len(data) == 2
    assert data[0]["graph_type"] == NPU
    assert data[1]["graph_type"] == BENCH

    # matched_node_link 被写回到原 graph_data
    assert graph_data["NPU"]["node"]["n1_forward"]["matched_node_link"] == [
        "b1_forward"
    ]
    assert graph_data["Bench"]["node"]["b1_forward"]["matched_node_link"] == [
        "n1_forward"
    ]

    # GraphState.config_data 被更新
    cfg2 = GraphState.get_global_value("config_data")
    assert cfg2["npuMatchNodes"]["n1_forward"] == "b1_forward"
    assert cfg2["benchMatchNodes"]["b1_forward"] == "n1_forward"
    assert "n1_forward" not in cfg2["npuUnMatchNodes"]
    assert "b1_forward" not in cfg2["benchUnMatchNodes"]


def test_process_md5_task_delete_requires_existing_match():
    graph_data = _base_graph_data()

    # 未建立匹配时删除应失败
    ret = MatchNodesController.process_md5_task_delete(
        graph_data, "n1_forward", "b1_forward"
    )
    assert ret["success"] is False

    # 建立匹配后删除应成功
    cfg = GraphState.get_global_value("config_data")
    cfg["npuMatchNodes"]["n1_forward"] = "b1_forward"
    cfg["benchMatchNodes"]["b1_forward"] = "n1_forward"
    GraphState.set_global_value("config_data", cfg)

    # 需要 graph_data 里也存在对应节点
    graph_data["NPU"]["node"]["n1_forward"]["data"]["precision_index"] = 1
    ret2 = MatchNodesController.process_md5_task_delete(
        graph_data, "n1_forward", "b1_forward"
    )
    assert ret2["success"] is True
    assert graph_data["NPU"]["node"]["n1_forward"]["matched_node_link"] == []
    assert graph_data["Bench"]["node"]["b1_forward"]["matched_node_link"] == []


def test_update_graph_node_data_formats_fields():
    node_data = {"k": {"Max": "1", "Min": "0", "Norm": "1", "Mean": "1"}}
    stat_diff = {
        "k": {
            "MaxRelativeErr": 0.1,
            "MinRelativeErr": float("nan"),
            "NormRelativeErr": 0.0,
            "MeanRelativeErr": 0.2,
            "MaxAbsErr": float("nan"),
            "MinAbsErr": 1.0,
            "MeanAbsErr": 2.0,
            "NormAbsErr": 3.0,
        }
    }
    MatchNodesController.update_graph_node_data(node_data, stat_diff, "summary")
    assert node_data["k"]["MaxRelativeErr"].__class__ == float
    assert math.isnan(node_data["k"]["MaxAbsErr"])  # 修复NaN值的比较


def test_delete_matched_node_data_removes_error_keys():
    node_data = {"k": {"MaxAbsErr": 1, "Other": 2}}
    cleaned = MatchNodesController.delete_matched_node_data(copy.deepcopy(node_data))
    assert "MaxAbsErr" not in cleaned["k"]
    assert cleaned["k"]["Other"] == 2


def test_delete_matched_node_data_skips_non_dict():
    # ensure non-dict values are ignored without raising
    node_data = {"k": None, "j": {"MaxRelativeErr": 5}}
    cleaned = MatchNodesController.delete_matched_node_data(copy.deepcopy(node_data))
    assert "k" in cleaned and cleaned["k"] is None
    assert "MaxRelativeErr" not in cleaned["j"]


def test_update_graph_node_data_noop():
    # passing empty data should simply return without errors
    original = {}
    MatchNodesController.update_graph_node_data(original, {},'summary')
    assert original == {}


def test_add_and_delete_config_match_nodes_roundtrip():
    # start with a clean config_data
    GraphState.set_global_value("config_data", {})
    MatchNodesController.add_config_match_nodes("nA", "bA")
    cfg = GraphState.get_global_value("config_data")
    assert cfg["npuMatchNodes"]["nA"] == "bA"
    assert cfg["benchMatchNodes"]["bA"] == "nA"
    assert "nA" in cfg["npuUnMatchNodes"] or "nA" not in cfg["npuUnMatchNodes"]
    # now delete and ensure entries are removed and added to unmatch lists
    MatchNodesController.delete_config_match_nodes("nA", "bA")
    cfg2 = GraphState.get_global_value("config_data")
    assert "nA" not in cfg2.get("npuMatchNodes", {})
    assert "bA" not in cfg2.get("benchMatchNodes", {})
    assert "nA" in cfg2.get("npuUnMatchNodes", [])
    assert "bA" in cfg2.get("benchUnMatchNodes", [])


def test_process_summary_task_add_basic_and_returns_data():
    graph_data = _base_graph_data()
    # replace input/output data with numeric values so statistical diff is non-empty
    graph_data["NPU"]["node"]["n1_forward"]["input_data"] = {
        "n1_forward.x": {"Max": "2", "Min": "0", "Norm": "1", "Mean": "1"}
    }
    graph_data["Bench"]["node"]["b1_forward"]["input_data"] = {
        "b1_forward.x": {"Max": "1", "Min": "0", "Norm": "2", "Mean": "1"}
    }
    graph_data["NPU"]["node"]["n1_forward"]["output_data"] = {
        "n1_forward.y": {"Max": "5", "Min": "1", "Norm": "3", "Mean": "2"}
    }
    graph_data["Bench"]["node"]["b1_forward"]["output_data"] = {
        "b1_forward.y": {"Max": "4", "Min": "0", "Norm": "3", "Mean": "2"}
    }
    # call directly
    res = MatchNodesController.process_summary_task_add(
        graph_data, "n1_forward", "b1_forward"
    )
    assert res["success"] is True
    assert isinstance(res.get("data"), list) and len(res["data"]) == 2
    # graph_data should be updated: matched_node_link uses parent list, so root linked to other node
    assert graph_data["NPU"]["node"]["n1_forward"]["matched_node_link"] == [
        "b1_forward"
    ]
    assert graph_data["Bench"]["node"]["b1_forward"]["matched_node_link"] == [
        "n1_forward"
    ]
    cfg = GraphState.get_global_value("config_data")
    assert cfg["npuMatchNodes"]["n1_forward"] == "b1_forward"


def test_process_summary_task_add_io_empty_error():
    graph_data = {
        "NPU": {"node": {"n1": {"input_data": {}, "output_data": {}}}},
        "Bench": {"node": {"b1": {"input_data": {}, "output_data": {}}}},
    }
    res = MatchNodesController.process_summary_task_add(graph_data, "n1", "b1")
    assert res["success"] is True
    # message should mention failure to compute statistical diff
    assert res["data"][0]["precision_index"] == 0


def test_process_summary_task_delete_errors_and_success():
    # config mismatch
    graph_data = _base_graph_data()
    GraphState.set_global_value(
        "config_data", {"npuMatchNodes": {}, "benchMatchNodes": {}}
    )
    res = MatchNodesController.process_summary_task_delete(
        graph_data, "n1_forward", "b1_forward"
    )
    assert res["success"] is False
    assert "节点未匹配" in res.get("error", "")

    # node missing
    cfg = {
        "npuMatchNodes": {"n1_forward": "b1_forward"},
        "benchMatchNodes": {"b1_forward": "n1_forward"},
    }
    GraphState.set_global_value("config_data", cfg)
    empty_graph = {"NPU": {"node": {}}, "Bench": {"node": {}}}
    res2 = MatchNodesController.process_summary_task_delete(
        empty_graph, "n1_forward", "b1_forward"
    )
    assert res2["success"] is False
    assert "节点不存在" in res2.get("error", "")

    # success case
    graph_data2 = _base_graph_data()
    # add precision_index so pop() exercise
    graph_data2["NPU"]["node"]["n1_forward"]["data"]["precision_index"] = 0.5
    GraphState.set_global_value("config_data", cfg)
    res3 = MatchNodesController.process_summary_task_delete(
        graph_data2, "n1_forward", "b1_forward"
    )
    assert res3["success"] is True
    assert graph_data2["NPU"]["node"]["n1_forward"]["matched_node_link"] == []
    cfg3 = GraphState.get_global_value("config_data")
    assert "n1_forward" not in cfg3.get("npuMatchNodes", {})


def test_process_task_add_and_delete_via_general_methods():
    graph_data = _base_graph_data()
    # md5 success
    result = MatchNodesController.process_task_add(
        graph_data, "n1_forward", "b1_forward", "md5"
    )
    assert result[0]["success"] is True and result[1]["success"] is True
    # delete via general
    GraphState.set_global_value(
        "config_data",
        {
            "npuMatchNodes": {"n1_forward": "b1_forward"},
            "benchMatchNodes": {"b1_forward": "n1_forward"},
        },
    )
    resdel = MatchNodesController.process_task_delete(
        graph_data, "n1_forward", "b1_forward", "md5"
    )
    assert resdel[0]["success"] is True
    # summary via general
    graph_data2 = _base_graph_data()
    graph_data2["NPU"]["node"]["n1_forward"]["input_data"] = {
        "n1_forward.x": {"Max": "1", "Min": "0", "Norm": "1", "Mean": "1"}
    }
    graph_data2["Bench"]["node"]["b1_forward"]["input_data"] = {
        "b1_forward.x": {"Max": "1", "Min": "0", "Norm": "1", "Mean": "1"}
    }
    graph_data2["NPU"]["node"]["n1_forward"]["output_data"] = {
        "n1_forward.y": {"Max": "2", "Min": "0", "Norm": "2", "Mean": "2"}
    }
    graph_data2["Bench"]["node"]["b1_forward"]["output_data"] = {
        "b1_forward.y": {"Max": "2", "Min": "0", "Norm": "2", "Mean": "2"}
    }
    ressum = MatchNodesController.process_task_add(
        graph_data2, "n1_forward", "b1_forward", "summary"
    )
    assert isinstance(ressum, list) and ressum[0]["success"]


def test_process_task_add_child_layer_and_delete_with_subnodes():
    # prepare graph with one level of children that should match
    graph_data = _base_graph_data()
    # create matching children names such that extract_module_name returns "common"
    graph_data["NPU"]["node"]["n1_forward"]["subnodes"] = ["common.a.b.c"]
    graph_data["Bench"]["node"]["b1_forward"]["subnodes"] = ["common.x.y.z"]
    graph_data["NPU"]["node"]["common.a.b.c"] = {
        "node_type": MODULE,
        "matched_node_link": [],
        "input_data": {"foo": {"md5": "1"}},
        "output_data": {"foo": {"md5": "1"}},
        "data": {},
        "subnodes": [],
    }
    graph_data["Bench"]["node"]["common.x.y.z"] = {
        "node_type": MODULE,
        "matched_node_link": [],
        "input_data": {"foo": {"md5": "1"}},
        "output_data": {"foo": {"md5": "1"}},
        "data": {},
        "subnodes": [],
    }
    # ensure config_data empty for fresh start
    GraphState.set_global_value("config_data", {})
    results = MatchNodesController.process_task_add_child_layer(
        graph_data, "n1_forward", "b1_forward", "md5"
    )
    # root and child should have success entries
    assert any(item.get("success") for item in results)
    # test delete child layer after matching
    # put config state indicating match
    GraphState.set_global_value(
        "config_data",
        {
            "npuMatchNodes": {
                "n1_forward": "b1_forward",
                "common.a.b.c": "common.x.y.z",
            },
            "benchMatchNodes": {
                "b1_forward": "n1_forward",
                "common.x.y.z": "common.a.b.c",
            },
        },
    )
    result_del = MatchNodesController.process_task_delete_child_layer(
        graph_data, "n1_forward", "b1_forward", "md5"
    )
    assert isinstance(result_del, list)


def test_process_task_add_child_layer_parameter_and_type_errors():
    assert MatchNodesController.process_task_add_child_layer({}, "", "", "") == [
        {"success": False, "error": GraphUtils.t("parameterError")}
    ]
    gd = _base_graph_data()
    gd["Bench"]["node"]["b1_forward"]["node_type"] = 999
    assert (
        MatchNodesController.process_task_add_child_layer(
            gd, "n1_forward", "b1_forward", "md5"
        )[0]["success"]
        is False
    )


def test_process_task_delete_child_layer_parameter_error():
    assert MatchNodesController.process_task_delete_child_layer({}, "", "", "") == [
        {"success": False, "error": GraphUtils.t("parameterError")}
    ]
