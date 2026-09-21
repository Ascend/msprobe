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
import sqlite3

import pytest

from plugins.tb_graph_ascend.hierarchy_plugin.server.app.repositories.graph_repo_db import (
    GraphRepoDB,
)
from plugins.tb_graph_ascend.hierarchy_plugin.server.app.utils.constant import (
    BENCH,
    NPU,
    SINGLE,
)
from plugins.tb_graph_ascend.hierarchy_plugin.server.app.utils.global_state import (
    GraphState,
)


def _init_sqlite_db(db_path: str):
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE tb_config (
                id INTEGER PRIMARY KEY,
                micro_steps INTEGER,
                tool_tip TEXT,
                overflow_check INTEGER,
                graph_type TEXT,
                node_colors TEXT,
                task TEXT,
                rank_list TEXT,
                step_list TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE tb_stack (
                id INTEGER PRIMARY KEY,
                stack_info TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE tb_nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_name TEXT,
                up_node TEXT,
                sub_nodes TEXT,
                node_type INTEGER,
                matched_node_link TEXT,
                precision_index REAL,
                overflow_level TEXT,
                matched_distributed TEXT,
                micro_step_id INTEGER,
                data_source TEXT,
                step INTEGER,
                rank INTEGER,
                node_order INTEGER,
                input_data TEXT,
                output_data TEXT,
                dump_data_dir TEXT,
                stack_id INTEGER,
                modified INTEGER,
                parallel_merge_info TEXT
            )
            """
        )

        # config
        conn.execute(
            """
            INSERT INTO tb_config(id, micro_steps, tool_tip, overflow_check, graph_type, node_colors, task, rank_list, step_list)
            VALUES(1, 3, ?, 1, 'compare', ?, 'summary', ?, ?)
            """,
            (
                json.dumps({"k": "v"}),
                json.dumps({"#FFFFFF": {"value": [0, 1], "description": "ok"}}),
                json.dumps([0]),
                json.dumps([1, 2]),
            ),
        )

        # stacks
        conn.execute(
            "INSERT INTO tb_stack(id, stack_info) VALUES(1, ?)", (json.dumps(["s1"]),)
        )

        # nodes: root + children
        nodes = [
            # root
            (
                "root",
                "",
                json.dumps(["n1", "n2"]),
                0,
                json.dumps([]),
                0.0,
                "NaN",
                json.dumps({}),
                -1,
                NPU,
                1,
                0,
                0,
            ),
            # children leaf nodes
            (
                "n1",
                "root",
                json.dumps([]),
                1,
                json.dumps(["b1"]),
                0.2,
                "low",
                json.dumps({}),
                -1,
                NPU,
                1,
                0,
                1,
            ),
            (
                "n2",
                "root",
                json.dumps([]),
                1,
                json.dumps([]),
                0.8,
                "high",
                json.dumps({}),
                -1,
                NPU,
                1,
                0,
                2,
            ),
            # bench nodes
            (
                "b1",
                "root",
                json.dumps([]),
                1,
                json.dumps(["n1"]),
                None,
                "NaN",
                json.dumps({}),
                -1,
                BENCH,
                1,
                0,
                3,
            ),
        ]
        for (
            node_name,
            up_node,
            sub_nodes,
            node_type,
            matched_node_link,
            precision_index,
            overflow_level,
            matched_distributed,
            micro_step_id,
            data_source,
            step,
            rank,
            node_order,
        ) in nodes:
            conn.execute(
                """
                INSERT INTO tb_nodes(
                    node_name, up_node, sub_nodes, node_type, matched_node_link, precision_index,
                    overflow_level, matched_distributed, micro_step_id, data_source, step, rank, node_order,
                    input_data, output_data, dump_data_dir, stack_id, modified, parallel_merge_info
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    node_name,
                    up_node,
                    sub_nodes,
                    node_type,
                    matched_node_link,
                    precision_index,
                    overflow_level,
                    matched_distributed,
                    micro_step_id,
                    data_source,
                    step,
                    rank,
                    node_order,
                    json.dumps({}),
                    json.dumps({}),
                    None,  # dump_data_dir
                    1,
                    0,
                    json.dumps([]),
                ),
            )

        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def repo_db(tmp_path):
    db_path = tmp_path / "t.vis.db"
    db_path.write_bytes(b"")  # 先创建空文件，满足安全校验
    os.chmod(db_path, 0o600)
    _init_sqlite_db(str(db_path))
    return GraphRepoDB(str(db_path))


def test_query_config_info_parses_fields(repo_db):
    cfg = repo_db.query_config_info()
    assert cfg["microSteps"] == 3
    assert cfg["overflowCheck"] is True
    assert cfg["isSingleGraph"] is False  # compare => 双图
    assert cfg["tooltips"] == {"k": "v"}
    assert isinstance(cfg["colors"], dict)
    assert cfg["task"] == "summary"
    assert cfg["ranks"] == [0]
    assert cfg["steps"] == [1, 2]


def test_query_root_nodes_returns_object(repo_db):
    root = repo_db.query_root_nodes(NPU, rank=0, step=1)
    assert root["node_name"] == "root"
    assert root["upnode"] == ""
    assert root["subnodes"] == ["n1", "n2"]


def test_query_sub_nodes_returns_children_sorted(repo_db):
    sub = repo_db.query_sub_nodes("root", NPU, rank=0, step=1)
    assert list(sub.keys()) == ["n1", "n2"]


def test_query_node_name_list_sorted_by_order(repo_db):
    names = repo_db.query_node_name_list(rank=0, step=1, micro_step=-1)
    assert names[:3] == ["root", "n1", "n2"]


def test_query_all_node_info_list_caches(repo_db):
    GraphState.set_global_value("all_node_info_cache", {})
    ret1 = repo_db.query_all_node_info_list(rank=0, step=1, micro_step=-1)
    assert ret1["npu_node_list"][:3] == ["root", "n1", "n2"]
    assert ret1["bench_node_list"] == ["b1"]

    # 第二次应直接命中缓存（至少结果一致）
    ret2 = repo_db.query_all_node_info_list(rank=0, step=1, micro_step=-1)
    assert ret2 == ret1
    cache = GraphState.get_global_value("all_node_info_cache")
    assert "0_1_-1" in cache


def test_query_all_matched_relations_parses_links(repo_db):
    GraphState.set_global_value("matched_relations_cache", {})
    rel = repo_db.query_all_matched_relations(rank=0, step=1, micro_step=-1)
    assert rel["npu_match_node"]["n1"] == "b1"
    assert "n2" in rel["npu_unmatch_node"]
    assert rel["bench_match_node"]["b1"] == "n1"


def test_query_node_info_joins_stack(repo_db):
    info = repo_db.query_node_info("n1", NPU, rank=0, step=1)
    assert info["node_name"] == "n1"
    assert info["stack_info"] == ["s1"]


def test_update_config_colors_persists(repo_db):
    new_colors = {"#000000": {"value": [0, 1], "description": "ok"}}
    assert repo_db.update_config_colors(new_colors) is True
    cfg = repo_db.query_config_info()
    assert cfg["colors"] == new_colors


def test_update_nodes_info_updates_precision_and_modified(repo_db):
    nodes_info = [
        {
            "node_name": "n2",
            "matched_node_link": ["b1"],
            "precision_index": 0.1,
            "input_data": {},
            "output_data": {},
            "graph_type": NPU,
        }
    ]
    assert repo_db.update_nodes_info(nodes_info, rank=0, step=1) is True

    # 再查回验证 precision_index 被更新
    info = repo_db.query_node_info("n2", NPU, rank=0, step=1)
    assert info["data"]["precision_index"] == 0.1
    assert info["modified"] == 1


def test_query_node_list_by_precision_filters(repo_db):
    meta = {"rank": 0, "step": 1, "microStep": -1}
    precision_range = {"pass": [0, 0.3], "warning": [0.3, 0.6], "error": [0.6, 1.1]}
    # n1 precision_index=0.2 => pass；n2=0.8 => error
    res = repo_db.query_node_list_by_precision(
        meta, precision_range, ["pass", "error"], is_filter_unmatch_nodes=False
    )
    names = {x["name"] for x in res}
    assert {"n1", "n2"} <= names


def test_query_up_nodes_traverses_hierarchy(repo_db):
    # n1 has parent root; should return both in ascending order
    up = repo_db.query_up_nodes("n1", NPU, rank=0, step=1)
    assert "root" in up and "n1" in up


def test_query_matched_nodes_info(repo_db):
    result = repo_db.query_matched_nodes_info("n1", "b1", rank=0, step=1)
    assert "n1" in result["NPU"]["node"]
    assert "b1" in result["Bench"]["node"]
    empty = repo_db.query_matched_nodes_info("", "", rank=0, step=1)
    assert empty == {"NPU": {"node": {}}, "Bench": {"node": {}}}


def test_query_node_and_sub_nodes(repo_db):
    data = repo_db.query_node_and_sub_nodes("root", "root", rank=0, step=1)
    assert "n1" in data["NPU"]["node"]
    assert "n2" in data["NPU"]["node"]


def test_query_matched_nodes_info_by_config(repo_db):
    cfg = {"n1": "b1"}
    result = repo_db.query_matched_nodes_info_by_config(cfg, rank=0, step=1)
    assert "n1" in result["NPU"]["node"]
    assert "b1" in result["Bench"]["node"]


def test_query_node_list_by_overflow(repo_db):
    lst = repo_db.query_node_list_by_overflow(
        step=1, rank=0, micro_step=-1, values=["low", "high"]
    )
    statuses = {x["status"] for x in lst}
    assert "low" in statuses and "high" in statuses


def test_query_node_info_by_data_source(repo_db):
    npu = repo_db.query_node_info_by_data_source(step=1, rank=0, data_source=NPU)
    bench = repo_db.query_node_info_by_data_source(step=1, rank=0, data_source=BENCH)
    assert "n1" in npu and "b1" in bench


def test_update_nodes_precision_error_and_modify_matched(repo_db):
    # precision update will close connection internally
    assert repo_db.update_nodes_precision_error([(0.42, 1, 0, "n1")]) is True
    info = repo_db.query_node_info("n1", NPU, rank=0, step=1)
    assert info["data"]["precision_index"] == 0.42

    # reopen connection for manual modification of data
    repo_db._initialize_db_connection()
    repo_db.conn.execute(
        "UPDATE tb_nodes SET modified = 1, matched_node_link = ? WHERE node_name = 'n2'",
        (json.dumps(["foo"]),),
    )
    repo_db.conn.commit()
    mod = repo_db.query_modify_matched_nodes_list(rank=0, step=1)
    assert mod.get("n2") == "foo"


def test_caches_and_invalid_source(repo_db):
    repo_db.conn.execute(
        "INSERT INTO tb_nodes(node_name,data_source,step,rank) VALUES('bad','XYZ',1,0)"
    )
    repo_db.conn.commit()
    info = repo_db.query_all_node_info_list(rank=0, step=1, micro_step=-1)
    assert "bad" not in info.get("npu_node_list", [])
    rel = repo_db.query_all_matched_relations(rank=0, step=1, micro_step=-1)
    assert "npu_match_node" in rel


def test_fetch_and_convert_rows_helper():
    # simple local dummy to avoid importing cross-file
    class DummyGraphRepoDBLocal(GraphRepoDB):
        def __init__(self):
            self.db_path = ":memory:"
            self.conn = None

    repo = DummyGraphRepoDBLocal()

    class FakeCursor:
        def fetchall(self):
            return [{"node_name": "x"}]

    out = repo._fetch_and_convert_rows(FakeCursor())
    assert "x" in out


def test_connection_failure_short_circuits(monkeypatch, repo_db):
    monkeypatch.setattr(repo_db, "_initialize_db_connection", lambda: None)
    assert repo_db.query_config_info() == {}
    assert repo_db.query_root_nodes(NPU, rank=0, step=1) == {}
    assert repo_db.query_node_name_list(rank=0, step=1, micro_step=-1) == []
    assert repo_db.query_all_node_info_list(rank=0, step=1, micro_step=-1) == {}
    assert repo_db.update_config_colors({}) is False
    assert repo_db.update_nodes_info([], rank=0, step=1) is False
    assert repo_db.update_nodes_precision_error([]) is False


def test_initialize_db_connection_permissions(monkeypatch, tmp_path):
    bad = tmp_path / "nope.db"
    bad.write_bytes(b"")
    from plugins.tb_graph_ascend.hierarchy_plugin.server.app.utils.graph_utils import (
        GraphUtils,
    )

    repo = GraphRepoDB(str(bad))
    monkeypatch.setattr(
        GraphUtils,
        "safe_check_load_file_path",
        lambda path, create=False: (False, "err"),
    )
    conn = repo._initialize_db_connection()
    assert conn is None
    assert repo.conn is None


def test_query_root_nodes_single_and_empty(repo_db):
    # SINGLE gets converted to NPU
    root = repo_db.query_root_nodes(SINGLE, rank=0, step=1)
    assert root["node_name"] == "root"
    # non-existing step returns empty
    empty = repo_db.query_root_nodes(NPU, rank=99, step=99)
    assert empty == {}


def test_methods_handle_exceptions(monkeypatch):
    class BrokenConn:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def execute(self, *args, **kwargs):
            raise sqlite3.OperationalError("boom")

        def executemany(self, *args, **kwargs):
            raise sqlite3.OperationalError("boom")

        def fetchall(self):
            raise sqlite3.OperationalError("boom")

        def commit(self):
            raise sqlite3.OperationalError("boom")

        def close(self):
            pass

    repo = GraphRepoDB(":memory:")
    monkeypatch.setattr(repo, "_initialize_db_connection", lambda: BrokenConn())
    # each method should catch and return a safe default
    assert repo.query_config_info() == {}
    assert repo.query_root_nodes(NPU, 0, 0) == {}
    assert repo.query_up_nodes("x", NPU, 0, 0) == {}
    assert repo.query_matched_nodes_info("a", "b", 0, 0) == {
        "NPU": {"node": {}},
        "Bench": {"node": {}},
    }
    assert repo.query_node_and_sub_nodes("a", "b", 0, 0) == {"NPU": {}, "Bench": {}}
    assert repo.query_matched_nodes_info_by_config({"a": "b"}, 0, 0) == {}
    assert repo.query_sub_nodes("x", NPU, 0, 0) == {}
    assert repo.query_node_info("x", NPU, 0, 0) == {}
    assert repo.query_node_name_list(0, 0, -1) == []
    assert repo.query_all_node_info_list(0, 0, -1) == {
        "npu_node_list": [],
        "bench_node_list": [],
    }
    assert repo.query_all_matched_relations(0, 0, -1) == {
        "npu_match_node": {},
        "bench_match_node": {},
        "npu_unmatch_node": [],
        "bench_unmatch_node": [],
    }
    assert repo.query_modify_matched_nodes_list(0, 0) == {}
    assert (
        repo.query_node_list_by_precision(
            {"rank": 0, "step": 0, "microStep": -1}, {}, [], False
        )
        == []
    )
    assert repo.query_node_list_by_overflow(0, 0, -1, []) == []
    assert repo.query_node_info_by_data_source(0, 0, NPU) == {}
    assert repo.update_config_colors({}) is False
    assert repo.update_nodes_info([], 0, 0) is False
    assert repo.update_nodes_precision_error([]) is False


def test_precision_with_unmatched_filter(repo_db):
    meta = {"rank": 0, "step": 1, "microStep": -1}
    precision_range = {"pass": [0, 0.3]}
    # mark n1 unmatched by clearing its matched link
    repo_db.conn.execute(
        "UPDATE tb_nodes SET matched_node_link = '' WHERE node_name='n1'"
    )
    repo_db.conn.commit()
    out = repo_db.query_node_list_by_precision(
        meta, precision_range, ["pass"], is_filter_unmatch_nodes=True
    )
    assert any(x["status"] == "unmatched" for x in out)


def test_update_nodes_info_exception(monkeypatch, repo_db):
    # patch connection to raise on executemany
    class BadConn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def executemany(self, *a, **k):
            raise Exception("fail")

        def close(self):
            pass

    monkeypatch.setattr(repo_db, "_initialize_db_connection", lambda: BadConn())
    assert (
        repo_db.update_nodes_info(
            [
                {
                    "node_name": "x",
                    "matched_node_link": [],
                    "input_data": {},
                    "output_data": {},
                    "precision_index": 0,
                    "graph_type": NPU,
                }
            ],
            0,
            0,
        )
        is False
    )


def test_update_precision_error_exception(monkeypatch):
    repo = GraphRepoDB(":memory:")

    class BadConn:
        def executemany(self, *a, **k):
            raise Exception("err")

        def commit(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(repo, "_initialize_db_connection", lambda: BadConn())
    assert repo.update_nodes_precision_error([(1, 2, 3, "n")]) is False


def test_initialize_db_connection_sqlite_error(monkeypatch, tmp_path):
    bad_path = tmp_path / "db"
    # make sqlite3.connect raise
    monkeypatch.setattr(
        sqlite3,
        "connect",
        lambda *args, **kwargs: (_ for _ in ()).throw(Exception("no sql")),
    )
    repo = GraphRepoDB(str(bad_path))
    conn = repo._initialize_db_connection()
    assert conn is None
