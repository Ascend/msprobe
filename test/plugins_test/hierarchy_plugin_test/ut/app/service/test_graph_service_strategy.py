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

import os

from plugins.tb_graph_ascend.hierarchy_plugin.server.app.service.graph_service_base import GraphServiceStrategy
from plugins.tb_graph_ascend.hierarchy_plugin.server.app.utils.global_state import GraphState


def test_load_converted_graph_data_lists_dirs_and_yaml(tmp_path, monkeypatch):
    # 构造目录结构
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "b").mkdir()
    (tmp_path / "a" / "x.yaml").write_text("k: v", encoding="utf-8")
    (tmp_path / "a" / "b" / "y.yaml").write_text("k: v", encoding="utf-8")

    # 绕过安全校验（仅验证遍历逻辑与返回结构）
    monkeypatch.setattr(
        "plugins.tb_graph_ascend.hierarchy_plugin.server.app.service.graph_service_base.GraphUtils.safe_check_load_file_path",
        lambda *_args, **_kwargs: (True, None),
    )

    ret = GraphServiceStrategy.load_converted_graph_data(str(tmp_path))
    assert ret["success"] is True
    data = ret["data"]
    assert "dirs" in data and "yaml_files" in data
    # 返回的是相对 logdir 的相对路径
    assert "a" in data["dirs"]
    assert os.path.join("a", "x.yaml") in data["yaml_files"]


def test_load_meta_dir_discovers_vis_db_files(tmp_path):
    # 准备 logdir 和 run 目录
    logdir = tmp_path / "logdir"
    run1 = logdir / "run1"
    run1.mkdir(parents=True)
    db_file = run1 / "tag1.vis.db"
    db_file.write_bytes(b"")  # 空文件足够通过安全校验与 only_check
    os.chmod(db_file, 0o600)

    GraphState.set_global_value("logdir", str(logdir))
    GraphState.set_global_value("runs", {})
    GraphState.set_global_value("first_run_tags", {})

    ret = GraphServiceStrategy.load_meta_dir()
    assert "data" in ret
    assert "run1" in ret["data"]
    assert ret["data"]["run1"]["type"] == "db"
    assert ret["data"]["run1"]["tags"] == ["tag1"]

    runs = GraphState.get_global_value("runs")
    assert "run1" in runs
