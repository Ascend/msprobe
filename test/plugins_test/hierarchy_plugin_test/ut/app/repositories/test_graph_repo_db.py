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

from plugins.tb_graph_ascend.hierarchy_plugin.server.app.repositories.graph_repo_db import GraphRepoDB


class DummyGraphRepoDB(GraphRepoDB):
    """轻量级子类，用于绕过真实数据库初始化，仅测试纯转换逻辑。"""

    def __init__(self):
        # 不调用父类 __init__，避免真实路径和权限校验
        self.db_path = ":memory:"
        self.conn = None


def test_convert_to_graph_json_structure():
    repo = DummyGraphRepoDB()
    npu_nodes = {"n1": {"node_name": "n1"}, "n2": {"node_name": "n2"}}
    bench_nodes = {"b1": {"node_name": "b1"}}

    result = repo._convert_to_graph_json(npu_nodes, bench_nodes)

    assert "NPU" in result and "Bench" in result
    assert result["NPU"]["node"] == npu_nodes
    assert result["Bench"]["node"] == bench_nodes


def test_convert_db_to_object_basic_fields():
    repo = DummyGraphRepoDB()
    raw = {
        "node_name": "test_node",
        "node_type": "1",
        "output_data": '{"key": "value"}',
        "input_data": '{"in": 1}',
        "up_node": "parent",
        "sub_nodes": '["child1", "child2"]',
        "matched_node_link": '["m1"]',
        "stack_info": '["stack1"]',
        "micro_step_id": "3",
        "precision_index": 0.5,
        "overflow_level": "low",
        "parallel_merge_info": '["p1"]',
        "matched_distributed": '["d1"]',
        "modified": "1",
    }

    obj = repo._convert_db_to_object(raw)

    assert obj["id"] == "test_node"
    assert obj["node_name"] == "test_node"
    assert obj["node_type"] == 1
    assert obj["upnode"] == "parent"
    assert obj["subnodes"] == ["child1", "child2"]
    assert obj["matched_node_link"] == ["m1"]
    assert obj["stack_info"] == ["stack1"]
    assert obj["micro_step_id"] == 3
    assert obj["data"]["precision_index"] == 0.5
    assert obj["data"]["overflow_level"] == "low"
