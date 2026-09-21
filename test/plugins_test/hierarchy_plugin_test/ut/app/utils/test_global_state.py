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

from plugins.tb_graph_ascend.hierarchy_plugin.server.app.utils.global_state import GraphState


def test_graph_state_defaults_exist_after_reset():
    GraphState.reset_global_state()
    assert GraphState.get_global_value("logdir") == ""
    assert GraphState.get_global_value("config_info") == {}
    assert isinstance(GraphState.get_global_value("config_data"), dict)


def test_graph_state_set_and_get():
    GraphState.set_global_value("logdir", "/tmp/logdir")
    assert GraphState.get_global_value("logdir") == "/tmp/logdir"


def test_graph_state_get_with_default_when_missing_key():
    assert GraphState.get_global_value("missing_key", default=123) == 123

