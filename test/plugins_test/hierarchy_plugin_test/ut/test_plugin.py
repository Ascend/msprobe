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
from unittest.mock import patch


from plugins.tb_graph_ascend.hierarchy_plugin.server.plugin import (
    GraphsPlugin,
    PLUGIN_NAME,
    PLUGIN_NAME_RUN_METADATA_WITH_GRAPH,
    DB_EXT,
)
from plugins.tb_graph_ascend.hierarchy_plugin.server.app.utils.global_state import (
    GraphState,
)


class MockContext:
    """Mock TensorBoard context for testing."""

    def __init__(self, logdir="/tmp/test_run"):
        self.logdir = logdir


def test_graphs_plugin_init(monkeypatch):
    """Test GraphsPlugin initialization."""
    context = MockContext(logdir="/tmp/test_run/")

    # Mock parent __init__
    monkeypatch.setattr(
        "tensorboard.plugins.base_plugin.TBPlugin.__init__", lambda self, ctx: None
    )

    plugin = GraphsPlugin(context)

    assert plugin.plugin_name == PLUGIN_NAME
    assert plugin.logdir == "/tmp/test_run"
    assert GraphState.get_global_value("logdir") == "/tmp/test_run"


def test_graphs_plugin_init_with_trailing_slash(monkeypatch):
    """Test GraphsPlugin initialization strips trailing slash."""
    context = MockContext(logdir="/tmp/test_run///")

    monkeypatch.setattr(
        "tensorboard.plugins.base_plugin.TBPlugin.__init__", lambda self, ctx: None
    )

    plugin = GraphsPlugin(context)

    assert plugin.logdir == "/tmp/test_run"


def test_graphs_plugin_init_with_home_expansion(monkeypatch):
    """Test GraphsPlugin initialization expands home directory."""
    context = MockContext(logdir="~/test_run")

    monkeypatch.setattr(
        "tensorboard.plugins.base_plugin.TBPlugin.__init__", lambda self, ctx: None
    )

    plugin = GraphsPlugin(context)

    assert plugin.logdir == os.path.expanduser("~/test_run")


def test_graphs_plugin_init_resets_global_state(monkeypatch):
    """Test GraphsPlugin initialization resets GraphState."""
    context = MockContext()

    # Set some global state before initialization
    GraphState.set_global_value("test_key", "test_value")
    assert GraphState.get_global_value("test_key") == "test_value"

    monkeypatch.setattr(
        "tensorboard.plugins.base_plugin.TBPlugin.__init__", lambda self, ctx: None
    )

    plugin = GraphsPlugin(context)

    # Global state should be reset
    assert GraphState.get_global_value("test_key") is None


def test_get_plugin_apps():
    """Test get_plugin_apps returns correct route mapping."""
    context = MockContext()

    with patch(
        "tensorboard.plugins.base_plugin.TBPlugin.__init__", lambda self, ctx: None
    ):
        plugin = GraphsPlugin(context)

    apps = plugin.get_plugin_apps()

    # Verify apps is a dictionary
    assert isinstance(apps, dict)

    # Verify expected routes are present
    expected_routes = [
        "/index.js",
        "/index.html",
        "/load_meta_dir",
        "/filterNodes",
        "/loadConvertedGraphData",
        "/convertToGraph",
        "/getConvertProgress",
        "/screen",
        "/loadGraphData",
        "/loadGraphConfigInfo",
        "/loadGraphAllNodeList",
        "/loadGraphMatchedRelations",
        "/changeNodeExpandState",
        "/updateHierarchyData",
        "/getNodeInfo",
        "/addMatchNodes",
        "/addMatchNodesByConfig",
        "/deleteMatchNodes",
        "/saveData",
        "/updateColors",
        "/saveMatchedRelations",
        "/updatePrecisionError",
    ]

    for route in expected_routes:
        assert route in apps, f"Route {route} not found in plugin apps"


def test_get_plugin_apps_all_routes_reference_graph_view(monkeypatch):
    """Test that all routes reference GraphView methods."""
    context = MockContext()

    with patch(
        "tensorboard.plugins.base_plugin.TBPlugin.__init__", lambda self, ctx: None
    ):
        plugin = GraphsPlugin(context)

    apps = plugin.get_plugin_apps()

    # Verify all routes have non-None values
    for route, handler in apps.items():
        assert handler is not None, f"Route {route} has None handler"


def test_is_active():
    """Test is_active always returns True."""
    context = MockContext()

    with patch(
        "tensorboard.plugins.base_plugin.TBPlugin.__init__", lambda self, ctx: None
    ):
        plugin = GraphsPlugin(context)

    assert plugin.is_active() is True


def test_data_plugin_names():
    """Test data_plugin_names returns correct tuple."""
    context = MockContext()

    with patch(
        "tensorboard.plugins.base_plugin.TBPlugin.__init__", lambda self, ctx: None
    ):
        plugin = GraphsPlugin(context)

    names = plugin.data_plugin_names()

    assert isinstance(names, tuple)
    assert len(names) == 2
    assert PLUGIN_NAME in names
    assert PLUGIN_NAME_RUN_METADATA_WITH_GRAPH in names
    assert names[0] == PLUGIN_NAME
    assert names[1] == PLUGIN_NAME_RUN_METADATA_WITH_GRAPH


def test_frontend_metadata():
    """Test frontend_metadata returns correct metadata."""
    context = MockContext()

    with patch(
        "tensorboard.plugins.base_plugin.TBPlugin.__init__", lambda self, ctx: None
    ):
        plugin = GraphsPlugin(context)

    metadata = plugin.frontend_metadata()

    # Verify metadata attributes
    assert metadata.es_module_path == "/index.js"
    assert metadata.disable_reload is True


def test_plugin_constants():
    """Test plugin constants are defined correctly."""
    assert PLUGIN_NAME == "graph_ascend"
    assert PLUGIN_NAME_RUN_METADATA_WITH_GRAPH == "graph_ascend_run_metadata_graph"
    assert DB_EXT is not None  # DB_EXT comes from Extension.DB.value
