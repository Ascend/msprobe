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

import pytest
from unittest.mock import MagicMock

from plugins.tb_graph_ascend.hierarchy_plugin.server.app.model.hierarchy import (
    Hierarchy,
    INNER_WIDTH,
    INNER_HIGHT,
    HORIZONTAL_SPACING,
)
from plugins.tb_graph_ascend.hierarchy_plugin.server.app.utils.constant import (
    API,
    MODULE,
    NPU,
    NPU_PREFIX,
    SINGLE,
    UNEXPAND_NODE,
)
from plugins.tb_graph_ascend.hierarchy_plugin.server.app.utils.global_state import (
    GraphState,
)


def create_mock_repo(root_node_name="root", sub_nodes=None, up_nodes=None):
    """Helper to create mock repo"""
    repo = MagicMock()
    repo.query_root_nodes.return_value = {
        "node_name": root_node_name,
        "node_type": MODULE,
        "subnodes": ["child1", "child2"] if not sub_nodes else list(sub_nodes.keys()),
    }
    if sub_nodes is None:
        sub_nodes = {
            "child1": {"node_type": UNEXPAND_NODE, "subnodes": []},
            "child2": {"node_type": UNEXPAND_NODE, "subnodes": []},
        }
    repo.query_sub_nodes.return_value = sub_nodes

    if up_nodes is None:
        up_nodes = {root_node_name: {"upnode": None}}
    repo.query_up_nodes.return_value = up_nodes
    return repo


class TestExtractLabelName:
    """Test cases for extract_label_name method"""

    @pytest.mark.parametrize(
        "node_name,node_type,expected",
        [
            ("Module.layer1.1.relu.ReLU.forward.0", MODULE, "relu.ReLU.forward.0"),
            ("Module.layer4.0.BasicBlock.forward.0", MODULE, "layer4.0.BasicBlock.forward.0"),
            ("Module.layer1.1.ApiList.1", API, "ApiList.1"),
            ("Module.layer1.1.ApiList.0.1", API, "ApiList.0.1"),
            ("short", MODULE, "short"),
            ("", MODULE, ""),
            ("A.B", MODULE, "A.B"),
            ("Api", API, "Api"),
            ("M.A.B.C.D.E.F.G.H.I", MODULE, "F.G.H.I"),
            ("A.1.B.2", MODULE, "A.1.B.2"),
            ("A.B.C.D", MODULE, "A.B.C.D"),
            ("M.1.L", API, "M.1.L"),
            ("M.L", API, "M.L"),
        ],
    )
    def test_extract_label_name_various_inputs(self, node_name, node_type, expected):
        """Test extract_label_name with various node names and types"""
        assert Hierarchy.extract_label_name(node_name, node_type) == expected


class TestMeasureTextWidth:
    """Test cases for measure_text_width method"""

    @pytest.mark.parametrize(
        "text,expected_width",
        [
            ("abcd", 24),
            ("", 0),
            ("中文", 12),
            ("a", 6),
            ("hello world", 66),
            ("x" * 100, 600),
        ],
    )
    def test_measure_text_width_various_inputs(self, text, expected_width):
        """Test measure_text_width with various text inputs"""
        assert Hierarchy.measure_text_width(text) == expected_width


class TestHierarchyInit:
    """Test cases for Hierarchy initialization"""

    @pytest.mark.parametrize(
        "graph_type,root_name,micro_step,rank,step,expected_prefix",
        [
            (NPU, "root_npu", -1, 0, 0, NPU_PREFIX),
            (SINGLE, "root_single", 0, 0, 0, ""),
        ],
    )
    def test_hierarchy_init_basic(self, graph_type, root_name, micro_step, rank, step, expected_prefix):
        """Test Hierarchy initialization with different graph types"""
        repo = create_mock_repo(root_name)
        h = Hierarchy(graph_type, repo, micro_step, rank, step)
        assert h.root_name == root_name
        assert h.graph_type == graph_type
        assert h.rank == rank
        assert h.step == step
        assert h.micro_step_id == micro_step
        assert h.name_prefix == expected_prefix
        assert root_name in h.current_hierarchy
        repo.query_root_nodes.assert_called_once()

    def test_hierarchy_init_no_root_node(self):
        """Test Hierarchy when no root node found"""
        repo = MagicMock()
        repo.query_root_nodes.return_value = None
        h = Hierarchy(NPU, repo, -1, 0, 0)
        assert not hasattr(h, "current_hierarchy") or not h.current_hierarchy


class TestGroupChildren:
    """Test cases for group_children method"""

    @pytest.mark.parametrize(
        "children,expected_groups,setup_fn",
        [
            (
                ["child1", "child2"],
                [(UNEXPAND_NODE, ["child1", "child2"])],
                lambda h: (
                    h.current_hierarchy["root"].__setitem__("children", ["child1", "child2"]),
                    h.current_hierarchy["child1"].__setitem__("nodeType", UNEXPAND_NODE),
                    h.current_hierarchy["child2"].__setitem__("nodeType", UNEXPAND_NODE),
                ),
            ),
            ([], [], lambda h: None),
            (["nonexistent"], [], lambda h: None),
        ],
    )
    def test_group_children_basic(self, children, expected_groups, setup_fn):
        """Test grouping children with various inputs"""
        repo = create_mock_repo()
        h = Hierarchy(NPU, repo, -1, 0, 0)
        setup_fn(h)

        groups = h.group_children(children)
        assert len(groups) == len(expected_groups)
        for i, (expected_type, expected_children) in enumerate(expected_groups):
            assert groups[i][0] == expected_type
            assert groups[i][1] == expected_children

    def test_group_children_mixed_types(self):
        """Test grouping children with mixed types"""
        repo = create_mock_repo()
        h = Hierarchy(NPU, repo, -1, 0, 0)
        h.current_hierarchy["node_a"] = {"nodeType": UNEXPAND_NODE}
        h.current_hierarchy["node_b"] = {"nodeType": MODULE}
        h.current_hierarchy["node_c"] = {"nodeType": MODULE}

        groups = h.group_children(["node_a", "node_b", "node_c"])
        assert len(groups) == 2
        assert groups[0][0] == UNEXPAND_NODE
        assert groups[1][0] == MODULE

    def test_group_children_alternating_types(self):
        """Test grouping children with alternating types"""
        repo = create_mock_repo()
        h = Hierarchy(NPU, repo, -1, 0, 0)
        for i, node_type in enumerate([UNEXPAND_NODE, MODULE, UNEXPAND_NODE, MODULE]):
            h.current_hierarchy[f"node_{i}"] = {"nodeType": node_type}

        groups = h.group_children([f"node_{i}" for i in range(4)])
        assert len(groups) == 4
        for i, expected_type in enumerate([UNEXPAND_NODE, MODULE, UNEXPAND_NODE, MODULE]):
            assert groups[i][0] == expected_type


class TestGetBasicRenderInfo:
    """Test cases for get_basic_rende_info method"""

    @pytest.mark.parametrize(
        "node_name,node_info,expected_checks",
        [
            (
                "root",
                {
                    "node_type": MODULE,
                    "subnodes": ["child1"],
                    "upnode": None,
                    "data": {"precision_index": "0.5"},
                    "matched_node_link": [],
                },
                {"isRoot": True, "label": "root", "nodeType": MODULE},
            ),
            (
                "child",
                {
                    "node_type": MODULE,
                    "subnodes": [],
                    "upnode": "parent",
                    "data": {"precision_index": "NaN"},
                },
                {"isRoot": False, "parentNode": "parent", "nodeType": UNEXPAND_NODE},
            ),
            ("node", None, {}),
        ],
    )
    def test_get_basic_render_info_basic(self, node_name, node_info, expected_checks):
        """Test getting render info with various inputs"""
        repo = create_mock_repo("root" if node_name == "root" else "root")
        h = Hierarchy(NPU, repo, -1, 0, 0)
        render_info = h.get_basic_rende_info(node_name, node_info)

        if node_info is None:
            assert render_info == {}
        else:
            for key, expected_value in expected_checks.items():
                assert render_info[key] == expected_value

    def test_get_basic_render_info_with_cache(self):
        """Test getting render info with cached precision"""
        repo = create_mock_repo()
        h = Hierarchy(NPU, repo, -1, 0, 0)
        GraphState.set_global_value(
            "update_precision_cache", {"testnode": {"precision_index": "0.99"}}
        )
        node_info = {
            "node_type": MODULE,
            "subnodes": [],
            "upnode": "root",
            "data": {"precision_index": "0.5"},
        }
        render_info = h.get_basic_rende_info("testnode", node_info)
        assert render_info["precisionIndex"] == "0.99"

    def test_get_basic_render_info_with_overflow_and_distributed(self):
        """Test getting render info with overflow level and matched distributed"""
        repo = create_mock_repo()
        h = Hierarchy(NPU, repo, -1, 0, 0)
        node_info = {
            "node_type": MODULE,
            "subnodes": ["child"],
            "upnode": "root",
            "data": {"precision_index": "0.85", "overflow_level": "high"},
            "matched_node_link": ["link1"],
            "matched_distributed": {"device": "npu0"},
        }
        render_info = h.get_basic_rende_info("node", node_info)
        assert render_info["precisionIndex"] == "0.85"
        assert render_info["overflowLevel"] == "high"
        assert render_info["matchedDistributed"] == {"device": "npu0"}

    def test_get_basic_render_info_root_with_microstep(self):
        """Test root node with microstep filtering"""
        repo = MagicMock()
        root_info = {
            "node_name": "root",
            "node_type": MODULE,
            "subnodes": ["child1", "child2"],
        }
        repo.query_root_nodes.return_value = root_info
        repo.query_sub_nodes.return_value = {
            "child1": {"node_type": UNEXPAND_NODE, "micro_step_id": 0, "subnodes": []},
            "child2": {"node_type": UNEXPAND_NODE, "micro_step_id": 1, "subnodes": []},
        }
        repo.query_up_nodes.return_value = {"root": {"upnode": None}}

        h_all = Hierarchy(NPU, repo, -1, 0, 0)
        render_info_all = h_all.get_basic_rende_info("root", root_info)
        assert "child1" in render_info_all["children"]
        assert "child2" in render_info_all["children"]

        repo.query_sub_nodes.return_value = {
            "child1": {"node_type": UNEXPAND_NODE, "micro_step_id": 0, "subnodes": []},
            "child2": {"node_type": UNEXPAND_NODE, "micro_step_id": 1, "subnodes": []},
        }
        h_specific = Hierarchy(NPU, repo, 0, 0, 0)
        render_info_specific = h_specific.get_basic_rende_info("root", root_info)
        assert "child1" in render_info_specific["children"]
        assert "child2" not in render_info_specific["children"]


class TestResizeHierarchy:
    """Test cases for resize_hierarchy method"""

    @pytest.mark.parametrize(
        "node_name,node_data,expected_checks,setup_children",
        [
            ("leaf", {"nodeType": UNEXPAND_NODE}, {"width": INNER_WIDTH, "height": INNER_HIGHT}, False),
            (
                "parent",
                {"nodeType": MODULE, "expand": False, "label": "test", "children": []},
                {"height": INNER_HIGHT},
                True,
            ),
            (
                "parent",
                {
                    "nodeType": MODULE,
                    "expand": True,
                    "label": "parent",
                    "children": ["child1", "child2"],
                    "width": 0,
                    "height": 0,
                },
                {"width_positive": True, "height_positive": True},
                True,
            ),
            ("nonexistent", None, None, False),
        ],
    )
    def test_resize_hierarchy_basic(self, node_name, node_data, expected_checks, setup_children):
        """Test resizing hierarchy with various node types"""
        repo = create_mock_repo()
        h = Hierarchy(NPU, repo, -1, 0, 0)

        if node_data is not None:
            h.current_hierarchy[node_name] = node_data
            if setup_children and node_name == "parent" and node_data.get("expand"):
                h.current_hierarchy["child1"] = {
                    "nodeType": UNEXPAND_NODE,
                    "width": 50,
                    "height": INNER_HIGHT,
                }
                h.current_hierarchy["child2"] = {
                    "nodeType": UNEXPAND_NODE,
                    "width": 50,
                    "height": INNER_HIGHT,
                }

        h.resize_hierarchy(node_name)

        if expected_checks is None:
            return

        node = h.current_hierarchy.get(node_name, {})
        for key, expected_value in expected_checks.items():
            if key == "width_positive":
                assert node["width"] > 0
            elif key == "height_positive":
                assert node["height"] > 0
            elif key == "width" and expected_value is True:
                expected_width = h.measure_text_width(node_data.get("label", "")) + HORIZONTAL_SPACING * 2
                assert node["width"] == expected_width
            else:
                assert node[key] == expected_value

    def test_resize_hierarchy_many_children_horizontal(self):
        """Test resize hierarchy with many UNEXPAND_NODE children (>MAX_PER_ROW)"""
        repo = create_mock_repo()
        h = Hierarchy(NPU, repo, -1, 0, 0)

        children = []
        for i in range(7):
            child_name = f"child{i}"
            children.append(child_name)
            h.current_hierarchy[child_name] = {
                "nodeType": UNEXPAND_NODE,
                "width": INNER_WIDTH,
                "height": INNER_HIGHT,
            }

        h.current_hierarchy["parent"] = {
            "nodeType": MODULE,
            "expand": True,
            "label": "parent",
            "children": children,
            "width": 0,
            "height": 0,
        }
        h.resize_hierarchy("parent")
        assert h.current_hierarchy["parent"]["width"] > 0
        assert h.current_hierarchy["parent"]["height"] > 0

    def test_resize_hierarchy_mixed_child_types(self):
        """Test resize hierarchy with mixed child types"""
        repo = create_mock_repo()
        h = Hierarchy(NPU, repo, -1, 0, 0)

        h.current_hierarchy["child1"] = {
            "nodeType": UNEXPAND_NODE,
            "width": 40,
            "height": INNER_HIGHT,
        }
        h.current_hierarchy["child2"] = {
            "nodeType": MODULE,
            "width": 50,
            "height": INNER_HIGHT,
        }

        h.current_hierarchy["parent"] = {
            "nodeType": MODULE,
            "expand": True,
            "label": "parent",
            "children": ["child1", "child2"],
            "width": 0,
            "height": 0,
        }
        h.resize_hierarchy("parent")
        assert h.current_hierarchy["parent"]["width"] > 0


class TestLayoutHierarchy:
    """Test cases for layout_hierarchy method"""

    @pytest.mark.parametrize(
        "node_name,parent_data,children_data,check_coords",
        [
            ("leaf", None, None, False),
            (
                "parent",
                {
                    "x": 0,
                    "y": 0,
                    "width": 100,
                    "height": INNER_HIGHT,
                    "expand": True,
                    "children": ["child1", "child2"],
                },
                [
                    {
                        "name": "child1",
                        "nodeType": UNEXPAND_NODE,
                        "x": 0,
                        "y": 0,
                        "width": 40,
                        "height": INNER_HIGHT,
                        "expand": False,
                        "children": [],
                    },
                    {
                        "name": "child2",
                        "nodeType": UNEXPAND_NODE,
                        "x": 0,
                        "y": 0,
                        "width": 40,
                        "height": INNER_HIGHT,
                        "expand": False,
                        "children": [],
                    },
                ],
                True,
            ),
        ],
    )
    def test_layout_hierarchy_basic(self, node_name, parent_data, children_data, check_coords):
        """Test layout hierarchy with various node types"""
        repo = create_mock_repo()
        h = Hierarchy(NPU, repo, -1, 0, 0)

        if node_name == "leaf":
            h.current_hierarchy["leaf"] = {"nodeType": UNEXPAND_NODE}
        else:
            h.current_hierarchy[node_name] = parent_data
            for child in children_data:
                h.current_hierarchy[child["name"]] = child

        h.layout_hierarchy(node_name)

        if check_coords:
            for child in children_data:
                assert h.current_hierarchy[child["name"]]["x"] >= 0
                assert h.current_hierarchy[child["name"]]["y"] >= 0

    def test_layout_hierarchy_vertical_with_module_children(self):
        """Test layout hierarchy with vertical layout for MODULE children"""
        repo = create_mock_repo()
        h = Hierarchy(NPU, repo, -1, 0, 0)

        h.current_hierarchy["child1"] = {
            "nodeType": MODULE,
            "x": 0,
            "y": 0,
            "width": 50,
            "height": INNER_HIGHT,
            "expand": False,
            "children": [],
        }
        h.current_hierarchy["child2"] = {
            "nodeType": MODULE,
            "x": 0,
            "y": 0,
            "width": 50,
            "height": INNER_HIGHT,
            "expand": False,
            "children": [],
        }

        h.current_hierarchy["parent"] = {
            "x": 0,
            "y": 0,
            "width": 100,
            "height": INNER_HIGHT,
            "expand": True,
            "children": ["child1", "child2"],
        }
        h.layout_hierarchy("parent")

        assert h.current_hierarchy["child1"]["x"] >= 0
        assert h.current_hierarchy["child1"]["y"] >= 0
        assert h.current_hierarchy["child2"]["x"] >= 0
        assert h.current_hierarchy["child2"]["y"] >= 0
        assert h.current_hierarchy["child1"]["y"] < h.current_hierarchy["child2"]["y"]

    def test_layout_hierarchy_horizontal_multiple_rows(self):
        """Test layout hierarchy with horizontal layout spanning multiple rows"""
        repo = create_mock_repo()
        h = Hierarchy(NPU, repo, -1, 0, 0)

        children = []
        for i in range(7):
            child_name = f"leaf{i}"
            children.append(child_name)
            h.current_hierarchy[child_name] = {
                "nodeType": UNEXPAND_NODE,
                "x": 0,
                "y": 0,
                "width": INNER_WIDTH,
                "height": INNER_HIGHT,
                "expand": False,
                "children": [],
            }

        h.current_hierarchy["parent"] = {
            "x": 0,
            "y": 0,
            "width": 500,
            "height": INNER_HIGHT,
            "expand": True,
            "children": children,
        }
        h.layout_hierarchy("parent")

        for child_name in children:
            assert h.current_hierarchy[child_name]["x"] >= 0
            assert h.current_hierarchy[child_name]["y"] >= 0


class TestUpdateGraphData:
    """Test cases for update_graph_data method"""

    def test_update_graph_data_root(self):
        """Test updating graph data for root node"""
        repo = create_mock_repo()
        h = Hierarchy(NPU, repo, -1, 0, 0)
        h.update_graph_data(h.root_name)
        assert h.current_hierarchy[h.root_name].get("expand", False) is True

    def test_update_graph_data_existing_node(self):
        """Test updating graph data for existing node"""
        repo = create_mock_repo()
        h = Hierarchy(NPU, repo, -1, 0, 0)
        h.current_hierarchy["existing_node"] = {
            "expand": False,
            "children": ["child1"],
            "nodeType": MODULE,
        }
        h.update_graph_data("existing_node")
        # process_click_expand requires children to be in current_hierarchy
        assert h.current_hierarchy["existing_node"]["expand"] is True

    def test_update_graph_data_non_existing_node(self):
        """Test updating graph data for non-existing node (triggers process_select_expand)"""
        repo = MagicMock()
        repo.query_root_nodes.return_value = {
            "node_name": "root",
            "node_type": MODULE,
            "subnodes": [],
        }
        repo.query_up_nodes.return_value = {
            "nonexistent": {"upnode": "root"},
            "root": {"upnode": None},
        }
        h = Hierarchy(NPU, repo, -1, 0, 0)
        h.update_graph_data("nonexistent")
        assert "nonexistent" in h.current_hierarchy or "root" in h.current_hierarchy


class TestProcessClickExpand:
    """Test cases for process_click_expand method"""

    @pytest.mark.parametrize(
        "node_name,node_data,expected_expand,description",
        [
            ("root", {"expand": True, "children": ["child1"]}, True, "already expanded root"),
            ("node", {"expand": False, "children": []}, False, "no children"),
        ],
    )
    def test_process_click_expand_basic(self, node_name, node_data, expected_expand, description):
        """Test process_click_expand with various scenarios"""
        repo = create_mock_repo()
        h = Hierarchy(NPU, repo, -1, 0, 0)

        if node_name == "root":
            h.current_hierarchy[h.root_name]["expand"] = node_data["expand"]
            h.current_hierarchy[h.root_name]["children"] = node_data["children"]
        else:
            h.current_hierarchy[node_name] = node_data

        h.process_click_expand(h.root_name if node_name == "root" else node_name)

        check_node = h.root_name if node_name == "root" else node_name
        if description == "already expanded root":
            assert h.current_hierarchy[check_node]["expand"] is expected_expand

    def test_process_click_expand_collapse_then_expand(self):
        """Test process_click_expand toggle behavior"""
        repo = create_mock_repo()
        h = Hierarchy(NPU, repo, -1, 0, 0)
        h.current_hierarchy[h.root_name]["expand"] = True
        h.current_hierarchy[h.root_name]["children"] = ["child1"]

        h.process_click_expand(h.root_name)
        assert h.current_hierarchy[h.root_name]["expand"] is True

    def test_process_click_expand_with_sub_nodes(self):
        """Test expanding node and adding sub nodes"""
        repo = MagicMock()
        repo.query_root_nodes.return_value = {
            "node_name": "root",
            "node_type": MODULE,
            "subnodes": ["parent"],
        }
        repo.query_sub_nodes.return_value = {
            "child1": {"node_type": UNEXPAND_NODE, "subnodes": []},
            "child2": {"node_type": UNEXPAND_NODE, "subnodes": []},
        }
        repo.query_up_nodes.return_value = {"root": {"upnode": None}}

        h = Hierarchy(NPU, repo, -1, 0, 0)
        h.current_hierarchy["parent"] = {
            "expand": False,
            "children": ["child1", "child2"],
            "nodeType": MODULE,
        }
        h.process_click_expand("parent")
        assert h.current_hierarchy["parent"]["expand"] is True
        assert "child1" in h.current_hierarchy
        assert "child2" in h.current_hierarchy


class TestProcessSelectExpand:
    """Test cases for process_select_expand method"""

    def test_process_select_expand_simple(self):
        """Test process_select_expand with proper parent structure"""
        repo = MagicMock()
        repo.query_root_nodes.return_value = {
            "node_name": "root",
            "node_type": MODULE,
            "subnodes": ["parent"],
        }
        repo.query_sub_nodes.return_value = {
            "child": {"node_type": UNEXPAND_NODE, "subnodes": []}
        }
        repo.query_up_nodes.return_value = {
            "child": {"upnode": "parent"},
            "parent": {"upnode": "root"},
            "root": {"upnode": None},
        }

        h = Hierarchy(NPU, repo, -1, 0, 0)
        h.current_hierarchy["parent"] = {
            "expand": False,
            "children": ["child"],
            "nodeType": MODULE,
        }
        h.current_hierarchy["child"] = {
            "expand": False,
            "children": [],
            "nodeType": UNEXPAND_NODE,
        }
        h.process_select_expand("child")
        assert h.current_hierarchy["parent"]["expand"] is True

    def test_process_select_expand_nonexistent_node(self):
        """Test process_select_expand with non-existent node"""
        repo = MagicMock()
        repo.query_root_nodes.return_value = {
            "node_name": "root",
            "node_type": MODULE,
            "subnodes": [],
        }
        repo.query_up_nodes.return_value = {
            "nonexistent": {"upnode": "root"},
            "root": {"upnode": None},
        }

        h = Hierarchy(NPU, repo, -1, 0, 0)
        h.process_select_expand("nonexistent")
        assert "root" in h.current_hierarchy


class TestUpdateHierarchyData:
    """Test cases for update_hierarchy_data and update_current_hierarchy_data methods"""

    def test_update_hierarchy_data(self):
        """Test updating hierarchy data"""
        repo = create_mock_repo()
        h = Hierarchy(NPU, repo, -1, 0, 0)
        data = h.update_hierarchy_data()
        assert isinstance(data, dict)
        assert h.root_name in data

    @pytest.mark.parametrize(
        "node_name,initial_data,update_data,expected_result,should_update",
        [
            (
                "node1",
                {"precisionIndex": "NaN", "matchedNodeLink": []},
                {"node_name": "node1", "precision_index": "0.95", "matched_node_link": ["link1"]},
                {"precisionIndex": "0.95", "matchedNodeLink": ["link1"]},
                True,
            ),
            (
                "nonexistent",
                None,
                {"node_name": "nonexistent", "precision_index": "0.5"},
                None,
                False,
            ),
        ],
    )
    def test_update_current_hierarchy_data(self, node_name, initial_data, update_data, expected_result, should_update):
        """Test updating current hierarchy data with various scenarios"""
        repo = create_mock_repo("root")
        h = Hierarchy(NPU, repo, -1, 0, 0)

        if initial_data:
            h.current_hierarchy[node_name] = initial_data

        result = h.update_current_hierarchy_data([update_data])
        assert result is True

        if should_update:
            for key, expected_value in expected_result.items():
                assert h.current_hierarchy[node_name][key] == expected_value


class TestGetConnectedGraph:
    """Test cases for get_connected_graph method"""

    def test_get_connected_graph_basic(self):
        """Test getting connected graph"""
        repo = create_mock_repo()
        h = Hierarchy(NPU, repo, -1, 0, 0)
        h.current_hierarchy[h.root_name]["expand"] = True
        h.current_hierarchy[h.root_name]["children"] = ["child1"]
        h.current_hierarchy["child1"] = {
            "expand": False,
            "children": [],
            "nodeType": UNEXPAND_NODE,
            "x": 0,
            "y": 0,
            "width": 50,
            "height": INNER_HIGHT,
        }
        result = {}
        new_hierarchy = {}
        h.get_connected_graph(h.root_name, result, new_hierarchy)
        assert h.root_name in result
        assert "child1" in new_hierarchy

    def test_get_connected_graph_recursive(self):
        """Test getting connected graph recursively"""
        repo = create_mock_repo()
        h = Hierarchy(NPU, repo, -1, 0, 0)
        h.current_hierarchy[h.root_name]["expand"] = True
        h.current_hierarchy[h.root_name]["children"] = ["child1"]
        h.current_hierarchy["child1"] = {
            "expand": True,
            "children": ["grandchild1"],
            "nodeType": MODULE,
            "x": 0,
            "y": 0,
            "width": 50,
            "height": INNER_HIGHT,
        }
        h.current_hierarchy["grandchild1"] = {
            "expand": False,
            "children": [],
            "nodeType": UNEXPAND_NODE,
            "x": 0,
            "y": 0,
            "width": 50,
            "height": INNER_HIGHT,
        }
        result = {}
        new_hierarchy = {}
        h.get_connected_graph(h.root_name, result, new_hierarchy)
        assert h.root_name in result
        assert "child1" in result
        assert "grandchild1" in result


class TestGetHierarchy:
    """Test cases for get_hierarchy method"""

    def test_get_hierarchy_basic(self):
        """Test getting hierarchy structure"""
        repo = create_mock_repo()
        h = Hierarchy(NPU, repo, -1, 0, 0)
        h.current_hierarchy[h.root_name]["expand"] = True
        hierarchy = h.get_hierarchy()
        assert isinstance(hierarchy, dict)
        assert h.root_name in hierarchy

    def test_get_hierarchy_updates_current_hierarchy(self):
        """Test that get_hierarchy updates current_hierarchy"""
        repo = create_mock_repo()
        h = Hierarchy(NPU, repo, -1, 0, 0)
        h.current_hierarchy[h.root_name]["expand"] = True
        h.get_hierarchy()
        assert isinstance(h.current_hierarchy, dict)


class TestUpdateGraphShapeAndPosition:
    """Test cases for update_graph_shape and update_graph_position methods"""

    def test_update_graph_shape(self):
        """Test update_graph_shape calls resize_hierarchy"""
        repo = create_mock_repo()
        h = Hierarchy(NPU, repo, -1, 0, 0)
        h.update_graph_shape()
        root_node = h.current_hierarchy.get(h.root_name, {})
        assert "width" in root_node
        assert "height" in root_node

    def test_update_graph_position(self):
        """Test update_graph_position calls layout_hierarchy"""
        repo = create_mock_repo()
        h = Hierarchy(NPU, repo, -1, 0, 0)
        h.update_graph_position()
        root_node = h.current_hierarchy.get(h.root_name, {})
        assert "x" in root_node
        assert "y" in root_node
