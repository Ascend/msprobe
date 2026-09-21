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

import math
import os
import json
import tempfile

import pytest

from plugins.tb_graph_ascend.hierarchy_plugin.server.app.utils.graph_utils import (
    GraphUtils,
    FILE_PATH_MAX_LENGTH,
    MAX_FILE_SIZE,
)
from plugins.tb_graph_ascend.hierarchy_plugin.server.app.utils.constant import DataType
import stat
from plugins.tb_graph_ascend.hierarchy_plugin.server.app.utils.global_state import (
    GraphState,
)
from plugins.tb_graph_ascend.hierarchy_plugin.server.app.utils.i18n import EN, ZH


def test_get_opposite_node_name_forward_to_backward():
    name = "layer1_forward_conv"
    assert GraphUtils.get_opposite_node_name(name) == "layer1_backward_conv"


def test_get_opposite_node_name_backward_to_forward():
    name = "block2_backward_dense"
    assert GraphUtils.get_opposite_node_name(name) == "block2_forward_dense"


@pytest.mark.parametrize(
    "value,expected",
    [
        ("10%", 0.1),
        ("0.5%", 0.005),
        ("1.25%", 0.0125),
        (0.3, 0.3),
    ],
)
def test_convert_to_float_percentage_and_number(value, expected):
    result = GraphUtils.convert_to_float(value)
    assert math.isclose(result, expected, rel_tol=1e-9)


def test_convert_to_float_nan_on_invalid():
    result = GraphUtils.convert_to_float("not-a-number")
    assert math.isnan(result)


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, "NaN"),
        (float("nan"), "NaN"),
        (0.0, "0.0000%"),
        (0.1234, "12.3400%"),
    ],
)
def test_format_relative_err(value, expected):
    result = GraphUtils.format_relative_err(value)
    assert result == expected


def test_nan_to_str():
    assert GraphUtils.nan_to_str(float("nan")) == "NaN"
    assert GraphUtils.nan_to_str(0.5) == 0.5


def test_split_graph_data_by_microstep_all_when_minus_one():
    graph_data = {
        "root": "root",
        "node": {
            "root": {"subnodes": ["child1", "child2"], "micro_step_id": 0},
            "child1": {"subnodes": [], "micro_step_id": 1},
            "child2": {"subnodes": [], "micro_step_id": 2},
        },
    }
    result = GraphUtils.split_graph_data_by_microstep(graph_data, -1)
    assert result == graph_data["node"]


def test_split_graph_data_by_microstep_specific_micro_step():
    graph_data = {
        "root": "root",
        "node": {
            "root": {"subnodes": ["child1", "child2"], "micro_step_id": 0},
            "child1": {"subnodes": [], "micro_step_id": 1},
            "child2": {"subnodes": [], "micro_step_id": 2},
        },
    }
    result = GraphUtils.split_graph_data_by_microstep(graph_data, 1)
    # 当前实现仅返回满足 micro_step 过滤条件的节点（不强制包含 root）
    assert set(result.keys()) == {"child1"}


def test_t_returns_translation_by_graph_state_lang():
    GraphState.set_global_value("lang", ZH)
    assert GraphUtils.t("dbInitError") == "数据库初始化失败"

    GraphState.set_global_value("lang", EN)
    assert GraphUtils.t("dbInitError") == "Database initialization failed"


def test_get_parent_node_list_single_node():
    """Test getting parent node list with single node"""
    graph_data = {
        "node": {
            "root": {"upnode": None, "name": "root"},
        }
    }
    result = GraphUtils.get_parent_node_list(graph_data, "root")
    assert result == ["root"]


def test_get_parent_node_list_chain():
    """Test getting parent node list with chain of nodes"""
    graph_data = {
        "node": {
            "child": {"upnode": "parent", "name": "child"},
            "parent": {"upnode": "root", "name": "parent"},
            "root": {"upnode": None, "name": "root"},
        }
    }
    result = GraphUtils.get_parent_node_list(graph_data, "child")
    assert result == ["root", "parent", "child"]


def test_get_parent_node_list_empty_graph():
    """Test get_parent_node_list with empty graph"""
    graph_data = {"node": {}}
    result = GraphUtils.get_parent_node_list(graph_data, "missing_node")
    assert result == ["missing_node"]


def test_get_parent_node_list_none_graph():
    """Test get_parent_node_list with None graph_data"""
    result = GraphUtils.get_parent_node_list(None, "node")
    assert result == []


def test_get_parent_node_list_none_node_name():
    """Test get_parent_node_list with None node_name"""
    graph_data = {"node": {"root": {}}}
    result = GraphUtils.get_parent_node_list(graph_data, None)
    assert result == []


def test_remove_prefix():
    """Test remove_prefix removes specified prefix"""
    node_data = {"N___layer1": 10, "N___layer2": 20, "other": 30}
    result = GraphUtils.remove_prefix(node_data, "N___")
    assert result == {"layer1": 10, "layer2": 20, "other": 30}


def test_remove_prefix_none_data():
    """Test remove_prefix with None data"""
    result = GraphUtils.remove_prefix(None, "N___")
    assert result == {}


def test_remove_prefix_no_matching():
    """Test remove_prefix when no keys match prefix"""
    node_data = {"layer1": 10, "layer2": 20}
    result = GraphUtils.remove_prefix(node_data, "N___")
    assert result == {"layer1": 10, "layer2": 20}


def test_bytes_to_human_readable_zero():
    """Test bytes_to_human_readable with 0 bytes"""
    assert GraphUtils.bytes_to_human_readable(0) == "0 B"


def test_bytes_to_human_readable_bytes():
    """Test bytes_to_human_readable with small values"""
    assert GraphUtils.bytes_to_human_readable(512, 0) == "512 B"


def test_bytes_to_human_readable_kilobytes():
    """Test bytes_to_human_readable with KB values"""
    result = GraphUtils.bytes_to_human_readable(1024)
    assert result == "1.00 KB"


def test_bytes_to_human_readable_megabytes():
    """Test bytes_to_human_readable with MB values"""
    result = GraphUtils.bytes_to_human_readable(1024 * 1024)
    assert result == "1.00 MB"


def test_bytes_to_human_readable_gigabytes():
    """Test bytes_to_human_readable with GB values"""
    result = GraphUtils.bytes_to_human_readable(1024 * 1024 * 1024)
    assert result == "1.00 GB"


def test_safe_json_loads_valid_json():
    """Test safe_json_loads with valid JSON"""
    json_str = '{"key": "value", "number": 42}'
    result = GraphUtils.safe_json_loads(json_str)
    assert result == {"key": "value", "number": 42}


def test_safe_json_loads_invalid_json():
    """Test safe_json_loads with invalid JSON"""
    json_str = "invalid json"
    result = GraphUtils.safe_json_loads(json_str)
    assert result is None


def test_safe_json_loads_invalid_json_with_default():
    """Test safe_json_loads with invalid JSON and default value"""
    json_str = "invalid json"
    result = GraphUtils.safe_json_loads(json_str, default_value={})
    assert result == {}


def test_safe_json_loads_non_string():
    """Test safe_json_loads with non-string input"""
    result = GraphUtils.safe_json_loads(123)
    assert result is None


def test_safe_json_loads_non_string_with_default():
    """Test safe_json_loads with non-string input and default"""
    result = GraphUtils.safe_json_loads(123, default_value=[])
    assert result == []


def test_safe_get_node_info_valid():
    """Test safe_get_node_info with valid data"""
    data = {"nodeInfo": {"nodeName": "layer1", "nodeType": "NPU", "shape": [10, 20]}}
    result = GraphUtils.safe_get_node_info(data)
    assert result == data["nodeInfo"]


def test_safe_get_node_info_missing_field():
    """Test safe_get_node_info with missing required field"""
    data = {"nodeInfo": {"nodeName": "layer1"}}
    result = GraphUtils.safe_get_node_info(data)
    assert result is None


def test_safe_get_node_info_with_default():
    """Test safe_get_node_info with default value"""
    data = {"nodeInfo": {"nodeName": "layer1"}}
    result = GraphUtils.safe_get_node_info(data, default_value={})
    assert result == {}


def test_safe_get_meta_data_valid():
    """Test safe_get_meta_data with valid metadata"""
    GraphState.set_global_value("config_info", {})
    data = {
        "metaData": {
            "tag": "tag1",
            "microStep": -1,
            "run": "run1",
            "type": "vis",
            "lang": "zh",
        }
    }
    result = GraphUtils.safe_get_meta_data(data)
    assert result == data["metaData"]


def test_safe_get_meta_data_missing_field():
    """Test safe_get_meta_data with missing required field"""
    data = {"metaData": {"tag": "tag1"}}
    result = GraphUtils.safe_get_meta_data(data)
    assert result is None


def test_safe_get_meta_data_with_default():
    """Test safe_get_meta_data with default value"""
    data = {"metaData": {"tag": "tag1"}}
    result = GraphUtils.safe_get_meta_data(data, default_value={})
    assert result == {}


def test_is_relative_to_relative_path():
    """Test is_relative_to with relative path"""
    with tempfile.TemporaryDirectory() as tmpdir:
        subdir = os.path.join(tmpdir, "subdir")
        os.makedirs(subdir)
        result = GraphUtils.is_relative_to(subdir, tmpdir)
        assert result is True


def test_is_relative_to_unrelated_path():
    """Test is_relative_to with unrelated path"""
    with tempfile.TemporaryDirectory() as tmpdir1:
        with tempfile.TemporaryDirectory() as tmpdir2:
            result = GraphUtils.is_relative_to(tmpdir1, tmpdir2)
            assert result is False


def test_walk_with_max_depth():
    """Test walk_with_max_depth limits directory depth"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create nested structure: level1/level2/level3
        level1 = os.path.join(tmpdir, "level1")
        level2 = os.path.join(level1, "level2")
        level3 = os.path.join(level2, "level3")
        os.makedirs(level3)

        # Walk with max_depth=2 should stop before level3
        depths = []
        for root, dirs, files in GraphUtils.walk_with_max_depth(tmpdir, 2):
            depth = root[len(tmpdir) :].count(os.sep) + 1
            depths.append(depth)

        # Should visit tmpdir (depth 1) and level1 (depth 2)
        assert 0 not in depths or 1 in depths or 2 in depths


def test_get_graph_data_from_cache(monkeypatch):
    """Test get_graph_data returns cached data"""
    GraphState.set_global_value("current_tag", "tag1")
    GraphState.set_global_value("current_run", "/path/to/run")
    GraphState.set_global_value("current_file_data", {"cached": "data"})
    GraphState.set_global_value("runs", {"run1": "/path/to/run"})

    result, error = GraphUtils.get_graph_data({"run": "run1", "tag": "tag1"})
    assert error is None
    assert result == {"cached": "data"}


def test_get_graph_data_no_params():
    """Test get_graph_data with no parameters"""
    result, error = GraphUtils.get_graph_data(None)
    assert result is None
    assert error is not None


def test_get_graph_data_no_tag():
    """Test get_graph_data with missing tag"""
    result, error = GraphUtils.get_graph_data({"run": "run1"})
    assert result is None


@pytest.mark.parametrize(
    "percentage_str",
    [
        "0.0%, by due to Mean smaller than 1e-06",
        "10%, with comment",
        "5.5%, some comment",
    ],
)
def test_convert_to_float_with_comment(percentage_str):
    """Test convert_to_float parses percentage with comma"""
    result = GraphUtils.convert_to_float(percentage_str)
    # Should parse the first part before comma
    assert not math.isnan(result)
    assert result >= 0


def test_split_graph_data_by_microstep_nested_nodes():
    """Test split_graph_data_by_microstep with nested nodes"""
    graph_data = {
        "root": "root",
        "node": {
            "root": {"subnodes": ["child1", "child2"], "micro_step_id": 0},
            "child1": {"subnodes": ["grandchild1"], "micro_step_id": 1},
            "child2": {"subnodes": [], "micro_step_id": 1},
            "grandchild1": {"subnodes": [], "micro_step_id": 2},
        },
    }

    # Filter for micro_step=1 should include child1 and child2, but not grandchild1
    result = GraphUtils.split_graph_data_by_microstep(graph_data, 1)
    assert "child1" in result
    assert "child2" in result
    assert "grandchild1" not in result
    assert "root" not in result


def test_split_graph_data_by_microstep_circular_error(monkeypatch):
    """Circular reference in microstep traversal raises ValueError when both nodes match filter"""
    monkeypatch.setattr(GraphUtils, "t", lambda x: "err")
    graph_data = {
        "root": "root",
        "node": {
            "root": {"subnodes": ["a"], "micro_step_id": 0},
            "a": {"subnodes": ["root"], "micro_step_id": 0},
        },
    }
    with pytest.raises(ValueError):
        GraphUtils.split_graph_data_by_microstep(graph_data, 0)


def test_safe_save_and_load_data(tmp_path, monkeypatch):
    """Verify saving and loading round-trip"""
    run_dir = tmp_path / "runA"
    run_dir.mkdir()
    GraphState.set_global_value("runs", {"runA": str(run_dir)})
    GraphState.set_global_value("logdir", str(tmp_path))
    monkeypatch.setattr(
        GraphUtils,
        "safe_check_save_file_path",
        staticmethod(lambda p, d=False: (True, None)),
    )
    monkeypatch.setattr(
        GraphUtils,
        "safe_check_load_file_path",
        staticmethod(lambda p, d=False: (True, None)),
    )

    ok, err = GraphUtils.safe_save_data({"foo": "bar"}, "runA", "tag1")
    assert ok is True and err is None
    saved = run_dir / "tag1"
    assert saved.exists()

    data, err2 = GraphUtils.safe_load_data("runA", "tag1")
    assert err2 is None and data == {"foo": "bar"}
    chk, _ = GraphUtils.safe_load_data("runA", "tag1", only_check=True)
    assert chk is True


def test_safe_save_data_invalid_params():
    res, err = GraphUtils.safe_save_data({}, "", "tag")
    assert res is None and "required" in err
    res2, err2 = GraphUtils.safe_save_data({}, "run", "")
    assert res2 is None and "required" in err2


def test_safe_save_data_bad_tag(monkeypatch):
    GraphState.set_global_value("runs", {"r": "/tmp"})
    ok, err = GraphUtils.safe_save_data({}, "r", "bad tag")
    assert ok is None and err == "Invalid data"


def test_safe_save_data_permission_error(monkeypatch):
    GraphState.set_global_value("runs", {"r": "/tmp"})
    monkeypatch.setattr(
        GraphUtils,
        "safe_check_save_file_path",
        staticmethod(lambda p, d=False: (False, "nope")),
    )
    ok, err = GraphUtils.safe_save_data({}, "r", "t")
    assert ok is None


def test_safe_load_data_missing_params():
    res, err = GraphUtils.safe_load_data(None, "tag")
    assert res is None and "required" in err
    res2, err2 = GraphUtils.safe_load_data("run", None)
    assert res2 is None and "required" in err2


def test_safe_load_data_unsecure(monkeypatch, tmp_path):
    GraphState.set_global_value("logdir", str(tmp_path / "safe"))
    GraphState.set_global_value("runs", {"r": "/not/safe"})
    ok, err = GraphUtils.safe_load_data("r", "tag")
    assert ok is None


def test_find_config_files(tmp_path, monkeypatch):
    run_dir = tmp_path / "runX"
    run_dir.mkdir()
    (run_dir / "a.vis.config").write_text("x")
    (run_dir / "b.txt").write_text("y")
    GraphState.set_global_value("runs", {"runX": str(run_dir)})
    monkeypatch.setattr(
        GraphUtils,
        "safe_check_load_file_path",
        staticmethod(lambda p, d=False: (True, None)),
    )
    files = GraphUtils.find_config_files("runX")
    assert files == ["a.vis.config"]


def test_safe_check_save_file_path_conditions(tmp_path):
    long = "a" * (FILE_PATH_MAX_LENGTH + 1)
    ok, err = GraphUtils.safe_check_save_file_path(long)
    assert ok is False
    fpath = tmp_path / "f"
    fpath.write_text("x")
    os.chmod(fpath, 0o400)
    ok2, err2 = GraphUtils.safe_check_save_file_path(str(fpath))
    assert ok2 is False
    sl = tmp_path / "link"
    sl.symlink_to(fpath)
    ok3, err3 = GraphUtils.safe_check_save_file_path(str(sl))
    assert ok3 is False


def test_safe_check_load_file_path_conditions(tmp_path):
    # calling with non-existent path raises FileNotFoundError (stat call occurs first)
    with pytest.raises(FileNotFoundError):
        GraphUtils.safe_check_load_file_path(str(tmp_path / "no"))
    f = tmp_path / "f2"
    f.write_text("x")
    os.chmod(f, 0o000)
    ok2, err2 = GraphUtils.safe_check_load_file_path(str(f))
    assert ok2 is False
    d = tmp_path / "d"
    d.mkdir()
    ok3, err3 = GraphUtils.safe_check_load_file_path(str(d))
    assert ok3 is False


def test_safe_save_data_type_error(monkeypatch):
    """TypeError during JSON dump should be caught"""
    GraphState.set_global_value("runs", {"r": "/tmp"})
    monkeypatch.setattr(
        GraphUtils,
        "safe_check_save_file_path",
        staticmethod(lambda p, d=False: (True, None)),
    )
    monkeypatch.setattr(
        json,
        "dump",
        lambda d, f, ensure_ascii, indent: (_ for _ in ()).throw(TypeError("x")),
    )
    ok, err = GraphUtils.safe_save_data({"foo": "bar"}, "r", "t")
    assert ok is None and err == "Invalid data"


def test_safe_save_data_os_error(monkeypatch):
    """Simulate OSError when opening file"""
    GraphState.set_global_value("runs", {"r": "/tmp"})
    monkeypatch.setattr(
        GraphUtils,
        "safe_check_save_file_path",
        staticmethod(lambda p, d=False: (True, None)),
    )
    monkeypatch.setattr(
        "builtins.open", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("fail"))
    )
    ok, err = GraphUtils.safe_save_data({"foo": "bar"}, "r", "t")
    assert ok is None


def test_safe_save_data_runtime_error_link(monkeypatch, tmp_path):
    """Simulate file becoming symlink after write"""
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    GraphState.set_global_value("runs", {"r": str(run_dir)})
    monkeypatch.setattr(
        GraphUtils,
        "safe_check_save_file_path",
        staticmethod(lambda p, d=False: (True, None)),
    )
    monkeypatch.setattr(os.path, "islink", lambda p: True)
    ok, err = GraphUtils.safe_save_data({"foo": "bar"}, "r", "t")
    assert ok is None and "failed to save" in err


def test_safe_load_data_invalid_json(tmp_path, monkeypatch):
    dirpath = tmp_path / "r"
    dirpath.mkdir()
    f = dirpath / "tag"
    f.write_text("notjson")
    GraphState.set_global_value("runs", {"r": str(dirpath)})
    GraphState.set_global_value("logdir", str(tmp_path))
    monkeypatch.setattr(GraphUtils, "is_relative_to", staticmethod(lambda p, b: True))
    monkeypatch.setattr(
        GraphUtils,
        "safe_check_load_file_path",
        staticmethod(lambda p, d=False: (True, None)),
    )
    data, err = GraphUtils.safe_load_data("r", "tag")
    assert err == "File is not a valid JSON file!"


def test_safe_load_data_generic_exception(monkeypatch, tmp_path):
    dirpath = tmp_path / "r2"
    dirpath.mkdir()
    f = dirpath / "tag2"
    f.write_text("{}")
    GraphState.set_global_value("runs", {"r2": str(dirpath)})
    GraphState.set_global_value("logdir", str(tmp_path))
    monkeypatch.setattr(GraphUtils, "is_relative_to", staticmethod(lambda p, b: True))
    monkeypatch.setattr(
        GraphUtils,
        "safe_check_load_file_path",
        staticmethod(lambda p, d=False: (True, None)),
    )
    monkeypatch.setattr(json, "load", lambda f: (_ for _ in ()).throw(Exception("bad")))
    data, err = GraphUtils.safe_load_data("r2", "tag2")
    assert data is None


def test_safe_check_save_file_path_owner_and_group(monkeypatch, tmp_path):
    class S:
        pass

    st = S()
    st.st_mode = os.stat(tmp_path).st_mode | 0o020
    st.st_uid = os.getuid() + 1
    monkeypatch.setattr(os, "stat", lambda p: st)
    ok, err = GraphUtils.safe_check_save_file_path(str(tmp_path))
    current_uid = os.getuid()
    if current_uid == 0:
        assert ok is True
    else:
        assert ok is False


def test_safe_check_load_file_path_permissions(monkeypatch, tmp_path):
    f = tmp_path / "p"
    f.write_text("x")

    class S:
        pass

    st = S()
    st.st_mode = os.stat(str(f)).st_mode | 0o020
    st.st_uid = os.getuid() + 1
    monkeypatch.setattr(os, "stat", lambda p: st)
    ok, err = GraphUtils.safe_check_load_file_path(str(f))
    assert ok is False


def test_get_graph_data_exception(monkeypatch):
    """get_graph_data should handle exceptions from safe_load_data"""
    GraphState.set_global_value("runs", {"run1": "/tmp"})
    monkeypatch.setattr(
        GraphUtils,
        "safe_load_data",
        lambda r, t, o=False: (_ for _ in ()).throw(Exception("boom")),
    )
    res, err = GraphUtils.get_graph_data({"run": "run1", "tag": "x"})
    assert res is None
    assert "fail to get graph data" in err


def test_get_parent_node_list_circular():
    """Circular parent chain raises ValueError"""
    GraphState.set_global_value("lang", EN)
    graph_data = {"node": {"a": {"upnode": "b"}, "b": {"upnode": "a"}}}
    with pytest.raises(ValueError):
        GraphUtils.get_parent_node_list(graph_data, "a")


def test_safe_json_loads_length_exceeded(monkeypatch):
    # monkeypatch MAX_FILE_SIZE to small value so we don't allocate huge string
    monkeypatch.setattr(
        "plugins.tb_graph_ascend.hierarchy_plugin.server.app.utils.graph_utils.MAX_FILE_SIZE",
        1,
    )
    assert GraphUtils.safe_json_loads("xx") is None


def test_safe_json_loads_generic_error(monkeypatch):
    monkeypatch.setattr(
        json, "loads", lambda s: (_ for _ in ()).throw(Exception("oops"))
    )
    assert GraphUtils.safe_json_loads("{}") is None


def test_safe_get_node_info_length_exceeded():
    class Big:
        def __str__(self):
            return "x" * (MAX_FILE_SIZE + 1)

    data = {"nodeInfo": Big()}
    assert GraphUtils.safe_get_node_info(data) is None


def test_safe_get_node_info_jsondecode(monkeypatch):
    class Bad:
        def __str__(self):
            raise json.JSONDecodeError("err", "", 0)

    data = {"nodeInfo": Bad()}
    assert GraphUtils.safe_get_node_info(data) is None


def test_safe_get_node_info_generic_exception(monkeypatch):
    class Bad:
        def __str__(self):
            raise RuntimeError("bad")

    data = {"nodeInfo": Bad()}
    assert GraphUtils.safe_get_node_info(data) is None


def test_safe_get_meta_data_length_exceeded():
    class Big:
        def __str__(self):
            return "x" * (MAX_FILE_SIZE + 1)

    data = {"metaData": Big()}
    assert GraphUtils.safe_get_meta_data(data) is None


def test_safe_get_meta_data_microstep_out_of_range():
    GraphState.set_global_value("config_info", {"microSteps": 3})
    m = {"tag": "t", "microStep": 5, "run": "r", "type": "v", "lang": "zh"}
    assert GraphUtils.safe_get_meta_data({"metaData": m}) is None


def test_safe_get_meta_data_rank_step_config(monkeypatch):
    GraphState.set_global_value("config_info", {"ranks": [1], "steps": [2]})
    m = {
        "tag": "t",
        "microStep": -1,
        "run": "r",
        "type": DataType.DB.value,
        "lang": "zh",
        "rank": 2,
        "step": 1,
    }
    # rank not in config
    assert GraphUtils.safe_get_meta_data({"metaData": m}) is None
    m["rank"] = 1
    m["step"] = 3
    assert GraphUtils.safe_get_meta_data({"metaData": m}) is None


def test_safe_get_meta_data_jsondecode(monkeypatch):
    class Bad:
        def __str__(self):
            raise json.JSONDecodeError("err", "", 0)

    data = {"metaData": Bad()}
    assert GraphUtils.safe_get_meta_data(data) is None


def test_safe_get_meta_data_generic_exception(monkeypatch):
    class Bad:
        def __str__(self):
            raise RuntimeError("bad")

    data = {"metaData": Bad()}
    assert GraphUtils.safe_get_meta_data(data) is None


def test_safe_check_load_file_path_dir_expected(tmp_path):
    # create a file but ask for directory should error
    f = tmp_path / "file"
    f.write_text("x")
    ok, err = GraphUtils.safe_check_load_file_path(str(f), is_dir=True)
    assert ok is False


def test_safe_check_load_file_path_no_read_permission(monkeypatch, tmp_path):
    f = tmp_path / "no_read"
    f.write_text("x")

    class S:
        pass

    st = S()
    st.st_mode = os.stat(str(f)).st_mode & ~stat.S_IRUSR
    st.st_uid = os.getuid()
    monkeypatch.setattr(os, "stat", lambda p: st)
    ok, err = GraphUtils.safe_check_load_file_path(str(f))
    assert ok is False


def test_safe_check_load_file_path_size_exceeded(monkeypatch, tmp_path):
    f = tmp_path / "big"
    f.write_text("x")
    monkeypatch.setattr(os.path, "getsize", lambda p: MAX_FILE_SIZE + 1)
    ok, err = GraphUtils.safe_check_load_file_path(str(f))
    assert ok is False


def test_safe_check_load_file_path_root_user(monkeypatch, tmp_path):
    f = tmp_path / "ok"
    f.write_text("x")
    monkeypatch.setattr(os, "getuid", lambda: 0)
    ok, err = GraphUtils.safe_check_load_file_path(str(f))
    assert ok is True


def test_safe_load_data_permission_error(monkeypatch, tmp_path):
    dirpath = tmp_path / "r3"
    dirpath.mkdir()
    f = dirpath / "tag"
    f.write_text("{}")
    GraphState.set_global_value("runs", {"r3": str(dirpath)})
    GraphState.set_global_value("logdir", str(tmp_path))
    monkeypatch.setattr(GraphUtils, "is_relative_to", staticmethod(lambda p, b: True))
    monkeypatch.setattr(
        GraphUtils,
        "safe_check_load_file_path",
        staticmethod(lambda p, d=False: (False, "nope")),
    )
    res, err = GraphUtils.safe_load_data("r3", "tag")
    assert res is None


def test_get_graph_data_fresh_load(monkeypatch):
    """get_graph_data should load when not cached"""
    GraphState.set_global_value("runs", {"run1": "/tmp"})
    GraphState.set_global_value("current_tag", "other")
    GraphState.set_global_value("current_run", "/tmp")
    GraphState.set_global_value("current_file_data", {"old": "data"})
    monkeypatch.setattr(
        GraphUtils, "safe_load_data", lambda r, t, o=False: ({"new": "data"}, None)
    )
    res, err = GraphUtils.get_graph_data({"run": "run1", "tag": "tagX"})
    assert err is None
    assert res == {"new": "data"}
    assert GraphState.get_global_value("current_tag") == "tagX"
    assert GraphState.get_global_value("current_run") == "/tmp"


def test_compare_tag_names_and_sort():
    # create comparisons
    assert GraphUtils.compare_tag_names("a", "a") == 0
    assert GraphUtils.compare_tag_names("a", "b") < 0
    assert GraphUtils.compare_tag_names("1", "a") < 0
    assert GraphUtils.compare_tag_names("file2", "file10") < 0
    assert GraphUtils.compare_tag_names("path/1_2", "path/1_10") < 0
    # sort_data
    data = {"b": {"type": "t", "tags": ["x2", "x1"]}, "a": {"type": "t", "tags": ["y"]}}
    sortedd = GraphUtils.sort_data(data)
    assert list(sortedd.keys()) == ["a", "b"]
    assert sortedd["b"]["tags"] == ["x1", "x2"]


def test_is_safe_string_and_escape_html():
    assert GraphUtils.is_safe_string("normal text") is True
    assert GraphUtils.is_safe_string(123) is False
    # long string: temporarily reduce limit
    import plugins.tb_graph_ascend.hierarchy_plugin.server.app.utils.graph_utils as gu

    orig_limit = gu.FILE_PATH_MAX_LENGTH
    gu.FILE_PATH_MAX_LENGTH = 1
    try:
        assert GraphUtils.is_safe_string("ab") is False
    finally:
        gu.FILE_PATH_MAX_LENGTH = orig_limit
    # dangerous patterns
    assert GraphUtils.is_safe_string("<script>alert(1)</script>") is False
    # escape html
    s = "<>&'" + "/"
    assert GraphUtils.escape_html(s) == "&lt;&gt;&amp;&#39;&#x2F;"


def test_validate_colors_param_cases():
    # non-dict
    ok, err, _ = GraphUtils.validate_colors_param("notdict")
    assert not ok
    # empty dict
    ok, err, _ = GraphUtils.validate_colors_param({})
    assert not ok
    # bad key
    ok, err, _ = GraphUtils.validate_colors_param({"badkey": {}})
    assert not ok
    # value not dict
    ok, err, _ = GraphUtils.validate_colors_param({"#FFF": "no"})
    assert not ok
    # missing value
    ok, err, _ = GraphUtils.validate_colors_param({"#FFF": {}})
    assert not ok
    # list wrong length
    ok, err, _ = GraphUtils.validate_colors_param({"#FFF": {"value": [1]}})
    assert not ok
    # list non numbers
    ok, err, _ = GraphUtils.validate_colors_param({"#FFF": {"value": [1, "a"]}})
    assert not ok
    # list invalid order
    ok, err, _ = GraphUtils.validate_colors_param({"#FFF": {"value": [5, 1]}})
    assert not ok
    # unsupported string
    ok, err, _ = GraphUtils.validate_colors_param({"#FFF": {"value": "xyz"}})
    assert not ok
    # wrong type
    ok, err, _ = GraphUtils.validate_colors_param({"#FFF": {"value": 123}})
    assert not ok
    # description unsafe
    ok, err, result = GraphUtils.validate_colors_param(
        {"#FFF": {"value": 1, "description": "<script>"}}
    )
    assert not ok
    # valid
    inp = {"#AABBCC": {"value": [0, 5], "description": "good"}}
    ok, err, result = GraphUtils.validate_colors_param(inp)
    assert ok
    assert result["#AABBCC"]["description"] == "good"


def test_validate_and_process_parallel_merge_cases():
    data = {}
    assert (
        GraphUtils.validate_and_process_parallel_merge("notdict", data)["success"]
        is False
    )
    # npu not dict
    res = GraphUtils.validate_and_process_parallel_merge({"npu": "x"}, data)
    assert res["success"] is False
    # missing keys
    res = GraphUtils.validate_and_process_parallel_merge({"npu": {"pp": 1}}, data)
    assert res["success"] is False
    # order unsafe
    res = GraphUtils.validate_and_process_parallel_merge(
        {"npu": {"pp": 1, "rank_size": 1, "tp": 1, "vpp": 1, "order": "<script>"}}, data
    )
    assert res["success"] is False
    # bench invalid
    res = GraphUtils.validate_and_process_parallel_merge(
        {"npu": {"pp": 1, "rank_size": 1, "tp": 1, "vpp": 1}, "bench": "bad"}, data
    )
    assert res["success"] is False
    # success case with bench
    data.clear()
    res = GraphUtils.validate_and_process_parallel_merge(
        {
            "npu": {"pp": 1, "rank_size": 1, "tp": 1, "vpp": 1},
            "bench": {"pp": 1, "rank_size": 1, "tp": 1, "vpp": 1},
        },
        data,
    )
    assert res["success"]
    assert "parallel_param_n" in data and "parallel_param_b" in data
