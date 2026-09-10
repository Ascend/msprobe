import json
from pathlib import Path
from unittest.mock import patch

import torch

from msprobe.logic_buffer_trace import logic_buffer_trace as trace


def test_default_trace_dir_uses_current_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(trace.DIR_ENV, raising=False)
    monkeypatch.delenv(trace.ACTIVE_DIR_ENV, raising=False)
    with patch.object(trace, "_debug_context_path", return_value=None), patch.object(
        trace, "_caller_model_dir", return_value=None
    ):
        assert trace.trace_dir() == tmp_path / "logic_buffer_run"


def test_safe_name_adds_hash_when_truncated():
    prefix = "kernel_" + "x" * 200
    first = trace._safe_name(prefix + "a")
    second = trace._safe_name(prefix + "b")
    assert len(first) <= 120
    assert len(second) <= 120
    assert first != second


def test_save_replay_args_uses_graph_specific_files(tmp_path, monkeypatch):
    monkeypatch.setenv(trace.TRACE_ENV, "1")
    monkeypatch.setenv(trace.DIR_ENV, str(tmp_path))
    with patch.object(trace, "_caller_model_dir", return_value=Path("/models/graph")):
        trace.save_replay_args([torch.tensor([1])], "graph_0_0", ["arg0_1"])
        trace.save_replay_args([torch.tensor([2]), 3], "graph_1_1", ["arg0_1", "arg1_1"])

    rows = [json.loads(line) for line in (tmp_path / "replay_args.jsonl").read_text().splitlines()]
    paths = [Path(row["path"]) for row in rows]
    assert len(set(paths)) == 2
    assert all(path.is_file() for path in paths)
    assert [row["arg_count"] for row in rows] == [1, 2]
    assert not (tmp_path / "replay_args.pt").exists()
