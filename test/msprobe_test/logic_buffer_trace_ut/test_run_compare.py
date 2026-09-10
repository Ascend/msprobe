import json
from types import SimpleNamespace

import pytest
import torch

from msprobe.logic_buffer_trace import run_compare


def _write_fx(path, names):
    args = ", ".join(names)
    path.write_text(f"def forward(self, {args}):\n    return {names[0]}\n", encoding="utf-8")


def test_compare_tensors_uses_atol():
    actual = torch.tensor([0.0])
    expected = torch.tensor([0.005])
    assert run_compare.compare_tensors(actual, expected, rtol=0.0, atol=0.01)["status"] == "ok"
    assert run_compare.compare_tensors(actual, expected, rtol=0.0, atol=0.001)["status"] == "value_mismatch"


def test_replay_args_match_model_and_signature(tmp_path):
    trace_dir = tmp_path / "trace"
    first_model = tmp_path / "first"
    second_model = tmp_path / "second"
    trace_dir.mkdir()
    first_model.mkdir()
    second_model.mkdir()
    first_fx = first_model / "fx_graph_runnable.py"
    second_fx = second_model / "fx_graph_runnable.py"
    _write_fx(first_fx, ["arg0_1"])
    _write_fx(second_fx, ["arg0_1"])

    first_args = trace_dir / "replay_args_graph_0.pt"
    second_args = trace_dir / "replay_args_graph_1.pt"
    torch.save([torch.tensor([1])], first_args)
    torch.save([torch.tensor([2])], second_args)
    rows = [
        {
            "time": 1,
            "path": str(first_args),
            "arg_count": 1,
            "arg_names": ["arg0_1"],
            "model_dir": str(first_model),
        },
        {
            "time": 2,
            "path": str(second_args),
            "arg_count": 1,
            "arg_names": ["arg0_1"],
            "model_dir": str(second_model),
        },
    ]
    (trace_dir / "replay_args.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    module = SimpleNamespace(__file__=str(first_fx), mod=object())
    args = run_compare.get_args_from_fx_module(module, str(trace_dir), first_fx)
    assert torch.equal(args[0], torch.tensor([1]))


def test_replay_args_reject_wrong_argument_count(tmp_path):
    fx_path = tmp_path / "fx_graph_runnable.py"
    _write_fx(fx_path, ["arg0_1", "arg1_1"])
    torch.save([torch.tensor([1])], tmp_path / "replay_args.pt")
    module = SimpleNamespace(__file__=str(fx_path), mod=object())

    with pytest.raises(ValueError, match="expected 2, got 1"):
        run_compare.get_args_from_fx_module(module, str(tmp_path), fx_path)


def test_transpose_and_permute_are_pure_views():
    assert run_compare._is_pure_view_fx({"fx_target": "torch.ops.aten.permute.default"})
    assert run_compare._is_pure_view_fx({"fx_target": "torch.ops.aten.transpose.int"})
