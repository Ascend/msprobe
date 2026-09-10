#!/usr/bin/env python3
"""Replay FX debug repro and output_code.py, then compare traced buffers.

Expected workflow:
  1. apply patch.py
  2. compile with INDUCTOR_LOGIC_BUFFER_TRACE=1
  3. run this script on the generated FX repro/output_code.py
"""
# pylint: disable=duplicate-code

from __future__ import annotations

import argparse
import ast
import csv
import html
import importlib.util
import inspect
import json
import math
import os
import re
import shutil
import sys
import time
import traceback
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

import torch


HERE = Path(__file__).resolve().parent
DEFAULT_TRACE_DIR = Path("logic_buffer_run")
TRUE = {"1", "true", "yes", "on"}
sys.dont_write_bytecode = True


APP_NAME = "logic-buffer-compare"
APP_VERSION = "0.3.0"

_ANSI = {
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "cyan": "\033[36m",
    "reset": "\033[0m",
}

_STATE_STYLE = {
    "INFO": ("cyan", "INFO"),
    "RUN": ("cyan", "RUN"),
    "DONE": ("green", "DONE"),
    "OK": ("green", "OK"),
    "WARN": ("yellow", "WARN"),
    "FAIL": ("red", "FAIL"),
    "SKIP": ("dim", "SKIP"),
}


class Console:
    def __init__(self, *, color: bool, quiet: bool = False) -> None:
        self.color = color
        self.quiet = quiet

    def style(self, value: str, style: str) -> str:
        if not self.color:
            return value
        return f"{_ANSI[style]}{value}{_ANSI['reset']}"

    def header(self) -> None:
        if self.quiet:
            return
        print(f"{self.style(APP_NAME, 'bold')} {self.style(APP_VERSION, 'dim')}")
        print("Replay FX and generated Inductor code, then compare traced buffers.")

    def section(self, title: str) -> None:
        if self.quiet:
            return
        print()
        print(self.style(title, "bold"))

    def item(
        self,
        state: str,
        label: str,
        detail: str = "",
        *,
        stderr: bool = False,
        force: bool = False,
    ) -> None:
        if self.quiet and not force:
            return
        style, token = _STATE_STYLE[state]
        tag = self.style(f"[{token:^5}]", style)
        suffix = f"  {detail}" if detail else ""
        print(f"  {tag} {label}{suffix}", file=sys.stderr if stderr else sys.stdout)

    def key_value(self, key: str, value: Any) -> None:
        if self.quiet:
            return
        print(f"  {key:<17} {value}")

    def step(self, label: str) -> "ConsoleStep":
        return ConsoleStep(self, label)


class ConsoleStep:
    def __init__(self, console: Console, label: str) -> None:
        self.console = console
        self.label = label
        self.started = 0.0

    def __enter__(self) -> "ConsoleStep":
        self.started = time.perf_counter()
        self.console.item("RUN", self.label)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        elapsed = time.perf_counter() - self.started
        if exc is None:
            self.console.item("DONE", self.label, _format_duration(elapsed))
        else:
            self.console.item(
                "FAIL",
                self.label,
                f"{_format_duration(elapsed)}; {exc}",
                stderr=True,
                force=True,
            )
        return False


def _format_duration(seconds: float) -> str:
    if seconds < 1.0:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60.0:
        return f"{seconds:.2f} s"
    minutes, remainder = divmod(seconds, 60.0)
    return f"{int(minutes)}m {remainder:.1f}s"


def _color_enabled(disabled: bool) -> bool:
    return (
        not disabled and "NO_COLOR" not in os.environ and os.environ.get("TERM", "") != "dumb" and sys.stdout.isatty()
    )


def _status_counts(comparisons: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for row in comparisons:
        counts[str(row.get("status", "unknown"))] += 1
    return dict(sorted(counts.items()))


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _call_order(value: Any, fallback: int) -> Tuple[int, Any, int]:
    if value is None:
        return (2, fallback, fallback)
    try:
        return (0, int(value), fallback)
    except (TypeError, ValueError):
        return (1, str(value), fallback)


def _is_triton_kernel(kind: Any, kernel_name: Any) -> bool:
    kind_text = str(kind or "").lower()
    name_text = str(kernel_name or "").lower()
    return "triton" in kind_text or name_text.startswith("triton_") or name_text.startswith("triton.")


def mark_first_triton_value_mismatch(
    comparisons: List[Dict[str, Any]],
    kernels: List[Dict[str, Any]],
) -> Dict[str, Any] | None:
    """Mark and summarize the earliest Triton call with a value mismatch.

    ``call_id`` is treated as execution order.  The original comparison list
    remains the JSON report's top-level object; the selected row is annotated
    in place so downstream readers can find the localization result without a
    report-format migration.
    """
    kernel_by_call_id: Dict[str, Dict[str, Any]] = {}
    for kernel in kernels:
        call_id = kernel.get("call_id")
        if call_id is not None:
            kernel_by_call_id[str(call_id)] = kernel

    candidates: List[Tuple[Tuple[int, Any, int], Dict[str, Any], Dict[str, Any]]] = []
    for index, row in enumerate(comparisons):
        if row.get("status") != "value_mismatch":
            continue

        embedded = row.get("kernel_row") or {}
        call_id = _first_not_none(row.get("call_id"), embedded.get("call_id"))
        metadata = kernel_by_call_id.get(str(call_id), {}) if call_id is not None else {}
        kernel_name = _first_not_none(
            row.get("kernel_name"),
            embedded.get("kernel_name"),
            metadata.get("kernel_name"),
        )
        kind = _first_not_none(
            row.get("kind"),
            embedded.get("kind"),
            metadata.get("kind"),
        )
        if not _is_triton_kernel(kind, kernel_name):
            continue

        candidates.append((_call_order(call_id, index), row, metadata))

    if not candidates:
        return None

    _, selected, metadata = min(candidates, key=lambda item: item[0])
    embedded = selected.get("kernel_row") or {}
    call_id = _first_not_none(selected.get("call_id"), embedded.get("call_id"), metadata.get("call_id"))
    kernel_name = _first_not_none(
        selected.get("kernel_name"),
        embedded.get("kernel_name"),
        metadata.get("kernel_name"),
    )

    selected["first_triton_value_mismatch"] = True
    selected["localization_result"] = "first_triton_value_mismatch"

    return {
        "result": "first_triton_value_mismatch",
        "kernel_name": kernel_name,
        "call_id": call_id,
        "fx_node": selected.get("fx_node"),
        "logical_buffer": selected.get("logical_buffer"),
        "call_arg": _first_not_none(selected.get("call_arg"), embedded.get("call_arg")),
        "bad_element_count": selected.get("bad_element_count"),
        "max_abs_diff": selected.get("max_abs_diff"),
        "mean_rel_abs_diff": selected.get("mean_rel_abs_diff"),
    }


def _require_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"{label} must be a file: {path}")
    return path


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def unique_module_name(prefix: str) -> str:
    return f"{prefix}_{os.getpid()}_{int(time.time() * 1000000)}"


def import_file(path: Path, prefix: str):
    name = unique_module_name(prefix)
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def resolve_fx_replay_path(path: Path, prefer_transformed: bool) -> Path:
    path = path.resolve()
    if not prefer_transformed:
        return path
    if path.name == "fx_graph_transformed_runnable.py":
        return path

    transformed = path.with_name("fx_graph_transformed_runnable.py")
    if transformed.exists():
        return transformed
    return path


def _find_one(candidates: List[Path], label: str) -> Path:
    unique = sorted({path.resolve() for path in candidates}, key=lambda p: (len(p.parts), str(p)))
    if not unique:
        raise FileNotFoundError(f"could not find {label}")
    if len(unique) > 1:
        preview = "\n".join(f"  {path}" for path in unique[:20])
        suffix = "" if len(unique) <= 20 else f"\n  ... {len(unique) - 20} more"
        raise RuntimeError(f"found multiple {label} candidates:\n{preview}{suffix}")
    return unique[0]


def resolve_paths_from_dir(root: Path, prefer_transformed: bool) -> Tuple[Path, Path]:
    root = root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"expected a directory: {root}")

    if prefer_transformed:
        fx_direct = root / "fx_graph_transformed_runnable.py"
        if not fx_direct.exists():
            fx_direct = root / "fx_graph_runnable.py"
    else:
        fx_direct = root / "fx_graph_runnable.py"
    output_direct = root / "output_code.py"
    if fx_direct.exists() and output_direct.exists():
        return fx_direct.resolve(), output_direct.resolve()

    fx_candidates: List[Path] = []
    if prefer_transformed:
        fx_candidates = list(root.rglob("fx_graph_transformed_runnable.py"))
    if not fx_candidates:
        fx_candidates = list(root.rglob("fx_graph_runnable.py"))
    fx_path = _find_one(fx_candidates, "FX runnable")

    output_same_dir = fx_path.with_name("output_code.py")
    if output_same_dir.exists():
        return fx_path, output_same_dir.resolve()

    output_candidates = list(root.rglob("output_code.py"))
    output_path = _find_one(output_candidates, "output_code.py")
    return fx_path, output_path


def clean_runtime_outputs(trace_dir: Path) -> None:
    for filename in ("fx_values.jsonl", "kernel_values.jsonl"):
        path = trace_dir / filename
        if path.exists():
            path.unlink()
    for dirname in ("fx_tensors", "kernel_tensors"):
        path = trace_dir / dirname
        if path.exists():
            shutil.rmtree(path)


VIEW_FX_OPS = (
    "aten.view",
    "aten.reshape",
    "aten.squeeze",
    "aten.unsqueeze",
    "aten.permute",
    "aten.transpose",
    "aten.as_strided",
    "aten.slice.",
    "aten.reshape_as",
    "aten.flatten",
    "aten.unflatten",
    "aten.expand",
    "aten.broadcast_to",
    "aten.contiguous",
)


def _is_pure_view_fx(row: Dict[str, Any]) -> bool:
    target = str(row.get("fx_target") or "")
    return any(v in target for v in VIEW_FX_OPS)


def build_maps(trace_dir: Path) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    live_lowering = read_jsonl(trace_dir / "lowering_live_fx_buffer_map.jsonl")
    lowering = live_lowering or read_jsonl(trace_dir / "lowering_fx_buffer_map.jsonl")
    kernels = read_jsonl(trace_dir / "kernel_arg_map.jsonl")

    buffer_to_fx: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in lowering:
        buf = row.get("buffer")
        fx = row.get("fx_node")
        if buf and fx:
            buffer_to_fx[str(buf)].append(row)
    # A buffer is often recorded under BOTH its computation producer and a pure
    # reshape/view of it (shared storage, e.g. gelu -> reshape of the same
    # buffer).  A view row only re-labels the same tensor, so when a buffer has
    # a real computation producer keep it and drop the view duplicate.
    for buf, rows in list(buffer_to_fx.items()):
        if len(rows) > 1:
            non_views = [r for r in rows if not _is_pure_view_fx(r)]
            if non_views:
                buffer_to_fx[buf] = non_views
    return buffer_to_fx, kernels


def post_call_logical_names(row: Dict[str, Any]) -> List[str]:
    output_logicals = row.get("output_logical_names")
    if output_logicals:
        raw = output_logicals
    else:
        logical_names = list(row.get("logical_names", []))
        role = row.get("role")
        if role == "input_output" and logical_names:
            raw = logical_names[-1:]
        elif role == "output":
            raw = logical_names
        else:
            raw = []

    result: List[str] = []
    seen = set()
    for item in raw:
        name = str(item)
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result


def wanted_fx_nodes(buffer_to_fx: Dict[str, List[Dict[str, Any]]], kernels: List[Dict[str, Any]]) -> List[str]:
    wanted = set()
    for kernel in kernels:
        for entry in kernel.get("dump_entries", []):
            for logical in post_call_logical_names(entry):
                for fx_row in buffer_to_fx.get(str(logical), []):
                    wanted.add(str(fx_row["fx_node"]))
    return sorted(wanted)


def forward_placeholder_names(text: str) -> List[str]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "forward":
            continue
        args = [arg.arg for arg in node.args.args]
        if args and args[0] == "self":
            args = args[1:]
        return args
    return []


def instrument_fx_source(fx_path: Path, dst_path: Path, nodes: Iterable[str]) -> None:
    wanted = set(nodes)
    text = fx_path.read_text(encoding="utf-8")
    placeholders = forward_placeholder_names(text)
    wanted_placeholders = [name for name in placeholders if name in wanted]

    import_line = "from torch._inductor.logic_buffer_trace import dump_fx_tensor as _logic_buffer_dump_fx_tensor\n"
    if import_line not in text:
        match = re.search(r"(^import torch\s*$)", text, flags=re.MULTILINE)
        if match:
            insert_at = match.end() + 1
            text = text[:insert_at] + import_line + text[insert_at:]
        else:
            text = import_line + text

    out_lines: List[str] = []
    assign_re = re.compile(r"^(\s*)([A-Za-z_][A-Za-z0-9_]*)\s*=")
    forward_re = re.compile(r"^(\s*)def\s+forward\(")
    inserted_placeholders = False
    for line in text.splitlines():
        out_lines.append(line)
        forward_match = forward_re.match(line)
        if not inserted_placeholders and wanted_placeholders and forward_match and line.rstrip().endswith(":"):
            indent = forward_match.group(1) + "    "
            for name in wanted_placeholders:
                out_lines.append(f"{indent}_logic_buffer_dump_fx_tensor({name!r}, {name})")
            inserted_placeholders = True
        match = assign_re.match(line)
        if not match:
            continue
        indent, name = match.group(1), match.group(2)
        if name in wanted:
            out_lines.append(f"{indent}_logic_buffer_dump_fx_tensor({name!r}, {name})")

    dst_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def _expected_arg_names(fx_module: Any, fx_path: Path | None = None) -> List[str]:
    source_path = fx_path or Path(getattr(fx_module, "__file__", ""))
    if source_path.is_file():
        names = forward_placeholder_names(source_path.read_text(encoding="utf-8"))
        if names:
            return names
    mod = getattr(fx_module, "mod", None)
    if mod is None:
        return []
    try:
        return list(inspect.signature(mod.forward).parameters)
    except (TypeError, ValueError):
        return []


def _load_replay_args(path: Path, expected_count: int) -> List[Any]:
    args = list(torch.load(path, weights_only=True))
    if len(args) != expected_count:
        raise ValueError(f"replay input count mismatch for {path}: expected {expected_count}, got {len(args)}")
    return args


def _replay_args_from_trace(save_dir: Path, expected_names: List[str], model_dir: Path | None) -> List[Any] | None:
    expected_count = len(expected_names)
    rows = read_jsonl(save_dir / "replay_args.jsonl")
    candidates: List[Tuple[int, Dict[str, Any], Path]] = []
    for row in rows:
        raw_path = row.get("path")
        if not raw_path:
            continue
        try:
            arg_count = int(row.get("arg_count", -1))
        except (TypeError, ValueError):
            continue
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = save_dir / path
        if not path.is_file() or arg_count != expected_count:
            continue
        row_names = [str(name) for name in (row.get("arg_names") or [])]
        row_model = row.get("model_dir")
        same_names = bool(expected_names and row_names == expected_names)
        same_model = bool(
            model_dir is not None and row_model and Path(str(row_model)).expanduser().resolve() == model_dir
        )
        if same_model and same_names:
            rank = 3
        elif same_model:
            rank = 2
        elif same_names:
            rank = 1
        else:
            continue
        candidates.append((rank, row, path))

    if candidates:
        candidates.sort(key=lambda item: (item[0], float(item[1].get("time", 0))), reverse=True)
        return _load_replay_args(candidates[0][2], expected_count)

    legacy_path = save_dir / "replay_args.pt"
    if legacy_path.is_file():
        return _load_replay_args(legacy_path, expected_count)

    if rows:
        available = set()
        for row in rows:
            try:
                available.add(int(row.get("arg_count", -1)))
            except (TypeError, ValueError):
                continue
        raise ValueError(
            "no replay inputs match the current FX graph "
            f"(expected names={expected_names}, count={expected_count}; available counts={sorted(available)})"
        )
    return None


def get_args_from_fx_module(fx_module: Any, save_dir: str | None, fx_path: Path | None = None):
    expected_names = _expected_arg_names(fx_module, fx_path)
    if not expected_names:
        raise RuntimeError("cannot determine the current FX graph input signature")
    if save_dir:
        source_path = fx_path or Path(getattr(fx_module, "__file__", ""))
        model_dir = source_path.expanduser().resolve().parent if source_path.is_file() else None
        replay_args = _replay_args_from_trace(Path(save_dir), expected_names, model_dir)
        if replay_args is not None:
            return replay_args

    if not hasattr(fx_module, "mod"):
        raise RuntimeError("fx_graph_runnable.py does not define global 'mod'")
    if not hasattr(fx_module, "load_args"):
        raise RuntimeError("fx_graph_runnable.py does not define load_args(reader)")
    from torch._dynamo.debug_utils import InputReader

    reader = InputReader(save_dir)
    fx_module.load_args(reader)
    args = list(reader.args)
    if len(args) != len(expected_names):
        raise ValueError(
            f"FX load_args produced the wrong input count: expected {len(expected_names)}, got {len(args)}"
        )
    return args


def clone_arg(arg: Any) -> Any:
    if isinstance(arg, torch.Tensor):
        with torch.no_grad():
            out = arg.detach().clone(memory_format=torch.preserve_format)
            out.requires_grad_(arg.requires_grad)
            return out
    return arg


def clone_args(args: List[Any]) -> List[Any]:
    return [clone_arg(arg) for arg in args]


def run_fx(fx_path: Path, trace_dir: Path, nodes: List[str], base_args: List[Any], save_dir: str | None):
    instrumented = trace_dir / "instrumented_fx_graph_runnable.py"
    instrument_fx_source(fx_path, instrumented, nodes)
    module = import_file(instrumented, "logic_buffer_fx")
    args = clone_args(base_args)
    with torch.no_grad():
        result = module.mod(*args)
    return result


def run_output(output_path: Path, base_args: List[Any]):
    module = import_file(output_path, "logic_buffer_output")
    if not hasattr(module, "call"):
        raise RuntimeError("output_code.py does not define call(args)")
    args = clone_args(base_args)
    with torch.no_grad():
        result = module.call(args)
    return result


def load_tensor(path: str) -> torch.Tensor:
    return torch.load(path, map_location="cpu", weights_only=True)


def _json_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return str(value)


def _flat_index_to_index(flat_index: int, shape: List[int]) -> List[int]:
    if not shape:
        return []
    coords: List[int] = []
    remaining = int(flat_index)
    for dim in reversed(shape):
        if dim == 0:
            coords.append(0)
            continue
        coords.append(remaining % int(dim))
        remaining //= int(dim)
    return list(reversed(coords))


def _top_error_elements(
    fx_tensor: torch.Tensor,
    kernel_tensor: torch.Tensor,
    abs_diff: torch.Tensor,
    rel_diff: torch.Tensor,
    *,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    if abs_diff.numel() == 0:
        return []
    flat_abs = abs_diff.reshape(-1)
    flat_rel = rel_diff.reshape(-1)
    flat_fx = fx_tensor.reshape(-1)
    flat_kernel = kernel_tensor.reshape(-1)
    k = min(limit, flat_abs.numel())
    values, indices = torch.topk(flat_abs, k=k, largest=True)
    shape = list(fx_tensor.shape)
    rows: List[Dict[str, Any]] = []
    for rank, (value, index_tensor) in enumerate(zip(values, indices)):
        flat_index = int(index_tensor.item())
        rows.append(
            {
                "rank": rank + 1,
                "flat_index": flat_index,
                "index": _flat_index_to_index(flat_index, shape),
                "fx_value": _json_scalar(flat_fx[flat_index]),
                "kernel_value": _json_scalar(flat_kernel[flat_index]),
                "abs_diff": _json_scalar(value),
                "rel_diff": _json_scalar(flat_rel[flat_index]),
            }
        )
    return rows


def compare_tensors(a: torch.Tensor, b: torch.Tensor, rtol: float, atol: float) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "fx_saved_shape": list(a.shape),
        "kernel_saved_shape": list(b.shape),
        "fx_dtype": str(a.dtype),
        "kernel_dtype": str(b.dtype),
    }
    if list(a.shape) != list(b.shape):
        result["status"] = "shape_mismatch"
        return result

    if a.dtype != b.dtype:
        result["dtype_mismatch"] = True

    aa_for_bad = a.to(torch.float32)
    bb_for_bad = b.to(torch.float32)
    bad_diff = (aa_for_bad - bb_for_bad).abs()
    bad_denom = torch.maximum(
        torch.maximum(aa_for_bad.abs(), bb_for_bad.abs()),
        torch.tensor(1e-12),
    )
    bad_rel = bad_diff / bad_denom
    floating = a.is_floating_point() or b.is_floating_point()
    close = torch.isclose(aa_for_bad, bb_for_bad, rtol=rtol, atol=atol) if floating else a == b
    result["bad_element_count"] = int((~close).sum().item())
    result["bad_element_rel_threshold"] = float(rtol)
    result["bad_element_abs_threshold"] = float(atol)
    result["top_error_elements"] = _top_error_elements(a, b, bad_diff, bad_rel)

    if floating:
        aa = a.to(torch.float32)
        bb = b.to(torch.float32)
        diff = (aa - bb).abs()
        denom = torch.maximum(torch.maximum(aa.abs(), bb.abs()), torch.tensor(1e-12))
        rel = diff / denom
        result["max_abs_diff"] = float(diff.max().item()) if diff.numel() else 0.0
        result["mean_abs_diff"] = float(diff.mean().item()) if diff.numel() else 0.0
        result["max_rel_abs_diff"] = float(rel.max().item()) if rel.numel() else 0.0
        result["mean_rel_abs_diff"] = float(rel.mean().item()) if rel.numel() else 0.0
        result["compare_metric"] = "torch.isclose"
        result["compare_threshold_rtol"] = float(rtol)
        result["compare_threshold_atol"] = float(atol)
        ok = bool(close.all().item())
    else:
        eq = a == b
        result["max_abs_diff"] = int((a.to(torch.int64) - b.to(torch.int64)).abs().max().item()) if a.numel() else 0
        result["mean_abs_diff"] = 0.0 if bool(eq.all().item()) else None
        result["compare_metric"] = "exact"
        result["compare_threshold"] = 0.0
        ok = bool(eq.all().item())

    result["status"] = "ok" if ok else "value_mismatch"
    return result


def latest_by_key(rows: List[Dict[str, Any]], key: str) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        value = row.get(key)
        if value is not None:
            result[str(value)] = row
    return result


def build_comparisons(
    trace_dir: Path, buffer_to_fx: Dict[str, List[Dict[str, Any]]], rtol: float, atol: float
) -> List[Dict[str, Any]]:
    fx_values = [row for row in read_jsonl(trace_dir / "fx_values.jsonl") if row.get("kind") == "fx_tensor"]
    kernel_values = [row for row in read_jsonl(trace_dir / "kernel_values.jsonl") if row.get("kind") == "kernel_tensor"]
    fx_by_node = latest_by_key(fx_values, "fx_node")

    comparisons: List[Dict[str, Any]] = []
    for kernel_row in kernel_values:
        kernel_path = kernel_row.get("path")
        if not kernel_path:
            continue
        try:
            kernel_tensor = load_tensor(str(kernel_path))
        except Exception as exc:
            comparisons.append(
                {
                    "status": "kernel_load_error",
                    "kernel_row": kernel_row,
                    "error": repr(exc),
                }
            )
            continue

        output_logicals = post_call_logical_names(kernel_row)
        if not output_logicals:
            comparisons.append(
                {
                    "status": "no_post_call_logical_output",
                    "kernel_name": kernel_row.get("kernel_name"),
                    "call_id": kernel_row.get("call_id"),
                    "call_arg": kernel_row.get("call_arg"),
                    "role": kernel_row.get("role"),
                    "all_logical_names": kernel_row.get("logical_names"),
                    "input_logical_names": kernel_row.get("input_logical_names"),
                    "output_logical_names": output_logicals,
                    "kernel_row": kernel_row,
                }
            )
            continue

        for logical in output_logicals:
            fx_rows = buffer_to_fx.get(str(logical), [])
            if not fx_rows:
                comparisons.append(
                    {
                        "status": "no_fx_for_logical_buffer",
                        "logical_buffer": logical,
                        "kernel_name": kernel_row.get("kernel_name"),
                        "call_id": kernel_row.get("call_id"),
                        "call_arg": kernel_row.get("call_arg"),
                        "role": kernel_row.get("role"),
                        "all_logical_names": kernel_row.get("logical_names"),
                        "input_logical_names": kernel_row.get("input_logical_names"),
                        "output_logical_names": output_logicals,
                        "kernel_path": kernel_path,
                        "kernel_row": kernel_row,
                    }
                )
                continue
            for fx_row in fx_rows:
                fx_node = str(fx_row["fx_node"])
                fx_value = fx_by_node.get(fx_node)
                if not fx_value or not fx_value.get("path"):
                    comparisons.append(
                        {
                            "status": "missing_fx_value",
                            "logical_buffer": logical,
                            "fx_node": fx_node,
                            "kernel_name": kernel_row.get("kernel_name"),
                            "call_id": kernel_row.get("call_id"),
                            "call_arg": kernel_row.get("call_arg"),
                            "role": kernel_row.get("role"),
                            "all_logical_names": kernel_row.get("logical_names"),
                            "input_logical_names": kernel_row.get("input_logical_names"),
                            "output_logical_names": output_logicals,
                            "kernel_path": kernel_path,
                            "kernel_row": kernel_row,
                        }
                    )
                    continue
                try:
                    fx_tensor = load_tensor(str(fx_value["path"]))
                    cmp = compare_tensors(fx_tensor, kernel_tensor, rtol, atol)
                except Exception as exc:
                    cmp = {"status": "compare_error", "error": repr(exc)}
                comparisons.append(
                    {
                        "logical_buffer": logical,
                        "fx_node": fx_node,
                        "kernel_name": kernel_row.get("kernel_name"),
                        "call_id": kernel_row.get("call_id"),
                        "call_arg": kernel_row.get("call_arg"),
                        "role": kernel_row.get("role"),
                        "all_logical_names": kernel_row.get("logical_names"),
                        "input_logical_names": kernel_row.get("input_logical_names"),
                        "output_logical_names": output_logicals,
                        "fx_path": fx_value.get("path"),
                        "kernel_path": kernel_path,
                        "fx_meta": fx_row,
                        **cmp,
                    }
                )
    mark_missing_fx_covered(comparisons)
    return comparisons


def mark_missing_fx_covered(comparisons: List[Dict[str, Any]]) -> None:
    comparable_by_logical: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in comparisons:
        logical = row.get("logical_buffer")
        if (
            logical is not None
            and row.get("status") != "missing_fx_value"
            and row.get("fx_node")
            and row.get("fx_path")
        ):
            comparable_by_logical[str(logical)].append(row)

    for row in comparisons:
        if row.get("status") != "missing_fx_value":
            continue
        logical = row.get("logical_buffer")
        if logical is None:
            continue
        alternatives = comparable_by_logical.get(str(logical), [])
        if not alternatives:
            continue
        row["status"] = "ok"
        row["covered_missing_fx_value"] = True
        row["covered_by_fx_nodes"] = [alt.get("fx_node") for alt in alternatives if alt.get("fx_node")]
        row["covered_by_statuses"] = [alt.get("status") for alt in alternatives if alt.get("status")]
        row["covered_by_reason"] = "same logical buffer has at least one tensor-valued FX comparison"


def write_text_report(
    path: Path,
    comparisons: List[Dict[str, Any]],
    localization: Dict[str, Any] | None = None,
) -> None:
    counts = defaultdict(int)
    for row in comparisons:
        counts[row.get("status", "unknown")] += 1

    lines = ["Logic Buffer Compare Report", ""]
    lines.append("Error localization:")
    if localization is None:
        lines.append("  first_triton_value_mismatch: not_found")
    else:
        lines.append("  result: first_triton_value_mismatch")
        lines.append(f"  kernel_name: {localization.get('kernel_name')}")
        lines.append(f"  call_id: {localization.get('call_id')}")
        lines.append(f"  fx_node: {localization.get('fx_node')}")
        lines.append(f"  logical_buffer: {localization.get('logical_buffer')}")
        lines.append(f"  call_arg: {localization.get('call_arg')}")
        lines.append(f"  bad_element_count: {localization.get('bad_element_count')}")
        lines.append(f"  max_abs_diff: {localization.get('max_abs_diff')}")
        lines.append(f"  mean_rel_abs_diff: {localization.get('mean_rel_abs_diff')}")
    lines.append("")
    lines.append("Status counts:")
    for key in sorted(counts):
        lines.append(f"  {key}: {counts[key]}")
    lines.append("")
    lines.append("Comparisons:")
    for row in comparisons:
        kernel_row = row.get("kernel_row") or {}
        kernel_name = row.get("kernel_name") or kernel_row.get("kernel_name")
        call_arg = row.get("call_arg") or kernel_row.get("call_arg")
        lines.append(
            f"  {row.get('status')} logical={row.get('logical_buffer')} "
            f"fx={row.get('fx_node')} kernel={kernel_name} "
            f"call_arg={call_arg} bad_elems={row.get('bad_element_count')} "
            f"max_abs={row.get('max_abs_diff')}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def write_csv_report(path: Path, comparisons: List[Dict[str, Any]]) -> None:
    fields = [
        "status",
        "logical_buffer",
        "fx_node",
        "kernel_name",
        "call_id",
        "call_arg",
        "role",
        "fx_saved_shape",
        "kernel_saved_shape",
        "fx_dtype",
        "kernel_dtype",
        "dtype_mismatch",
        "compare_metric",
        "compare_threshold",
        "compare_threshold_rtol",
        "compare_threshold_atol",
        "bad_element_rel_threshold",
        "bad_element_abs_threshold",
        "bad_element_count",
        "mean_abs_diff",
        "max_abs_diff",
        "mean_rel_abs_diff",
        "max_rel_abs_diff",
        "all_logical_names",
        "input_logical_names",
        "output_logical_names",
        "fx_path",
        "kernel_path",
        "error",
        "covered_missing_fx_value",
        "covered_by_fx_nodes",
        "covered_by_statuses",
        "covered_by_reason",
        "first_triton_value_mismatch",
        "localization_result",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in comparisons:
            kernel_row = row.get("kernel_row") or {}
            out = {field: row.get(field) for field in fields}
            for field in (
                "kernel_name",
                "call_id",
                "call_arg",
                "role",
                "all_logical_names",
                "input_logical_names",
                "output_logical_names",
                "kernel_path",
            ):
                if out.get(field) in (None, "") and field in kernel_row:
                    out[field] = kernel_row.get(field)
            writer.writerow({field: _csv_value(out.get(field)) for field in fields})


def _source_context(path: Path, needle: str, radius: int = 2) -> List[Dict[str, Any]]:
    if not path.exists() or not needle:
        return []

    lines = path.read_text(encoding="utf-8").splitlines()
    contexts: List[Dict[str, Any]] = []
    for index, line in enumerate(lines):
        if needle not in line:
            continue
        start = max(0, index - radius)
        end = min(len(lines), index + radius + 1)
        contexts.append(
            {
                "line": index + 1,
                "text": line.strip(),
                "context": [{"line": i + 1, "text": lines[i]} for i in range(start, end)],
            }
        )
    return contexts


def _compact_fx_value(row: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if row is None:
        return None
    out = dict(row)
    if "repr" in out and isinstance(out["repr"], str) and len(out["repr"]) > 1000:
        out["repr"] = out["repr"][:1000] + "...<truncated>"
    return out


def write_missing_fx_diagnostics(
    path: Path,
    trace_dir: Path,
    fx_path: Path,
    comparisons: List[Dict[str, Any]],
) -> int:
    fx_values = [row for row in read_jsonl(trace_dir / "fx_values.jsonl") if row.get("kind") == "fx_tensor"]
    fx_by_node = latest_by_key(fx_values, "fx_node")
    by_logical: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in comparisons:
        logical = row.get("logical_buffer")
        if logical is not None:
            by_logical[str(logical)].append(row)

    missing = [row for row in comparisons if row.get("status") == "missing_fx_value"]
    diagnostics: List[Dict[str, Any]] = []
    instrumented = trace_dir / "instrumented_fx_graph_runnable.py"

    for row in missing:
        fx_node = str(row.get("fx_node", ""))
        logical = str(row.get("logical_buffer", ""))
        fx_value = fx_by_node.get(fx_node)
        if fx_value is None:
            reason = "fx_node_not_dumped"
        elif not fx_value.get("is_tensor"):
            reason = f"fx_node_dumped_non_tensor:{fx_value.get('type')}"
        elif not fx_value.get("path"):
            reason = "fx_node_dumped_without_path"
        else:
            reason = "unknown_missing_fx_path"

        alternatives = []
        for other in by_logical.get(logical, []):
            if other is row or other.get("status") == "missing_fx_value":
                continue
            alternatives.append(
                {
                    "status": other.get("status"),
                    "fx_node": other.get("fx_node"),
                    "fx_path": other.get("fx_path"),
                    "mean_rel_abs_diff": other.get("mean_rel_abs_diff"),
                    "max_abs_diff": other.get("max_abs_diff"),
                }
            )

        diagnostics.append(
            {
                "reason": reason,
                "logical_buffer": logical,
                "fx_node": fx_node,
                "kernel_name": row.get("kernel_name"),
                "call_id": row.get("call_id"),
                "call_arg": row.get("call_arg"),
                "fx_value_row": _compact_fx_value(fx_value),
                "same_logical_alternatives": alternatives,
                "instrumented_source_context": _source_context(instrumented, fx_node),
                "fx_source_context": _source_context(fx_path, fx_node),
            }
        )

    write_json(path, diagnostics)
    return len(diagnostics)


def _shorten(value: str, limit: int = 80) -> str:
    value = str(value)
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 1)] + "…"


def _safe_dom_id(prefix: str, value: Any) -> str:
    raw = re.sub(r"[^A-Za-z0-9_.:-]", "_", str(value))
    return f"{prefix}-{raw}"


def _ast_unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return type(node).__name__


def _fx_op_label(value: ast.AST) -> str:
    if isinstance(value, ast.Call):
        return _shorten(_ast_unparse(value.func), 72)
    if isinstance(value, ast.Subscript):
        return "getitem"
    if isinstance(value, ast.BinOp):
        return type(value.op).__name__
    if isinstance(value, ast.UnaryOp):
        return type(value.op).__name__
    if isinstance(value, ast.Tuple):
        return "tuple"
    if isinstance(value, ast.List):
        return "list"
    return _shorten(_ast_unparse(value), 72)


def _names_loaded(node: ast.AST) -> Set[str]:
    names: Set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
            names.add(child.id)
    return names


def _find_forward_fn(tree: ast.AST) -> ast.FunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "forward":
            return node
    return None


def _find_named_fn(tree: ast.AST, name: str) -> ast.FunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _full_attr(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _full_attr(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return _ast_unparse(node)


def _is_buffer_like_name(name: str) -> bool:
    return bool(re.match(r"^(buf\d+|arg\d+_\d+)$", name))


def _tensor_names_in_expr(node: ast.AST) -> List[str]:
    return sorted(name for name in _names_loaded(node) if _is_buffer_like_name(name))


def _iter_runtime_statements(stmts: List[ast.stmt]) -> Iterable[ast.stmt]:
    for stmt in stmts:
        if isinstance(stmt, (ast.With, ast.AsyncWith, ast.If, ast.Try)):
            bodies: List[List[ast.stmt]] = []
            if hasattr(stmt, "body"):
                bodies.append(list(getattr(stmt, "body")))
            if isinstance(stmt, ast.If):
                bodies.append(list(stmt.orelse))
            if isinstance(stmt, ast.Try):
                bodies.extend([list(handler.body) for handler in stmt.handlers])
                bodies.append(list(stmt.orelse))
                bodies.append(list(stmt.finalbody))
            for body in bodies:
                yield from _iter_runtime_statements(body)
            continue
        yield stmt


def parse_output_kernel_calls(output_path: Path) -> List[Dict[str, Any]]:
    if not output_path.exists() or not output_path.is_file():
        return []
    try:
        tree = ast.parse(output_path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []

    call_fn = _find_named_fn(tree, "call")
    if call_fn is None:
        return []

    calls: List[Dict[str, Any]] = []
    for stmt in _iter_runtime_statements(call_fn.body):
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            if isinstance(call.func, ast.Attribute) and call.func.attr == "run":
                kernel_name = _full_attr(call.func.value)
                calls.append(
                    {
                        "kernel_name": kernel_name,
                        "kind": "triton",
                        "inputs": _dedupe_keep_order(name for arg in call.args for name in _tensor_names_in_expr(arg)),
                        "outputs": [],
                        "line": getattr(stmt, "lineno", None),
                    }
                )
                continue

            kernel_name = _full_attr(call.func)
            if kernel_name.startswith("extern_kernels."):
                outputs = _dedupe_keep_order(
                    name for kw in call.keywords if kw.arg == "out" for name in _tensor_names_in_expr(kw.value)
                )
                inputs = _dedupe_keep_order(name for arg in call.args for name in _tensor_names_in_expr(arg))
                calls.append(
                    {
                        "kernel_name": kernel_name,
                        "kind": "extern",
                        "inputs": inputs,
                        "outputs": outputs,
                        "line": getattr(stmt, "lineno", None),
                    }
                )
            continue

        if not isinstance(stmt, ast.Assign) or not isinstance(stmt.value, ast.Call):
            continue
        call = stmt.value
        kernel_name = _full_attr(call.func)
        if not (
            kernel_name.startswith("extern_kernels.")
            or kernel_name.startswith("torch.ops.aten.")
            or kernel_name.startswith("torch.ops.npu.")
            or kernel_name.startswith("aten.")
        ):
            continue

        outputs = [
            target.id for target in stmt.targets if isinstance(target, ast.Name) and _is_buffer_like_name(target.id)
        ]
        inputs = _dedupe_keep_order(
            name for arg in list(call.args) + [kw.value for kw in call.keywords] for name in _tensor_names_in_expr(arg)
        )
        calls.append(
            {
                "kernel_name": kernel_name,
                "kind": "extern",
                "inputs": inputs,
                "outputs": outputs,
                "line": getattr(stmt, "lineno", None),
            }
        )

    return calls


def _kernel_names_compatible(recorded: str, parsed: str) -> bool:
    if recorded == parsed:
        return True
    if parsed.endswith(recorded):
        return True
    if recorded.startswith("extern_kernels.") and parsed.startswith("extern_kernels."):
        return recorded.split(".", 1)[1] == parsed.split(".", 1)[1]
    return False


def enrich_kernels_from_output_code(
    kernels: List[Dict[str, Any]],
    output_path: Path,
) -> List[Dict[str, Any]]:
    parsed_calls = parse_output_kernel_calls(output_path)
    if not parsed_calls:
        return kernels

    rows = [dict(row) for row in sorted(kernels, key=lambda row: int(row.get("call_id", 0)))]
    parsed_index = 0
    for row in rows:
        recorded_name = str(row.get("kernel_name", ""))
        matched = None
        for i in range(parsed_index, min(len(parsed_calls), parsed_index + 8)):
            parsed = parsed_calls[i]
            if _kernel_names_compatible(recorded_name, str(parsed.get("kernel_name", ""))):
                matched = parsed
                parsed_index = i + 1
                break
        if matched is None and parsed_index < len(parsed_calls):
            parsed = parsed_calls[parsed_index]
            if str(row.get("kind", "")).startswith("triton") == (parsed.get("kind") == "triton"):
                matched = parsed
                parsed_index += 1
        if matched is None:
            continue

        row["_parsed_inputs"] = list(matched.get("inputs") or [])
        row["_parsed_outputs"] = list(matched.get("outputs") or [])
        row["_source_line"] = matched.get("line")
    return rows


def _symint_arg_names(fx_path: Path) -> Set[str]:
    """Names of forward args that are dynamic-shape scalars (``reader.symint``).

    Dynamic models pass shape values (e.g. batch=224) as scalar args declared in
    load_args as ``reader.symint(N)  # arg2_1``.  These are not tensors and have
    no dataflow consumers, so they are filtered out of the FX graph.
    """
    names: Set[str] = set()
    try:
        for line in fx_path.read_text(encoding="utf-8").splitlines():
            if "reader.symint" not in line:
                continue
            m = re.search(r"#\s*(arg\d+_1)", line)
            if m:
                names.add(m.group(1))
    except Exception:  # nosec B110
        pass
    return names


def parse_fx_graph(fx_path: Path) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]]]:
    text = fx_path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    forward = _find_forward_fn(tree)
    if forward is None:
        return [], []

    nodes: List[Dict[str, Any]] = []
    edges: List[Tuple[str, str]] = []
    known: Set[str] = set()
    seen_edges: Set[Tuple[str, str]] = set()
    symint_args = _symint_arg_names(fx_path)

    args = [arg.arg for arg in forward.args.args]
    if args and args[0] == "self":
        args = args[1:]
    for index, name in enumerate(args):
        if name in symint_args:
            continue
        nodes.append(
            {
                "id": name,
                "label": name,
                "kind": "placeholder",
                "op": "placeholder",
                "line": forward.lineno,
                "order": index,
            }
        )
        known.add(name)

    order = len(nodes)
    for stmt in forward.body:
        if not isinstance(stmt, ast.Assign):
            continue
        if isinstance(stmt.value, ast.Constant) and stmt.value.value is None:
            continue

        targets = [target for target in stmt.targets if isinstance(target, ast.Name)]
        if not targets:
            continue
        target = targets[0].id
        if target.startswith("_logic_buffer_"):
            continue

        node = {
            "id": target,
            "label": target,
            "kind": "op",
            "op": _fx_op_label(stmt.value),
            "line": getattr(stmt, "lineno", None),
            "order": order,
            "expr": _shorten(_ast_unparse(stmt.value), 180),
        }
        order += 1
        nodes.append(node)

        for source in sorted(_names_loaded(stmt.value)):
            if source == target or source not in known:
                continue
            edge = (source, target)
            if edge not in seen_edges:
                seen_edges.add(edge)
                edges.append(edge)
        known.add(target)

    return nodes, edges


def _entry_input_logicals(entry: Dict[str, Any]) -> List[str]:
    values = list(entry.get("input_logical_names") or [])
    if not values and entry.get("role") in ("input", "input_output"):
        values = list(entry.get("logical_names") or [])
    return [str(item) for item in values]


def _entry_output_logicals(entry: Dict[str, Any]) -> List[str]:
    values = list(entry.get("output_logical_names") or [])
    if entry.get("role") in ("output", "input_output"):
        # For in-place (input_output) kernels, ``logical_names`` also lists the
        # input buffer that is modified in place.  That input is produced by an
        # upstream kernel (its own FX node), so it must not be treated as an
        # output of this kernel -- otherwise two kernels would both map to the
        # same FX node and the graph would show duplicate mappings.
        inputs = set(entry.get("input_logical_names") or [])
        for item in entry.get("logical_names") or []:
            if item in inputs:
                continue
            if item not in values:
                values.append(item)
    return [str(item) for item in values]


def _dedupe_keep_order(values: Iterable[str]) -> List[str]:
    result: List[str] = []
    seen: Set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _parse_alias_map(output_path: Path | None) -> Dict[str, str]:
    """Parse output_code.py for buffer aliases created by codegen.

    ``buf80 = reinterpret_tensor(buf88, ...)`` (a slice of a stack/concat output)
    and ``buf4 = buf3[0]`` (getitem) both make ``buf80``/``buf4`` an alias of a
    base buffer.  Kernel producers write the alias while consumers read the base,
    so without resolving them the call graph is full of dangling nodes.
    """
    aliases: Dict[str, str] = {}
    if output_path is None or not output_path.exists():
        return aliases
    try:
        for line in output_path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\s*(\w+)\s*=\s*reinterpret_tensor\((\w+)", line)
            if m:
                aliases[m.group(1)] = m.group(2)
                continue
            m = re.match(r"\s*(\w+)\s*=\s*(\w+)\[(\d+)\]", line)
            if m:
                aliases[m.group(1)] = m.group(2)
    except Exception:  # nosec B110
        pass
    return aliases


def _parse_alias_slices(output_path: Path | None) -> Dict[str, Dict[str, int]]:
    """Parse output_code.py for the SLICE each reinterpret_tensor alias covers.

    A stack/concat fusion writes its output buffer slice-by-slice::

        buf29 = empty_strided((1, 512), (512, 1), device='npu', ...)
        buf21 = reinterpret_tensor(buf29, (1, 64), (512, 1), 0)     # slice [0:64]
        buf22 = reinterpret_tensor(buf29, (1, 64), (512, 1), 64)    # slice [64:128]

    Returns ``{alias: {"base": base, "offset": off, "size": numel}}`` for aliases
    that are genuine slices (offset > 0 or smaller than the base).  Pure views
    (offset 0, full size) are omitted.  This lets the visualization label a
    kernel as writing ``cat[0:64]`` instead of collapsing every slice producer
    onto the single ``cat`` node.
    """
    slices: Dict[str, Dict[str, int]] = {}
    if output_path is None or not output_path.exists():
        return slices
    base_numel: Dict[str, int] = {}
    try:
        lines = output_path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            m = re.match(r"\s*(\w+)\s*=\s*empty_strided\((\([^)]*\))\s*,\s*(\([^)]*\))", line)
            if m:
                sizes = re.findall(r"\d+", m.group(2))
                if sizes:
                    numel = 1
                    for s in sizes:
                        numel *= int(s)
                    base_numel[m.group(1)] = numel
        for line in lines:
            m = re.match(
                r"\s*(\w+)\s*=\s*reinterpret_tensor\((\w+),\s*(\([^)]*\))\s*,\s*(\([^)]*\))\s*,\s*(\d+)\)", line
            )
            if not m:
                continue
            alias, base, sizes_s, _strides_s, offset_s = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
            sizes = re.findall(r"\d+", sizes_s)
            if not sizes:
                continue
            numel = 1
            for s in sizes:
                numel *= int(s)
            offset = int(offset_s)
            base_n = base_numel.get(base, 0)
            if offset > 0 or (base_n and numel < base_n):
                slices[alias] = {"base": base, "offset": offset, "size": numel}
    except Exception:  # nosec B110
        pass
    return slices


def _build_cat_operand_map(
    fx_path: Path | None,
    buffer_to_fx: Dict[str, List[Dict[str, Any]]],
    slice_map: Dict[str, Dict[str, int]],
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Map slice-written stack/concat buffers to their cat/stack operand nodes.

    A stack/concat output buffer ``buf88`` (fx node ``cat``) is written
    slice-by-slice by parallel kernels.  The slice at element-offset ``off`` is
    operand ``i`` of the cat node, where ``i`` is the RANK of ``off`` among the
    sorted offsets -- NOT ``off // size``.  For a 3-D cat the offset steps by
    the last-dim chunk (64) while ``size`` is the full element count (e.g.
    1728), so ``off // size`` collapses every slice onto operand 0.
    Returns ``{base_buffer: {offset: {"node": operand, "idx": i,
    "chunk": uniform_step_or_None}}}`` derived from the transformed FX graph's
    cat/stack operand list.
    """
    operand_map: Dict[str, Dict[int, Dict[str, Any]]] = {}
    if fx_path is None or not fx_path.exists():
        return operand_map
    try:
        nodes, _edges = parse_fx_graph(fx_path)
    except Exception:
        return operand_map
    by_id = {str(n.get("id")): n for n in nodes}
    # base buffer -> its fx node name
    base2fx: Dict[str, str] = {}
    for buf, rows in buffer_to_fx.items():
        for r in rows:
            fx = str(r.get("fx_node", ""))
            if fx in by_id:
                base2fx[str(buf)] = fx
    # group slice aliases by base buffer
    by_base: Dict[str, List[Dict[str, int]]] = {}
    for sm in slice_map.values():
        by_base.setdefault(str(sm["base"]), []).append(sm)
    for base, sms in by_base.items():
        node = by_id.get(base2fx.get(base, ""))
        if node is None:
            continue
        expr = str(node.get("expr", ""))
        m = re.search(r"\[([^\]]+)\]", expr)
        if not m:
            continue
        operands = [x.strip() for x in m.group(1).split(",")]
        if not operands:
            continue
        offsets = sorted({int(sm["offset"]) for sm in sms})
        if not offsets:
            continue
        # uniform chunk = step between sorted offsets (the cat-dim slice width);
        # None if offsets aren't evenly spaced.
        steps = {offsets[i + 1] - offsets[i] for i in range(len(offsets) - 1)}
        chunk = next(iter(steps)) if len(steps) == 1 else None
        opmap: Dict[int, Dict[str, Any]] = {}
        for i, off in enumerate(offsets):
            if i < len(operands):
                opmap[off] = {"node": str(operands[i]), "idx": i, "chunk": chunk}
        operand_map[base] = opmap
    return operand_map


def _resolve_alias(name: Any, aliases: Dict[str, str]) -> str:
    name = str(name)
    seen: Set[str] = set()
    while name in aliases and name not in seen:
        seen.add(name)
        name = aliases[name]
    return name


def _kernel_true_output_logicals(
    kernel_row: Dict[str, Any],
    aliases: Dict[str, str] | None = None,
    buffer_to_fx: Dict[str, List[Dict[str, Any]]] | None = None,
) -> List[str]:
    """Output logicals excluding in-place reuse of an input buffer.

    A fused kernel may write an alias of one of its own inputs (e.g. softmax
    fused into the bmm buffer: ``buf14 = reinterpret_tensor(buf8)`` while buf8
    is also the input).  Resolving such an alias to its base would make the
    kernel claim the input's producer FX node / buffer.  We only keep outputs
    that are not aliases of this kernel's inputs.

    Exception: if the RAW output buffer has its own buffer->fx mapping, it is
    kept as-is (not alias-resolved).  A storage alias (reinterpret_tensor) is
    not a VALUE alias: an in-place fused kernel (layer-norm / softmax into the
    input's storage) or an extern multi-output call records its output under a
    buffer name whose buffer->fx entry is the correct attribution, even though
    resolving it lands on an input base (gpt2: ``buf30`` aliases input ``buf17``
    but maps to ``native_layer_norm_2``) or on an unrelated raw-output name
    (bert ``npu_fusion_attention``: ``buf7`` aliases ``buf6`` but maps to the
    attention node).  Falling back to the alias base would lose that mapping.
    """
    aliases = aliases or {}
    buffer_to_fx = buffer_to_fx or {}
    input_bases: Set[str] = set()
    for entry in kernel_row.get("entries", []):
        for logical in _entry_input_logicals(entry):
            input_bases.add(_resolve_alias(logical, aliases))
    # Triton kernels record their inputs in the entries; their _parsed_inputs
    # (from `.run(bufX, ...)`) also include the in-place output buffer and must
    # not be treated as an input.  Extern kernels rely on _parsed_inputs.
    if not str(kernel_row.get("kind", "")).startswith("triton"):
        for item in kernel_row.get("_parsed_inputs", []) or []:
            input_bases.add(_resolve_alias(str(item), aliases))

    result: List[str] = []
    for entry in kernel_row.get("entries", []):
        for logical in _entry_output_logicals(entry):
            raw = str(logical)
            if buffer_to_fx.get(raw):
                # own mapping: correct attribution, keep raw name
                if raw not in result:
                    result.append(raw)
                continue
            resolved = _resolve_alias(raw, aliases)
            if resolved in input_bases:
                continue
            if resolved not in result:
                result.append(resolved)
    for item in kernel_row.get("_parsed_outputs", []) or []:
        raw = str(item)
        if buffer_to_fx.get(raw):
            if raw not in result:
                result.append(raw)
            continue
        resolved = _resolve_alias(raw, aliases)
        if resolved not in input_bases and resolved not in result:
            result.append(resolved)
    return result


def kernel_fx_nodes(
    kernel_row: Dict[str, Any],
    buffer_to_fx: Dict[str, List[Dict[str, Any]]],
    aliases: Dict[str, str] | None = None,
    slice_map: Dict[str, Dict[str, int]] | None = None,
    operand_map: Dict[str, Dict[int, str]] | None = None,
) -> List[str]:
    """FX node(s) a kernel maps to, with slice/operand attribution.

    Parallel kernels that each write one slice of a stack/concat output
    (``buf21 = reinterpret_tensor(buf29, (1,64), (512,1), 0)`` -> base ``buf29``
    maps to the ``cat`` node) all collapse onto the same ``cat`` node.  With
    ``slice_map`` the label gains ``cat[0:64]``, ...; with ``operand_map``
    (from :func:`_build_cat_operand_map`) the kernel is additionally attributed
    to the cat node's i-th OPERAND (e.g. ``squeeze_2``), which is the correct
    logical owner of the slice it computes.
    """
    aliases = aliases or {}
    slice_map = slice_map or {}
    operand_map = operand_map or {}
    # per resolved base buffer: slice labels + operand hits from raw outputs
    slice_labels: Dict[str, List[str]] = {}
    operand_hits: Dict[str, List[str]] = {}
    for entry in kernel_row.get("entries", []):
        for logical in _entry_output_logicals(entry):
            _add_slice_info(slice_labels, operand_hits, logical, slice_map, operand_map)
    for item in kernel_row.get("_parsed_outputs", []) or []:
        _add_slice_info(slice_labels, operand_hits, item, slice_map, operand_map)

    fx_nodes: List[str] = []
    for resolved in _kernel_true_output_logicals(kernel_row, aliases, buffer_to_fx):
        for fx_row in buffer_to_fx.get(resolved, []):
            fx = fx_row.get("fx_node")
            if not fx:
                continue
            ops = operand_hits.get(str(resolved))
            labels = slice_labels.get(str(resolved))
            if ops:
                for op in ops:
                    fx_nodes.append(op)
            if labels:
                for lab in labels:
                    fx_nodes.append(f"{fx}{lab}")
            elif not ops:
                fx_nodes.append(str(fx))
    return _dedupe_keep_order(fx_nodes)


def _add_slice_info(
    slice_labels: Dict[str, List[str]],
    operand_hits: Dict[str, List[str]],
    logical: Any,
    slice_map: Dict[str, Dict[str, int]],
    operand_map: Dict[str, Dict[int, Dict[str, Any]]],
) -> None:
    raw = str(logical)
    if raw not in slice_map:
        return
    sm = slice_map[raw]
    off, size = int(sm["offset"]), int(sm["size"])
    base = str(sm["base"])
    info = (operand_map.get(base) or {}).get(off)
    if info is not None:
        # operand attribution: offset rank (NOT off//size, which collapses 3-D
        # cat slices onto operand 0)
        node = str(info["node"])
        if node not in operand_hits.setdefault(base, []):
            operand_hits[base].append(node)
        # slice label with the real cat-dim chunk (uniform offset step)
        chunk = info.get("chunk") or size
        label = f"[{off}:{off + chunk}]"
        if label not in slice_labels.setdefault(base, []):
            slice_labels[base].append(label)
    else:
        label = f"[{off}:{off + size}]"
        if label not in slice_labels.setdefault(base, []):
            slice_labels[base].append(label)


def bad_element_label(stat: Dict[str, Any]) -> str:
    rtol = stat.get("bad_element_rel_threshold")
    atol = stat.get("bad_element_abs_threshold")
    if rtol is None or atol is None:
        return "bad elements"
    return f"bad rtol={float(rtol):g}/atol={float(atol):g}"


def _empty_compare_stat() -> Dict[str, Any]:
    return {
        "comparison_count": 0,
        "bad_element_sum": 0,
        "bad_element_max": 0,
        "max_abs_diff": None,
        "max_rel_abs_diff": None,
        "bad_element_rel_threshold": None,
        "bad_element_abs_threshold": None,
        "statuses": {},
    }


def _add_compare_stat(stat: Dict[str, Any], row: Dict[str, Any]) -> None:
    stat["comparison_count"] += 1
    status = str(row.get("status", "unknown"))
    stat["statuses"][status] = int(stat["statuses"].get(status, 0)) + 1
    bad_count = row.get("bad_element_count")
    if bad_count is not None:
        bad_count = int(bad_count)
        stat["bad_element_sum"] += bad_count
        stat["bad_element_max"] = max(int(stat["bad_element_max"]), bad_count)
    for key in ("bad_element_rel_threshold", "bad_element_abs_threshold"):
        if row.get(key) is not None:
            stat[key] = float(row[key])
    for key in ("max_abs_diff", "max_rel_abs_diff"):
        value = row.get(key)
        if value is None:
            continue
        value = float(value)
        if stat[key] is None or value > float(stat[key]):
            stat[key] = value


def compare_stats_by_call_id(comparisons: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = defaultdict(_empty_compare_stat)
    for row in comparisons:
        call_id = row.get("call_id")
        if call_id is None:
            kernel_row = row.get("kernel_row") or {}
            call_id = kernel_row.get("call_id")
        if call_id is None:
            continue
        _add_compare_stat(stats[str(call_id)], row)
    return dict(stats)


def compare_stats_by_fx_node(comparisons: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = defaultdict(_empty_compare_stat)
    for row in comparisons:
        fx_node = row.get("fx_node")
        if not fx_node:
            continue
        _add_compare_stat(stats[str(fx_node)], row)
    return dict(stats)


def compare_details_by_call_id(comparisons: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    def safe_metric(value: Any) -> Any:
        return None if value is None else _json_scalar(value)

    details: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in comparisons:
        call_id = row.get("call_id")
        if call_id is None:
            kernel_row = row.get("kernel_row") or {}
            call_id = kernel_row.get("call_id")
        if call_id is None:
            continue
        details[str(call_id)].append(
            {
                "status": row.get("status"),
                "logical_buffer": row.get("logical_buffer"),
                "fx_node": row.get("fx_node"),
                "kernel_name": row.get("kernel_name"),
                "call_id": call_id,
                "call_arg": row.get("call_arg"),
                "role": row.get("role"),
                "bad_element_count": row.get("bad_element_count"),
                "mean_abs_diff": safe_metric(row.get("mean_abs_diff")),
                "max_abs_diff": safe_metric(row.get("max_abs_diff")),
                "mean_rel_abs_diff": safe_metric(row.get("mean_rel_abs_diff")),
                "max_rel_abs_diff": safe_metric(row.get("max_rel_abs_diff")),
                "top_error_elements": row.get("top_error_elements") or [],
            }
        )
    return dict(details)


def _kernel_call_instance_id(
    row: Dict[str, Any],
    order: int,
    occurrence: int,
) -> str:
    call_id = str(row.get("call_id", order))
    source_line = row.get("_source_line")
    if source_line is not None:
        return f"k{call_id}_l{source_line}_o{occurrence}"
    return f"k{call_id}_o{occurrence}"


def build_kernel_call_graph(
    kernels: List[Dict[str, Any]],
    buffer_to_fx: Dict[str, List[Dict[str, Any]]],
    compare_stats: Dict[str, Dict[str, Any]] | None = None,
    compare_details: Dict[str, List[Dict[str, Any]]] | None = None,
    aliases: Dict[str, str] | None = None,
    slice_map: Dict[str, Dict[str, int]] | None = None,
    operand_map: Dict[str, Dict[int, str]] | None = None,
) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]]]:
    aliases = aliases or {}
    slice_map = slice_map or {}
    operand_map = operand_map or {}
    rows = sorted(kernels, key=lambda row: int(row.get("call_id", 0)))
    nodes: List[Dict[str, Any]] = []
    edges: List[Tuple[str, str]] = []
    seen_edges: Set[Tuple[str, str]] = set()
    producers: Dict[str, Set[str]] = {}
    occurrences: Dict[Tuple[str, Any], int] = defaultdict(int)

    for order, row in enumerate(rows):
        call_id = int(row.get("call_id", order))
        occurrence_key = (str(row.get("kernel_name", "")), row.get("_source_line"))
        occurrence = occurrences[occurrence_key]
        occurrences[occurrence_key] += 1
        node_id = _kernel_call_instance_id(row, order, occurrence)
        entries = list(row.get("entries") or [])
        input_logicals = _dedupe_keep_order(
            _resolve_alias(logical, aliases) for entry in entries for logical in _entry_input_logicals(entry)
        )
        if row.get("_parsed_inputs") and not str(row.get("kind", "")).startswith("triton"):
            input_logicals = _dedupe_keep_order(
                list(input_logicals) + [_resolve_alias(str(item), aliases) for item in row.get("_parsed_inputs", [])]
            )
        # Graph edges keep ALL outputs (including in-place writes of an input
        # buffer), so an in-place kernel still connects to the consumers of the
        # buffer it modified.  The FX mapping uses _kernel_true_output_logicals
        # to avoid claiming the input's producer node.
        output_logicals = _dedupe_keep_order(
            _resolve_alias(logical, aliases) for entry in entries for logical in _entry_output_logicals(entry)
        )
        if row.get("_parsed_outputs"):
            output_logicals = _dedupe_keep_order(
                list(output_logicals) + [_resolve_alias(str(item), aliases) for item in row.get("_parsed_outputs", [])]
            )

        for logical in input_logicals:
            for producer in producers.get(logical, set()):
                if producer == node_id:
                    continue
                edge = (producer, node_id)
                if edge not in seen_edges:
                    seen_edges.add(edge)
                    edges.append(edge)

        nodes.append(
            {
                "id": node_id,
                "label": f"#{call_id} {row.get('kernel_name', '<kernel>')}",
                "kernel_name": row.get("kernel_name", "<kernel>"),
                "kind": row.get("kind", "kernel_call"),
                "call_id": call_id,
                "call_instance_id": node_id,
                "occurrence": occurrence,
                "order": order,
                "inputs": input_logicals,
                "outputs": output_logicals,
                "fx_nodes": kernel_fx_nodes(row, buffer_to_fx, aliases, slice_map, operand_map),
                "source_line": row.get("_source_line"),
                "compare_stats": (compare_stats or {}).get(str(call_id), {}),
                "compare_details": (compare_details or {}).get(str(call_id), []),
            }
        )

        for logical in output_logicals:
            producers.setdefault(logical, set()).add(node_id)

    return nodes, edges


def _compute_generations(
    nodes: List[Dict[str, Any]],
    edges: List[Tuple[str, str]],
) -> Dict[str, int]:
    node_ids = {str(node["id"]) for node in nodes}
    children: Dict[str, List[str]] = {node_id: [] for node_id in node_ids}
    indegree: Dict[str, int] = {node_id: 0 for node_id in node_ids}
    for src, dst in edges:
        if src not in node_ids or dst not in node_ids:
            continue
        children[src].append(dst)
        indegree[dst] += 1

    order_index = {str(node["id"]): int(node.get("order", i)) for i, node in enumerate(nodes)}
    ready = sorted(
        [node_id for node_id, degree in indegree.items() if degree == 0], key=lambda x: order_index.get(x, 0)
    )
    generation: Dict[str, int] = {node_id: 0 for node_id in node_ids}
    visited: List[str] = []
    while ready:
        node_id = ready.pop(0)
        visited.append(node_id)
        for child in sorted(children.get(node_id, []), key=lambda x: order_index.get(x, 0)):
            generation[child] = max(generation.get(child, 0), generation[node_id] + 1)
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
        ready.sort(key=lambda x: (generation.get(x, 0), order_index.get(x, 0)))

    if len(visited) != len(node_ids):
        for node_id in node_ids:
            generation.setdefault(node_id, 0)
    return generation


def _topo_order(node_ids: Set[str], edges: List[Tuple[str, str]]) -> List[str]:
    children: Dict[str, List[str]] = {node_id: [] for node_id in node_ids}
    indegree: Dict[str, int] = {node_id: 0 for node_id in node_ids}
    for src, dst in edges:
        if src in node_ids and dst in node_ids:
            children[src].append(dst)
            indegree[dst] += 1
    ready = deque(node_id for node_id, degree in indegree.items() if degree == 0)
    order: List[str] = []
    while ready:
        node_id = ready.popleft()
        order.append(node_id)
        for child in children[node_id]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    for node_id in node_ids:
        if node_id not in order:
            order.append(node_id)
    return order


def layout_graph(
    nodes: List[Dict[str, Any]],
    edges: List[Tuple[str, str]],
    *,
    node_w: int,
    node_h: int,
    col_gap: int,
    row_gap: int,
) -> Dict[str, Any]:
    """Layered layout that keeps edges short.

    Leaf/input nodes are sunk down to just above their earliest consumer so an
    op's operands sit next to it instead of stretching a long fly-line across
    the canvas.  Within each layer, nodes are ordered with a median sweep and
    then pulled toward the barycenter of their neighbours, keeping the graph
    compact and readable like a hand-drawn data-flow graph.
    """
    node_ids = {str(node["id"]) for node in nodes}
    order_index = {str(node["id"]): int(node.get("order", i)) for i, node in enumerate(nodes)}
    edges = [(src, dst) for src, dst in edges if src in node_ids and dst in node_ids]

    parents: Dict[str, List[str]] = {node_id: [] for node_id in node_ids}
    children: Dict[str, List[str]] = {node_id: [] for node_id in node_ids}
    indegree: Dict[str, int] = {node_id: 0 for node_id in node_ids}
    for src, dst in edges:
        parents[dst].append(src)
        children[src].append(dst)
        indegree[dst] += 1

    # 1. baseline longest-path layers, then sink source (leaf) nodes to just
    #    above their earliest consumer so inputs are placed next to the op that
    #    uses them.  Sources have no incoming edges, so this is always valid.
    layer = _compute_generations(nodes, edges)
    for node_id in node_ids:
        if indegree[node_id] == 0 and children[node_id]:
            layer[node_id] = max(0, min(layer[c] for c in children[node_id]) - 1)
    topo = _topo_order(node_ids, edges)
    for node_id in topo:
        if parents[node_id]:
            layer[node_id] = max(layer[node_id], max(layer[p] + 1 for p in parents[node_id]))

    layers: Dict[int, List[str]] = defaultdict(list)
    for node_id in node_ids:
        layers[layer[node_id]].append(node_id)
    for ids in layers.values():
        ids.sort(key=lambda node_id: (order_index.get(node_id, 0), node_id))

    # 2. crossing-reducing ordering inside each layer (median-based sweeps).
    pos: Dict[str, int] = {}
    for ids in layers.values():
        for index, node_id in enumerate(ids):
            pos[node_id] = index

    def _median(values: List[int]) -> float:
        if not values:
            return float("inf")
        values = sorted(values)
        return float(values[len(values) // 2])

    for _ in range(6):
        for gen in sorted(layers):
            layers[gen].sort(
                key=lambda node_id: (
                    _median([pos.get(p, 0) for p in parents.get(node_id, [])])
                    if parents.get(node_id)
                    else float(order_index.get(node_id, 0)),
                    order_index.get(node_id, 0),
                )
            )
            for index, node_id in enumerate(layers[gen]):
                pos[node_id] = index
        for gen in sorted(layers, reverse=True):
            layers[gen].sort(
                key=lambda node_id: (
                    _median([pos.get(c, 0) for c in children.get(node_id, [])])
                    if children.get(node_id)
                    else float(pos.get(node_id, order_index.get(node_id, 0))),
                    order_index.get(node_id, 0),
                )
            )
            for index, node_id in enumerate(layers[gen]):
                pos[node_id] = index

    # 3. wrap very wide layers into stacked sub-rows so the canvas does not
    #    stretch into an unreadable horizontal band.
    max_per_row = 56
    subrow_gap = 8
    subrows: Dict[int, List[List[str]]] = {}
    subrow_index: Dict[str, int] = {}
    for gen in sorted(layers):
        ids = layers[gen]
        wrapped = [ids[i : i + max_per_row] for i in range(0, len(ids), max_per_row)]
        subrows[gen] = wrapped
        for sub_index, sub in enumerate(wrapped):
            for node_id in sub:
                subrow_index[node_id] = sub_index

    # 4. x-coordinates: place each sub-row's nodes in the ordered sequence with
    #    a fixed pitch.  Because a sub-row holds at most ``max_per_row`` nodes,
    #    its span never exceeds the per-row budget, so the canvas stays compact
    #    and no node overlaps another (different sub-rows are offset vertically).
    x: Dict[str, float] = {}
    for gen, wrapped in subrows.items():
        for sub in wrapped:
            for index, node_id in enumerate(sub):
                x[node_id] = 24 + index * (node_w + col_gap)

    # 5. y positions: each layer is a vertical block that fully contains all of
    #    its wrapped sub-rows, so every edge still points strictly downward.
    layer_top: Dict[int, float] = {}
    cursor = 24.0
    for gen in sorted(subrows):
        layer_top[gen] = cursor
        cursor += len(subrows[gen]) * (node_h + subrow_gap)

    positioned: Dict[str, Dict[str, Any]] = {}
    max_x = 0
    max_y = 0
    for node_id in node_ids:
        xx = x[node_id]
        yy = layer_top[layer[node_id]] + subrow_index.get(node_id, 0) * (node_h + subrow_gap)
        positioned[node_id] = {"x": xx, "y": yy, "w": node_w, "h": node_h}
        max_x = max(max_x, xx + node_w + 24)
        max_y = max(max_y, yy + node_h + 24)

    return {
        "positions": positioned,
        "width": max(max_x, 640),
        "height": max(max_y, 420),
    }


def layout_kernel_call_graph(
    nodes: List[Dict[str, Any]],
    edges: List[Tuple[str, str]],
    *,
    node_w: int,
    node_h: int,
    col_gap: int,
    row_gap: int,
) -> Dict[str, Any]:
    """Lay out kernel calls as a vertical timeline, keeping consumers near producers."""
    node_ids = {str(node["id"]) for node in nodes}
    parents: Dict[str, List[str]] = {node_id: [] for node_id in node_ids}
    for src, dst in edges:
        if src in node_ids and dst in node_ids:
            parents[dst].append(src)

    ordered_nodes = sorted(nodes, key=lambda node: (int(node.get("order", 0)), str(node.get("id"))))
    lane_by_node: Dict[str, int] = {}

    for node in ordered_nodes:
        node_id = str(node["id"])
        parent_lanes = [lane_by_node[parent] for parent in parents.get(node_id, []) if parent in lane_by_node]
        if parent_lanes:
            lane = int(round(sum(parent_lanes) / len(parent_lanes)))
        else:
            lane = 0

        lane_by_node[node_id] = max(0, lane)

    positioned: Dict[str, Dict[str, Any]] = {}
    max_x = 0
    max_y = 0
    for row, node in enumerate(ordered_nodes):
        node_id = str(node["id"])
        lane = lane_by_node.get(node_id, 0)
        x = 24 + lane * (node_w + col_gap)
        y = 24 + row * (node_h + row_gap)
        positioned[node_id] = {"x": x, "y": y, "w": node_w, "h": node_h}
        max_x = max(max_x, x + node_w + 24)
        max_y = max(max_y, y + node_h + 24)

    return {
        "positions": positioned,
        "width": max(max_x, 640),
        "height": max(max_y, 420),
    }


def add_layout_slack(layout: Dict[str, Any], *, right: int, bottom: int) -> Dict[str, Any]:
    out = dict(layout)
    out["positions"] = layout["positions"]
    out["width"] = int(out.get("width", 0)) + right
    out["height"] = int(out.get("height", 0)) + bottom
    return out


def _render_edges(
    edges: List[Tuple[str, str]],
    positions: Dict[str, Dict[str, Any]],
    *,
    css_class: str,
) -> str:
    paths: List[str] = []
    for src, dst in edges:
        if src not in positions or dst not in positions:
            continue
        a = positions[src]
        b = positions[dst]
        x1 = a["x"] + a["w"] / 2
        y1 = a["y"] + a["h"]
        x2 = b["x"] + b["w"] / 2
        y2 = b["y"]
        mid = max(24.0, (y2 - y1) / 2)
        d = f"M{x1:.1f},{y1:.1f} C{x1:.1f},{y1 + mid:.1f} {x2:.1f},{y2 - mid:.1f} {x2:.1f},{y2:.1f}"
        paths.append(f'<path class="{css_class}" d="{d}"></path>')
    return "\n".join(paths)


def _format_compare_bad_line(stat: Dict[str, Any]) -> str:
    return (
        f"{bad_element_label(stat)} s={stat.get('bad_element_sum', 0)} "
        f"m={stat.get('bad_element_max', 0)} "
        f"c={stat.get('comparison_count', 0)}"
    )


def _render_fx_nodes(
    nodes: List[Dict[str, Any]],
    positions: Dict[str, Dict[str, Any]],
    fx_compare_stats: Dict[str, Dict[str, Any]],
) -> str:
    pieces: List[str] = []
    for node in nodes:
        node_id = str(node["id"])
        if node_id not in positions:
            continue
        p = positions[node_id]
        dom_id = _safe_dom_id("fx", node_id)
        compare_stat = fx_compare_stats.get(node_id, _empty_compare_stat())
        bad_line = _format_compare_bad_line(compare_stat)
        title = html.escape(
            f"{node.get('expr') or node.get('op') or ''}\\n{bad_line}\\nstatuses: {compare_stat.get('statuses', {})}",
            quote=True,
        )
        label = html.escape(_shorten(str(node.get("label", node_id)), 32))
        op = html.escape(_shorten(str(node.get("op", "")), 38))
        kind = html.escape(str(node.get("kind", "")))
        line = node.get("line")
        extra_class = " fx-has-bad" if int(compare_stat.get("bad_element_sum", 0)) > 0 else ""
        pieces.append(
            f'''<div id="{dom_id}" class="node fx-node fx-{kind}{extra_class}" data-fx-name="{html.escape(node_id, quote=True)}" '''
            f'''style="left:{p["x"]}px;top:{p["y"]}px;width:{p["w"]}px;height:{p["h"]}px" title="{title}">'''
            f'''<div class="node-title">{label}</div><div class="node-sub">{op}</div>'''
            f'''<div class="node-bad">{html.escape(bad_line)}</div>'''
            f'''<div class="node-foot">line {html.escape(str(line or ""))}</div></div>'''
        )
    return "\n".join(pieces)


def _render_kernel_nodes(nodes: List[Dict[str, Any]], positions: Dict[str, Dict[str, Any]]) -> str:
    pieces: List[str] = []
    for node in nodes:
        node_id = str(node["id"])
        if node_id not in positions:
            continue
        p = positions[node_id]
        dom_id = _safe_dom_id("kernel", node_id)
        fx_nodes = list(node.get("fx_nodes") or [])
        fx_json = html.escape(json.dumps(fx_nodes, ensure_ascii=False), quote=True)
        details_json = html.escape(
            json.dumps(node.get("compare_details") or [], ensure_ascii=False, default=str, allow_nan=False),
            quote=True,
        )
        outputs = ", ".join(node.get("outputs") or [])
        inputs = ", ".join(node.get("inputs") or [])
        compare_stats = node.get("compare_stats") or {}
        bad_line = _format_compare_bad_line(compare_stats)
        title = html.escape(
            f"{node.get('kernel_name')}\\ninputs: {inputs}\\noutputs: {outputs}\\n{bad_line}\\nfx: {', '.join(fx_nodes)}",
            quote=True,
        )
        kind = str(node.get("kind", "kernel_call"))
        label = html.escape(_shorten(str(node.get("kernel_name", node_id)), 42))
        call_id = html.escape(str(node.get("call_id", "")))
        source_line = node.get("source_line")
        instance = html.escape(str(node.get("call_instance_id", node_id)))
        out_text = html.escape(_shorten(outputs, 42))
        fx_text = html.escape(_shorten(", ".join(fx_nodes), 48))
        line_text = f"L{source_line}" if source_line is not None else "L?"
        pieces.append(
            f'''<div id="{dom_id}" class="node kernel-node kernel-{html.escape(kind)}" data-fx-nodes="{fx_json}" data-compare-details="{details_json}" '''
            f'''style="left:{p["x"]}px;top:{p["y"]}px;width:{p["w"]}px;height:{p["h"]}px" title="{title}">'''
            f'''<div class="node-title">#{call_id} {html.escape(line_text)} {label}</div><div class="node-sub">out: {out_text}</div>'''
            f'''<div class="node-bad">{html.escape(bad_line)}</div>'''
            f'''<div class="node-foot">{instance} fx: {fx_text}</div></div>'''
        )
    return "\n".join(pieces)


def _graph_html(
    *,
    trace_dir: Path,
    fx_path: Path,
    output_path: Path,
    fx_nodes: List[Dict[str, Any]],
    fx_edges: List[Tuple[str, str]],
    kernel_nodes: List[Dict[str, Any]],
    kernel_edges: List[Tuple[str, str]],
    fx_layout: Dict[str, Any],
    kernel_layout: Dict[str, Any],
    fx_compare_stats: Dict[str, Dict[str, Any]],
) -> str:
    fx_edge_svg = _render_edges(fx_edges, fx_layout["positions"], css_class="edge fx-edge")
    kernel_edge_svg = _render_edges(kernel_edges, kernel_layout["positions"], css_class="edge kernel-edge")
    fx_node_html = _render_fx_nodes(fx_nodes, fx_layout["positions"], fx_compare_stats)
    kernel_node_html = _render_kernel_nodes(kernel_nodes, kernel_layout["positions"])
    stats = {
        "fx_nodes": len(fx_nodes),
        "fx_edges": len(fx_edges),
        "kernel_calls": len(kernel_nodes),
        "kernel_edges": len(kernel_edges),
    }
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Logic Buffer FX / Kernel Graph</title>
<style>
html, body {{
  margin: 0;
  background: #f6f8fb;
  color: #172033;
  font-family: 'Google Sans', 'Segoe UI', sans-serif;
}}
.header {{
  position: sticky;
  top: 0;
  z-index: 5;
  background: rgba(255, 255, 255, 0.96);
  border-bottom: 1px solid #d7dde8;
  padding: 14px 18px;
  box-shadow: 0 2px 12px rgba(18, 31, 56, 0.08);
}}
.header h1 {{
  margin: 0 0 6px;
  font-size: 20px;
}}
.meta {{
  font-size: 12px;
  color: #5f6f89;
  line-height: 1.45;
  word-break: break-all;
}}
.graph-grid {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  gap: 12px;
  padding: 12px;
  height: calc(100vh - 132px);
  min-height: 520px;
  box-sizing: border-box;
}}
.pane {{
  background: #ffffff;
  border: 1px solid #d7dde8;
  border-radius: 10px;
  height: 100%;
  min-height: 0;
  overflow: auto;
  overscroll-behavior: contain;
  cursor: grab;
  box-shadow: 0 8px 28px rgba(18, 31, 56, 0.08);
}}
.pane.dragging {{
  cursor: grabbing;
  user-select: none;
}}
.pane-title {{
  position: sticky;
  top: 0;
  z-index: 3;
  background: #ffffff;
  border-bottom: 1px solid #e1e6ef;
  padding: 10px 12px;
  font-weight: 700;
}}
.hint {{
  margin-left: 8px;
  font-weight: 400;
  color: #718096;
  font-size: 12px;
}}
.canvas {{
  position: relative;
  overflow: visible;
  background-image: radial-gradient(#dfe5ef 0.8px, transparent 0.8px);
  background-size: 18px 18px;
  transform-origin: 0 0;
}}
.edges-svg {{
  position: absolute;
  inset: 0;
  overflow: visible;
  pointer-events: none;
}}
.edge {{
  fill: none;
  stroke: #9aa8bb;
  stroke-width: 1.1;
  opacity: 0.42;
}}
.kernel-edge {{
  stroke: #7a8799;
}}
.node {{
  position: absolute;
  box-sizing: border-box;
  border: 1px solid #2f3b4f;
  border-radius: 8px;
  background: #f9fbff;
  padding: 6px 8px;
  overflow: hidden;
  box-shadow: 0 3px 10px rgba(17, 24, 39, 0.08);
  transition: border-color 120ms ease, box-shadow 120ms ease, opacity 120ms ease, transform 120ms ease;
}}
.node-title {{
  font-size: 11px;
  font-weight: 800;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}
.node-sub {{
  margin-top: 2px;
  font-size: 10px;
  color: #314155;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}
.node-foot {{
  margin-top: 2px;
  font-size: 9px;
  color: #69768a;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}
.node-bad {{
  margin-top: 2px;
  font-size: 9px;
  color: #9f1239;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}
.fx-placeholder {{
  background: #eef6ff;
  border-color: #5b8def;
}}
.fx-op {{
  background: #ffffff;
}}
.fx-has-bad {{
  border-color: #be123c;
  background: #fff7f8;
}}
.kernel-triton_kernel_call {{
  background: #eaf7ef;
  border-color: #248a4b;
}}
.kernel-extern_kernel_call {{
  background: #fff7df;
  border-color: #ba8425;
}}
.kernel-node:hover {{
  transform: translateY(-1px);
  box-shadow: 0 8px 24px rgba(17, 24, 39, 0.16);
}}
.fx-node.highlight {{
  border-color: #e11d48;
  box-shadow: 0 0 0 3px rgba(225, 29, 72, 0.24), 0 8px 22px rgba(225, 29, 72, 0.20);
  background: #fff1f2;
  z-index: 2;
}}
.kernel-node.active {{
  border-color: #e11d48;
  box-shadow: 0 0 0 3px rgba(225, 29, 72, 0.24), 0 8px 22px rgba(225, 29, 72, 0.20);
  z-index: 2;
}}
.dimmed .fx-node:not(.highlight) {{
  opacity: 0.28;
}}
#hover-info {{
  margin-top: 6px;
  font-size: 12px;
  color: #334155;
}}
.toolbar {{
  position: sticky;
  top: 37px;
  z-index: 4;
  display: flex;
  gap: 6px;
  padding: 6px 10px;
  background: rgba(255,255,255,0.94);
  border-bottom: 1px solid #edf1f7;
}}
.tool-button {{
  border: 1px solid #cad3e2;
  border-radius: 6px;
  background: #f8fafc;
  color: #334155;
  padding: 3px 7px;
  font-size: 11px;
  cursor: pointer;
}}
.tool-button:hover {{
  background: #edf2f7;
}}
.detail-drawer {{
  position: fixed;
  top: 0;
  right: 0;
  z-index: 20;
  width: min(720px, 48vw);
  height: 100vh;
  background: #ffffff;
  border-left: 1px solid #cbd5e1;
  box-shadow: -18px 0 44px rgba(15, 23, 42, 0.20);
  transform: translateX(102%);
  transition: transform 180ms ease;
  display: flex;
  flex-direction: column;
}}
.detail-drawer.open {{
  transform: translateX(0);
}}
.drawer-header {{
  padding: 14px 16px;
  border-bottom: 1px solid #e2e8f0;
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
}}
.drawer-title {{
  font-size: 15px;
  font-weight: 800;
  line-height: 1.35;
}}
.drawer-close {{
  border: 1px solid #cbd5e1;
  border-radius: 6px;
  background: #f8fafc;
  cursor: pointer;
  padding: 4px 8px;
}}
.drawer-body {{
  overflow: auto;
  padding: 12px 16px 24px;
  font-size: 12px;
}}
.detail-card {{
  border: 1px solid #dbe3ee;
  border-radius: 8px;
  padding: 10px;
  margin-bottom: 12px;
  background: #fbfdff;
}}
.detail-card h3 {{
  margin: 0 0 6px;
  font-size: 13px;
}}
.detail-meta {{
  color: #475569;
  line-height: 1.45;
  margin-bottom: 8px;
}}
.detail-table {{
  width: 100%;
  border-collapse: collapse;
  font-variant-numeric: tabular-nums;
}}
.detail-table th,
.detail-table td {{
  border-top: 1px solid #e2e8f0;
  padding: 4px 5px;
  text-align: left;
  vertical-align: top;
}}
.detail-table th {{
  color: #475569;
  font-weight: 700;
}}
@media (max-width: 1100px) {{
  .graph-grid {{
    grid-template-columns: 1fr;
    height: auto;
  }}
  .pane {{
    height: 78vh;
  }}
  .detail-drawer {{
    width: min(92vw, 720px);
  }}
}}
</style>
</head>
<body>
<div class="header">
  <h1>FX / Kernel Call Graph</h1>
  <div class="meta">trace_dir: {html.escape(str(trace_dir))}</div>
  <div class="meta">fx: {html.escape(str(fx_path))}</div>
  <div class="meta">output_code: {html.escape(str(output_path))}</div>
  <div class="meta">stats: {html.escape(json.dumps(stats, ensure_ascii=False))}</div>
  <div id="hover-info">Hover a kernel node to highlight corresponding FX nodes.</div>
</div>
<div class="graph-grid">
  <section id="fx-pane" class="pane">
    <div class="pane-title">FX Transformed Graph <span class="hint">top-to-bottom topological generations</span></div>
    <div class="toolbar"><button class="tool-button" data-pane="fx-pane" data-action="zoom-in">+</button><button class="tool-button" data-pane="fx-pane" data-action="zoom-out">-</button><button class="tool-button" data-pane="fx-pane" data-action="reset">reset</button></div>
    <div class="canvas" style="width:{fx_layout['width']}px;height:{fx_layout['height']}px">
      <svg class="edges-svg" width="{fx_layout['width']}" height="{fx_layout['height']}">{fx_edge_svg}</svg>
      {fx_node_html}
    </div>
  </section>
  <section id="kernel-pane" class="pane">
    <div class="pane-title">Kernel Call Graph <span class="hint">vertical call order, lanes follow direct producers</span></div>
    <div class="toolbar"><button class="tool-button" data-pane="kernel-pane" data-action="zoom-in">+</button><button class="tool-button" data-pane="kernel-pane" data-action="zoom-out">-</button><button class="tool-button" data-pane="kernel-pane" data-action="reset">reset</button></div>
    <div class="canvas" style="width:{kernel_layout['width']}px;height:{kernel_layout['height']}px">
      <svg class="edges-svg" width="{kernel_layout['width']}" height="{kernel_layout['height']}">{kernel_edge_svg}</svg>
      {kernel_node_html}
    </div>
  </section>
</div>
<aside id="detail-drawer" class="detail-drawer" aria-hidden="true">
  <div class="drawer-header">
    <div>
      <div id="drawer-title" class="drawer-title">Kernel Compare Details</div>
      <div id="drawer-subtitle" class="meta"></div>
    </div>
    <button id="drawer-close" class="drawer-close">close</button>
  </div>
  <div id="drawer-body" class="drawer-body"></div>
</aside>
<script>
const fxPane = document.getElementById('fx-pane');
const hoverInfo = document.getElementById('hover-info');
const detailDrawer = document.getElementById('detail-drawer');
const drawerTitle = document.getElementById('drawer-title');
const drawerSubtitle = document.getElementById('drawer-subtitle');
const drawerBody = document.getElementById('drawer-body');
const drawerClose = document.getElementById('drawer-close');
function escapeHtml(value) {{
  return String(value ?? '').replace(/[&<>"']/g, (ch) => ({{
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  }}[ch]));
}}
function fmtNumber(value) {{
  if (value === null || value === undefined || value === '') return '';
  if (typeof value === 'number') {{
    if (!Number.isFinite(value)) return String(value);
    const abs = Math.abs(value);
    if (abs !== 0 && (abs < 1e-4 || abs >= 1e5)) return value.toExponential(4);
    return String(Math.round(value * 1000000) / 1000000);
  }}
  return String(value);
}}
function renderTopErrorTable(elements) {{
  if (!elements || !elements.length) {{
    return '<div class="detail-meta">No top error elements recorded.</div>';
  }}
  const rows = elements.map((item) => `
    <tr>
      <td>${{escapeHtml(item.rank)}}</td>
      <td>${{escapeHtml(JSON.stringify(item.index ?? []))}}</td>
      <td>${{escapeHtml(fmtNumber(item.fx_value))}}</td>
      <td>${{escapeHtml(fmtNumber(item.kernel_value))}}</td>
      <td>${{escapeHtml(fmtNumber(item.abs_diff))}}</td>
      <td>${{escapeHtml(fmtNumber(item.rel_diff))}}</td>
    </tr>
  `).join('');
  return `
    <table class="detail-table">
      <thead><tr><th>#</th><th>index</th><th>fx</th><th>kernel</th><th>abs</th><th>rel</th></tr></thead>
      <tbody>${{rows}}</tbody>
    </table>
  `;
}}
function openKernelDetails(kernelNode) {{
  let details = [];
  try {{
    details = JSON.parse(kernelNode.dataset.compareDetails || '[]');
  }} catch (e) {{
    details = [];
  }}
  drawerTitle.textContent = kernelNode.querySelector('.node-title')?.textContent || 'Kernel Compare Details';
  drawerSubtitle.textContent = kernelNode.querySelector('.node-sub')?.textContent || '';
  if (!details.length) {{
    drawerBody.innerHTML = '<div class="detail-card">No compare details recorded for this kernel call.</div>';
  }} else {{
    drawerBody.innerHTML = details.map((row) => `
      <section class="detail-card">
        <h3>buffer=${{escapeHtml(row.call_arg)}} fx=${{escapeHtml(row.fx_node)}} logical=${{escapeHtml(row.logical_buffer)}}</h3>
        <div class="detail-meta">
          status=${{escapeHtml(row.status)}} bad=${{escapeHtml(row.bad_element_count)}}<br>
          max_abs=${{escapeHtml(fmtNumber(row.max_abs_diff))}} mean_abs=${{escapeHtml(fmtNumber(row.mean_abs_diff))}}<br>
          max_rel=${{escapeHtml(fmtNumber(row.max_rel_abs_diff))}} mean_rel=${{escapeHtml(fmtNumber(row.mean_rel_abs_diff))}}
        </div>
        ${{renderTopErrorTable(row.top_error_elements || [])}}
      </section>
    `).join('');
  }}
  detailDrawer.classList.add('open');
  detailDrawer.setAttribute('aria-hidden', 'false');
}}
function closeKernelDetails() {{
  detailDrawer.classList.remove('open');
  detailDrawer.setAttribute('aria-hidden', 'true');
}}
drawerClose.addEventListener('click', closeKernelDetails);
function clearHighlights() {{
  document.body.classList.remove('dimmed');
  document.querySelectorAll('.fx-node.highlight').forEach(n => n.classList.remove('highlight'));
  document.querySelectorAll('.kernel-node.active').forEach(n => n.classList.remove('active'));
  hoverInfo.textContent = 'Hover a kernel node to highlight corresponding FX nodes.';
}}
function highlightFx(kernelNode) {{
  clearHighlights();
  kernelNode.classList.add('active');
  let fxNodes = [];
  try {{
    fxNodes = JSON.parse(kernelNode.dataset.fxNodes || '[]');
  }} catch (e) {{
    fxNodes = [];
  }}
  if (!fxNodes.length) {{
    hoverInfo.textContent = 'No FX node mapping recorded for ' + kernelNode.textContent.trim();
    return;
  }}
  document.body.classList.add('dimmed');
  let first = null;
  for (const name of fxNodes) {{
    // fx_nodes may carry a slice annotation (e.g. "cat[0:64]"); highlight the
    // base node ("cat") but display the annotated label.
    const baseName = String(name).replace(/\\[[^\\]]*\\]$/, '');
    const id = 'fx-' + baseName.replace(/[^A-Za-z0-9_\\-:.]/g, '_');
    const node = document.getElementById(id);
    if (!node) continue;
    node.classList.add('highlight');
    if (!first) first = node;
  }}
  hoverInfo.textContent = 'Kernel maps to FX nodes: ' + fxNodes.join(', ');
  if (first) {{
    panZoom['fx-pane'].centerNode(first);
  }}
}}
document.querySelectorAll('.kernel-node').forEach(node => {{
  node.addEventListener('mouseenter', () => highlightFx(node));
  node.addEventListener('mouseleave', clearHighlights);
  node.addEventListener('click', (event) => {{
    event.stopPropagation();
    openKernelDetails(node);
  }});
}});
function setupPanZoom(paneId) {{
  const pane = document.getElementById(paneId);
  const canvas = pane.querySelector('.canvas');
  let scale = 1.0;
  let dragging = false;
  let startX = 0;
  let startY = 0;
  let startLeft = 0;
  let startTop = 0;
  function applyScale(nextScale, clientX, clientY) {{
    nextScale = Math.max(0.18, Math.min(2.5, nextScale));
    const beforeX = (pane.scrollLeft + clientX - pane.getBoundingClientRect().left) / scale;
    const beforeY = (pane.scrollTop + clientY - pane.getBoundingClientRect().top) / scale;
    scale = nextScale;
    canvas.style.transform = 'scale(' + scale + ')';
    canvas.style.width = (parseFloat(canvas.dataset.baseWidth) * scale) + 'px';
    canvas.style.height = (parseFloat(canvas.dataset.baseHeight) * scale) + 'px';
    pane.scrollLeft = beforeX * scale - (clientX - pane.getBoundingClientRect().left);
    pane.scrollTop = beforeY * scale - (clientY - pane.getBoundingClientRect().top);
  }}
  canvas.dataset.baseWidth = parseFloat(canvas.style.width);
  canvas.dataset.baseHeight = parseFloat(canvas.style.height);
  canvas.style.width = (parseFloat(canvas.dataset.baseWidth) * scale) + 'px';
  canvas.style.height = (parseFloat(canvas.dataset.baseHeight) * scale) + 'px';
  pane.addEventListener('wheel', (event) => {{
    if (!event.ctrlKey && !event.metaKey) return;
    event.preventDefault();
    const factor = event.deltaY < 0 ? 1.12 : 0.89;
    applyScale(scale * factor, event.clientX, event.clientY);
  }}, {{passive: false}});
  pane.addEventListener('mousedown', (event) => {{
    if (event.button !== 0) return;
    if (event.target.closest('.node') || event.target.closest('button')) return;
    dragging = true;
    pane.classList.add('dragging');
    startX = event.clientX;
    startY = event.clientY;
    startLeft = pane.scrollLeft;
    startTop = pane.scrollTop;
    event.preventDefault();
  }});
  window.addEventListener('mousemove', (event) => {{
    if (!dragging) return;
    pane.scrollLeft = startLeft - (event.clientX - startX);
    pane.scrollTop = startTop - (event.clientY - startY);
  }});
  window.addEventListener('mouseup', () => {{
    dragging = false;
    pane.classList.remove('dragging');
  }});
  return {{
    centerNode: (node) => {{
      const targetX = (node.offsetLeft + node.offsetWidth / 2) * scale;
      const targetY = (node.offsetTop + node.offsetHeight / 2) * scale;
      pane.scrollTo({{
        left: Math.max(0, targetX - pane.clientWidth / 2),
        top: Math.max(0, targetY - pane.clientHeight / 2),
        behavior: 'smooth',
      }});
    }},
    zoomIn: () => applyScale(scale * 1.18, pane.getBoundingClientRect().left + pane.clientWidth / 2, pane.getBoundingClientRect().top + pane.clientHeight / 2),
    zoomOut: () => applyScale(scale / 1.18, pane.getBoundingClientRect().left + pane.clientWidth / 2, pane.getBoundingClientRect().top + pane.clientHeight / 2),
    reset: () => {{
      scale = 1.0;
      canvas.style.transform = 'scale(1)';
      canvas.style.width = canvas.dataset.baseWidth + 'px';
      canvas.style.height = canvas.dataset.baseHeight + 'px';
      pane.scrollLeft = 0;
      pane.scrollTop = 0;
    }},
  }};
}}
const panZoom = {{
  'fx-pane': setupPanZoom('fx-pane'),
  'kernel-pane': setupPanZoom('kernel-pane'),
}};
document.querySelectorAll('.tool-button').forEach(button => {{
  button.addEventListener('click', () => {{
    const ctl = panZoom[button.dataset.pane];
    if (!ctl) return;
    if (button.dataset.action === 'zoom-in') ctl.zoomIn();
    if (button.dataset.action === 'zoom-out') ctl.zoomOut();
    if (button.dataset.action === 'reset') ctl.reset();
  }});
}});
</script>
</body>
</html>
"""


def write_graph_html(
    path: Path,
    trace_dir: Path,
    fx_path: Path,
    output_path: Path,
    buffer_to_fx: Dict[str, List[Dict[str, Any]]],
    kernels: List[Dict[str, Any]],
    comparisons: List[Dict[str, Any]],
) -> None:
    fx_nodes, fx_edges = parse_fx_graph(fx_path)
    kernels = enrich_kernels_from_output_code(kernels, output_path)
    call_stats = compare_stats_by_call_id(comparisons)
    call_details = compare_details_by_call_id(comparisons)
    fx_stats = compare_stats_by_fx_node(comparisons)
    aliases = _parse_alias_map(output_path)
    slice_map = _parse_alias_slices(output_path)
    kernel_nodes, kernel_edges = build_kernel_call_graph(
        kernels,
        buffer_to_fx,
        call_stats,
        call_details,
        aliases,
        slice_map,
        _build_cat_operand_map(fx_path, buffer_to_fx, slice_map),
    )
    fx_layout = layout_graph(
        fx_nodes,
        fx_edges,
        node_w=160,
        node_h=58,
        col_gap=28,
        row_gap=40,
    )
    fx_layout = add_layout_slack(fx_layout, right=900, bottom=900)
    kernel_layout = layout_kernel_call_graph(
        kernel_nodes,
        kernel_edges,
        node_w=280,
        node_h=78,
        col_gap=36,
        row_gap=22,
    )
    path.write_text(
        _graph_html(
            trace_dir=trace_dir,
            fx_path=fx_path,
            output_path=output_path,
            fx_nodes=fx_nodes,
            fx_edges=fx_edges,
            kernel_nodes=kernel_nodes,
            kernel_edges=kernel_edges,
            fx_layout=fx_layout,
            kernel_layout=kernel_layout,
            fx_compare_stats=fx_stats,
        ),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=APP_NAME,
        description=(
            "Replay an FX debug repro and generated output_code.py, then compare the traced logical-buffer values."
        ),
        epilog=(
            "examples:\n"
            "  %(prog)s /path/to/torch_compile_debug/.../model__0_forward_1.0\n"
            "  %(prog)s --fx fx_graph_runnable.py --output output_code.py\n"
            "  %(prog)s TRACE_DIR --skip-run --fail-on-mismatch"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {APP_VERSION}",
    )

    source = parser.add_argument_group("input selection")
    source.add_argument(
        "directory",
        nargs="?",
        type=Path,
        help=(
            "directory containing fx_graph*_runnable.py and output_code.py; "
            "files are discovered recursively when necessary"
        ),
    )
    source.add_argument("--fx", type=Path, metavar="PATH", help="explicit FX runnable path")
    source.add_argument(
        "--output",
        type=Path,
        metavar="PATH",
        help="explicit output_code.py path",
    )
    source.add_argument(
        "--trace-dir",
        type=Path,
        metavar="DIR",
        default=None,
        help="trace metadata, tensor dumps, and report output directory",
    )
    source.add_argument(
        "--save-dir",
        metavar="DIR",
        default=None,
        help="saved replay-input directory used by fx_graph_runnable.load_args",
    )

    comparison = parser.add_argument_group("comparison")
    comparison.add_argument(
        "--rtol",
        type=float,
        default=1e-2,
        help="relative tolerance used by torch.isclose (default: %(default)g)",
    )
    comparison.add_argument(
        "--atol",
        type=float,
        default=1e-2,
        help="absolute tolerance used by torch.isclose (default: %(default)g)",
    )
    comparison.add_argument(
        "--max-elements",
        type=int,
        default=1048576,
        metavar="N",
        help="maximum tensor elements captured by tracing hooks (default: %(default)s)",
    )
    comparison.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="exit with status 2 when any comparison is non-OK",
    )

    execution = parser.add_argument_group("execution")
    execution.add_argument(
        "--skip-run",
        action="store_true",
        help="skip FX/output replay and compare existing tensor dumps only",
    )
    execution.add_argument(
        "--no-prefer-transformed",
        action="store_true",
        help="do not prefer fx_graph_transformed_runnable.py when available",
    )
    execution.add_argument(
        "--shape-override",
        action="append",
        default=[],
        metavar="sN=VAL",
        help="replay at a different dynamic shape, e.g. --shape-override s77=4 "
        "(repeatable); updates the shape scalar arg and resizes inputs",
    )
    execution.add_argument(
        "--quiet",
        action="store_true",
        help="suppress normal progress and summary output",
    )
    execution.add_argument(
        "--no-color",
        action="store_true",
        help="disable ANSI colors even when stdout is a terminal",
    )
    execution.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="print a Python traceback when execution fails",
    )
    return parser


def _shape_symbol_args(output_path: Path) -> Dict[str, int]:
    """Parse output_code.py `sN = argM_1` lines -> {symbol: arg_index}.

    Inductor passes dynamic-shape values to generated code as scalar args,
    e.g. ``s77 = arg2_1``; this maps the symbol back to the args slot.
    """
    result: Dict[str, int] = {}
    try:
        for line in output_path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\s*(s\d+)\s*=\s*(arg\d+_1)", line)
            if m:
                try:
                    result[m.group(1)] = int(m.group(2)[3:-2])
                except ValueError:
                    pass
    except Exception:  # nosec B110
        pass
    return result


def _symbol_tensor_dims(trace_dir: Path) -> Dict[str, List[Tuple[int, int]]]:
    """Map a dynamic-shape symbol (sN) to (arg_index, dim) of the input tensors.

    From lowering_fx_buffer_map: buffer="arg4_1", size=["s77","3","s53","s53"]
    means s77 is dim 0 of args[4] and s53 is dims 2/3.
    """
    result: Dict[str, List[Tuple[int, int]]] = {}
    for row in read_jsonl(trace_dir / "lowering_fx_buffer_map.jsonl"):
        buf = str(row.get("buffer", ""))
        size = row.get("size") or []
        if not (buf.startswith("arg") and buf.endswith("_1")):
            continue
        try:
            arg_idx = int(buf[3:-2])
        except ValueError:
            continue
        for dim, item in enumerate(size):
            if re.fullmatch(r"s\d+", str(item)):
                result.setdefault(str(item), []).append((arg_idx, dim))
    return result


def apply_shape_override(
    base_args,
    output_path: Path,
    trace_dir: Path,
    overrides: List[str],
):
    """Replay at a different dynamic shape.

    ``overrides`` look like "s77=4".  Dynamic shapes reach output_code as
    scalar args (``s77 = arg2_1``), so we update the scalar arg and resize the
    input tensors whose symbolic dim equals the symbol.  Tensor values are
    re-initialised with random data of the new size -- both the FX replay and
    the kernel replay consume the same args, so the comparison stays valid.
    """
    if not overrides:
        return base_args
    shape_args = _shape_symbol_args(output_path)
    symbol_dims = _symbol_tensor_dims(trace_dir)
    base_args = list(base_args)
    for spec in overrides:
        name, _, val_s = spec.partition("=")
        try:
            val = int(val_s)
        except ValueError:
            print(f"  [WARN ] ignoring bad shape override: {spec!r}")
            continue
        if name in shape_args:
            base_args[shape_args[name]] = val
        for arg_idx, dim in symbol_dims.get(name, []):
            if arg_idx >= len(base_args):
                continue
            t = base_args[arg_idx]
            if isinstance(t, torch.Tensor):
                shp = list(t.shape)
                shp[dim] = val
                base_args[arg_idx] = torch.randn(*shp, device=t.device, dtype=t.dtype)
    return base_args


def run(args: argparse.Namespace, parser: argparse.ArgumentParser, console: Console) -> int:
    started = time.perf_counter()
    prefer_transformed = not args.no_prefer_transformed

    with console.step("Resolve input artifacts"):
        if args.directory is not None:
            dir_fx, dir_output = resolve_paths_from_dir(
                args.directory,
                prefer_transformed,
            )
            fx_path = args.fx.resolve() if args.fx is not None else dir_fx
            output_path = args.output.resolve() if args.output is not None else dir_output
        else:
            if args.fx is None or args.output is None:
                parser.error("provide either DIRECTORY or both --fx and --output")
            fx_path = resolve_fx_replay_path(args.fx, prefer_transformed)
            output_path = args.output.resolve()

        fx_path = _require_file(fx_path, "FX runnable")
        output_path = _require_file(output_path, "output_code.py")

    if args.trace_dir is not None:
        trace_dir = args.trace_dir.resolve()
    elif args.directory is not None:
        trace_dir = args.directory.resolve() / "logic_buffer_run"
    else:
        trace_dir = DEFAULT_TRACE_DIR.resolve()
    trace_dir.mkdir(parents=True, exist_ok=True)

    save_dir = args.save_dir if args.save_dir is not None else str(trace_dir)
    os.environ["INDUCTOR_LOGIC_BUFFER_TRACE"] = "1"
    os.environ["INDUCTOR_LOGIC_BUFFER_DIR"] = str(trace_dir)
    os.environ["INDUCTOR_LOGIC_BUFFER_MAX_ELEMS"] = str(args.max_elements)

    console.section("Configuration")
    console.key_value("FX replay", fx_path)
    console.key_value("Generated code", output_path)
    console.key_value("Trace directory", trace_dir)
    console.key_value("Replay inputs", save_dir)
    console.key_value("Tolerance", f"rtol={args.rtol:g}, atol={args.atol:g}")
    console.key_value("Capture limit", f"{args.max_elements:,} elements")
    console.key_value("Replay mode", "existing dumps" if args.skip_run else "FX + generated code")

    console.section("Execution")
    with console.step("Load lowering and kernel metadata"):
        buffer_to_fx, kernels = build_maps(trace_dir)
        nodes = wanted_fx_nodes(buffer_to_fx, kernels)

    console.item(
        "INFO",
        "Metadata",
        f"{len(buffer_to_fx)} logical buffers, {len(kernels)} kernel calls, {len(nodes)} FX nodes selected",
    )
    if not nodes:
        console.item(
            "WARN",
            "No FX nodes selected",
            "check the trace directory and ensure compilation used INDUCTOR_LOGIC_BUFFER_TRACE=1",
        )

    if args.skip_run:
        console.item("SKIP", "Replay", "using existing fx/kernel tensor dumps")
    else:
        with console.step("Prepare replay inputs"):
            clean_runtime_outputs(trace_dir)
            fx_for_inputs = import_file(fx_path, "logic_buffer_fx_inputs")
            base_args = get_args_from_fx_module(fx_for_inputs, save_dir, fx_path)

        console.item("INFO", "Replay inputs", f"{len(base_args)} argument(s) loaded")
        if args.shape_override:
            base_args = apply_shape_override(base_args, output_path, trace_dir, args.shape_override)
            console.item(
                "INFO",
                "Shape override",
                "; ".join(args.shape_override),
            )

        with console.step("Replay instrumented FX graph"):
            run_fx(fx_path, trace_dir, nodes, base_args, save_dir)

        with console.step("Replay generated Inductor code"):
            run_output(output_path, base_args)

    with console.step("Compare traced tensors"):
        comparisons = build_comparisons(
            trace_dir,
            buffer_to_fx,
            args.rtol,
            args.atol,
        )
        localization = mark_first_triton_value_mismatch(comparisons, kernels)

    json_report = trace_dir / "compare_report.json"
    text_report = trace_dir / "compare_report.txt"
    csv_report = trace_dir / "compare_report.csv"
    diagnostics_report = trace_dir / "missing_fx_value_diagnostics.json"
    graph_html_path = trace_dir / "logic_buffer_graph.html"

    with console.step("Generate reports and graph"):
        write_json(json_report, comparisons)
        write_text_report(text_report, comparisons, localization)
        write_csv_report(csv_report, comparisons)
        missing_diag_count = write_missing_fx_diagnostics(
            diagnostics_report,
            trace_dir,
            fx_path,
            comparisons,
        )
        write_graph_html(
            graph_html_path,
            trace_dir,
            fx_path,
            output_path,
            buffer_to_fx,
            kernels,
            comparisons,
        )

    counts = _status_counts(comparisons)
    ok = counts.get("ok", 0)
    non_ok = len(comparisons) - ok

    console.section("Comparison summary")
    console.key_value("Total", len(comparisons))
    console.key_value("OK", ok)
    console.key_value("Non-OK", non_ok)
    for status, count in counts.items():
        if status == "ok":
            continue
        console.item("WARN", status, str(count))

    if counts.get("value_mismatch", 0):
        console.section("Error localization")
        if localization is None:
            console.item(
                "WARN",
                "First Triton mismatch",
                "value mismatches were found, but none could be mapped to Triton kernel metadata",
            )
        else:
            console.item(
                "FAIL",
                "First Triton mismatch",
                str(localization.get("kernel_name") or "<unknown kernel>"),
            )
            console.key_value("Call ID", localization.get("call_id"))
            console.key_value("FX node", localization.get("fx_node"))
            console.key_value("Logical buffer", localization.get("logical_buffer"))
            console.key_value("Call argument", localization.get("call_arg"))
            console.key_value("Bad elements", localization.get("bad_element_count"))
            console.key_value("Max abs diff", localization.get("max_abs_diff"))
            console.key_value("Mean rel diff", localization.get("mean_rel_abs_diff"))

    console.section("Artifacts")
    console.item("OK", "JSON report", str(json_report))
    console.item("OK", "Text report", str(text_report))
    console.item("OK", "CSV report", str(csv_report))
    console.item("OK", "Interactive graph", str(graph_html_path))
    if missing_diag_count:
        console.item(
            "WARN",
            "Missing-FX diagnostics",
            f"{diagnostics_report} ({missing_diag_count} unresolved entry/entries)",
        )

    elapsed = time.perf_counter() - started
    console.section("Result")
    if not comparisons:
        console.item(
            "WARN",
            "No comparisons were produced",
            f"completed in {_format_duration(elapsed)}",
        )
    elif non_ok == 0:
        console.item(
            "OK",
            "All comparisons passed",
            f"{ok}/{len(comparisons)} in {_format_duration(elapsed)}",
        )
    else:
        console.item(
            "WARN",
            "Comparison completed with non-OK results",
            f"{non_ok}/{len(comparisons)} non-OK in {_format_duration(elapsed)}",
        )

    if args.fail_on_mismatch and non_ok:
        return 2
    return 0


def main() -> int:
    try:
        importlib.import_module("torch._inductor.logic_buffer_trace")
    except ImportError:
        try:
            from .logic_monkeypatch import _load_helper
        except ImportError:
            from logic_monkeypatch import _load_helper
        _load_helper()

    parser = build_parser()
    args = parser.parse_args()

    if args.directory is None and (args.fx is None or args.output is None):
        parser.error("provide either DIRECTORY or both --fx and --output")
    if args.rtol < 0 or args.atol < 0:
        parser.error("--rtol and --atol must be non-negative")
    if args.max_elements <= 0:
        parser.error("--max-elements must be greater than zero")

    console = Console(
        color=_color_enabled(args.no_color),
        quiet=args.quiet,
    )
    console.header()

    try:
        return run(args, parser, console)
    except KeyboardInterrupt:
        console.item(
            "FAIL",
            "Interrupted",
            "operation cancelled by user",
            stderr=True,
            force=True,
        )
        return 130
    except Exception as exc:
        console.item(
            "FAIL",
            "Execution failed",
            str(exc),
            stderr=True,
            force=True,
        )
        if args.verbose:
            traceback.print_exc()
        else:
            print(
                "  hint: rerun with --verbose for a full traceback",
                file=sys.stderr,
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
