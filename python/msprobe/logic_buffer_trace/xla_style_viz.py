#!/usr/bin/env python3
"""Render the logic-buffer FX graph in the style of XLA's hlo_graph_dumper.

Emits a DOT graph (topology only — node colors/shapes/labels and edges), then
delegates layout to the Graphviz dot engine exactly like XLA does
(rankdir=TB + graphviz.layout(dot, "svg", "dot")).  No coordinates are computed
here; leaf placement is Graphviz's global edge-length/crossing optimization.

Colors/scheme copied from xla/service/hlo_graph_dumper.cc (ColorScheme /
NodeColorsForScheme / GetInstructionColor), material palette.
"""
# pylint: disable=duplicate-code

from __future__ import annotations

import argparse
import ast
import os
import shutil
import subprocess  # nosec B404
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

# --- XLA ColorScheme -> (fill, stroke, font), from hlo_graph_dumper.cc:193 ---
XLA_COLORS = {
    "blue": ("#bbdefb", "#8aacc8", "black"),
    "brown": ("#bcaaa4", "#8c7b75", "black"),
    "darkblue": ("#1565c0", "#003c8f", "white"),
    "darkgreen": ("#2e7d32", "#005005", "white"),
    "darkorange": ("#ffb74d", "#c88719", "black"),
    "darkred": ("#b71c1c", "#7f0000", "white"),
    "gray": ("#cfd8dc", "#9ea7aa", "black"),
    "green": ("#c8e6c9", "#97b498", "black"),
    "orange": ("#ffe0b2", "#cbae82", "black"),
    "purple": ("#e1bee7", "#af8eb5", "black"),
    "red": ("#ffcdd2", "#cb9ca1", "black"),
    "white": ("white", "#9e9e9e", "black"),
    "yellow": ("#fff9c4", "#cbc693", "black"),
}

# torch op name -> XLA ColorScheme, mirroring GetInstructionColor's categories:
#   white  = elementwise;  green = data movement;  darkblue = dot/conv;
#   purple = reduction / batch-norm;  yellow = broadcast;  gray = trivial.
WHITE_OPS = {
    "add",
    "mul",
    "sub",
    "div",
    "sqrt",
    "rsqrt",
    "reciprocal",
    "exp",
    "log",
    "log2",
    "log10",
    "tanh",
    "sigmoid",
    "neg",
    "abs",
    "clamp",
    "maximum",
    "minimum",
    "leaky_relu",
    "softmax",
    "pow",
    "gelu",
    "silu",
    "erf",
    "square",
    "hardtanh",
    "selu",
    "hardsigmoid",
    "hardswish",
    "relu6",
    "relu",
    "dropout",
    "clone",
    "gt",
    "lt",
    "ge",
    "le",
    "eq",
    "ne",
    "to",
    "type_as",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sign",
    "ceil",
    "floor",
    "round",
    "trunc",
    "frac",
    "copysign",
    "convert_element_type",
    "item",
    "is_non_overlapping_and_dense",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "logical_and",
    "logical_or",
    "logical_not",
    "where",
    "remainder",
    "fmod",
    "floor_divide",
    "true_divide",
    "rsub",
    "prelu",
    "addcmul",
}
GREEN_OPS = {
    "view",
    "reshape",
    "transpose",
    "permute",
    "flatten",
    "unflatten",
    "unsqueeze",
    "squeeze",
    "cat",
    "concat",
    "split",
    "chunk",
    "slice",
    "narrow",
    "repeat",
    "expand",
    "roll",
    "flip",
    "rot90",
    "t",
    "moveaxis",
    "swapaxes",
    "stack",
    "select",
    "getitem",
    "unbind",
    "slice_scatter",
    "select_scatter",
    "copy",
    "alias",
    "detach",
}
DARKBLUE_OPS = {
    "convolution",
    "conv1d",
    "conv2d",
    "conv3d",
    "_convolution",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    "mm",
    "matmul",
    "addmm",
    "addmv",
    "bmm",
    "baddbmm",
    "linear",
    "dot",
}
PURPLE_OPS = {
    "mean",
    "sum",
    "amax",
    "amin",
    "max",
    "min",
    "prod",
    "argmax",
    "argmin",
    "std",
    "var",
    "norm",
    "logsumexp",
    "all",
    "any",
    "cumsum",
    "cumprod",
    "count_nonzero",
    "reduce",
    "native_batch_norm",
    "_native_batch_norm_legit",
    "_native_batch_norm_legit_no_training",
    "batch_norm",
    "layer_norm",
    "group_norm",
    "instance_norm",
    "max_pool2d_with_indices",
    "max_pool1d",
    "avg_pool2d",
    "avg_pool1d",
    "adaptive_avg_pool2d",
    "max_pool2d",
    "max_unpool2d",
    "scatter",
    "gather",
    "index_select",
}
YELLOW_OPS = {
    "broadcast_to",
    "expand_as",
    "repeat_interleave",
    "as_strided",
    "constant_pad_nd",
    "upsample_bilinear2d",
    "upsample_nearest2d",
    "pad",
    "reflection_pad2d",
    "replication_pad2d",
    "broadcast",
}


def opname(op: str) -> str:
    """torch.ops.aten.convolution.default -> convolution ; getitem -> getitem."""
    if op.startswith("torch.ops."):
        parts = op.split(".")
        if len(parts) >= 4:
            return parts[3]
    return op


def op_color(op: str) -> str:
    name = opname(op)
    if name in WHITE_OPS:
        return "white"
    if name in GREEN_OPS:
        return "green"
    if name in DARKBLUE_OPS:
        return "darkblue"
    if name in PURPLE_OPS:
        return "purple"
    if name in YELLOW_OPS:
        return "yellow"
    if name in ("clone", "alias", "detach", "copy_", "input"):
        return "gray"
    # unknown opcode: XLA falls through to the default kWhite
    return "white"


def html_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _operands_in_order(stmt: ast.Assign) -> List[str]:
    """Names loaded by the RHS, in source order (gives operand indices)."""
    result: List[str] = []
    seen = set()
    for node in ast.walk(stmt.value):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in seen:
                seen.add(node.id)
                result.append(node.id)
    return result


def parse_fx(fx_path: Path) -> Tuple[List[Dict], List[Tuple[str, str, int]]]:
    """Parse a fx_graph*_runnable.py forward body into nodes and ordered edges.

    Returns (nodes, edges) where each node is
      {id, kind(placeholder|op), op, expr}
    and each edge is (operand_id, target_id, operand_index).
    """
    text = fx_path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    forward = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "forward":
            forward = node
            break
    if forward is None:
        raise RuntimeError("no forward() found in " + str(fx_path))

    nodes: List[Dict] = []
    edges: List[Tuple[str, str, int]] = []
    known = set()
    # dynamic-shape scalar args (reader.symint) are not tensors; skip them
    import re as _re

    symint = set()
    for _ln in text.splitlines():
        if "reader.symint" in _ln:
            _m = _re.search(r"#\s*(arg\d+_1)", _ln)
            if _m:
                symint.add(_m.group(1))
    for arg in forward.args.args:
        if arg.arg == "self" or arg.arg in symint:
            continue
        nodes.append({"id": arg.arg, "kind": "placeholder", "op": "", "expr": ""})
        known.add(arg.arg)

    for stmt in forward.body:
        if not isinstance(stmt, ast.Assign):
            continue
        # skip `; arg = None` memory-free statements and storage allocations
        if isinstance(stmt.value, ast.Constant) and stmt.value.value is None:
            continue
        targets = [t for t in stmt.targets if isinstance(t, ast.Name)]
        if not targets:
            continue
        target = targets[0].id
        if target.startswith("_logic_buffer_"):
            continue
        if target in known:  # re-assignment -> keep the first (the real op)
            continue
        if isinstance(stmt.value, ast.Call):
            op = ast.unparse(stmt.value.func)
        else:
            op = type(stmt.value).__name__
        operands = [o for o in _operands_in_order(stmt) if o in known]
        nodes.append(
            {
                "id": target,
                "kind": "op",
                "op": op,
                "expr": ast.unparse(stmt.value),
            }
        )
        for idx, operand in enumerate(operands):
            edges.append((operand, target, idx))
        known.add(target)

    return nodes, edges


def build_dot(fx_path: Path, title: str = "FX Graph (XLA style)", fx_stats: Dict[str, Dict] | None = None) -> str:
    nodes, edges = parse_fx(fx_path)
    node_ids = {n["id"] for n in nodes}
    edges = [(s, t, i) for s, t, i in edges if s in node_ids and t in node_ids]

    # find output node(s): nodes with no outgoing edges (sinks of the DAG)
    has_out = {s for s, _, _ in edges}
    all_nodes = {n["id"] for n in nodes}
    roots = sorted(all_nodes - has_out)

    out: List[str] = []
    out.append("digraph G {")
    out.append("rankdir = TB;")
    out.append("compound = true;")
    out.append("labelloc = t;")
    out.append(f"label = <<b>{html_escape(title)}</b>>;")

    out.append("  // ---- nodes ----")
    for n in nodes:
        nid = n["id"]
        if n["kind"] == "placeholder":
            label = f"<<b>{html_escape(nid)}</b><br/>(Parameter)>"
            color = "orange"
        else:
            op = opname(n["op"])
            label = f"<<b>{html_escape(nid)}</b><br/>{html_escape(op)}>"
            color = op_color(n["op"])
        fill, stroke, font = XLA_COLORS[color]
        tooltip = html_escape(n.get("expr", "") or "")
        if fx_stats is not None:
            stat = fx_stats.get(nid)
            if stat and int(stat.get("bad_element_sum", 0) or 0) > 0:
                fill, stroke, font = XLA_COLORS["red"]
                tooltip += "  [bad elem sum=%s]" % stat.get("bad_element_sum")
        out.append(
            f'  "{nid}" [label={label}, shape=box, '
            f'style="filled", fillcolor="{fill}", color="{stroke}", '
            f'fontcolor="{font}", tooltip="{tooltip}"];'
        )

    out.append("  // ---- root (graph output) ----")
    for rid in roots:
        out.append(
            f'  "root_{rid}" [label=ROOT, shape=circle, style="filled", '
            f'fillcolor="#f5f5f5", color="#9e9e9e", fontcolor="black"];'
        )
        out.append(f'  "{rid}" -> "root_{rid}";')

    out.append("  // ---- edges ----")
    # group by target to know operand count for headlabels
    from collections import defaultdict

    by_target = defaultdict(list)
    for s, t, i in edges:
        by_target[t].append((s, i))
    for s, t, i in edges:
        label_attr = ""
        if len(by_target[t]) > 1:
            label_attr = f' headlabel="{i}", labeldistance=2'
        out.append(f'  "{s}" -> "{t}" [{label_attr.strip()}] ;')

    out.append("}")
    return "\n".join(out)


def render_svg(dot_text: str, node_script: Path) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".dot", delete=False) as f:
        f.write(dot_text)
        dot_path = f.name
    try:
        node = shutil.which("node")
        if node is None:
            raise RuntimeError("node executable was not found")
        # The executable is resolved explicitly and subprocess never invokes a shell.
        proc = subprocess.run(  # nosec B603
            [node, str(node_script), dot_path],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"node render failed:\n{proc.stderr}")
        return proc.stdout
    finally:
        os.unlink(dot_path)


HTML_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Logic Buffer FX Graph (XLA style)</title>
<style>
html, body {{ margin: 0; background: #f6f8fb; color: #172033; }}
.header {{ padding: 12px 16px; background: rgba(255,255,255,.96); border-bottom: 1px solid #d7dde8; }}
.header h1 {{ margin: 0; font-size: 18px; }}
.meta {{ font-size: 12px; color: #5f6f89; word-break: break-all; }}
.pane {{ margin: 12px; padding: 8px; background: #fff; border: 1px solid #d7dde8; border-radius: 8px; }}
svg {{ display: block; margin: 0 auto; }}
/* XLA-style hover: highlight a node and its incident edges */
svg g.node {{ cursor: pointer; }}
svg g.node:hover > ellipse, svg g.node:hover > polygon,
svg g.node:hover > rect {{ stroke-width: 3; }}
svg g.node:hover ~ g.edge path {{ stroke: #d32f2f; stroke-width: 2.2; }}
svg g.node:hover ~ g.edge polygon {{ fill: #d32f2f; stroke: #d32f2f; }}
</style></head>
<body>
<div class="header">
  <h1>Logic Buffer FX Graph — XLA/hlo_graph_dumper style</h1>
  <div class="meta">trace: {trace_dir}</div>
  <div class="meta">fx: {fx_path}</div>
</div>
<div class="pane">
{svg}
</div>
</body></html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument(
        "--fx", type=Path, default=None, help="explicit fx_graph*_runnable.py (default: auto in model_dir)"
    )
    parser.add_argument(
        "--node-script",
        type=Path,
        default=Path(tempfile.gettempdir()) / "viz_render" / "render_svg.js",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output HTML path (default: <model_dir>/logic_buffer_run/logic_buffer_graph_xla.html)",
    )
    parser.add_argument("--dot-only", action="store_true", help="write the DOT text and exit (no graphviz render)")
    args = parser.parse_args()

    model_dir = args.model_dir
    if args.fx is not None:
        fx_path = args.fx
    else:
        cand = model_dir / "fx_graph_transformed_runnable.py"
        if not cand.exists():
            cand = model_dir / "fx_graph_runnable.py"
        fx_path = cand
    if not fx_path.exists():
        parser.error(f"FX runnable not found: {fx_path}")

    trace_dir = model_dir / "logic_buffer_run"
    out_path = args.out or (trace_dir / "logic_buffer_graph_xla.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dot = build_dot(fx_path)

    dot_path = out_path.with_suffix(".dot")
    dot_path.write_text(dot, encoding="utf-8")
    print(f"DOT written to {dot_path}")

    if args.dot_only:
        return 0

    svg = render_svg(dot, args.node_script)
    html = HTML_TEMPLATE.format(
        trace_dir=html_escape(str(trace_dir)),
        fx_path=html_escape(str(fx_path)),
        svg=svg,
    )
    out_path.write_text(html, encoding="utf-8")
    print(f"HTML written to {out_path} (svg {len(svg)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
