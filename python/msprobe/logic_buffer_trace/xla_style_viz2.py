#!/usr/bin/env python3
"""Two-pane logic-buffer visualization, both panes rendered XLA/graphviz style.

Left  pane = FX graph,  right pane = kernel call graph.  Both are emitted as
DOT and laid out by the Graphviz dot engine (@hpcc-js/wasm), exactly like XLA's
hlo_graph_dumper.  The interactive framework (pan/zoom, hover a kernel to
highlight its FX nodes, click a kernel for per-buffer error details) is carried
over from the original hand-coded layout version.
"""
# pylint: disable=duplicate-code

from __future__ import annotations

import argparse
import html as _html
import importlib.util
import json
import os
import re
import shutil
import subprocess  # nosec B404
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

try:
    from . import run_compare as rc
    from .xla_style_viz import XLA_COLORS, build_dot, html_escape, parse_fx
except ImportError:  # Allow direct execution of this file during migration.
    import run_compare as rc
    from xla_style_viz import XLA_COLORS, build_dot, html_escape, parse_fx

NODE_SCRIPT = Path(tempfile.gettempdir()) / "viz_render" / "render_svg.js"


def render_svg(dot_text: str, node_script: Path = NODE_SCRIPT) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".dot", delete=False) as f:
        f.write(dot_text)
        dot_path = f.name
    try:
        node = shutil.which("node")
        if node is None:
            raise RuntimeError("node executable was not found")
        # The executable is resolved explicitly and subprocess never invokes a shell.
        proc = subprocess.run(  # nosec B603
            [node, str(node_script), dot_path], capture_output=True, text=True, timeout=180, check=False
        )
        if proc.returncode != 0:
            raise RuntimeError("node render failed:\n" + proc.stderr)
        return proc.stdout
    finally:
        os.unlink(dot_path)


def build_kernel_dot(kernel_nodes: List[Dict[str, Any]], kernel_edges: List[Tuple[str, str]]) -> str:
    out = [
        "digraph G {",
        "rankdir = TB;",
        "compound = true;",
        "labelloc = t;",
        "label = <<b>Kernel Call Graph (XLA style)</b>>;",
    ]
    for kn in kernel_nodes:
        nid = kn["id"]
        name = str(kn.get("kernel_name", ""))
        call_id = kn.get("call_id")
        if "convolution" in name or "addmm" in name or "mm" in name or name.startswith("extern_kernels"):
            color = "darkblue"
        elif "triton" in name:
            color = "purple"
        else:
            color = "white"
        fill, stroke, font = XLA_COLORS[color]
        short = name.replace("torch.ops.aten.", "").replace(".default", "").replace("extern_kernels.", "")
        label = f"<<b>#{call_id}</b><br/>{html_escape(short)}>"
        kind = str(kn.get("kind", ""))
        tooltip = html_escape(f"#{call_id} {name}  ({kind})")
        out.append(
            f'  "{nid}" [label={label}, shape=box, style="filled", '
            f'fillcolor="{fill}", color="{stroke}", fontcolor="{font}", '
            f'tooltip="{tooltip}"];'
        )
    for s, t in kernel_edges:
        out.append(f'  "{s}" -> "{t}";')
    # invisible sequence edges: chain kernels by call order so graphviz lays
    # them out as one vertical timeline (like the original hand-coded pane),
    # even when a kernel has no shared logical buffer with its neighbours.
    by_call = {kn.get("call_id"): kn["id"] for kn in kernel_nodes if kn.get("call_id") is not None}
    seq = sorted(by_call)
    for a, b in zip(seq, seq[1:]):
        out.append(f'  "{by_call[a]}" -> "{by_call[b]}" [style="invis"];')
    out.append("}")
    return "\n".join(out)


def sanitize_id(name: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.:-]", "_", str(name))


def _annotate_node(svg: str, check: Any, make_open: Any) -> str:
    """Rewrite only the node opening tag, leaving the nested graphviz
    structure (the inner <g id=...a_nodeN> anchor wrapper and the matching
    closing </g>) untouched.  Matching the whole group with a non-greedy
    `.*?</g>` would cut off at the inner </g> and produce malformed SVG.
    """

    def repl(m: re.Match) -> str:
        name = _html.unescape(m.group(1))
        if not check(name):
            return m.group(0)
        return make_open(name) + "<title>" + m.group(1) + "</title>"

    return re.sub(r'<g id="node\d+" class="node">\s*<title>(.*?)</title>', repl, svg, flags=re.S)


def annotate_fx_svg(svg: str, node_ids: set) -> str:
    """Give every graphviz node a stable id + class for the JS."""
    return _annotate_node(
        svg,
        lambda name: name in node_ids,
        lambda name: (
            f'<g id="fx-{sanitize_id(name)}" class="node fx-node" data-fx-name="{_html.escape(name, quote=True)}">'
        ),
    )


def annotate_kernel_svg(svg: str, kernel_by_id: Dict[str, Dict]) -> str:
    def make_open(nid: str) -> str:
        kn = kernel_by_id.get(nid)
        if kn is None:
            return '<g id="node" class="node">'
        fx_json = json.dumps(kn.get("fx_nodes", []), ensure_ascii=False)
        det_json = json.dumps(kn.get("compare_details", []), ensure_ascii=False, default=str, allow_nan=False)
        return (
            f'<g id="kernel-{sanitize_id(nid)}" class="node kernel-node" '
            f'data-fx-nodes="{_html.escape(fx_json, quote=True)}" '
            f'data-compare-details="{_html.escape(det_json, quote=True)}" '
            f'data-call-id="{kn.get("call_id", "")}">'
        )

    return _annotate_node(svg, lambda nid: nid in kernel_by_id, make_open)


HTML_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Logic Buffer FX / Kernel Graph (XLA style)</title>
<style>
html, body {{ margin: 0; background: #f6f8fb; color: #172033; font-family: 'Google Sans', 'Segoe UI', sans-serif; }}
.header {{ position: sticky; top: 0; z-index: 5; background: rgba(255,255,255,.96);
  border-bottom: 1px solid #d7dde8; padding: 12px 18px; box-shadow: 0 2px 12px rgba(18,31,56,.08); }}
.header h1 {{ margin: 0 0 6px; font-size: 20px; }}
.meta {{ font-size: 12px; color: #5f6f89; word-break: break-all; }}
.graph-grid {{ display: grid; grid-template-columns: minmax(0,1fr) minmax(0,1fr); gap: 12px;
  padding: 12px; height: calc(100vh - 150px); min-height: 520px; box-sizing: border-box; }}
.pane {{ background:#fff; border:1px solid #d7dde8; border-radius:10px; overflow:auto;
  overscroll-behavior: contain; cursor: grab; box-shadow: 0 8px 28px rgba(18,31,56,.08); }}
.pane.dragging {{ cursor: grabbing; user-select: none; }}
.pane-title {{ position: sticky; top:0; z-index:3; background:#fff; border-bottom:1px solid #e1e6ef;
  padding:10px 12px; font-weight:700; }}
.hint {{ margin-left:8px; font-weight:400; color:#718096; font-size:12px; }}
.pane svg {{ display:block; }}
.toolbar {{ position: sticky; top: 37px; z-index: 4; display:flex; gap:6px; padding:6px 10px;
  background: rgba(255,255,255,.94); border-bottom:1px solid #edf1f7; }}
.tool-button {{ border:1px solid #cad3e2; border-radius:6px; background:#f8fafc; color:#334155;
  padding:3px 7px; font-size:11px; cursor:pointer; }}
.tool-button:hover {{ background:#edf2f7; }}
#hover-info {{ margin-top: 6px; font-size: 12px; color: #334155; }}
/* XLA-style node emphasis */
g.fx-node, g.kernel-node {{ cursor: pointer; }}
g.fx-node.highlight polygon, g.fx-node.highlight rect, g.fx-node.highlight ellipse,
g.kernel-node.active polygon, g.kernel-node.active rect, g.kernel-node.active ellipse {{
  stroke: #e11d48 !important; stroke-width: 4 !important;
  filter: drop-shadow(0 0 6px rgba(225,29,72,.9)); }}
/* dimming of unrelated nodes is applied via these JS-added classes */
g.fx-node.fx-dim, g.kernel-node.k-dim {{ opacity: 0.30; }}
.detail-drawer {{ position: fixed; top:0; right:0; z-index:20; width:min(720px,48vw); height:100vh;
  background:#fff; border-left:1px solid #cbd5e1; box-shadow:-18px 0 44px rgba(15,23,42,.20);
  transform: translateX(102%); transition: transform 180ms ease; display:flex; flex-direction:column; }}
.detail-drawer.open {{ transform: translateX(0); }}
.drawer-header {{ padding:14px 16px; border-bottom:1px solid #e2e8f0; display:flex; align-items:flex-start;
  justify-content:space-between; gap:12px; }}
.drawer-title {{ font-size:15px; font-weight:800; line-height:1.35; }}
.drawer-close {{ border:1px solid #cbd5e1; border-radius:6px; background:#f8fafc; cursor:pointer; padding:4px 8px; }}
.drawer-body {{ overflow:auto; padding:12px 16px 24px; font-size:12px; }}
.detail-card {{ border:1px solid #dbe3ee; border-radius:8px; padding:10px; margin-bottom:12px; background:#fbfdff; }}
.detail-card h3 {{ margin:0 0 6px; font-size:13px; }}
.detail-meta {{ color:#475569; line-height:1.45; margin-bottom:8px; }}
.detail-table {{ width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }}
.detail-table th, .detail-table td {{ border-top:1px solid #e2e8f0; padding:4px 5px; text-align:left; }}
.detail-table th {{ color:#475569; font-weight:700; }}
</style></head>
<body>
<div class="header">
  <h1>FX / Kernel Call Graph — XLA/hlo_graph_dumper style</h1>
  <div class="meta">trace_dir: {trace_dir}</div>
  <div class="meta">fx: {fx_path}   output: {output_path}</div>
  <div id="hover-info">Hover a kernel node to highlight corresponding FX nodes.</div>
</div>
<div class="graph-grid">
  <section id="fx-pane" class="pane">
    <div class="pane-title">FX Transformed Graph <span class="hint">Graphviz dot layout</span></div>
    <div class="toolbar"><button class="tool-button" data-pane="fx-pane" data-action="zoom-in">+</button>
      <button class="tool-button" data-pane="fx-pane" data-action="zoom-out">-</button>
      <button class="tool-button" data-pane="fx-pane" data-action="reset">reset</button></div>
    {fx_svg}
  </section>
  <section id="kernel-pane" class="pane">
    <div class="pane-title">Kernel Call Graph <span class="hint">Graphviz dot layout</span></div>
    <div class="toolbar"><button class="tool-button" data-pane="kernel-pane" data-action="zoom-in">+</button>
      <button class="tool-button" data-pane="kernel-pane" data-action="zoom-out">-</button>
      <button class="tool-button" data-pane="kernel-pane" data-action="reset">reset</button></div>
    {kernel_svg}
  </section>
</div>
<aside id="detail-drawer" class="detail-drawer" aria-hidden="true">
  <div class="drawer-header">
    <div><div id="drawer-title" class="drawer-title">Kernel Compare Details</div>
      <div id="drawer-subtitle" class="meta"></div></div>
    <button id="drawer-close" class="drawer-close">close</button>
  </div>
  <div id="drawer-body" class="drawer-body"></div>
</aside>
<script>
const hoverInfo = document.getElementById('hover-info');
const detailDrawer = document.getElementById('detail-drawer');
const drawerTitle = document.getElementById('drawer-title');
const drawerSubtitle = document.getElementById('drawer-subtitle');
const drawerBody = document.getElementById('drawer-body');
document.getElementById('drawer-close').addEventListener('click', () => {{
  detailDrawer.classList.remove('open'); detailDrawer.setAttribute('aria-hidden','true');
}});
function sanitizeId(name) {{ return String(name ?? '').replace(/[^A-Za-z0-9_.:-]/g, '_'); }}
function escapeHtml(value) {{ return String(value ?? '').replace(/[&<>"']/g,
  (ch) => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[ch])); }}
function fmtNumber(v) {{
  if (v === null || v === undefined || v === '') return '';
  if (typeof v === 'number') {{
    if (!Number.isFinite(v)) return String(v);
    const a = Math.abs(v);
    if (a !== 0 && (a < 1e-4 || a >= 1e5)) return v.toExponential(4);
    return String(Math.round(v*1e6)/1e6);
  }}
  return String(v);
}}
function renderTopErrorTable(elements) {{
  if (!elements || !elements.length) return '<div class="detail-meta">No top error elements recorded.</div>';
  const rows = elements.map((it) => `<tr><td>${{escapeHtml(it.rank)}}</td><td>${{escapeHtml(JSON.stringify(it.index ?? []))}}</td>
    <td>${{escapeHtml(fmtNumber(it.fx_value))}}</td><td>${{escapeHtml(fmtNumber(it.kernel_value))}}</td>
    <td>${{escapeHtml(fmtNumber(it.abs_diff))}}</td><td>${{escapeHtml(fmtNumber(it.rel_diff))}}</td></tr>`).join('');
  return `<table class="detail-table"><thead><tr><th>#</th><th>index</th><th>fx</th><th>kernel</th><th>abs</th><th>rel</th></tr></thead><tbody>${{rows}}</tbody></table>`;
}}
const fxNodeEls = Array.from(document.querySelectorAll('.fx-node'));
const kernelNodeEls = Array.from(document.querySelectorAll('.kernel-node'));
const fxByIndex = new Map(fxNodeEls.map((el) => [el.dataset.fxName, el]));
function clearHighlights() {{
  for (const el of fxNodeEls) {{
    el.classList.remove('highlight', 'fx-dim');
  }}
  for (const el of kernelNodeEls) {{
    el.classList.remove('active', 'k-dim');
  }}
  hoverInfo.textContent = 'Hover a kernel node to highlight corresponding FX nodes.';
}}
function highlightFx(kernelNode) {{
  clearHighlights();
  kernelNode.classList.add('active');
  for (const el of kernelNodeEls) {{
    if (el !== kernelNode) el.classList.add('k-dim');
  }}
  let fxNodes = [];
  try {{ fxNodes = JSON.parse(kernelNode.dataset.fxNodes || '[]'); }} catch(e) {{ fxNodes = []; }}
  if (!fxNodes.length) {{
    hoverInfo.textContent = 'No FX node mapping recorded for this kernel.';
    return;
  }}
  let first = null;
  const wanted = new Set(fxNodes);
  for (const el of fxNodeEls) {{
    if (wanted.has(el.dataset.fxName)) {{
      el.classList.add('highlight');
      if (!first) first = el;
    }} else {{
      el.classList.add('fx-dim');
    }}
  }}
  hoverInfo.textContent = 'Kernel maps to FX nodes: ' + fxNodes.join(', ');
  if (first) centerOn(first, document.getElementById('fx-pane'));
}}
function openKernelDetails(kernelNode) {{
  let details = [];
  try {{ details = JSON.parse(kernelNode.dataset.compareDetails || '[]'); }} catch(e) {{ details = []; }}
  const titleEl = kernelNode.querySelector('.node-title, text, title');
  drawerTitle.textContent = titleEl ? titleEl.textContent.trim() : 'Kernel Compare Details';
  drawerSubtitle.textContent = '';
  if (!details.length) {{
    drawerBody.innerHTML = '<div class="detail-card">No compare details recorded for this kernel call.</div>';
  }} else {{
    drawerBody.innerHTML = details.map((row) => `
      <section class="detail-card">
        <h3>buffer=${{escapeHtml(row.call_arg)}} fx=${{escapeHtml(row.fx_node)}} logical=${{escapeHtml(row.logical_buffer)}}</h3>
        <div class="detail-meta">status=${{escapeHtml(row.status)}} bad=${{escapeHtml(row.bad_element_count)}}<br>
          max_abs=${{escapeHtml(fmtNumber(row.max_abs_diff))}} mean_abs=${{escapeHtml(fmtNumber(row.mean_abs_diff))}}<br>
          max_rel=${{escapeHtml(fmtNumber(row.max_rel_abs_diff))}} mean_rel=${{escapeHtml(fmtNumber(row.mean_rel_abs_diff))}}</div>
        ${{renderTopErrorTable(row.top_error_elements || [])}}
      </section>`).join('');
  }}
  detailDrawer.classList.add('open');
  detailDrawer.setAttribute('aria-hidden','false');
}}
for (const node of kernelNodeEls) {{
  node.addEventListener('mouseenter', () => highlightFx(node));
  node.addEventListener('mouseleave', clearHighlights);
  node.addEventListener('click', (e) => {{ e.stopPropagation(); openKernelDetails(node); }});
}}
function centerOn(el, pane) {{
  if (!el) return;
  const r = el.getBoundingClientRect();
  const pr = pane.getBoundingClientRect();
  pane.scrollTo({{
    left: pane.scrollLeft + (r.left - pr.left) - pr.width / 2,
    top: pane.scrollTop + (r.top - pr.top) - pr.height / 2,
    behavior: 'smooth',
  }});
}}
function setupPanZoom(paneId) {{
  const pane = document.getElementById(paneId);
  const svg = pane.querySelector('svg');
  const baseW = parseFloat(svg.getAttribute('width')) || 1000;
  const baseH = parseFloat(svg.getAttribute('height')) || 1000;
  const PX = 4/3;
  let scale = 1;
  function apply(s) {{
    scale = Math.max(0.04, Math.min(8, s));
    svg.style.width = (baseW * PX * scale) + 'px';
    svg.style.height = (baseH * PX * scale) + 'px';
  }}
  // default zoom is natural size (scale 1) so node text stays readable;
  // the graph may be very large, that is expected.
  apply(1);
  let dragging=false, sx=0, sy=0, sl=0, st=0;
  pane.addEventListener('wheel', (e) => {{
    if (!e.ctrlKey && !e.metaKey) return;
    e.preventDefault();
    apply(scale * (e.deltaY < 0 ? 1.12 : 0.89));
  }}, {{passive:false}});
  pane.addEventListener('mousedown', (e) => {{
    if (e.button !== 0) return;
    if (e.target.closest('a')) return;
    dragging=true; sx=e.clientX; sy=e.clientY; sl=pane.scrollLeft; st=pane.scrollTop;
    pane.classList.add('dragging'); e.preventDefault();
  }});
  window.addEventListener('mousemove', (e) => {{
    if (!dragging) return;
    pane.scrollLeft = sl - (e.clientX - sx);
    pane.scrollTop = st - (e.clientY - sy);
  }});
  window.addEventListener('mouseup', () => {{ dragging=false; pane.classList.remove('dragging'); }});
  return {{
    zoomIn: () => apply(scale*1.18), zoomOut: () => apply(scale/1.18), reset: () => apply(1),
    centerOn: (el) => centerOn(el, pane),
  }};
}}
const panZoom = {{ 'fx-pane': setupPanZoom('fx-pane'), 'kernel-pane': setupPanZoom('kernel-pane') }};
document.querySelectorAll('.tool-button').forEach(btn => btn.addEventListener('click', () => {{
  const ctl = panZoom[btn.dataset.pane]; if (!ctl) return;
  if (btn.dataset.action === 'zoom-in') ctl.zoomIn();
  if (btn.dataset.action === 'zoom-out') ctl.zoomOut();
  if (btn.dataset.action === 'reset') ctl.reset();
}}));
</script>
</body></html>
"""


def _full_fx_values(fx_path: Path, save_dir: Path):
    """Replay the FX graph and return {fx_node: tensor} for EVERY node.

    The trace's lowering map records fused relu values as ``lazy`` (buffer=None),
    so the static buffer->fx map has no entry for those kernels.  Their output
    tensor is nonetheless bit-identical to a relu FX node, so we can recover the
    mapping dynamically by replaying the FX graph and matching by value.
    """
    text = fx_path.read_text(encoding="utf-8")
    text = "_VALS = {}\n" + text
    assign_re = re.compile(r"^(\s*)([A-Za-z_][A-Za-z0-9_]*)\s*=")
    out = []
    for line in text.splitlines():
        out.append(line)
        m = assign_re.match(line)
        if not m:
            continue
        name = m.group(2)
        if line.lstrip().startswith(("class ", "def ", "import ", "from ", "@")):
            continue
        if name.startswith("_") or name == "self":
            continue
        out.append(f"{m.group(1)}_VALS[{name!r}] = {name}")
    inst_path = save_dir / "_fx_all_values.py"
    inst_path.write_text("\n".join(out) + "\n", encoding="utf-8")

    spec = importlib.util.spec_from_file_location("_fx_all_values", str(inst_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_fx_all_values"] = mod
    spec.loader.exec_module(mod)
    args = rc.get_args_from_fx_module(mod, str(save_dir), fx_path)
    with torch.no_grad():
        mod.mod(*args)
    return {k: v.detach().float().cpu() for k, v in mod._VALS.items() if isinstance(v, torch.Tensor)}


def value_map_unmapped(buffer_to_fx, kernels, trace_dir: Path, fx_path: Path):
    """Add buffer->fx entries for kernel outputs the static map is missing.

    Only buffers that appear as a kernel output but have no lowering-map entry
    are considered.  They are matched to an FX node with identical shape and an
    (essentially) identical value.
    """
    # kernel output buffers without a static mapping
    have = set()
    for buf in buffer_to_fx:
        have.add(str(buf))
    missing = {}
    for k in kernels:
        for e in k.get("dump_entries", []):
            for logical in rc.post_call_logical_names(e):
                if str(logical) not in have and str(logical) not in missing:
                    missing[str(logical)] = k.get("call_id")
    if not missing:
        return {}

    kv_rows = [
        json.loads(line) for line in (trace_dir / "kernel_values.jsonl").read_text().splitlines() if line.strip()
    ]
    kpath = {
        str(r.get("call_arg")): Path(r["path"]) for r in kv_rows if r.get("kind") == "kernel_tensor" and r.get("path")
    }

    fx_vals = _full_fx_values(fx_path, trace_dir)
    # group fx values by shape
    by_shape = {}
    for name, v in fx_vals.items():
        by_shape.setdefault(tuple(v.shape), []).append((name, v))

    extra = {}
    for buf in missing:
        if buf not in kpath:
            continue
        kt = torch.load(kpath[buf], map_location="cpu", weights_only=True).detach().float()
        if not isinstance(kt, torch.Tensor):
            continue
        best, best_score = None, None
        for name, v in by_shape.get(tuple(kt.shape), ()):
            d = (v - kt).abs().max().item()
            if best_score is None or d < best_score:
                best, best_score = name, d
        if best is not None and best_score < 1e-6:
            extra[str(buf)] = [{"fx_node": best, "buffer": buf, "value_match_max_abs": float(best_score)}]
    return extra


def replay_fx_on_cpu(trace_dir: Path, fx_path: Path, wanted) -> int:
    """Replay the FX graph on CPU and dump the wanted node values.

    The trace's kernel values are NPU; the FX graph itself is pure aten on the
    passed arguments, so replaying it on CPU gives an independent reference and
    lets the comparison be "NPU kernels vs CPU FX".  Writes fx_values.jsonl and
    fx_tensors/*.pt in the same format dump_fx_tensor uses.
    """
    wanted = set(wanted)
    fxv = trace_dir / "fx_values.jsonl"
    fxt = trace_dir / "fx_tensors"
    if fxv.exists():
        fxv.unlink()
    if fxt.exists():
        shutil.rmtree(fxt)
    fxt.mkdir(parents=True, exist_ok=True)

    outdir = str(trace_dir.resolve())
    prelude = (
        "_OUT_DIR = %r\n"
        "_WANTED = %r\n"
        "_ROWS = []\n"
        "import os, time, json\n"
        "import torch as _torch\n"
        "def _logic_dump(name, value):\n"
        "    if name not in _WANTED:\n"
        "        return\n"
        "    if isinstance(value, _torch.Tensor):\n"
        "        p = os.path.join(_OUT_DIR, 'fx_tensors', name + '.pt')\n"
        "        _torch.save(value.detach().cpu(), p)\n"
        "        _ROWS.append({'kind':'fx_tensor','time':time.time(),'fx_node':name,'path':p,'is_tensor':True})\n"
        "    else:\n"
        "        _ROWS.append({'kind':'fx_tensor','time':time.time(),'fx_node':name,'path':None,'is_tensor':False,'type':type(value).__name__})\n"
        "\n"
    ) % (outdir, sorted(wanted))
    text = prelude + fx_path.read_text(encoding="utf-8")
    assign_re = re.compile(r"^(\s*)([A-Za-z_][A-Za-z0-9_]*)\s*=")
    out = []
    for line in text.splitlines():
        out.append(line)
        m = assign_re.match(line)
        if not m:
            continue
        name = m.group(2)
        if line.lstrip().startswith(("class ", "def ", "import ", "from ", "@")):
            continue
        if name.startswith("_") or name == "self":
            continue
        if name in wanted:
            out.append(f"{m.group(1)}_logic_dump({name!r}, {name})")
    # the compiled FX graph hardcodes the capture-time device (npu) when it
    # creates masks/iota buffers (e.g. torch.full(..., device=ids.device)).
    # For a CPU replay we patch those to CPU so the whole graph stays on CPU.
    text2 = "\n".join(out) + "\n"
    text2 = text2.replace("device(type='npu', index=0)", "device(type='cpu')")
    text2 = text2.replace("device(type='npu')", "device(type='cpu')")
    inst = trace_dir / "instrumented_fx_graph_runnable_cpu.py"
    inst.write_text(text2, encoding="utf-8")

    spec = importlib.util.spec_from_file_location("_fx_cpu_dump", str(inst))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_fx_cpu_dump"] = mod
    spec.loader.exec_module(mod)
    args = rc.get_args_from_fx_module(mod, str(trace_dir), fx_path)
    cpu_args = [a.detach().cpu() if isinstance(a, torch.Tensor) else a for a in args]
    with torch.no_grad():
        mod.mod(*cpu_args)
    rows = list(mod._ROWS)
    with fxv.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return len(rows)


def _ensure_runtime_dumps(
    trace_dir: Path,
    fx_path: Path,
    output_path: Path,
    buffer_to_fx: Dict[str, List[Dict[str, Any]]],
    kernels: List[Dict[str, Any]],
) -> None:
    """Re-dump fx/kernel tensors when the trace's runtime dumps are missing.

    ``run_compare.main`` wipes ``fx_values.jsonl``/``kernel_values.jsonl`` at the
    start of its replay, and a failed replay (e.g. a CPU replay of an NPU-only
    op) can leave them absent.  The numeric comparison silently degrades to
    ``missing_fx_value`` without them, so re-dump on the trace's native device
    before comparing.
    """
    need_fx = not (trace_dir / "fx_values.jsonl").exists()
    need_kernel = not (trace_dir / "kernel_values.jsonl").exists()
    if not (need_fx or need_kernel):
        return
    print(
        "[viz] missing runtime dumps (fx=%s, kernel=%s); re-dumping via FX + output replay" % (need_fx, need_kernel),
        flush=True,
    )
    if not _register_trace_helper():
        print("[viz] cannot import torch._inductor.logic_buffer_trace; skipping re-dump", flush=True)
        return
    os.environ.setdefault("INDUCTOR_LOGIC_BUFFER_TRACE", "1")
    os.environ.setdefault("INDUCTOR_LOGIC_BUFFER_DIR", str(trace_dir))
    nodes = rc.wanted_fx_nodes(buffer_to_fx, kernels)
    fx_mod = rc.import_file(fx_path, "logic_buffer_fx_inputs")
    base_args = rc.get_args_from_fx_module(fx_mod, str(trace_dir), fx_path)
    if need_fx:
        rc.run_fx(fx_path, trace_dir, nodes, base_args, str(trace_dir))
    if need_kernel:
        rc.run_output(output_path, base_args)


def _register_trace_helper() -> bool:
    """Make ``torch._inductor.logic_buffer_trace`` importable for the re-dump.

    In monkey-patch mode the helper module only exists in the compile process;
    register it from the packaged ``logic_buffer_trace.py`` when available.
    """
    try:
        importlib.import_module("torch._inductor.logic_buffer_trace")

        return True
    except ImportError:
        pass
    try:
        try:
            from . import logic_monkeypatch as lm
        except ImportError:
            import logic_monkeypatch as lm  # type: ignore
        lm._load_helper()
        return True
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--fx", type=Path, default=None)
    parser.add_argument("--node-script", type=Path, default=NODE_SCRIPT)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--cpu", action="store_true", help="replay the FX graph on CPU (NPU kernels vs CPU FX)")
    parser.add_argument(
        "--fx-device",
        choices=("npu", "cpu"),
        default="npu",
        help="DEPRECATED: use --cpu (kept for backward compatibility)",
    )
    args = parser.parse_args()
    node_script = args.node_script
    fx_device = "cpu" if args.cpu else args.fx_device

    model_dir = args.model_dir
    if args.fx is not None:
        fx_path = args.fx
    else:
        cand = model_dir / "fx_graph_transformed_runnable.py"
        fx_path = cand if cand.exists() else model_dir / "fx_graph_runnable.py"
    output_path = model_dir / "output_code.py"
    trace_dir = model_dir / "logic_buffer_run"

    # ---- data prep (same as run_compare's skip-run path) ----
    buffer_to_fx, kernels = rc.build_maps(trace_dir)
    if fx_device == "cpu":
        wanted = rc.wanted_fx_nodes(buffer_to_fx, kernels)
        n = replay_fx_on_cpu(trace_dir, fx_path, wanted)
        print("[cpu] replayed FX on CPU, dumped %d wanted nodes" % n)
    _ensure_runtime_dumps(trace_dir, fx_path, output_path, buffer_to_fx, kernels)
    comparisons = rc.build_comparisons(trace_dir, buffer_to_fx, 1e-2, 1e-2)
    rc.mark_first_triton_value_mismatch(comparisons, kernels)
    kernels = rc.enrich_kernels_from_output_code(kernels, output_path)
    # NOTE: dynamic value-matching (value_map_unmapped) is deliberately NOT used
    # here to establish kernel->fx mappings -- dynamic numeric equality must not
    # be a mapping basis in the visualization.  It remains only as a test-side
    # check in tests/suite_analyze.py (dynamic_value_check).  Hover mapping is
    # purely static: lowering map + alias/slice/operand resolution.
    fx_stats = rc.compare_stats_by_fx_node(comparisons)
    call_stats = rc.compare_stats_by_call_id(comparisons)
    call_details = rc.compare_details_by_call_id(comparisons)
    slice_map = rc._parse_alias_slices(output_path)
    kernel_nodes, kernel_edges = rc.build_kernel_call_graph(
        kernels,
        buffer_to_fx,
        call_stats,
        call_details,
        rc._parse_alias_map(output_path),
        slice_map,
        rc._build_cat_operand_map(fx_path, buffer_to_fx, slice_map),
    )

    # ---- DOT + render ----
    fx_dot = build_dot(fx_path, "FX Graph (XLA style)", fx_stats)
    kernel_dot = build_kernel_dot(kernel_nodes, kernel_edges)
    fx_svg = annotate_fx_svg(render_svg(fx_dot, node_script), {n["id"] for n in parse_fx(fx_path)[0]})
    kernel_by_id = {kn["id"]: kn for kn in kernel_nodes}
    kernel_svg = annotate_kernel_svg(render_svg(kernel_dot, node_script), kernel_by_id)

    out_path = args.out or (trace_dir / "logic_buffer_graph_xla_dual.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    (trace_dir / "logic_buffer_graph_xla_dual.dot").write_text(fx_dot + "\n\n" + kernel_dot, encoding="utf-8")
    html = HTML_TEMPLATE.format(
        trace_dir=_html.escape(str(trace_dir)),
        fx_path=_html.escape(str(fx_path)),
        output_path=_html.escape(str(output_path)),
        fx_svg=fx_svg,
        kernel_svg=kernel_svg,
    )
    out_path.write_text(html, encoding="utf-8")
    fx_count = fx_svg.count('class="node fx-node"')
    print("HTML written to %s (fx nodes %d, kernel nodes %d)" % (out_path, fx_count, len(kernel_nodes)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
