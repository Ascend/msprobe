"""Monkey-patch alternative to patch.py: collect logic-buffer trace info by
wrapping Inductor methods instead of editing torch source files.

Usage:  import logic_monkeypatch; logic_monkeypatch.install()
(do this BEFORE the torch.compile call, in the same process as the compile)

Equivalence: each hook wraps the same host method patch.py edits and calls the
same helper function (logic_buffer_trace) at the same point, so the trace
output is equivalent.  No torch file is modified; the helper is loaded under
the name ``torch._inductor.logic_buffer_trace`` so its relative imports work
without copying the file into site-packages.
"""
# pylint: disable=duplicate-code

import importlib.util
import os
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
HELPER_SRC = HERE / "logic_buffer_trace.py"
TRUE = {"1", "true", "yes", "on"}
_INSTALLED = False


def _trace_on() -> bool:
    return os.environ.get("INDUCTOR_LOGIC_BUFFER_TRACE", "").strip().lower() in TRUE


def _load_helper():
    """Load logic_buffer_trace.py as torch._inductor.logic_buffer_trace so its
    internal ``from . import ir`` / ``from .virtualized import V`` work.
    """
    import torch._inductor as inductor

    spec = importlib.util.spec_from_file_location("torch._inductor.logic_buffer_trace", HELPER_SRC)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["torch._inductor.logic_buffer_trace"] = mod
    spec.loader.exec_module(mod)
    inductor.logic_buffer_trace = mod
    return mod


def _safe(fn, *args):
    try:
        fn(*args)
    except Exception:
        if os.environ.get("LOGIC_MONKEYPATCH_DEBUG"):
            traceback.print_exc()


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    h = _load_helper()
    from torch._inductor import ir
    import torch._inductor.codegen.triton as triton_mod
    import torch._inductor.codegen.wrapper as wrapper_mod
    import torch._inductor.debug as debug_mod
    from torch._inductor.graph import GraphLowering
    from torch._inductor.scheduler import Scheduler

    # ---- 1. graph lowering: GraphLowering.run_node -> record_lowering_boundary ----
    _run_node_orig = GraphLowering.run_node

    def run_node(self, n):
        result = _run_node_orig(self, n)
        if _trace_on():
            _safe(h.record_lowering_boundary, self, n, result)
        return result

    GraphLowering.run_node = run_node

    # ---- 2. scheduler DCE: Scheduler.__init__ -> record_live_fx_buffer_map_after_dce ----
    _sched_init_orig = Scheduler.__init__

    def sched_init(self, *a, **kw):
        _sched_init_orig(self, *a, **kw)
        if _trace_on():
            _safe(h.record_live_fx_buffer_map_after_dce, self)

    Scheduler.__init__ = sched_init

    # ---- 3. ir copy_input: ExternKernel.copy_input -> record_copy_input_materialization ----
    _copy_input_orig = ir.ExternKernel.copy_input

    def copy_input(x):
        pw = _copy_input_orig(x)
        if _trace_on():
            _safe(h.record_copy_input_materialization, x, pw)
        return pw

    ir.ExternKernel.copy_input = staticmethod(copy_input)

    # ---- 4-6. ir extern / multi-output codegen ----
    def wrap_codegen(cls, name, recorder):
        orig = getattr(cls, name)

        def codegen(self, wrapper):
            orig(self, wrapper)
            if _trace_on():
                try:
                    recorder(self, wrapper)
                except Exception:
                    if os.environ.get("LOGIC_MONKEYPATCH_DEBUG"):
                        traceback.print_exc()

        setattr(cls, name, codegen)

    def _extern_out(self, wrapper):
        output_ref = (
            self.output_view.codegen_reference() if getattr(self, "output_view", None) else self.codegen_reference()
        )
        call_id, entries = h.collect_extern_kernel_call(self.get_kernel_name(), self, output_ref)
        if call_id is not None and entries:
            h.codegen_kernel_runtime_dump(wrapper, self.get_kernel_name(), call_id, entries)

    def _extern_alloc(self, wrapper):
        call_id, entries = h.collect_extern_kernel_call(self.get_kernel_name(), self, self.get_name())
        if call_id is not None and entries:
            h.codegen_kernel_runtime_dump(wrapper, self.get_kernel_name(), call_id, entries)

    def _multi_output(self, wrapper):
        call_id, entries = h.collect_single_output_multi_output_call(self, self.codegen_reference())
        if call_id is not None and entries:
            parent = self.inputs[0]
            h.codegen_kernel_runtime_dump(wrapper, parent.get_kernel_name(), call_id, entries)

    wrap_codegen(ir.ExternKernelOut, "codegen", _extern_out)
    wrap_codegen(ir.ExternKernelAlloc, "codegen", _extern_alloc)
    wrap_codegen(ir.MultiOutput, "codegen", _multi_output)

    # ---- 7-8. triton: TritonKernel.call_kernel -> collect + dump ----
    _call_kernel_orig = triton_mod.TritonKernel.call_kernel

    def call_kernel(self, name, node=None, deallocate_ws=True):
        from torch._inductor.virtualized import V as _V

        call_id = None
        entries = []
        if _trace_on():
            try:
                _, call_args, precompile_args, _ = self.args.python_argdefs()
                for i in range(len(call_args)):
                    if _V.graph.is_unspec_arg(call_args[i]):
                        call_args[i] = call_args[i] + ".item()"
                call_id, entries = h.collect_triton_kernel_call(name, self, call_args, precompile_args)
            except Exception:
                if os.environ.get("LOGIC_MONKEYPATCH_DEBUG"):
                    traceback.print_exc()
        result = _call_kernel_orig(self, name, node, deallocate_ws)
        if call_id is not None and entries:
            _safe(h.codegen_kernel_runtime_dump, _V.graph.wrapper_code, name, call_id, entries)
        return result

    triton_mod.TritonKernel.call_kernel = call_kernel

    # ---- 9. wrapper save args: PythonWrapperCodegen.write_args ----
    _write_args_orig = wrapper_mod.PythonWrapperCodegen.write_args

    def write_args(self, input_names):
        _write_args_orig(self, input_names)
        if _trace_on():
            try:
                from torch._inductor.virtualized import V as _V

                input_names = list(_V.graph.graph_input_names)
                names = ", ".join(input_names)
                graph_key = "graph_%s_%s" % (
                    getattr(_V.graph, "graph_id", "unknown"),
                    getattr(_V.graph, "post_grad_graph_id", "unknown"),
                )
                self.prefix.writeline("import os as _inductor_logic_buffer_os")
                self.prefix.writeline(
                    "if _inductor_logic_buffer_os.environ.get('INDUCTOR_LOGIC_BUFFER_TRACE', '').strip().lower() in ('1', 'true', 'yes', 'on'):"
                )
                with self.prefix.indent():
                    self.prefix.writeline(
                        "from torch._inductor.logic_buffer_trace import save_replay_args as _inductor_logic_buffer_save_replay_args"
                    )
                    self.prefix.writeline(
                        "_inductor_logic_buffer_save_replay_args([%s], graph_key=%r, arg_names=%r)"
                        % (names, graph_key, input_names)
                    )
            except Exception:
                if os.environ.get("LOGIC_MONKEYPATCH_DEBUG"):
                    traceback.print_exc()

    wrapper_mod.PythonWrapperCodegen.write_args = write_args

    # ---- 10-13. debug hooks ----
    # 12. debug dir: wrap DebugContext.__enter__ (not __init__).
    #    torch only creates the debug dir in __enter__ (and only when
    #    config.trace.enabled).  Hooking there lets us point
    #    INDUCTOR_LOGIC_BUFFER_DIR at <torch_dir>/logic_buffer_run so the trace
    #    lands in the SAME folder as output_code.py / fx_graph_transformed* --
    #    exactly what the hard-coded patch (patch.py) does.  If torch didn't
    #    create a dir (trace.enabled off), create one here so the trace still
    #    has a home and fopen() won't assert on a None _path.
    _debug_enter_orig = debug_mod.DebugContext.__enter__

    def debug_enter(self):
        result = _debug_enter_orig(self)
        if _trace_on() and self._path is None:
            try:
                import torch._inductor.config as cfg
                from torch._dynamo.utils import get_debug_dir

                debug_dir = cfg.trace.debug_dir or get_debug_dir()
                import torch._inductor.debug as d

                folder = d.get_aot_graph_name()
                import itertools

                for n in itertools.count():
                    dirname = os.path.join(debug_dir, "torchinductor", f"{folder}.{n}")
                    if not os.path.exists(dirname):
                        os.makedirs(dirname)
                        self._path = dirname
                        break
            except Exception:
                if os.environ.get("LOGIC_MONKEYPATCH_DEBUG"):
                    traceback.print_exc()
        if _trace_on() and self._path is not None:
            if not os.environ.get("INDUCTOR_LOGIC_BUFFER_DIR", "").strip():
                os.environ["INDUCTOR_LOGIC_BUFFER_DIR"] = self.filename("logic_buffer_run")
                os.environ["INDUCTOR_LOGIC_BUFFER_ACTIVE_DIR"] = self.filename("logic_buffer_run")
        return result

    debug_mod.DebugContext.__enter__ = debug_enter

    # 13. transformed runnable save.
    #    torch 2.10: the fx_* debug methods live in DebugFormatter, NOT
    #    DebugContext.  DebugContext.__getattr__ dispatches to
    #    DebugFormatter(self) when config.trace.enabled (or, with the
    #    hard-coded patch, when INDUCTOR_LOGIC_BUFFER_TRACE is set).  So we
    #    wrap DebugFormatter.fx_graph_transformed -- that is the method that
    #    actually runs and writes fx_graph_transformed.py.
    if hasattr(debug_mod.DebugFormatter, "fx_graph_transformed"):
        _fxt_orig = debug_mod.DebugFormatter.fx_graph_transformed

        def fx_graph_transformed(self, gm, inputs):
            _fxt_orig(self, gm, inputs)
            if _trace_on():
                try:
                    from torch._dynamo.repro.after_aot import save_graph_repro

                    trace_dir = os.environ.get("INDUCTOR_LOGIC_BUFFER_DIR") or self.filename("logic_buffer_run")
                    os.makedirs(trace_dir, exist_ok=True)
                    with self.fopen("fx_graph_transformed_runnable.py") as fd:
                        save_graph_repro(fd, gm, inputs, "inductor", save_dir=trace_dir)
                except Exception:
                    if os.environ.get("LOGIC_MONKEYPATCH_DEBUG"):
                        traceback.print_exc()

        debug_mod.DebugFormatter.fx_graph_transformed = fx_graph_transformed

    _INSTALLED = True
    print("[logic_monkeypatch] installed %d hooks" % 14)


def maybe_auto_viz(model_dir: str = "") -> None:
    """Generate the interactive HTML right after collection (opt-in).

    Call at the END of your script, AFTER the compiled forward -- that is when
    the trace (kernel/fx dumps) is complete.  Gated by
    ``INDUCTOR_LOGIC_BUFFER_AUTO_VIZ=1`` so normal collect-only runs are not
    slowed by the viz replay.  Writes ``<model_dir>/logic_buffer_run/graph.html``.
    """
    if os.environ.get("INDUCTOR_LOGIC_BUFFER_AUTO_VIZ", "").strip().lower() not in TRUE:
        return
    try:
        import subprocess  # nosec B404

        mdir = model_dir
        if not mdir:
            root = Path("torch_compile_debug")
            outputs = sorted(
                root.rglob("output_code.py") if root.exists() else (),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            mdir = str(outputs[0].parent.resolve()) if outputs else None
        if not mdir:
            print("[auto-viz] no model dir (output_code.py) found; skip", flush=True)
            return
        out = os.path.join(mdir, "logic_buffer_run", "graph.html")
        viz = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xla_style_viz2.py")
        print("[auto-viz] generating %s ..." % out, flush=True)
        # sys.executable is trusted and subprocess never invokes a shell.
        subprocess.run(  # nosec B603
            [sys.executable, viz, mdir, "--out", out], check=False
        )
        print("[auto-viz] done -> %s" % out, flush=True)
    except Exception:
        if os.environ.get("LOGIC_MONKEYPATCH_DEBUG"):
            traceback.print_exc()
