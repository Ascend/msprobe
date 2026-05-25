#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
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

"""精度对比工具 - 对任意模型做 eager vs compile 精度比对"""

from __future__ import annotations

import re
import os
import hashlib
import contextlib
import types
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from msprobe.core.common.file_utils import create_directory


# 数据结构

@dataclass
class TensorDiff:
    max_abs:  float
    mean_abs: float
    max_rel:  float
    allclose: bool
    shape:    tuple

    def __str__(self):
        ok = 'OK' if self.allclose else '!!'
        return (f'[{ok}] max_abs={self.max_abs:.3e}  '
                f'mean_abs={self.mean_abs:.3e}  '
                f'max_rel={self.max_rel:.3e}  shape={self.shape}')


@dataclass
class ModuleDiff:
    name:        str
    fwd_input:   Optional[List[TensorDiff]] = None
    fwd_output:  Optional[List[TensorDiff]] = None
    grad_input:  Optional[List[TensorDiff]] = None
    grad_output: Optional[List[TensorDiff]] = None
    note:        str = ''


@dataclass
class CompareResult:
    loss_eager:    float
    loss_compiled: float
    diffs:         List[ModuleDiff] = field(default_factory=list)
    cast_dtype:    Optional[torch.dtype] = None

    @property
    def loss_diff(self):
        import math
        if math.isnan(self.loss_eager):
            return float('nan')
        return abs(self.loss_eager - self.loss_compiled)

    @property
    def all_pass(self):
        def _ok(diffs):
            return diffs is None or all(d.allclose for d in diffs)
        return all(
            d.note.startswith('SKIP') or d.note == 'IGNORED' or (
                _ok(d.fwd_input) and _ok(d.fwd_output) and
                _ok(d.grad_input) and _ok(d.grad_output)
            )
            for d in self.diffs
        )


@dataclass
class _GradSlot:
    value: Optional[torch.Tensor] = None


# ─────────────────────────────────────────────────────────────
# 内部工具
# ─────────────────────────────────────────────────────────────

_WRAP_ATTR   = '_pc_wrapped'
_IGNORE_ATTR = '_pc_ignored'


def _to_f32_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu()
    if isinstance(x, tuple):
        return tuple(_to_f32_cpu(t) for t in x)
    if isinstance(x, list):
        return [_to_f32_cpu(t) for t in x]
    return x


def _build_grad_slots(x):
    if isinstance(x, torch.Tensor):
        return _GradSlot()
    if isinstance(x, tuple):
        return tuple(_build_grad_slots(t) for t in x)
    if isinstance(x, list):
        return [_build_grad_slots(t) for t in x]
    return x


def _iter_tensor_slots(value, slots):
    if isinstance(value, torch.Tensor) and isinstance(slots, _GradSlot):
        yield value, slots
        return
    if isinstance(value, tuple) and isinstance(slots, tuple):
        for item, slot in zip(value, slots):
            yield from _iter_tensor_slots(item, slot)
        return
    if isinstance(value, list) and isinstance(slots, list):
        for item, slot in zip(value, slots):
            yield from _iter_tensor_slots(item, slot)
        return


def _materialize_grad_slots(x):
    if isinstance(x, _GradSlot):
        return x.value
    if isinstance(x, tuple):
        return tuple(_materialize_grad_slots(t) for t in x)
    if isinstance(x, list):
        return [_materialize_grad_slots(t) for t in x]
    return x


def _register_tensor_grad_hooks(handles: list, store: _TensorStore,
                                name: str, key: str, value):
    entry = store.bwd.setdefault(name, {'grad_input': None, 'grad_output': None})
    slots = _build_grad_slots(value)
    entry[key] = slots

    for tensor, slot in _iter_tensor_slots(value, slots):
        if not tensor.requires_grad:
            continue

        @torch.compiler.disable
        def save_grad(grad, out=slot):
            out.value = _to_f32_cpu(grad)

        handles.append(tensor.register_hook(save_grad))


def _cmp_tensor(a: torch.Tensor, b: torch.Tensor) -> TensorDiff:
    if a.shape != b.shape:
        return TensorDiff(float('inf'), float('inf'), float('inf'), False,
                          (tuple(a.shape), tuple(b.shape)))
    diff = (a - b).abs()
    return TensorDiff(
        max_abs  = diff.max().item(),
        mean_abs = diff.mean().item(),
        max_rel  = (diff / (a.abs() + 1e-8)).max().item(),
        allclose = torch.allclose(a, b, atol=1e-4, rtol=1e-3),
        shape    = tuple(a.shape),
    )


def _cmp_list(ea, eb) -> Optional[List[TensorDiff]]:
    if ea is None or eb is None:
        return None
    if isinstance(ea, torch.Tensor):
        ea, eb = [ea], [eb]
    pairs = list(_iter_tensor_pairs(ea, eb))
    return [_cmp_tensor(a, b) for a, b in pairs] or None


def _iter_tensor_pairs(ea, eb):
    if isinstance(ea, torch.Tensor) and isinstance(eb, torch.Tensor):
        yield ea, eb
        return
    if isinstance(ea, tuple) and isinstance(eb, tuple):
        for a, b in zip(ea, eb):
            yield from _iter_tensor_pairs(a, b)
        return
    if isinstance(ea, list) and isinstance(eb, list):
        for a, b in zip(ea, eb):
            yield from _iter_tensor_pairs(a, b)
        return


def _normalize_name(name: str) -> str:
    name = re.sub(r'^_orig_mod\.', '', name)
    name = re.sub(r'\._orig_mod', '', name)
    name = re.sub(r'\.module\.', '.', name)
    name = re.sub(r'\.module$', '', name)
    return name


def _is_orig_mod_node(name: str) -> bool:
    return name == '_orig_mod' or name.endswith('._orig_mod') or \
           name == 'module' or name.endswith('.module')


class _TensorStore:
    def __init__(self):
        self.fwd_in:  Dict[str, list]  = {}
        self.fwd_out: Dict[str, list]  = {}
        self.bwd:     Dict[str, dict]  = {}

    def clear(self):
        self.fwd_in.clear()
        self.fwd_out.clear()
        self.bwd.clear()


def _register_hooks(model: nn.Module, store: _TensorStore,
                    scoped_prefixes: Optional[Set[str]],
                    ignored_prefixes: Set[str],
                    capture_input: bool) -> list:
    handles = []
    for raw_name, module in model.named_modules():
        if not raw_name or _is_orig_mod_node(raw_name):
            continue
        name = _normalize_name(raw_name)

        if scoped_prefixes is not None:
            if not any(name == p or name.startswith(p + '.') or p == ''
                       for p in scoped_prefixes):
                continue

        if any(name == p or name.startswith(p + '.') for p in ignored_prefixes):
            continue

        if capture_input:
            def make_pre(n):
                @torch.compiler.disable
                def hook(mod, inp):
                    store.fwd_in[n] = _to_f32_cpu(inp)
                return hook
            handles.append(module.register_forward_pre_hook(make_pre(name)))

        def make_fwd(n):
            @torch.compiler.disable
            def hook(mod, inp, out):
                store.fwd_out[n] = _to_f32_cpu(out)
                _register_tensor_grad_hooks(handles, store, n, 'grad_input', inp)
                _register_tensor_grad_hooks(handles, store, n, 'grad_output', out)
            return hook

        handles.append(module.register_forward_hook(make_fwd(name)))
    return handles


def _remove_hooks(handles: list):
    for h in handles:
        h.remove()


# single_pass 模式相关

def _register_single_pass_hooks(compiled_model: nn.Module,
                                 store: _TensorStore,
                                 ignored_prefixes: Set[str],
                                 capture_input: bool):
    from torch._dynamo.eval_frame import OptimizedModule
    handles = []
    sv_map: Dict[str, dict] = {}

    for raw_name, mod in compiled_model.named_modules():
        if not raw_name or _is_orig_mod_node(raw_name):
            continue

        if isinstance(mod, OptimizedModule):
            orig_mod    = mod._orig_mod
            cast_dtype_ = None
        elif isinstance(mod, _CastWrapper) and isinstance(mod.module, OptimizedModule):
            orig_mod    = mod.module._orig_mod
            cast_dtype_ = mod.cast_dtype
        else:
            continue

        name = _normalize_name(raw_name)
        if any(name == p or name.startswith(p + '.') for p in ignored_prefixes):
            continue
        _saved: Dict[str, object] = {}
        sv_map[name] = _saved

        def make_pre(n, sv):
            @torch.compiler.disable
            def hook(m, inp):
                sv['inp_orig'] = tuple(inp)
                if capture_input:
                    store.fwd_in[n] = _to_f32_cpu(inp)
            return hook

        def make_fwd(n, om, sv, cd):
            @torch.compiler.disable
            def hook(m, inp, compiled_out):
                orig_inp = sv.get('inp_orig', inp)
                device_type = 'npu' if hasattr(torch, 'npu') and torch.npu.is_available() else 'cuda'
                with torch.no_grad():
                    with (torch.autocast(device_type=device_type, dtype=cd)
                          if cd is not None else contextlib.nullcontext()):
                        eager_out = om(*[x.detach() if isinstance(x, torch.Tensor) else x
                                         for x in orig_inp])
                store.fwd_out[n] = _cmp_list(
                    _to_f32_cpu(eager_out),
                    _to_f32_cpu(compiled_out),
                )

                c_out = compiled_out if isinstance(compiled_out, torch.Tensor) else \
                        next((t for t in compiled_out if isinstance(t, torch.Tensor)), None)
                if c_out is not None and c_out.requires_grad:
                    c_out.retain_grad()
                    sv['c_out'] = c_out
            return hook

        def make_bwd(n, om, sv):
            @torch.compiler.disable
            def hook(m, grad_in, grad_out):
                pass
            return hook

        handles.append(mod.register_forward_pre_hook(make_pre(name, _saved)))
        handles.append(mod.register_forward_hook(make_fwd(name, orig_mod, _saved, cast_dtype_)))
        handles.append(mod.register_full_backward_hook(make_bwd(name, orig_mod, _saved)))

    return handles, sv_map


# cast_dtype 相关

class _CastWrapper(nn.Module):
    def __init__(self, module: nn.Module, cast_dtype: torch.dtype):
        super().__init__()
        self.module    = module
        self.cast_dtype = cast_dtype

    @torch.compiler.disable
    def forward(self, *args, **kwargs):
        device_type = 'npu' if hasattr(torch, 'npu') and torch.npu.is_available() else 'cuda'
        with torch.autocast(device_type=device_type, dtype=self.cast_dtype):
            return self.module(*args, **kwargs)


# Graph dump

_gd_counter = 0


def _gd_next_name():
    global _gd_counter
    n = f"__compiled_fn_{_gd_counter}"
    _gd_counter += 1
    return n


def _gd_write(dump_dir, prefix, graph_name, src):
    safe = graph_name.replace(" ", "_")
    content = f"# === {graph_name} ===\n{src}\n"
    digest = hashlib.md5(content.encode()).hexdigest()[:8]
    path = os.path.join(dump_dir, f"{prefix}.{safe}.{digest}.py")
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(content)
    return path


@contextlib.contextmanager
def _graph_dump_ctx(dump_dir: str):
    global _gd_counter
    _gd_counter = 0
    create_directory(dump_dir)

    import torch._dynamo.utils as du
    orig_fn   = du.lazy_format_graph_code
    orig_code = orig_fn.__code__

    def _patched_lazy_format_graph_code(name, gm, maybe_id=None, **kwargs):
        _KW = ("Captured Graph", "Joint Graph", "Forward Graph", "Backward Graph")
        for kw in _KW:
            if kw.lower() in name.lower():
                try:
                    src  = gm.print_readable(print_output=False)
                    path = _gd_write(_gd_dump_dir, _gd_next_name(), name.strip(), src)
                    print(f"  [graph_dump] {name.strip():30s} -> {path}")
                except Exception as e:
                    print(f"  [graph_dump] 捕获失败: {e}")
                break
        return _gd_orig_fn(name, gm, maybe_id=maybe_id, **kwargs)

    orig_impl = types.FunctionType(
        orig_code, orig_fn.__globals__,
        orig_fn.__name__, orig_fn.__defaults__, orig_fn.__closure__,
    )
    orig_fn.__globals__.update(
        _gd_dump_dir  = dump_dir,
        _gd_orig_fn   = orig_impl,
        _gd_next_name = _gd_next_name,
        _gd_write     = _gd_write,
    )
    orig_fn.__code__ = _patched_lazy_format_graph_code.__code__
    try:
        yield
    finally:
        orig_fn.__code__ = orig_code


# PrecisionChecker

class PrecisionChecker:
    """精度对比工具，支持 wrap/ignore/cast_dtype/capture_input 等功能"""

    def __init__(
        self,
        backend:       str                    = 'aot_eager',
        threshold:     float                  = 1e-4,
        dump_graphs:   bool                   = False,
        graph_dir:     str                    = './graph_dump',
        cast_dtype:    Optional[torch.dtype]  = None,
        capture_input: bool                   = True,
        single_pass:   bool                   = True,
    ):
        self.backend       = backend
        self.threshold     = threshold
        self.dump_graphs   = dump_graphs
        self.graph_dir     = graph_dir
        self.cast_dtype    = cast_dtype
        self.capture_input = capture_input
        self.single_pass   = single_pass
        self._wrapped_ids: Dict[int, str] = {}
        self._ignored_ids: Set[int]       = set()

    # wrap API

    def wrap(self, module: nn.Module, name: str = None) -> nn.Module:
        if name is None:
            name = type(module).__name__
        self._wrapped_ids[id(module)] = name
        setattr(module, _WRAP_ATTR, True)
        return module

    def wrap_by_policy(self, model: nn.Module,
                       module_types: tuple) -> nn.Module:
        for mod_name, mod in model.named_modules():
            if isinstance(mod, tuple(module_types)):
                _ = self.wrap(mod, name=mod_name)
        return model

    def wrap_all_children(self, model: nn.Module,
                          depth: int = 1) -> nn.Module:
        _CONTAINERS = (nn.ModuleList, nn.ModuleDict, nn.Sequential)

        def _recurse(mod: nn.Module, prefix: str, remaining: int):
            for cname, child in mod.named_children():
                full_name = f'{prefix}.{cname}' if prefix else cname
                if isinstance(child, _CONTAINERS):
                    _ = _recurse(child, full_name, remaining)
                elif remaining <= 0 or not any(True for _ in child.named_children()):
                    _ = self.wrap(child, name=full_name)
                else:
                    _recurse(child, full_name, remaining - 1)

        _recurse(model, '', depth)
        return model

    # ignore API

    def ignore(self, module: nn.Module) -> nn.Module:
        self._ignored_ids.add(id(module))
        setattr(module, _IGNORE_ATTR, True)
        return module

    def ignore_by_policy(self, model: nn.Module,
                         module_types: tuple) -> nn.Module:
        for _, mod in model.named_modules():
            if isinstance(mod, tuple(module_types)):
                _ = self.ignore(mod)
        return model

    # 内部辅助方法

    def _save_rng_state(self):
        cpu_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() and torch.cuda.is_initialized() else None
        npu_state = None
        try:
            import torch_npu
            if torch.npu.is_available() and torch.npu.is_initialized():
                npu_state = torch.npu.get_rng_state_all()
        except (ImportError, AttributeError):
            pass
        return cpu_state, cuda_state, npu_state

    def _restore_rng_state(self, cpu_state, cuda_state, npu_state):
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        if npu_state is not None:
            torch.npu.set_rng_state_all(npu_state)

    def _compute_grad_input(self, compiled_model, name, sv, sp_store):
        from torch._dynamo.eval_frame import OptimizedModule
        c_out = sv.get('c_out')
        orig_inp = sv.get('inp_orig')
        orig_mod = None
        for raw, mod in compiled_model.named_modules():
            if _normalize_name(raw) == name and isinstance(mod, OptimizedModule):
                orig_mod = mod._orig_mod
                break
            elif _normalize_name(raw) == name and isinstance(mod, _CastWrapper) and isinstance(mod.module, OptimizedModule):
                orig_mod = mod.module._orig_mod
                break
        if c_out is None or c_out.grad is None or orig_inp is None or orig_mod is None:
            sp_store.bwd[name] = {'grad_input': None, 'grad_output': None}
            return

        go = c_out.grad.detach()
        re_args = tuple(
            x.detach().requires_grad_(True) if isinstance(x, torch.Tensor) and x.is_floating_point() else x
            for x in orig_inp
        )
        try:
            with torch.enable_grad():
                e_out = orig_mod(*re_args)
            e_leaves = [a for a in re_args if isinstance(a, torch.Tensor) and a.requires_grad]
            eager_gin = list(torch.autograd.grad(e_out, e_leaves, grad_outputs=go, allow_unused=True))
            eager_gin = [g for g in eager_gin if g is not None]

            c_re_args = tuple(
                x.detach().requires_grad_(True) if isinstance(x, torch.Tensor) and x.is_floating_point() else x
                for x in orig_inp
            )
            comp_mod = None
            for raw, mod in compiled_model.named_modules():
                if _normalize_name(raw) == name and isinstance(mod, OptimizedModule):
                    comp_mod = mod
                    break
            if comp_mod is not None:
                torch._dynamo.reset()
                with torch.enable_grad():
                    c_re_out = comp_mod(*c_re_args)
                c_leaves = [a for a in c_re_args if isinstance(a, torch.Tensor) and a.requires_grad]
                comp_gin = list(torch.autograd.grad(c_re_out, c_leaves, grad_outputs=go, allow_unused=True))
                comp_gin = [g for g in comp_gin if g is not None]
            else:
                comp_gin = []

            sp_store.bwd[name] = {
                'grad_input': _cmp_list(_to_f32_cpu(eager_gin), _to_f32_cpu(comp_gin)) if eager_gin and comp_gin else None,
                'grad_output': None,
            }
        except Exception:
            sp_store.bwd[name] = {'grad_input': None, 'grad_output': None}

    def _wrap_and_compile_modules(self, model: nn.Module, in_place: bool = False):
        target_model = model if in_place else deepcopy(model)
        orig_modules = dict(model.named_modules())
        wrapped_names = {n for n, m in orig_modules.items() if getattr(m, _WRAP_ATTR, False) and n != ''}

        for name, orig_mod in orig_modules.items():
            if not getattr(orig_mod, _WRAP_ATTR, False):
                continue
            if any(name != w and name.startswith(w + '.') for w in wrapped_names):
                continue

            if name == '':
                if self.cast_dtype:
                    target_model = _CastWrapper(target_model, self.cast_dtype)
                cm = torch.compile(target_model, backend=self.backend, fullgraph=False)
                return cm

            parent_name, _, child_name = name.rpartition('.')
            parent = target_model if parent_name == '' else target_model.get_submodule(parent_name)
            child = target_model.get_submodule(name)
            if self.cast_dtype:
                child = _CastWrapper(child, self.cast_dtype)
            cc = torch.compile(child, backend=self.backend, fullgraph=False)
            setattr(parent, child_name, cc)

        return target_model

    # auto-mode: install / collect

    def install(self, model: nn.Module) -> nn.Module:
        if not self.single_pass:
            raise RuntimeError("install() 仅支持 single_pass=True")
        self._install_model = model
        ignored_prefixes = self._collect_prefixes(model, _IGNORE_ATTR)

        _ = self._wrap_and_compile_modules(model, in_place=True)

        self._sp_store = _TensorStore()
        torch._dynamo.reset()
        self._sp_handles, self._sp_sv_map = _register_single_pass_hooks(
            model, self._sp_store, ignored_prefixes, self.capture_input,
        )
        self._install_ignored_prefixes = ignored_prefixes
        self._install_loss_c: Optional[float] = None
        return model

    def record_loss(self, loss: torch.Tensor):
        self._install_loss_c = loss.item() if isinstance(loss, torch.Tensor) else float(loss)

    def collect(self, loss: Optional[torch.Tensor] = None) -> 'CompareResult':
        if loss is not None:
            self.record_loss(loss)

        model           = self._install_model
        ignored_prefixes = self._install_ignored_prefixes
        sv_map          = self._sp_sv_map
        sp_store        = self._sp_store

        for name, sv in sv_map.items():
            self._compute_grad_input(model, name, sv, sp_store)

        _remove_hooks(self._sp_handles)
        diffs = self._build_diffs_single_pass(sp_store, ignored_prefixes)
        loss_c = self._install_loss_c if self._install_loss_c is not None else float('nan')
        return CompareResult(float('nan'), loss_c, diffs, self.cast_dtype)

    # compare

    def compare(self, fn: Callable[[nn.Module], torch.Tensor],
                model: nn.Module) -> CompareResult:
        if self.single_pass:
            return self._compare_single_pass(fn, model)
        return self._compare_two_pass(fn, model)

    def _compare_two_pass(self, fn, model):
        eager_model   = self._build_eager_cast(model)
        compiled_model = self._build_compiled(model)

        scoped_prefixes  = self._collect_prefixes(model, _WRAP_ATTR)
        ignored_prefixes = self._collect_prefixes(model, _IGNORE_ATTR)

        e_store = _TensorStore()
        c_store = _TensorStore()
        h_e = _register_hooks(eager_model,    e_store, scoped_prefixes,
                               ignored_prefixes, self.capture_input)
        h_c = _register_hooks(compiled_model, c_store, scoped_prefixes,
                               ignored_prefixes, self.capture_input)

        rng_state = self._save_rng_state()

        eager_model.train()
        loss_e = fn(eager_model).item()
        eager_model.zero_grad()

        self._restore_rng_state(*rng_state)

        compiled_model.train()
        torch._dynamo.reset()
        if self.dump_graphs:
            with _graph_dump_ctx(self.graph_dir):
                loss_c = fn(compiled_model).item()
        else:
            loss_c = fn(compiled_model).item()
        compiled_model.zero_grad()

        _remove_hooks(h_e)
        _remove_hooks(h_c)

        from torch._dynamo.eval_frame import OptimizedModule
        wrapper_names = {
            _normalize_name(raw)
            for raw, mod in compiled_model.named_modules()
            if raw and isinstance(mod, OptimizedModule)
        }
        if isinstance(compiled_model, OptimizedModule):
            wrapper_names.add('')

        diffs = self._build_diffs(e_store, c_store, wrapper_names,
                                  ignored_prefixes)
        return CompareResult(loss_e, loss_c, diffs, self.cast_dtype)

    def _compare_single_pass(self, fn, model):
        from torch._dynamo.eval_frame import OptimizedModule
        compiled_model = self._build_compiled(model)
        ignored_prefixes = self._collect_prefixes(model, _IGNORE_ATTR)

        compiled_model.train()
        try:
            warmup_loss = fn(compiled_model)
            if warmup_loss.requires_grad:
                warmup_loss.backward()
        except Exception as e:
            import warnings
            warnings.warn(f"Warmup forward failed: {e}")

        compiled_model.zero_grad()
        torch._dynamo.reset()

        sp_store = _TensorStore()
        handles, sv_map = _register_single_pass_hooks(
            compiled_model, sp_store, ignored_prefixes, self.capture_input,
        )

        optimized_modules = []
        for raw_name, mod in compiled_model.named_modules():
            if isinstance(mod, OptimizedModule):
                optimized_modules.append(raw_name)
            elif isinstance(mod, _CastWrapper) and isinstance(mod.module, OptimizedModule):
                optimized_modules.append(raw_name)

        if not optimized_modules:
            import warnings
            warnings.warn(
                f"single_pass mode: no OptimizedModule found after warmup. "
                f"Checked {len(list(compiled_model.named_modules()))} modules."
            )
        else:
            print(f"[DEBUG] Found {len(optimized_modules)} OptimizedModule(s): {optimized_modules[:5]}...")

        compiled_model.train()
        if self.dump_graphs:
            with _graph_dump_ctx(self.graph_dir):
                loss_c = fn(compiled_model).item()
        else:
            loss_c = fn(compiled_model).item()

        for name, sv in sv_map.items():
            self._compute_grad_input(compiled_model, name, sv, sp_store)

        compiled_model.zero_grad()
        _remove_hooks(handles)

        diffs = self._build_diffs_single_pass(sp_store, ignored_prefixes)
        return CompareResult(float('nan'), loss_c, diffs, self.cast_dtype)

    # report

    def report(self, result: CompareResult, csv_path: Optional[str] = None):
        W = 72
        dtype_tag = f'  cast_dtype={result.cast_dtype}' if result.cast_dtype else ''
        import math
        print(f"\n{'='*W}")
        if math.isnan(result.loss_eager):
            print(f"  Loss  compiled={result.loss_compiled:.6f}"
                  f"  (single_pass, eager loss not computed){dtype_tag}")
        else:
            print(f"  Loss  eager={result.loss_eager:.6f}  "
                  f"compiled={result.loss_compiled:.6f}  "
                  f"diff={result.loss_diff:.3e}{dtype_tag}")
        print(f"{'='*W}")

        sections = [('FORWARD INPUT',  'fwd_input'),
                    ('FORWARD OUTPUT', 'fwd_output'),
                    ('BACKWARD',       'bwd')]

        for title, key in sections:
            if key == 'fwd_input' and not self.capture_input:
                continue
            rows = []
            for d in result.diffs:
                if d.note:
                    flag = 'skip' if d.note.startswith('SKIP') or d.note == 'IGNORED' else 'WARN'
                    rows.append((flag, d.name, d.note))
                    continue
                if key == 'bwd':
                    for sub in ('grad_input', 'grad_output'):
                        val = getattr(d, sub)
                        rows.append((_flag_fmt(val), f"{d.name}.{sub}", _fmt_list(val)))
                else:
                    val = getattr(d, key)
                    rows.append((_flag_fmt(val), d.name, _fmt_list(val)))

            if not rows:
                continue
            print(f"\n  {title}")
            print(f"  {'-'*68}")
            for flag, name, detail in rows:
                print(f"  {flag:4s}  {name:52s}  {detail}")

        print(f"\n{'='*W}")
        print(f"  RESULT: {'ALL PASS' if result.all_pass else 'FAILED'}"
              f"  (atol=1e-4 rtol=1e-3)")
        print(f"{'='*W}\n")

        if csv_path:
            self._write_csv_report(result, csv_path)
            print(f"CSV report saved to: {csv_path}\n")

    def _write_csv_report(self, result: CompareResult, csv_path: str):
        import csv
        import math

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            writer.writerow([
                'module_name', 'check_type', 'tensor_index', 'status',
                'max_abs_diff', 'mean_abs_diff', 'max_rel_diff', 'shape', 'note'
            ])

            loss_status = 'PASS' if result.all_pass else 'FAIL'
            if math.isnan(result.loss_eager):
                writer.writerow([
                    'LOSS', 'loss', 0, loss_status,
                    'N/A', 'N/A', 'N/A',
                    f'compiled={result.loss_compiled:.6f}',
                    'single_pass mode'
                ])
            else:
                writer.writerow([
                    'LOSS', 'loss', 0, loss_status,
                    f'{result.loss_diff:.6e}', 'N/A', 'N/A',
                    f'eager={result.loss_eager:.6f} compiled={result.loss_compiled:.6f}',
                    ''
                ])

            for d in result.diffs:
                if d.note:
                    status = 'SKIP' if d.note.startswith('SKIP') or d.note == 'IGNORED' else 'WARN'
                    writer.writerow([
                        d.name, 'note', 0, status,
                        '', '', '', '', d.note
                    ])
                    continue

                if self.capture_input and d.fwd_input is not None:
                    for idx, td in enumerate(d.fwd_input):
                        writer.writerow([
                            d.name, 'fwd_input', idx,
                            'PASS' if td.allclose else 'FAIL',
                            f'{td.max_abs:.6e}', f'{td.mean_abs:.6e}',
                            f'{td.max_rel:.6e}', str(td.shape), ''
                        ])

                if d.fwd_output is not None:
                    for idx, td in enumerate(d.fwd_output):
                        writer.writerow([
                            d.name, 'fwd_output', idx,
                            'PASS' if td.allclose else 'FAIL',
                            f'{td.max_abs:.6e}', f'{td.mean_abs:.6e}',
                            f'{td.max_rel:.6e}', str(td.shape), ''
                        ])

                if d.grad_input is not None:
                    for idx, td in enumerate(d.grad_input):
                        writer.writerow([
                            d.name, 'grad_input', idx,
                            'PASS' if td.allclose else 'FAIL',
                            f'{td.max_abs:.6e}', f'{td.mean_abs:.6e}',
                            f'{td.max_rel:.6e}', str(td.shape), ''
                        ])

                if d.grad_output is not None:
                    for idx, td in enumerate(d.grad_output):
                        writer.writerow([
                            d.name, 'grad_output', idx,
                            'PASS' if td.allclose else 'FAIL',
                            f'{td.max_abs:.6e}', f'{td.mean_abs:.6e}',
                            f'{td.max_rel:.6e}', str(td.shape), ''
                        ])

    # 内部方法

    def _collect_prefixes(self, model: nn.Module, attr: str) -> Set[str]:
        prefixes = set()
        for name, mod in model.named_modules():
            if getattr(mod, attr, False):
                prefixes.add(name)
        return prefixes if prefixes else None if attr == _WRAP_ATTR else set()

    def _build_eager_cast(self, model: nn.Module) -> nn.Module:
        if not self.cast_dtype:
            return model
        eager_model  = deepcopy(model)
        orig_modules = dict(model.named_modules())
        wrapped_names = {n
                         for n, m in orig_modules.items()
                         if getattr(m, _WRAP_ATTR, False) and n != ''}
        for name, orig_mod in orig_modules.items():
            if not getattr(orig_mod, _WRAP_ATTR, False):
                continue
            if any(name != w and name.startswith(w + '.') for w in wrapped_names):
                continue
            if name == '':
                return _CastWrapper(eager_model, self.cast_dtype)
            parent_name, _, child_name = name.rpartition('.')
            parent = eager_model if parent_name == '' else \
                     eager_model.get_submodule(parent_name)
            child  = eager_model.get_submodule(name)
            setattr(parent, child_name, _CastWrapper(child, self.cast_dtype))
        return eager_model

    def _build_compiled(self, model: nn.Module) -> nn.Module:
        return self._wrap_and_compile_modules(model, in_place=False)

    def _build_diffs(self, e: _TensorStore, c: _TensorStore,
                     wrapper_names: Set[str],
                     ignored_prefixes: Set[str]) -> List[ModuleDiff]:
        all_names = sorted(
            set(e.fwd_out) | set(c.fwd_out) |
            set(e.fwd_in)  | set(c.fwd_in)  |
            set(e.bwd)     | set(c.bwd)      |
            ignored_prefixes
        )
        diffs = []
        for name in all_names:
            d = ModuleDiff(name=name)

            if any(name == p or name.startswith(p + '.') for p in ignored_prefixes):
                d.note = 'IGNORED'
                diffs.append(d)
                continue

            # Check if this module is inside a compiled wrapper (hooks can't fire inside fused graph)
            in_compiled = any(
                name.startswith(w + '.') for w in wrapper_names if w
            )

            if name in e.fwd_out and name in c.fwd_out:
                d.fwd_output = _cmp_list(e.fwd_out[name], c.fwd_out[name])
            elif name in e.fwd_out and name not in c.fwd_out:
                if in_compiled:
                    d.note = 'SKIP_inside_compiled'
                else:
                    d.note = 'MISSING_fwd_in_compiled'
            elif name not in e.fwd_out and name in c.fwd_out:
                d.note = 'MISSING_fwd_in_eager'

            if self.capture_input:
                if name in e.fwd_in and name in c.fwd_in:
                    d.fwd_input = _cmp_list(e.fwd_in[name], c.fwd_in[name])

            if name in e.bwd and name in c.bwd:
                d.grad_input  = _cmp_list(
                    _materialize_grad_slots(e.bwd[name]['grad_input']),
                    _materialize_grad_slots(c.bwd[name]['grad_input'])
                )
                d.grad_output = _cmp_list(
                    _materialize_grad_slots(e.bwd[name]['grad_output']),
                    _materialize_grad_slots(c.bwd[name]['grad_output'])
                )
            elif name in e.bwd and name not in c.bwd:
                if name in wrapper_names:
                    d.note = d.note or 'SKIP_compiled_wrapper'
                elif in_compiled:
                    d.note = d.note or 'SKIP_inside_compiled'
                else:
                    d.note = d.note or 'MISSING_bwd_in_compiled'

            diffs.append(d)
        return diffs

    def _build_diffs_single_pass(self, store: _TensorStore,
                                  ignored_prefixes: Set[str]) -> List[ModuleDiff]:
        all_names = sorted(
            set(store.fwd_out) | set(store.fwd_in) |
            set(store.bwd)     | ignored_prefixes
        )
        diffs = []
        for name in all_names:
            d = ModuleDiff(name=name)
            if any(name == p or name.startswith(p + '.') for p in ignored_prefixes):
                d.note = 'IGNORED'
                diffs.append(d)
                continue
            if name in store.fwd_out:
                d.fwd_output = store.fwd_out[name]
            if self.capture_input and name in store.fwd_in:
                inp = store.fwd_in[name]
                if inp is not None:
                    if isinstance(inp, list):
                        d.fwd_input = [TensorDiff(0.0, 0.0, 0.0, True, tuple(t.shape))
                                       for t in inp
                                       if isinstance(t, torch.Tensor)]
                    elif isinstance(inp, torch.Tensor):
                        d.fwd_input = [TensorDiff(0.0, 0.0, 0.0, True, tuple(inp.shape))]
            if name in store.bwd:
                d.grad_input  = store.bwd[name].get('grad_input')
                d.grad_output = store.bwd[name].get('grad_output')
            diffs.append(d)
        return diffs


# 报告格式化工具

def _fmt_list(val) -> str:
    if val is None:
        return 'None'
    if isinstance(val, list):
        parts = [str(v) for v in val if v is not None]
        return ' | '.join(parts) if parts else 'None'
    return str(val)


def _flag_fmt(val) -> str:
    if val is None:
        return 'pass'
    if isinstance(val, list):
        failed = any(not v.allclose for v in val if v is not None)
        return 'FAIL' if failed else 'pass'
    return 'pass'
