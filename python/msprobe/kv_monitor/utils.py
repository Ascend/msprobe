from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import time
from typing import Any

from msprobe.kv_monitor.const import (
    RANDOM_SUFFIX_RE,
)


def now_epoch() -> float:
    return round(time.time(), 4)


def get_external_request_id(request_id: str) -> str:
    return RANDOM_SUFFIX_RE.sub("", request_id)


_rank_cache: int | None = None


def distributed_rank() -> int:
    global _rank_cache
    if _rank_cache is not None:
        return _rank_cache
    try:
        import torch

        if torch.distributed.is_initialized():
            _rank_cache = torch.distributed.get_rank()
            return _rank_cache
    except Exception:
        return 0
    return 0


def step_is_prefill(computed_tokens, prompt_tokens) -> bool:
    return bool(computed_tokens < prompt_tokens)


def get_field(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def attr_chain(target: Any, *names: str) -> Any:
    for name in names:
        target = getattr(target, name, None)
        if target is None:
            return None
    return target


def wrap_class_method(target_cls, method_name: str, guard_attr: str, wrapper_factory) -> bool:
    if getattr(target_cls, guard_attr, False):
        return False
    original = getattr(target_cls, method_name)
    setattr(target_cls, method_name, wrapper_factory(original))
    setattr(target_cls, guard_attr, True)
    return True


class PostImportPatcher(importlib.abc.MetaPathFinder):
    def __init__(self, target_module, callback):
        self.target = target_module
        self.callback = callback
        self.done = False

    def find_spec(self, fullname, path, target=None):
        if fullname == self.target and not self.done:
            self.done = True
            spec = importlib.util.find_spec(fullname)
            if spec is None:
                return None
            loader = _PostImportPatcherLoader(spec.loader, self.callback)
            is_package = spec.submodule_search_locations is not None
            patched = importlib.machinery.ModuleSpec(fullname, loader, origin=spec.origin, is_package=is_package)
            # 包的子模块搜索路径必须原样带回：丢了它 importlib 会把包当普通模块处理，
            # 之后 `import 包.子模块` 会失败。
            patched.submodule_search_locations = spec.submodule_search_locations
            return patched
        return None


class _PostImportPatcherLoader(importlib.abc.Loader):
    def __init__(self, original_loader, callback):
        self.original_loader = original_loader
        self.callback = callback

    def create_module(self, spec):
        return self.original_loader.create_module(spec)

    def exec_module(self, module):
        self.original_loader.exec_module(module)
        self.callback()
