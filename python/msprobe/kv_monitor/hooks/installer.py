from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import partial, wraps
from typing import Any, Callable

from msprobe.core.common.log import logger
from msprobe.kv_monitor import config as kv_config

from msprobe.kv_monitor.utils import (
    PostImportPatcher,
    wrap_class_method,
)


@dataclass(frozen=True)
class PatchTarget:
    """在 module.cls.method 上挂 hook"""

    hook: str
    module: str
    cls: str
    method: str


_MONITOR = "monitor"
_RUNNER = "runner"
_DUMMY = "dummy"
_TRANSFER_KV_CACHE = "transfer_kv_cache"
_MODEL_RUNNER_V1 = "vllm_ascend.worker.model_runner_v1"
_NPU_MODEL_RUNNER = "NPUModelRunner"

# 内置补丁目标表
_BUILTIN_PATCH_TARGETS = (
    PatchTarget(_MONITOR, _MODEL_RUNNER_V1, _NPU_MODEL_RUNNER, "sample_tokens"),
    PatchTarget(_RUNNER, _MODEL_RUNNER_V1, _NPU_MODEL_RUNNER, "execute_model"),
    PatchTarget(_DUMMY, _MODEL_RUNNER_V1, _NPU_MODEL_RUNNER, "_dummy_run"),
    PatchTarget(
        _TRANSFER_KV_CACHE,
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector",
        "KVCacheRecvingThread",
        "_transfer_kv_cache_all_groups",
    ),
)


class Installer:
    _ASCEND_CONFIG_MODULE = "vllm_ascend.ascend_config"
    _ASCEND_CONFIG_GATE_ATTR = "_kv_monitor_gate_installed"
    _ASCEND_CONFIG_INIT_METHOD = "init_ascend_config"

    def __init__(self) -> None:
        self._actions_cache: dict[str, Callable] | None = None
        self._effective: dict[str, PatchTarget] | None = None
        self._deferred: set[str] = set()
        self._probe_gate_checked: bool = False
        self._installed: bool = False
        self._post_import_callbacks: dict[str, list[Callable]] = {}
        self._post_import_finders: dict[str, Any] = {}

    def _actions(self) -> dict[str, Callable]:
        """hook 名 -> 安装函数。延迟 import 避免循环依赖；结果缓存。"""
        if self._actions_cache is None:
            from msprobe.kv_monitor.hooks import monitor, runner, dummy, transfer_kv_cache

            self._actions_cache = {
                _MONITOR: monitor.apply_patch,
                _RUNNER: runner.apply_patch,
                _DUMMY: dummy.apply_patch,
                _TRANSFER_KV_CACHE: transfer_kv_cache.apply_patch,
            }
        return self._actions_cache

    def _effective_targets(self) -> dict[str, PatchTarget]:
        if self._effective is None:
            self._effective = {target.hook: target for target in _BUILTIN_PATCH_TARGETS}
        return self._effective

    def _action_for(self, target: PatchTarget) -> Callable:
        return self._actions()[target.hook]

    def _warn_target_skipped(self, target: PatchTarget, message: str, *args) -> None:
        logger.warning(f"[kv-monitor] patch target {target.hook}: " + (message % args))

    def _install_now(self, target: PatchTarget) -> None:
        module = sys.modules.get(target.module)
        if module is None:
            self._warn_target_skipped(target, "module %s not in sys.modules, hook skipped", target.module)
            return
        cls = getattr(module, target.cls, None)
        if cls is None:
            self._warn_target_skipped(target, "class %s not found in %s, hook skipped", target.cls, target.module)
            return
        if not hasattr(cls, target.method):
            self._warn_target_skipped(
                target, "method %s.%s.%s not found, hook skipped", target.module, target.cls, target.method
            )
            return
        self._action_for(target)(cls, target.method)

    def _apply_or_defer(self, target: PatchTarget) -> None:
        """目标模块已导入 → 立即安装；未导入 → 延迟到导入后。"""
        if target.module not in sys.modules:
            self._defer(target)
            return
        self._install_now(target)

    def _defer(self, target: PatchTarget) -> None:
        if target.hook in self._deferred:
            return
        self._deferred.add(target.hook)
        logger.info(f"[kv-monitor] patch target {target.hook} waits for module {target.module}")
        self._register_post_import(target.module, partial(self._install_now, target))

    def _install_probe_if_enabled(self, vllm_config) -> None:
        if self._probe_gate_checked:
            return
        probe_enabled = kv_config.DEFAULT_CONFIG.probe_gate(vllm_config)
        self._probe_gate_checked = True
        self._unhook_init_ascend_config()
        logger.info(f"[kv-monitor] probe_enabled: {probe_enabled}")
        if probe_enabled:
            self._install_all_targets()

    def _install_all_targets(self) -> None:
        for target in self._effective_targets().values():
            self._apply_or_defer(target)

    def _unhook_init_ascend_config(self) -> None:
        module = sys.modules.get(Installer._ASCEND_CONFIG_MODULE)
        if module is None:
            return
        wrapper = getattr(module, Installer._ASCEND_CONFIG_INIT_METHOD, None)
        original = getattr(wrapper, "__wrapped__", None)
        if original is not None:
            setattr(module, Installer._ASCEND_CONFIG_INIT_METHOD, original)

    def _make_ascend_config_wrapper(self, original):
        @wraps(original)
        def _wrapped(vllm_config, *args, **kwargs):
            config = original(vllm_config, *args, **kwargs)
            self._install_probe_if_enabled(vllm_config)
            return config

        return _wrapped

    def _make_ascend_config_callback(self) -> Callable:
        def _on_import() -> None:
            module = sys.modules.get(Installer._ASCEND_CONFIG_MODULE)
            if module is None or not hasattr(module, Installer._ASCEND_CONFIG_INIT_METHOD):
                logger.warning(
                    f"[kv-monitor] {Installer._ASCEND_CONFIG_MODULE}."
                    f"{Installer._ASCEND_CONFIG_INIT_METHOD} not found, hooks skipped"
                )
                return
            if wrap_class_method(
                module,
                method_name=Installer._ASCEND_CONFIG_INIT_METHOD,
                guard_attr=Installer._ASCEND_CONFIG_GATE_ATTR,
                wrapper_factory=self._make_ascend_config_wrapper,
            ):
                logger.info(
                    f"[kv-monitor] gate hooked on {Installer._ASCEND_CONFIG_MODULE}."
                    f"{Installer._ASCEND_CONFIG_INIT_METHOD}"
                )

        return _on_import

    def _register_post_import(self, module_name: str, callback: Callable) -> None:
        if module_name in sys.modules:
            callback()
            return
        callbacks = self._post_import_callbacks.setdefault(module_name, [])
        callbacks.append(callback)
        if len(callbacks) == 1:
            finder = PostImportPatcher(module_name, lambda: self._run_post_import(module_name))
            self._post_import_finders[module_name] = finder
            sys.meta_path.insert(0, finder)

    def _detach_finder(self, module_name: str) -> None:
        finder = self._post_import_finders.pop(module_name, None)
        if finder is None:
            return
        try:
            sys.meta_path.remove(finder)
        except ValueError:
            # finder 可能已被外部移除，忽略
            pass

    def _run_post_import(self, module_name: str) -> None:
        callbacks = self._post_import_callbacks.pop(module_name, [])
        self._detach_finder(module_name)
        for callback in callbacks:
            callback()

    def install(self) -> None:
        if self._installed:
            return
        self._installed = True
        self._register_post_import(Installer._ASCEND_CONFIG_MODULE, self._make_ascend_config_callback())

    def reset(self) -> None:
        self._actions_cache = None
        self._effective = None
        self._deferred.clear()
        self._probe_gate_checked = False
        self._installed = False
        self._post_import_callbacks.clear()
        for module_name in list(self._post_import_finders):
            self._detach_finder(module_name)
        self._post_import_finders.clear()


INSTALLER = Installer()
