from __future__ import annotations

import os

from msprobe.core.common.const import FileCheckConst
from msprobe.core.common.file_utils import check_and_get_real_path, check_path_before_create


class WatchConfigError(ValueError):
    pass


class Config:
    def __init__(self, values: dict | None = None):
        self._values: dict | None = values
        self._enabled: bool = False
        self._max_layers: int | None = None
        self._stats_dir: str | None = None
        if values is not None:
            self.check()

    def load(self, vllm_config=None) -> "Config | None":
        if self._values is not None:
            return self
        values = self._resolve_values(vllm_config)
        if values is None:
            return None
        self._values = values
        try:
            self.check()
        except Exception:
            self._values = None
            raise
        return self

    def check(self) -> None:
        """校验配置并把结果缓存到实例属性；load 后只调用一次。"""
        values = self._cfg()
        enabled = values.get("kv_stats_enable", False)
        if not isinstance(enabled, bool):
            raise WatchConfigError("[kv-config] 'kv_stats_enable' must be a boolean")
        max_layers = values.get("kv_stats_max_layers")
        if max_layers is not None and (
            not isinstance(max_layers, int) or isinstance(max_layers, bool) or max_layers < 0
        ):
            raise WatchConfigError("[kv-config] 'kv_stats_max_layers' must be a non-negative integer")
        stats_dir = values.get("kv_stats_dir")
        if stats_dir is not None:
            if not isinstance(stats_dir, str) or not stats_dir.strip():
                raise WatchConfigError("[kv-config] 'kv_stats_dir' must be a non-empty path")
            stats_dir = check_and_get_real_path(stats_dir, FileCheckConst.WRITE_ABLE, must_exist=False)
            check_path_before_create(stats_dir)
            if os.path.exists(stats_dir) and not os.path.isdir(stats_dir):
                raise WatchConfigError(f"[kv-config] 'kv_stats_dir' must be a directory: {stats_dir}")
        if enabled and stats_dir is None:
            raise WatchConfigError("[kv-config] watch config is missing 'kv_stats_dir'")
        self._enabled = enabled
        self._max_layers = max_layers
        self._stats_dir = stats_dir

    def enabled(self) -> bool:
        return self._enabled

    def max_layers(self) -> int | None:
        return self._max_layers

    def stats_dir(self) -> str | None:
        return self._stats_dir

    def probe_gate(self, vllm_config=None) -> bool:
        config = self.load(vllm_config)
        if config is None:
            return False
        return config.enabled()

    def _cfg(self) -> dict:
        if self._values is None:
            raise WatchConfigError("[kv-config] watch config is not loaded (call load(vllm_config) first)")
        return self._values

    def _resolve_values(self, vllm_config=None) -> dict | None:
        """采集配置直接取自 additional_config["kv_trace"]（字典，不再读取配置文件）。"""
        additional = getattr(vllm_config, "additional_config", None)
        if not isinstance(additional, dict):
            return None
        values = additional.get("kv_trace")
        return values if isinstance(values, dict) else None


# 进程级默认配置源
DEFAULT_CONFIG = Config()
