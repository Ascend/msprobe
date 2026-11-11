from __future__ import annotations

import os


class WatchConfigError(ValueError):
    pass


class Config:
    def __init__(self, values: dict | None = None):
        self._values: dict | None = values

    def load(self, vllm_config=None) -> "Config | None":
        if self._values is not None:
            return self
        values = self._resolve_values(vllm_config)
        if values is None:
            return None
        self._values = values
        return self

    def enabled(self) -> bool:
        return self._cfg.get("kv_stats_enable", False) is True

    def max_layers(self) -> int | None:
        return self._get_non_negative_int("kv_stats_max_layers")

    def require_stats_dir(self) -> str:
        stats_dir = self._cfg.get("kv_stats_dir")
        if not isinstance(stats_dir, str) or not stats_dir.strip():
            raise WatchConfigError("[kv-config] watch config is missing 'kv_stats_dir'")
        if not os.path.isabs(stats_dir):
            # 相对路径按 worker 进程的 CWD 解析，P/D 两个进程未必一致，容易写到意外位置；
            # 这里直接拒绝，要求显式绝对路径。
            raise WatchConfigError(f"[kv-config] 'kv_stats_dir' must be an absolute path, got: {stats_dir!r}")
        return stats_dir

    def probe_gate(self, vllm_config=None) -> bool:
        config = self.load(vllm_config)
        if config is None or not config.enabled():
            return False
        config.require_stats_dir()
        return True

    @property
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

    def _get_non_negative_int(self, key: str) -> int | None:
        value = self._cfg.get(key)
        if value is None:
            return None
        if not isinstance(value, int) or isinstance(value, bool):
            return None
        return max(value, 0)


# 进程级默认配置源
DEFAULT_CONFIG = Config()
