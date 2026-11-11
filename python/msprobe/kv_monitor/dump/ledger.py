from __future__ import annotations

import json
import os
import threading

from msprobe.kv_monitor import config as kv_config
from msprobe.kv_monitor.engine.adapter import adapt_to
from msprobe.kv_monitor.utils import distributed_rank, now_epoch


class WriteLedger:
    def __init__(self):
        self._lock = threading.Lock()

    def record(self, obj: dict, vllm_config=None) -> None:
        self.record_many([obj], vllm_config=vllm_config)

    def record_many(self, rows: list, vllm_config=None) -> None:
        """批量写入：一次 makedirs + 一次 open/append，避免逐行系统调用。

        账本位于 execute_model 热路径，逐行 open/close 在慢盘/NFS 上会显著放大步时延。
        """
        if not rows:
            return
        try:
            config = kv_config.DEFAULT_CONFIG.load(vllm_config)
            if config is None:
                return
            base_dir = config.require_stats_dir().rstrip("/")
            built = [self._build_row(row) for row in rows]
            with self._lock:
                self._write_rows(base_dir, built)
        except Exception as exc:
            # 账本自身失败不应打断推理：统一落一行错误记录（再失败也只放弃）
            self.record_error(exc, vllm_config)

    def record_error(self, error, vllm_config=None) -> None:
        try:
            config = kv_config.DEFAULT_CONFIG.load(vllm_config)
            if config is None:
                return
            base_dir = config.require_stats_dir().rstrip("/")
            row = self._build_row({"write": "msprobe_error", "error": str(error)})
            with self._lock:
                self._write_rows(base_dir, [row])
        except Exception:
            return

    def _build_row(self, obj: dict) -> dict:
        row = {**obj}
        row.setdefault("tid", threading.current_thread().ident)
        row.setdefault("time", now_epoch())
        return row

    def _write_rows(self, base_dir: str, rows: list) -> None:
        rank_dir = os.path.join(base_dir, f"rank{distributed_rank()}")
        os.makedirs(rank_dir, exist_ok=True)
        with open(os.path.join(rank_dir, "ledger.jsonl"), "a", encoding="utf-8") as file_handle:
            for row in rows:
                file_handle.write(json.dumps(row, default=str) + "\n")


default_ledger = WriteLedger()


def record_rows(target, rows) -> None:
    """批量落账；调用方已算好行数据时用它，避免重复计算。"""
    try:
        if not rows:
            return
        default_ledger.record_many(rows, vllm_config=adapt_to(target).vllm_config())
    except Exception as exc:
        default_ledger.record_error(exc)


def record_patched_call(target, build_rows, *args, **kwargs) -> None:
    try:
        rows = build_rows(target, *args, **kwargs)
    except Exception as exc:
        default_ledger.record_error(exc)
        return
    record_rows(target, rows)
