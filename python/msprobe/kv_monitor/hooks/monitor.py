from __future__ import annotations

from functools import wraps
from typing import Any, NamedTuple

import numpy as np
import torch

from msprobe.core.common.log import logger
from msprobe.kv_monitor import config as kv_config
from msprobe.kv_monitor.collect.collector import KVStatsCollector
from msprobe.kv_monitor.collect.compute import RequestMeta
from msprobe.kv_monitor.dump.policy import MonitorRuntime
from msprobe.kv_monitor.engine.adapter import adapt_to
from msprobe.kv_monitor.utils import step_is_prefill, wrap_class_method

from msprobe.kv_monitor.const import ROLE_CONSUMER, ROLE_PRODUCER


_MONITOR_ATTR = "_kv_monitor"


class CaptureBundle(NamedTuple):
    attn_metadata: Any
    token_rows: list
    request_meta: dict | None
    finished_req_ids: set = frozenset()


class KVMonitor:
    def __init__(self, runner) -> None:
        self._adapter = adapt_to(runner)
        self._runtime: MonitorRuntime | None = None
        self._stream = None

    def _config(self) -> kv_config.Config:
        return kv_config.DEFAULT_CONFIG.load(self._adapter.vllm_config())

    def _max_layers(self) -> int | None:
        config = self._config()
        if not config.enabled():
            return 0
        return config.max_layers()

    def _stats_dir(self) -> str:
        return self._config().require_stats_dir()

    def _enabled(self) -> bool:
        return self._max_layers() != 0

    def capture(self) -> CaptureBundle | None:
        if not self._enabled():
            return None

        self._ensure_runtime()

        adapter = self._adapter
        context = adapter.observe_step()
        if context is None:
            return None

        observation = adapter.observe_batch(context.scheduler_output)
        token_rows = self._token_rows(observation, context.scheduler_output)
        request_meta = self._request_meta(context.scheduler_output, observation)
        finished = context.finished_req_ids
        return CaptureBundle(context.attn_metadata, token_rows, request_meta, set(finished))

    def _token_rows(self, observation, scheduler_output) -> list:
        adapter = self._adapter
        try:
            tokens = [observation.num_scheduled.get(rid, 0) for rid in adapter.req_ids()]
            collector = self._ensure_runtime().collector
            positions = adapter.token_positions(scheduler_output, tokens, observation.total_scheduled)
            return collector.make_token_rows(
                np.asarray(tokens, dtype=np.int32),
                observation.total_scheduled,
                positions,
            )
        except Exception as e:
            logger.warning(f"[kv-cache-watch] build token rows failed, error: {e}")
            return []

    def _request_meta(self, scheduler_output, observation) -> dict | None:
        adapter = self._adapter
        if scheduler_output is None:
            logger.warning("[kv-cache-watch] build request meta failed: scheduler_output unavailable")
            return None
        try:
            req_ids = adapter.req_ids()
            progress = adapter.request_progress(scheduler_output)
            request_meta: dict = {}
            for index, request_id in enumerate(req_ids):
                prompt_len = int(progress.prompt_tokens[index])
                is_prefill = step_is_prefill(progress.computed_tokens[index], prompt_len)
                request_meta[request_id] = RequestMeta(
                    ROLE_PRODUCER if is_prefill else ROLE_CONSUMER,
                    prompt_len,
                )
            return request_meta
        except Exception as e:
            logger.warning(f"[kv-cache-watch] build request meta failed, error: {e}")
            return None

    def collect_and_dump(self, captured_state: CaptureBundle) -> None:
        try:
            runtime = self._ensure_runtime()
            # 必须在切换 stream 之前取主流：进 with 之后 current_stream() 就是 stats_stream，
            # 那时再 wait_stream(current_stream()) 等于自己等自己（空操作）。
            main_stream = torch.npu.current_stream()
            stats_stream = self._ensure_stream()
            stats_stream.wait_stream(main_stream)
            with torch.npu.stream(stats_stream):
                snapshot = runtime.collector.build_snapshot(
                    captured_state.attn_metadata,
                    captured_state.token_rows,
                    captured_state.request_meta,
                    captured_state.finished_req_ids,
                )
            runtime.write_snapshot(snapshot)
        except Exception as e:
            logger.warning(f"[kv-cache-watch] snapshot/dump failed, error: {e}")

    def _ensure_runtime(self) -> MonitorRuntime:
        if self._runtime is None:
            self._runtime = MonitorRuntime(
                collector=KVStatsCollector(self._adapter, self._config()),
                stats_dir_provider=self._stats_dir,
            )
        return self._runtime

    def _ensure_stream(self):
        if self._stream is None:
            self._stream = torch.npu.Stream()
        return self._stream


def apply_patch(cls, method_name: str = "sample_tokens") -> None:
    wrap_class_method(
        cls,
        method_name=method_name,
        guard_attr="_kv_monitor_injected",
        wrapper_factory=_make_sample_tokens_wrapper,
    )
    logger.info(f"[kv-cache-watch] msprobe KV Cache monitor injected into {cls.__name__}")


def _get_kv_monitor(runner) -> KVMonitor:
    monitor = getattr(runner, _MONITOR_ATTR, None)
    if monitor is None:
        monitor = KVMonitor(runner)
        try:
            setattr(runner, _MONITOR_ATTR, monitor)
        except Exception:
            return monitor
    return monitor


def _make_sample_tokens_wrapper(original_sample_tokens):
    @wraps(original_sample_tokens)
    def _wrapped_sample_tokens(self, *args, **kwargs):
        # pre_sample_state 会在 sample_tokens 内被消费置空，状态须在其调用前读取
        monitor = _get_kv_monitor(self)
        try:
            captured_state = monitor.capture()
        except Exception as e:
            logger.warning(f"[kv-cache-watch] capture failed, error: {e}")
            captured_state = None

        result = original_sample_tokens(self, *args, **kwargs)

        if captured_state is not None:
            monitor.collect_and_dump(captured_state)
        return result

    return _wrapped_sample_tokens
