
from __future__ import annotations

from typing import Any

from msprobe.kv_monitor.engine.adapter import (
    EngineAdapter,
    RequestProgress,
)


class _NullFacts:

    def role(self) -> str: return "mixed"
    def vllm_config(self) -> Any: return {}
    def speculative_tokens(self) -> int: return 0
    def block_size(self) -> int: return 0


class _NullBatch:

    def req_ids(self) -> list: return []
    def observe_batch(self, scheduler_output: Any) -> Any: return None


class _NullKvCache:

    def kv_components(self, module: Any) -> Any: return None
    def slot_mapping(self, attn_metadata: Any, layer_name: str) -> Any: return None
    def slot_values(self, count: int) -> list | None: return None
    def resolve_kv_meta(self, attn_metadata: Any) -> tuple[list[int] | None, Any]: return None, None
    def token_positions(self, scheduler_output: Any, counts: list, total: int) -> Any: return []
    def cp_geometry(self) -> Any: return None


class _NullLoopState:

    def observe_step(self) -> Any: return None
    def layers(self) -> Any: return {}
    def arange(self, count: int) -> Any: return []
    def positions(self, count: int) -> Any: return []
    def request_progress(self, scheduler_output: Any = None) -> RequestProgress:
        return RequestProgress(prompt_tokens=[], computed_tokens=[])


class _NullAdapter(
    EngineAdapter,
    _NullFacts,
    _NullBatch,
    _NullKvCache,
    _NullLoopState,
):

    __slots__ = ()
