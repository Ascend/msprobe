from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from msprobe.kv_monitor import config as kv_config
from msprobe.kv_monitor.collect.compute import (
    EntryBlock,
    StepSnapshot,
    TokenRow,
    cache_block_size,
    extract_request_windows,
    gather_mean_variance,
    group_rows,
    rows_for_role,
)
from msprobe.kv_monitor.collect.slots import Segment, window_segments
from msprobe.kv_monitor.engine.adapter import EngineAdapter, RankSlotMapping

from msprobe.kv_monitor.const import PAD_SLOT_ID, ROLE_PRODUCER


@dataclass
class _Scan:
    """跨层复用的解算结果：历史窗口对所有层相同，按块大小分桶。"""

    windows: dict[str, tuple[list[int], int]] | None = None
    decode_segments: dict[int, list[Segment]] = field(default_factory=dict)


# ===================== 采集器 =====================


class KVStatsCollector:
    def __init__(self, adapter: EngineAdapter, config: kv_config.Config):
        self._adapter = adapter
        self._max_layers = config.max_layers()
        self._role = self._adapter.role()
        self._block_size: int = 0
        self._request_local_steps: dict[str, int] = {}
        self._spec_token_count: int | None = None
        self._cache_dtype = ""

    def _cp_geometry(self):
        """适配器若提供 CP 几何则取用；测试替身没有该方法时按非 CP 处理。"""
        getter = getattr(self._adapter, "cp_geometry", None)
        return getter() if callable(getter) else None

    def _spec_tokens(self) -> int:
        if self._spec_token_count is None:
            self._spec_token_count = self._adapter.speculative_tokens()
        return self._spec_token_count

    def _capture_dtype(self, actual: str) -> None:
        if not self._cache_dtype:
            self._cache_dtype = actual

    def make_token_rows(
        self,
        num_scheduled_tokens: np.ndarray,
        total_scheduled_tokens: int,
        positions: Any = None,
    ) -> list[TokenRow]:
        if total_scheduled_tokens == 0:
            return []
        request_ids = self._adapter.req_ids()
        num_reqs = len(request_ids)
        request_indices = np.repeat(self._adapter.arange(num_reqs), num_scheduled_tokens[:num_reqs])
        sample_count = min(total_scheduled_tokens, len(request_indices))
        if positions is None:
            positions = self._adapter.positions(sample_count)
        return [
            TokenRow(
                row,
                request_ids[int(request_indices[row])],
                int(positions[row]),
                int(request_indices[row]),
            )
            for row in range(sample_count)
        ]

    def build_snapshot(
        self,
        attn_metadata: Any,
        token_rows: list[TokenRow],
        request_meta: dict | None = None,
        finished_req_ids: set[str] | None = None,
    ) -> StepSnapshot:
        p_rows, d_rows = rows_for_role(self._role, token_rows, request_meta) or ([], [])
        request_stats: dict[str, list] = {}
        scan = _Scan()
        collected_layers = 0

        request_seq_lens: dict[str, int] = {}
        if d_rows:
            scan.windows = extract_request_windows(attn_metadata, d_rows, self._block_size, self._adapter)
            for request_id, entry in (scan.windows or {}).items():
                if entry:
                    request_seq_lens[request_id] = int(entry[1])
        for request_id, (_rows, positions) in group_rows(p_rows).items():
            if positions:
                request_seq_lens.setdefault(request_id, max(positions) + 1)

        for layer_name, module in self._adapter.layers().items():
            if not self._collect_layer(layer_name, module, attn_metadata, p_rows, d_rows, request_stats, scan):
                continue
            collected_layers += 1
            if self._max_layers is not None and collected_layers >= self._max_layers:
                break

        return StepSnapshot(
            self._role == ROLE_PRODUCER,
            self._block_size,
            request_stats,
            self._advance_request_steps(request_stats, finished_req_ids),
            self._cache_dtype,
            request_meta,
            frozenset(finished_req_ids or ()),
            request_seq_lens,
        )

    def _collect_layer(
        self,
        layer_name: str,
        module: Any,
        attn_metadata: Any,
        p_rows: list[TokenRow],
        d_rows: list[TokenRow],
        request_stats: dict,
        scan: _Scan,
    ) -> bool:
        components = self._adapter.kv_components(module)
        sample_cache = next((c for c in components if isinstance(c, torch.Tensor)), None)
        if sample_cache is None:
            return False
        slot_mapping = self._adapter.slot_mapping(attn_metadata, layer_name)
        if slot_mapping is None:
            return False
        if sample_cache.ndim >= 2 and self._block_size == 0:
            self._block_size = cache_block_size(sample_cache)
        if self._block_size <= 0:
            return False

        segments = self._prefill_segments(p_rows, slot_mapping)
        segments += self._decode_segments(attn_metadata, d_rows, scan)
        if not segments:
            return False
        return self._collect(layer_name, components, segments, request_stats)

    def _prefill_segments(self, token_rows: list[TokenRow], slot_mapping: torch.Tensor) -> list[Segment]:
        """本步被调度的 token：位置来自 token 行，槽位按行号直接索引 slot_mapping。

        PCP/DCP 下引擎会把不属于本 rank 的位置写成 PAD_SLOT_ID，这里一并过滤掉。
        """
        segments = []
        for request_id, (rows, positions) in group_rows(token_rows).items():
            index = torch.as_tensor(rows, dtype=torch.long, device=slot_mapping.device)
            slots = slot_mapping[index].cpu().numpy()
            keep = slots != PAD_SLOT_ID
            if not bool(keep.all()):
                positions = [position for position, kept in zip(positions, keep.tolist()) if kept]
                slots = slots[keep]
                if not positions:
                    continue
            segments.append(Segment(request_id, positions, slots))
        return segments

    def _decode_segments(self, attn_metadata: Any, d_rows: list[TokenRow], scan: _Scan) -> list[Segment]:
        """decode 侧要全量重测历史窗口，解算结果与层无关，按块大小分桶跨层复用。"""
        if not d_rows:
            return []
        if scan.windows is None:
            scan.windows = extract_request_windows(attn_metadata, d_rows, self._block_size, self._adapter)
        if not scan.windows:
            return []
        segments = scan.decode_segments.get(self._block_size)
        if segments is None:
            segments = window_segments(RankSlotMapping(self._block_size, self._cp_geometry()), scan.windows)
            scan.decode_segments[self._block_size] = segments
        return segments

    def _collect(
        self,
        layer_name: str,
        components: tuple,
        segments: list[Segment],
        request_stats: dict,
    ) -> bool:
        """一次读设备：所有请求的槽位拼在一起取值，再按段切回各自的位置。"""
        slots_concat = np.concatenate([segment.slots for segment in segments])
        gathered, used = gather_mean_variance(components, slots_concat)
        if gathered is None:
            return False
        self._capture_dtype(str(components[used[0]].dtype))
        slots_list = slots_concat.tolist()
        num_used = len(used)
        offset = 0
        for segment in segments:
            upper = offset + len(segment.slots)
            components_of_request = request_stats.setdefault(segment.request_id, {})
            for index, component in enumerate(used):
                components_of_request.setdefault(layer_name, {})[component] = EntryBlock(
                    segment.positions,
                    slots_list[offset:upper],
                    gathered[index][offset:upper].tolist(),
                    gathered[num_used + index][offset:upper].tolist(),
                )
            offset = upper
        return True

    def _advance_request_steps(
        self,
        request_stats: dict,
        finished_req_ids: set[str] | None = None,
    ) -> dict[str, int]:
        for request_id in finished_req_ids or ():
            self._request_local_steps.pop(request_id, None)
        request_steps: dict[str, int] = {}
        for request_id in request_stats:
            local_step = self._request_local_steps.get(request_id, 0)
            request_steps[request_id] = local_step
            self._request_local_steps[request_id] = local_step + 1
        return request_steps
