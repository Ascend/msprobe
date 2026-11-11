from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, NamedTuple

import numpy as np
import torch

from msprobe.core.common.log import logger
from msprobe.kv_monitor.const import (
    REQUEST_INDEX_UNKNOWN,
    ROLE_CONSUMER,
    ROLE_MIXED,
    ROLE_PRODUCER,
)
from msprobe.kv_monitor.engine.adapter import EngineAdapter


class TokenRow(NamedTuple):
    row: int
    request_id: str
    position: int
    request_index: int = REQUEST_INDEX_UNKNOWN


class EntryBlock(NamedTuple):
    positions: list
    slots: list
    means: list
    variances: list


# ===================== 块表提取 =====================


def extract_request_windows(
    attn_metadata: Any,
    token_rows: list[TokenRow],
    block_size: int,
    adapter: EngineAdapter,
) -> dict[str, tuple[list[int], int]]:
    result: dict[str, tuple[list[int], int]] = {}
    try:
        if block_size <= 0 or not isinstance(attn_metadata, dict) or not attn_metadata:
            return result
        seq_lens, block_table = adapter.resolve_kv_meta(attn_metadata)
        if seq_lens is None or block_table is None:
            return result
        num_rows = int(block_table.shape[0])
        rows_by_request: dict[str, int] = {}
        for _row, request_id, _position, request_index in token_rows:
            if request_id in rows_by_request:
                continue
            if request_index < 0 or request_index >= num_rows or request_index >= len(seq_lens):
                continue
            rows_by_request[request_id] = request_index
        if not rows_by_request:
            return result
        wanted_items = list(rows_by_request.items())
        wanted = torch.as_tensor(
            [request_index for _request_id, request_index in wanted_items],
            dtype=torch.long,
            device=block_table.device,
        )
        cpu_rows = block_table.index_select(0, wanted).cpu().tolist()
        for (request_id, request_index), row in zip(wanted_items, cpu_rows):
            result[request_id] = ([int(b) for b in row], int(seq_lens[request_index]))
    except Exception as e:
        logger.error(f"[kv-monitor] extract_request_windows failed, error: {e}")
        return {}
    return result


# ===================== slot 索引与统计 =====================


def cache_block_size(cache: torch.Tensor) -> int:
    return int(cache.shape[1])


def slot_indices(cache: torch.Tensor, slots: torch.Tensor):
    block_size = cache_block_size(cache)
    max_slot = cache.shape[0] * block_size
    slot_tensor = slots.to(device=cache.device, dtype=torch.long, non_blocking=True).reshape(-1)
    valid = (slot_tensor >= 0) & (slot_tensor < max_slot)
    safe_slots = slot_tensor.clamp(0, max_slot - 1)
    block_ids = torch.div(safe_slots, block_size, rounding_mode="floor")
    return valid, block_ids, safe_slots % block_size


def masked_mean_variance(values: torch.Tensor, valid: torch.Tensor):
    variances, means = torch.var_mean(values, dim=1, unbiased=False)
    zeros = torch.zeros_like(means)
    return torch.where(valid, means, zeros), torch.where(valid, variances.clamp_min(0.0), zeros)


def gather_mean_variance(components: Iterable[Any], slots: Any) -> tuple[np.ndarray | None, list[int]]:
    gathered_device, used_component_indices = gather_mean_variance_device(components, slots)
    if gathered_device is None:
        return None, []
    return gathered_device.to(device="cpu", dtype=torch.float64).numpy(), used_component_indices


def gather_mean_variance_device(components: Iterable[Any], slots: Any) -> tuple[torch.Tensor | None, list[int]]:
    slots_tensor = torch.as_tensor(slots)
    total_slots = int(slots_tensor.numel())
    means_parts: list[torch.Tensor] = []
    variances_parts: list[torch.Tensor] = []
    used_component_indices: list[int] = []
    for component_index, cache in enumerate(components):
        if not isinstance(cache, torch.Tensor) or cache.ndim < 2 or cache.numel() == 0:
            continue
        valid, block_ids, block_offsets = slot_indices(cache, slots_tensor)
        values = cache[block_ids, block_offsets].float().reshape(total_slots, -1)
        means, variances = masked_mean_variance(values, valid)
        means_parts.append(means)
        variances_parts.append(variances)
        used_component_indices.append(component_index)
    if not means_parts:
        return None, []
    return torch.stack(means_parts + variances_parts), used_component_indices


# ===================== 请求分组 / 窗口判定 =====================


def cumsum_bounds(lengths: list[int]) -> np.ndarray:
    bounds = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(lengths, out=bounds[1:])
    return bounds


def request_spans(request_ids: list[str], lengths: list[int]) -> list[tuple[str, int, int]]:
    bounds = cumsum_bounds(lengths)
    return [(request_id, int(bounds[index]), int(bounds[index + 1])) for index, request_id in enumerate(request_ids)]


def group_rows(token_rows: list[TokenRow]) -> dict[str, tuple[list[int], list[int]]]:
    grouped: dict[str, tuple[list[int], list[int]]] = {}
    for row, request_id, position, _ in token_rows:
        entry = grouped.get(request_id)
        if entry is None:
            grouped[request_id] = ([row], [position])
        else:
            entry[0].append(row)
            entry[1].append(position)
    return grouped


def is_producer_meta(meta: Any) -> bool:
    return getattr(meta, "role", None) == ROLE_PRODUCER


def split_by_meta(token_rows: list[TokenRow], request_meta: dict | None):
    if request_meta is None:
        return None
    p_rows: list[TokenRow] = []
    d_rows: list[TokenRow] = []
    for token_row in token_rows:
        if is_producer_meta(request_meta.get(token_row.request_id)):
            p_rows.append(token_row)
        else:
            d_rows.append(token_row)
    return p_rows, d_rows


def rows_for_role(
    role: str, token_rows: list[TokenRow], request_meta: dict | None
) -> tuple[list[TokenRow], list[TokenRow]] | None:
    if role == ROLE_MIXED:
        return split_by_meta(token_rows, request_meta)
    if role == ROLE_PRODUCER:
        return token_rows, []
    if role == ROLE_CONSUMER:
        return [], token_rows
    return None


class RequestMeta(NamedTuple):
    role: str
    prompt_len: int | None = None


@dataclass(slots=True, frozen=True)
class StepSnapshot:
    is_prefill: bool
    block_size: int
    # request_id → 层模块路径 → 分量序号 → 该层该分量在本步的逐 token 统计
    request_stats: dict[str, dict[str, dict[int, "EntryBlock"]]]
    request_steps: dict
    dtype: str
    request_meta: dict | None = None
    finished_req_ids: frozenset = frozenset()
    request_seq_lens: dict = field(default_factory=dict)
