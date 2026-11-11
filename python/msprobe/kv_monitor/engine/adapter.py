from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Any, Iterator, NamedTuple, Protocol

import numpy as np
import torch

from msprobe.core.common.log import logger
from msprobe.kv_monitor.utils import attr_chain
from msprobe.kv_monitor.const import (
    ROLE_CONSUMER,
    ROLE_PRODUCER,
    ROLE_MIXED,
)


_ADAPTER_ATTR = "_kv_monitor_adapter"

# pre_sample_state 元组布局
_SCHEDULER_OUTPUT_INDEX = 0
_ATTN_METADATA_INDEX = 7

_KV_COMPONENTS = 2

# seq_lens 候选字段
_SEQ_LENS_FIELDS = ("seq_lens_list", "_seq_lens_cpu", "seq_lens_cpu", "seq_lens")

# 块表候选字段名
_BLOCK_TABLE_FIELDS = ("block_tables", "block_table_tensor")

_BLOCK_SIZE = "block_size"

_mamba_blocklist_cache: tuple | None = None


def _mamba_blocklist() -> tuple:
    global _mamba_blocklist_cache
    if _mamba_blocklist_cache is None:
        try:
            from vllm.model_executor.layers.mamba.abstract import MambaBase

            _mamba_blocklist_cache = (MambaBase,)
        except ImportError:
            _mamba_blocklist_cache = ()
    return _mamba_blocklist_cache


# ================= 对象级纯变换（调用方手里只有当步对象，没有 engine） =================


def _seq_lens_candidates(layer_meta: Any) -> Iterator[Any]:
    for name in _SEQ_LENS_FIELDS:
        yield getattr(layer_meta, name, None)


def _block_table_candidates(layer_meta: Any) -> Iterator[Any]:
    for name in _BLOCK_TABLE_FIELDS:
        yield getattr(layer_meta, name, None)


def _as_int_list(value: Any) -> list[int] | None:
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value] if value else None
    if torch.is_tensor(value):
        if value.dim() != 1 or value.numel() == 0:
            return None
        try:
            return [int(item) for item in value.detach().cpu().tolist()]
        except Exception:
            return None
    if isinstance(value, np.ndarray) and value.ndim == 1 and value.size:
        return [int(item) for item in value.tolist()]
    return None


class LocalTokens(NamedTuple):
    positions: list[int]
    slots: np.ndarray


class CpGeometry(NamedTuple):
    """PCP / DCP 下的 KV 分片几何（与引擎 block_table 的规则保持一致）。"""

    total_cp: int  # pcp_size * dcp_size
    rank: int  # dcp_size * pcp_rank + dcp_rank
    interleave: int  # cp_kv_cache_interleave_size
    physical_block_size: int  # 物理块大小（等于缓存张量的 shape[1]）
    blocks_per_phys_block: int  # hybrid 模式下每个逻辑块含的物理块数


class RankSlotMapping:
    def __init__(self, block_size: int, cp: CpGeometry | None = None) -> None:
        self.block_size = int(block_size)
        self.cp = cp

    def local_tokens(self, window: int, blocks: Any) -> LocalTokens | None:
        window = int(window)
        if window <= 0 or self.block_size <= 0 or self._block_count(blocks) == 0:
            return None
        positions = np.arange(window, dtype=np.int64)
        if self.cp is not None:
            positions = positions[self._owned_mask(positions)]
            if positions.size == 0:
                return None
        elif self._block_count(blocks) < self._blocks_needed(window):
            return None
        slots = self._slots_for_positions(positions, blocks)
        if slots is None:
            return None
        return LocalTokens(positions.tolist(), slots)

    def _owned_mask(self, positions: np.ndarray) -> np.ndarray:
        """PCP/DCP 下本 rank 只持有 1/total_cp 的位置（按 interleave 交错切分）。"""
        cp = self.cp
        if cp is None:
            return np.ones(positions.shape, dtype=bool)
        virtual_size = cp.physical_block_size * cp.total_cp
        virtual_offsets = np.remainder(positions, virtual_size)
        return (virtual_offsets // cp.interleave) % cp.total_cp == cp.rank

    def _block_count(self, blocks: Any) -> int:
        if blocks is None:
            return 0
        try:
            return len(blocks)
        except TypeError:
            return 0

    def _blocks_needed(self, count: int) -> int:
        if count <= 0 or self.block_size <= 0:
            return 0
        return (count - 1) // self.block_size + 1

    def _slots_for_positions(self, positions: np.ndarray, blocks: Any) -> np.ndarray | None:
        block_size = self.block_size
        if blocks is None or len(blocks) == 0 or block_size <= 0:
            return None
        positions = np.asarray(positions, dtype=np.int64)
        if positions.size == 0:
            return None
        cp = self.cp
        block_array = np.asarray(blocks, dtype=np.int64)
        if cp is None:
            block_index, block_offset = np.divmod(positions, block_size)
            if int(block_index.max()) >= block_array.size:
                return None
            return block_array[block_index] * block_size + block_offset
        virtual_size = cp.physical_block_size * cp.total_cp
        physical_index, virtual_offsets = np.divmod(positions, virtual_size)
        local_offsets = (
            virtual_offsets // (cp.total_cp * cp.interleave) * cp.interleave + virtual_offsets % cp.interleave
        )
        block_index = physical_index * cp.blocks_per_phys_block + local_offsets // block_size
        if int(block_index.max()) >= block_array.size:
            return None
        return block_array[block_index] * block_size + local_offsets % block_size


@dataclass(frozen=True)
class BatchObservation:
    num_scheduled: dict
    total_scheduled: int


@dataclass(frozen=True)
class StepContext:
    pre_sample_state: Any
    scheduler_output: Any
    attn_metadata: Any
    finished_req_ids: frozenset = frozenset()


@dataclass(frozen=True)
class RequestProgress:
    prompt_tokens: Any
    computed_tokens: Any


class FactsPort(Protocol):
    def role(self) -> str: ...

    def vllm_config(self) -> Any: ...

    def speculative_tokens(self) -> int: ...

    def block_size(self) -> int | None: ...


class BatchPort(Protocol):
    def req_ids(self) -> list: ...

    def observe_batch(self, scheduler_output: Any) -> BatchObservation | None: ...


class KvCachePort(Protocol):
    def kv_components(self, module: Any) -> tuple: ...

    def slot_mapping(self, attn_metadata: Any, layer_name: str) -> torch.Tensor | None: ...

    def slot_values(self, count: int) -> list | None: ...

    def token_positions(self, scheduler_output: Any, counts: list, total: int) -> Any: ...

    def cp_geometry(self) -> CpGeometry | None: ...

    def resolve_kv_meta(self, attn_metadata: Any) -> tuple[list[int] | None, Any]: ...


class StepPort(Protocol):
    def observe_step(self) -> StepContext | None: ...

    def layers(self) -> Any: ...

    def arange(self, count: int) -> Any: ...

    def positions(self, count: int) -> Any: ...

    def request_progress(self, scheduler_output: Any = None) -> RequestProgress: ...


class EngineAdapter(ABC):
    """引擎适配器的名义标记：不含任何方法，端口是结构化 Protocol。"""

    __slots__ = ()


@dataclass(frozen=True)
class VllmEngineAdapter(EngineAdapter):
    engine: Any

    def observe_batch(self, scheduler_output: Any) -> BatchObservation | None:
        if scheduler_output is None or self.engine.input_batch is None:
            return None
        return BatchObservation(
            num_scheduled=scheduler_output.num_scheduled_tokens,
            total_scheduled=scheduler_output.total_num_scheduled_tokens,
        )

    def observe_step(self) -> StepContext | None:
        state = self.engine.execute_model_state
        if state is None or len(state) <= _ATTN_METADATA_INDEX:
            return None
        scheduler_output = state[_SCHEDULER_OUTPUT_INDEX]
        attn_metadata = state[_ATTN_METADATA_INDEX]
        if (
            not (isinstance(attn_metadata, dict) and attn_metadata)
            or getattr(scheduler_output, "num_scheduled_tokens", None) is None
        ):
            logger.warning("[kv-monitor] execute_model_state layout unexpected, skip this step")
            return None
        return StepContext(
            pre_sample_state=state,
            scheduler_output=scheduler_output,
            attn_metadata=attn_metadata,
            finished_req_ids=frozenset(getattr(scheduler_output, "finished_req_ids", None) or ()),
        )

    def request_progress(self, scheduler_output: Any = None) -> RequestProgress:
        if scheduler_output is None:
            return RequestProgress(
                prompt_tokens=self.engine.input_batch.num_prompt_tokens,
                computed_tokens=self.engine.input_batch.num_computed_tokens_cpu,
            )
        num_prompt_tokens, num_computed_tokens = self._tokens_of(scheduler_output)
        return RequestProgress(
            prompt_tokens=num_prompt_tokens,
            computed_tokens=num_computed_tokens,
        )

    def slot_values(self, count: int) -> list | None:
        slot_mapping = self._slot_mapping_from_block_table(self.engine.input_batch.block_table)
        if slot_mapping is None:
            return None
        gpu_mapping = getattr(slot_mapping, "gpu", None)
        if gpu_mapping is None:
            return None
        return slot_mapping.gpu[:count].tolist()

    def token_positions(self, scheduler_output: Any, counts: list, total: int) -> Any:
        """本步每个采样 token 的全局位置。

        PCP（prefill context parallel）会在拆分后改写 runner 的 positions 缓冲，
        但 slot_mapping 是在拆分前按完整 token 集算好的；此时按 computed + offset
        还原位置，才能与 slot_mapping 对齐。
        """
        if int(getattr(self.engine, "pcp_size", 1) or 1) <= 1:
            return self.positions(total)
        progress = self.request_progress(scheduler_output)
        computed = progress.computed_tokens[: len(counts)]
        chunks = [
            np.asarray(computed[i], dtype=np.int64) + np.arange(int(count), dtype=np.int64)
            for i, count in enumerate(counts)
            if int(count) > 0
        ]
        if not chunks:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(chunks)[:total]

    def cp_geometry(self) -> CpGeometry | None:
        """读取 PCP/DCP 几何；未开启 CP 或信息不全时返回 None（回落到非 CP 公式）。"""
        try:
            from vllm.distributed import get_dcp_group, get_pcp_group

            pcp, dcp = get_pcp_group(), get_dcp_group()
            pcp_size, dcp_size = int(pcp.world_size), int(dcp.world_size)
            pcp_rank = int(pcp.rank_in_group) if pcp_size > 1 else 0
            dcp_rank = int(dcp.rank_in_group) if dcp_size > 1 else 0
        except Exception:
            return None
        total_cp = pcp_size * dcp_size
        if total_cp <= 1:
            return None
        parallel = getattr(self.engine.vllm_config, "parallel_config", None)
        interleave = int(getattr(parallel, "cp_kv_cache_interleave_size", 1) or 1)
        table = getattr(self.engine.input_batch, "block_table", None)
        if table is not None and not hasattr(table, "physical_block_size"):
            inner = getattr(table, "block_tables", None)
            if inner:
                table = inner[0]
        physical = int(getattr(table, "physical_block_size", 0) or 0)
        blocks_per_phys = int(getattr(table, "blocks_per_phys_block", 1) or 1)
        if physical <= 0:
            logger.warning("[kv-monitor] cp enabled but physical_block_size unavailable, fall back to non-cp mapping")
            return None
        return CpGeometry(
            total_cp, dcp_size * pcp_rank + dcp_rank, max(interleave, 1), physical, max(blocks_per_phys, 1)
        )

    def resolve_kv_meta(self, attn_metadata: Any) -> tuple[list[int] | None, Any]:
        layer_meta = next(iter(attn_metadata.values()))
        seq_lens = None
        for candidate in _seq_lens_candidates(layer_meta):
            resolved = _as_int_list(candidate)
            if resolved is not None:
                seq_lens = resolved
                break
        block_table = None
        for candidate in _block_table_candidates(layer_meta):
            if torch.is_tensor(candidate) and candidate.ndim == 2:
                block_table = candidate
                break
        return seq_lens, block_table

    def _slot_mapping_from_block_table(self, block_table: Any) -> Any:
        slot_mapping = getattr(block_table, "slot_mapping", None)
        if slot_mapping is None:
            try:
                slot_mapping = getattr(block_table[0], "slot_mapping", None)
            except Exception:
                slot_mapping = None
        return slot_mapping

    # —— 引擎级 ——

    def req_ids(self) -> list:
        batch = self.engine.input_batch
        return list(batch.req_ids[: batch.num_reqs])

    def arange(self, count: int) -> Any:
        return self.engine.arange_np[:count]

    def positions(self, count: int) -> Any:
        return self.engine._positions_np_buf[:count]

    def layers(self) -> Any:
        context = self.engine.compilation_config.static_forward_context
        mamba_types = _mamba_blocklist()
        if not mamba_types:
            return context
        return {name: module for name, module in context.items() if not isinstance(module, mamba_types)}

    def role(self) -> str:
        is_producer = getattr(self.engine, "is_kv_producer", False)
        is_consumer = getattr(self.engine, "is_kv_consumer", False)
        if is_producer and not is_consumer:
            return ROLE_PRODUCER
        if is_consumer and not is_producer:
            return ROLE_CONSUMER
        return ROLE_MIXED

    def speculative_tokens(self) -> int:
        config = self.engine.vllm_config.speculative_config
        if config is None:
            return 0
        tokens = config.num_speculative_tokens
        return max(int(tokens), 0) if tokens else 0

    def vllm_config(self) -> Any:
        return getattr(self.engine, "vllm_config", None)

    def block_size(self) -> int | None:
        block_size = getattr(self.engine, _BLOCK_SIZE, None)
        if not block_size:
            block_size = attr_chain(self.engine, "vllm_config", "cache_config", _BLOCK_SIZE)
        return int(block_size) if block_size else None

    def kv_components(self, module: Any) -> tuple:
        kv_cache = getattr(module, "kv_cache", None)
        if isinstance(kv_cache, torch.Tensor):
            return (kv_cache,)
        if not isinstance(kv_cache, (list, tuple)):
            return ()
        if len(kv_cache) == 1 and isinstance(kv_cache[0], (list, tuple)):
            kv_cache = kv_cache[0]  # [[K, V]] 这类单元素嵌套：先脱一层外壳
        return tuple(kv_cache[:_KV_COMPONENTS])

    def slot_mapping(self, attn_metadata: Any, layer_name: str) -> torch.Tensor | None:
        layer_metas = tuple(attn_metadata) if isinstance(attn_metadata, list) else (attn_metadata,)
        slots: list[torch.Tensor] = []
        for metadata_dict in layer_metas:
            metadata = metadata_dict.get(layer_name)
            if metadata is None:
                continue
            for source in (
                metadata,
                getattr(metadata, "decode", None),
                getattr(metadata, "prefill", None),
            ):
                if source is None:
                    continue
                slot_mapping = getattr(source, "slot_mapping", None)
                if not isinstance(slot_mapping, torch.Tensor):
                    continue
                if slot_mapping.ndim == 1:
                    flat_mapping = slot_mapping.reshape(-1)
                # shape[-1] == 2 表示 (block, offset) 二元组，与 KV_COMPONENTS（K/V 分量数）无关。
                elif slot_mapping.ndim > 1 and slot_mapping.shape[-1] == 2:
                    block_size = getattr(source, _BLOCK_SIZE, None)
                    if not (isinstance(block_size, int) and block_size > 0):
                        continue
                    slot_mapping = slot_mapping.to(torch.long)
                    flat_mapping = (slot_mapping[..., 0] * block_size + slot_mapping[..., 1]).reshape(-1)
                else:
                    continue
                slots.append(flat_mapping)
        if not slots:
            return None
        return torch.cat(slots) if len(slots) > 1 else slots[0]

    def _tokens_of(self, scheduler_output: Any) -> tuple[Any, Any]:
        num_prompt_tokens = getattr(scheduler_output, "num_prompt_tokens", None)
        num_computed_tokens = getattr(scheduler_output, "num_computed_tokens_cpu", None)
        if num_prompt_tokens is None or num_computed_tokens is None:
            batch = self.engine.input_batch
            return batch.num_prompt_tokens, batch.num_computed_tokens_cpu
        return num_prompt_tokens, num_computed_tokens


def adapt_to(engine: Any) -> EngineAdapter:
    adapter: EngineAdapter
    if isinstance(engine, EngineAdapter):
        adapter = engine
    else:
        cached = getattr(engine, _ADAPTER_ATTR, None)
        if cached is not None:
            adapter = cached
        else:
            adapter = VllmEngineAdapter(engine)
            # 引擎对象可能禁止挂属性（frozen/slots），失败只是每次重建适配器
            try:
                setattr(engine, _ADAPTER_ATTR, adapter)
            except Exception:  # nosec B110
                pass
    return adapter
