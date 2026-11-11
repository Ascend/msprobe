from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from typing import Callable, Iterable, Optional, Protocol

from msprobe.core.common.log import logger
from msprobe.kv_monitor.dump import writer
from msprobe.kv_monitor.collect.compute import StepSnapshot, is_producer_meta
from msprobe.kv_monitor.utils import distributed_rank, get_external_request_id
from msprobe.kv_monitor.utils import now_epoch

from msprobe.kv_monitor.const import (
    ROLE_CONSUMER,
    ROLE_PRODUCER,
)


_ENTRY_FMT = '{"pos":%s,"slot":%s,"mean":%s,"var":%s}'
_PAYLOAD_HEAD_FMT = '{"meta":%s,"per_token_stats":{'
_PAYLOAD_LAYER_FMT = '%s"%s":{'
_PAYLOAD_BLOCK_FMT = '%s"%s":[%s]'


class ArchiveLayout:
    """落盘目录布局：{stats_dir}/rank{rank}/{external_request_id}/step{N}.json。"""

    def __init__(
        self,
        stats_dir_provider: Callable[[], str],
        rank_provider: Callable[[], int] = distributed_rank,
    ):
        self._stats_dir_provider = stats_dir_provider
        self._rank_provider = rank_provider

    def rank_root(self) -> Optional[str]:
        stats_dir = (self._stats_dir_provider() or "").strip()
        if not stats_dir:
            logger.warning("[kv-dump-policy] stats_dir is empty, skip this step")
            return None
        return f"{stats_dir.rstrip('/')}/rank{self._rank_provider()}"

    @classmethod
    def request_dir(cls, rank_root: str, request_id: str) -> str:
        return f"{rank_root}/{get_external_request_id(request_id)}"

    @staticmethod
    def timestamp() -> float:
        return now_epoch()


def _render_scalar(value) -> str:
    """标量转 JSON 文本：numpy 标量先降级；NaN/Inf 由 json.dumps 写成 NaN/Infinity。"""
    return json.dumps(value.item() if hasattr(value, "item") else value)


def _entry_scalars(stat, index: int) -> tuple:
    return (
        _render_scalar(stat.positions[index]),
        _render_scalar(stat.slots[index]),
        _render_scalar(stat.means[index]),
        _render_scalar(stat.variances[index]),
    )


def _meta_json(meta: dict) -> str:
    return json.dumps(meta, separators=(",", ":"))


def _entry_body(stat, index: int) -> str:
    return _ENTRY_FMT % _entry_scalars(stat, index)


def _payload_chunks(meta_str: str, layers) -> Iterable[str]:
    """layers: 层模块路径 → 分量序号 → 条目字节串序列（沿用 request_stats 的两级结构）。"""
    yield _PAYLOAD_HEAD_FMT % meta_str
    for layer_index, (layer, components) in enumerate(layers.items()):
        yield _PAYLOAD_LAYER_FMT % ("" if layer_index == 0 else ",", layer)
        for index, (component, bodies) in enumerate(components.items()):
            yield _PAYLOAD_BLOCK_FMT % ("" if index == 0 else ",", component, ",".join(bodies))
        yield "}"
    yield "}}"


@dataclass
class CachedBlock:
    slots: list
    means: list
    variances: list
    entry_bodies: list[str]

    @classmethod
    def from_stat(cls, stat) -> "CachedBlock":
        bodies = [_entry_body(stat, i) for i in range(len(stat.positions))]
        slots, means, variances = _block_state(stat)
        return cls(
            slots=slots,
            means=means,
            variances=variances,
            entry_bodies=bodies,
        )

    def _window_unchanged(self, stat) -> bool:
        return (
            len(stat.positions) == len(self.slots)
            and stat.slots == self.slots
            and stat.means == self.means
            and stat.variances == self.variances
        )

    def _try_append_growth(self, stat, build_body: Callable) -> bool:
        new_count = len(stat.positions)
        prev_count = len(self.slots)
        if not (new_count > prev_count):
            return False
        if not (
            stat.slots[:prev_count] == self.slots
            and stat.means[:prev_count] == self.means
            and stat.variances[:prev_count] == self.variances
        ):
            return False
        new_bodies = [build_body(stat, i) for i in range(prev_count, new_count)]
        self.slots.extend(stat.slots[prev_count:])
        self.means.extend(stat.means[prev_count:])
        self.variances.extend(stat.variances[prev_count:])
        self.entry_bodies.extend(new_bodies)
        return True

    def _rebuild(self, stat, new_bodies: list[str]) -> None:
        self.slots, self.means, self.variances = _block_state(stat)
        self.entry_bodies = new_bodies


@dataclass
class CachedRequest:
    layers: dict[str, dict[str, CachedBlock]]


def _block_state(stat) -> tuple[list, list, list]:
    return list(stat.slots), list(stat.means), list(stat.variances)


def _meta_dict(
    role: str,
    request_id: str,
    time_value: float,
    prompt_len: int,
    step: int,
    seq_len: int,
    snapshot: StepSnapshot,
) -> dict:
    return {
        "role": role,
        "req_id": get_external_request_id(request_id),
        "time": time_value,
        "block_size": snapshot.block_size,
        "prompt_len": prompt_len,
        "dtype": snapshot.dtype,
        "step": step,
        "seq_len": seq_len,
    }


def _resolve_prompt_len(
    request_id: str,
    request_meta: dict | None,
    cache: dict[str, int] | None = None,
) -> int | None:
    if cache is not None and request_id in cache:
        return cache[request_id]
    meta = request_meta.get(request_id) if request_meta else None
    if meta is None or meta.prompt_len is None:
        return None
    prompt_len = int(meta.prompt_len)
    if cache is not None:
        cache[request_id] = prompt_len
    return prompt_len


def _stat_count(request_stats) -> int:
    return sum(len(components) for components in request_stats.values())


def _block_count_matches(cached_request, request_stats) -> bool:
    if cached_request is None:
        return False
    total = sum(len(components) for components in cached_request.layers.values())
    return total == _stat_count(request_stats)


class DumpPolicy(Protocol):
    def write(self, snapshot: StepSnapshot, layout: ArchiveLayout) -> None: ...


WritePlanEntry = tuple[DumpPolicy, StepSnapshot]


class _DirCreatingPolicy:
    def _ensure_request_dir(self, layout: ArchiveLayout, rank_root: Optional[str], request_id: str) -> Optional[str]:
        """建目录与写文件采用同一套 best-effort 语义：失败只告警并跳过该请求。"""
        if not rank_root:
            return None
        request_dir = layout.request_dir(rank_root, request_id)
        try:
            os.makedirs(request_dir, exist_ok=True)
        except OSError as exc:
            logger.warning(f"[kv-dump-policy] create dir failed, skip request: {request_dir}: {exc}")
            return None
        return request_dir


class ProducerDumpPolicy(_DirCreatingPolicy):
    def __init__(self, sink) -> None:
        super().__init__()
        self.prompt_len_cache: dict[str, int] = {}
        self._sink = sink

    @staticmethod
    def _bodies_by_layer(request_stats) -> tuple[dict[str, dict[int, tuple]], int]:
        layers: dict[str, dict[int, tuple]] = {}
        max_pos = -1
        for layer, components in request_stats.items():
            bodies_by_component = layers.setdefault(layer, {})
            for component, stat in components.items():
                bodies = []
                for i in range(len(stat.positions)):
                    pos = int(stat.positions[i])
                    max_pos = max(max_pos, pos)
                    bodies.append(_entry_body(stat, i))
                bodies_by_component[component] = tuple(bodies)
        return layers, max_pos

    def write(self, snapshot: StepSnapshot, layout: ArchiveLayout) -> None:
        time_value = layout.timestamp()
        rank_root = layout.rank_root()
        if rank_root is None:
            return
        for request_id, request_stats in snapshot.request_stats.items():
            step = snapshot.request_steps.get(request_id, 0)
            prompt_len = _resolve_prompt_len(request_id, snapshot.request_meta, self.prompt_len_cache)
            request_dir = self._ensure_request_dir(layout, rank_root, request_id)
            if request_dir is None:
                continue
            layers, max_pos = self._bodies_by_layer(request_stats)
            seq_len = snapshot.request_seq_lens.get(request_id) or (max_pos + 1 if max_pos >= 0 else 0)
            meta_str = _meta_json(
                _meta_dict(
                    ROLE_PRODUCER,
                    request_id,
                    time_value,
                    prompt_len,
                    step,
                    seq_len,
                    snapshot,
                )
            )
            path = f"{request_dir}/step{step}.json"
            self._sink.submit(path, _payload_chunks(meta_str, layers))
        for request_id in snapshot.finished_req_ids:
            self.prompt_len_cache.pop(request_id, None)


class ConsumerDumpPolicy(_DirCreatingPolicy):
    def __init__(self, sink):
        super().__init__()
        self._sink = sink
        self.request_cache: dict[str, CachedRequest] = {}
        self.prompt_len_cache: dict[str, int] = {}

    def write(self, snapshot: StepSnapshot, layout: ArchiveLayout) -> None:
        time_value = layout.timestamp()
        rank_root = layout.rank_root()
        if rank_root is None:
            return
        for request_id, request_stats in snapshot.request_stats.items():
            step = snapshot.request_steps.get(request_id, 0)
            payload = self._build_payload(request_id, request_stats, step, snapshot, time_value)
            if payload is None:
                continue
            request_dir = self._ensure_request_dir(layout, rank_root, request_id)
            if request_dir is None:
                continue
            self._sink.submit(f"{request_dir}/step{step}.json", payload)
        for request_id in snapshot.finished_req_ids:
            self.request_cache.pop(request_id, None)
            self.prompt_len_cache.pop(request_id, None)

    def _build_payload(self, request_id, request_stats, step, snapshot, time_value) -> Iterable[str] | None:
        prompt_len = _resolve_prompt_len(request_id, snapshot.request_meta, self.prompt_len_cache)
        last_stat = None
        for components in request_stats.values():
            for stat in components.values():
                last_stat = stat
        seq_len = snapshot.request_seq_lens.get(request_id) or (len(last_stat.positions) if last_stat else 0)
        meta_str = _meta_json(
            _meta_dict(
                ROLE_CONSUMER,
                request_id,
                time_value,
                prompt_len,
                step,
                seq_len,
                snapshot,
            )
        )
        cached_request = self._refresh_request_cache(request_id, request_stats)
        layers = {
            layer: {component: tuple(block.entry_bodies) for component, block in components.items()}
            for layer, components in cached_request.layers.items()
        }
        return _payload_chunks(meta_str, layers)

    def _refresh_request_cache(self, request_id, request_stats) -> CachedRequest:
        cached_request = self.request_cache.get(request_id)
        if not _block_count_matches(cached_request, request_stats):
            cached_request = self._serialize_all_blocks(request_stats)
        else:
            self._update_blocks(cached_request, request_stats)
        self.request_cache[request_id] = cached_request
        return cached_request

    def _serialize_all_blocks(self, request_stats) -> CachedRequest:
        layers: dict[str, dict[int, CachedBlock]] = {}
        for layer, components in request_stats.items():
            for component, stat in components.items():
                layers.setdefault(layer, {})[component] = CachedBlock.from_stat(stat)
        return CachedRequest(layers=layers)

    def _update_blocks(self, cached_request: CachedRequest, request_stats) -> None:
        for layer, components in request_stats.items():
            cached_components = cached_request.layers.setdefault(layer, {})
            for component, stat in components.items():
                cached_block = cached_components.get(component)
                if cached_block is None:
                    cached_components[component] = CachedBlock.from_stat(stat)
                    continue
                if cached_block._window_unchanged(stat):
                    continue
                if cached_block._try_append_growth(stat, _entry_body):
                    continue
                cached_block._rebuild(stat, [_entry_body(stat, i) for i in range(len(stat.positions))])


def _split_snapshot_by_meta(
    snapshot: StepSnapshot,
) -> tuple[StepSnapshot | None, StepSnapshot | None]:
    p_stats: dict = {}
    d_stats: dict = {}
    for request_id, stats in snapshot.request_stats.items():
        if is_producer_meta(snapshot.request_meta.get(request_id)):
            p_stats[request_id] = stats
        else:
            d_stats[request_id] = stats

    def _subset(stats: dict) -> StepSnapshot:
        return replace(
            snapshot,
            request_stats=stats,
            request_steps={rid: snapshot.request_steps.get(rid, 0) for rid in stats},
            request_meta=snapshot.request_meta,
        )

    return (
        _subset(p_stats) if p_stats else None,
        _subset(d_stats) if d_stats else None,
    )


class MonitorRuntime:
    def __init__(self, collector, stats_dir_provider: Callable[[], str], sink=None):
        self.collector = collector
        self._layout = ArchiveLayout(stats_dir_provider)
        sink = sink if sink is not None else writer.default_writer
        self.producer_policy = ProducerDumpPolicy(sink=sink)
        self.consumer_policy = ConsumerDumpPolicy(sink=sink)

    def _write_plan(self, snapshot: StepSnapshot) -> list[WritePlanEntry]:
        if snapshot.request_meta is None:
            policy = self.producer_policy if snapshot.is_prefill else self.consumer_policy
            return [(policy, snapshot)]
        p_subset, d_subset = _split_snapshot_by_meta(snapshot)
        plan: list[WritePlanEntry] = []
        if p_subset is not None:
            plan.append((self.producer_policy, p_subset))
        if d_subset is not None:
            plan.append((self.consumer_policy, d_subset))
        return plan

    def write_snapshot(self, snapshot: StepSnapshot) -> None:
        for policy, subset in self._write_plan(snapshot):
            policy.write(subset, self._layout)
