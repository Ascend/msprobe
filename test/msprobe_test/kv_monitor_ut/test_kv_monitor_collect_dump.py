
import json
import os
import sys
from types import SimpleNamespace

import pytest

from msprobe.kv_monitor import config as kv_config
from msprobe.kv_monitor.collect.collector import KVStatsCollector, _Scan
from msprobe.kv_monitor.collect.compute import RequestMeta, StepSnapshot, TokenRow
from msprobe.kv_monitor.collect.slots import window_segments
from msprobe.kv_monitor.dump import policy as dump
from msprobe.kv_monitor.dump import writer as kv_writer
from msprobe.kv_monitor.dump.policy import (
    ArchiveLayout,
    ConsumerDumpPolicy,
    MonitorRuntime,
    ProducerDumpPolicy,
)
from msprobe.kv_monitor.engine.adapter import RankSlotMapping, adapt_to


class _RecordingSink:

    def __init__(self):
        self.files: dict[str, str] = {}

    def submit(self, path, payload):
        self.files[path] = payload if isinstance(payload, str) else "".join(payload)

    def pending(self):
        return 0


def _stat(positions, means=None, slots=None, variances=None):
    count = len(positions)
    return SimpleNamespace(
        positions=list(positions),
        slots=list(slots if slots is not None else range(count)),
        means=list(means if means is not None else [0.5] * count),
        variances=list(variances if variances is not None else [0.25] * count),
    )


def _snapshot(role, stats, *, step=0, prompt_len=4, is_prefill=False, **kw):
    meta = {rid: RequestMeta(role, prompt_len) for rid in stats}
    return StepSnapshot(
        is_prefill,
        kw.get("block_size", 16),
        stats,
        {rid: step for rid in stats},
        kw.get("dtype", "float16"),
        meta,
        frozenset(kw.get("finished_req_ids") or ()),
    )


@pytest.fixture()
def layout(tmp_path, monkeypatch):
    monkeypatch.setattr(dump, "now_epoch", lambda: 0.0)
    return ArchiveLayout(lambda: str(tmp_path), rank_provider=lambda: 0)


def test_producer_payload_byte_golden(layout, tmp_path):
    sink = kv_writer.AsyncDumpWriter(worker_threads=0)
    policy = ProducerDumpPolicy(sink=sink)
    policy.write(
        _snapshot("P", {"req-a": {"layer0": {0: _stat([0, 1], means=[0.5, 1.0], slots=[3, 4], variances=[0.25, 0.0])}}},
                  is_prefill=True),
        layout,
    )
    raw = (tmp_path / "rank0" / "req-a" / "step0.json").read_text(encoding="utf-8")

    assert raw.split('"per_token_stats"', 1)[1] == (
        ':{"layer0":{"0":[{"pos":0,"slot":3,"mean":0.5,"var":0.25},'
        '{"pos":1,"slot":4,"mean":1.0,"var":0.0}]}}}'
    )
    assert '"meta":{' in raw and '"meta": {' not in raw
    assert '"pos": ' not in raw
    assert '"role":"P"' in raw and '"role": "P"' not in raw

    meta = json.loads(raw)["meta"]
    assert meta == {
        "role": "P",
        "req_id": "req-a",
        "time": ArchiveLayout.timestamp(),
        "block_size": 16,
        "prompt_len": 4,
        "dtype": "float16",
        "step": 0,
        "seq_len": 2,
    }


def test_consumer_payload_compact_dialect(layout, tmp_path):
    sink = _RecordingSink()
    ConsumerDumpPolicy(sink=sink).write(
        _snapshot("D", {"req-d": {"layer0": {0: _stat([0], means=[2.0], slots=[7], variances=[1.5])}}}),
        layout,
    )
    body = next(iter(sink.files.values()))
    assert body.split('"per_token_stats"', 1)[1] == ':{"layer0":{"0":[{"pos":0,"slot":7,"mean":2.0,"var":1.5}]}}}'
    assert '"meta":{' in body and '"meta": {' not in body


def test_caches_cleared_on_request_finish(layout, tmp_path):
    sink = _RecordingSink()
    snap_p = _snapshot("P", {"r1": {"layer0": {0: _stat([0])}}}, is_prefill=True)
    producer = ProducerDumpPolicy(sink=sink)
    producer.write(snap_p, layout)
    assert "r1" in producer.prompt_len_cache
    producer.write(
        _snapshot("P", {"r1": {"layer0": {0: _stat([0])}}}, is_prefill=True, finished_req_ids={"r1"}),
        layout,
    )
    assert "r1" not in producer.prompt_len_cache

    snap_d = _snapshot("D", {"r2": {"layer0": {0: _stat([0])}}})
    consumer = ConsumerDumpPolicy(sink=sink)
    consumer.write(snap_d, layout)
    assert "r2" in consumer.request_cache and "r2" in consumer.prompt_len_cache
    consumer.write(
        _snapshot("D", {"r2": {"layer0": {0: _stat([0])}}}, finished_req_ids={"r2"}),
        layout,
    )
    assert "r2" not in consumer.request_cache and "r2" not in consumer.prompt_len_cache


def test_meta_key_contract(layout, tmp_path):
    sink = kv_writer.AsyncDumpWriter(worker_threads=0)
    plain = ProducerDumpPolicy(sink=sink)
    plain.write(_snapshot("P", {"r": {"layer0": {0: _stat([0])}}}, is_prefill=True), layout)
    base_keys = list(json.loads((tmp_path / "rank0" / "r" / "step0.json").read_text())["meta"])
    assert base_keys == [
        "role", "req_id", "time", "block_size", "prompt_len",
        "dtype", "step", "seq_len",
    ]

    other_layout = ArchiveLayout(lambda: str(tmp_path), rank_provider=lambda: 3)
    ProducerDumpPolicy(sink=sink).write(
        _snapshot("P", {"r2": {"layer0": {0: _stat([0])}}}, is_prefill=True), other_layout
    )
    other_meta = json.loads((tmp_path / "rank3" / "r2" / "step0.json").read_text())["meta"]
    assert list(other_meta) == base_keys


def test_consumer_increment_append_then_rebuild(layout):
    sink = _RecordingSink()
    policy = ConsumerDumpPolicy(sink=sink)

    policy.write(_snapshot("D", {"r": {"layer0": {0: _stat([0, 1], means=[0.5, 0.5])}}}), layout)
    policy.write(_snapshot("D", {"r": {"layer0": {0: _stat([0, 1, 2], means=[0.5, 0.5, 0.5])}}}), layout)
    appended = _consumer_file(sink)["per_token_stats"]["layer0"]["0"]
    assert len(appended) == 3 and [e["pos"] for e in appended] == [0, 1, 2]

    sink.files.clear()
    policy.write(_snapshot("D", {"r": {"layer0": {0: _stat([0, 1, 2], means=[9.0, 0.5, 0.5])}}}), layout)
    rebuilt = _consumer_file(sink)["per_token_stats"]["layer0"]["0"]
    assert [e["mean"] for e in rebuilt] == [9.0, 0.5, 0.5]


def test_consumer_stale_key_not_evicted_pin(layout):
    sink = _RecordingSink()
    policy = ConsumerDumpPolicy(sink=sink)
    policy.write(_snapshot("D", {"r": {"layer0": {0: _stat([0])}, "layer1": {0: _stat([0])}}}), layout)
    policy.write(_snapshot("D", {"r": {"layer0": {0: _stat([0])}, "layer2": {0: _stat([0])}}}, step=1), layout)

    assert set(_consumer_file(sink)["per_token_stats"]) == {"layer0", "layer1", "layer2"}
    assert set(policy.request_cache["r"].layers) == {"layer0", "layer1", "layer2"}


def test_window_unchanged_compares_three_lists(layout):
    slot = _stat([0, 1], means=[1.0, 2.0], variances=[0.1, 0.2])
    cached = dump.CachedBlock(slots=[0, 1], means=[1.0, 2.0], variances=[0.1, 0.2], entry_bodies=[])
    assert cached._window_unchanged(slot)
    assert not cached._window_unchanged(_stat([0, 1], means=[1.0, 2.5], variances=[0.1, 0.2]))
    assert not cached._window_unchanged(_stat([0], means=[1.0], variances=[0.1]))


def test_producer_writes_only_through_sink(layout, tmp_path):
    sink = _RecordingSink()
    ProducerDumpPolicy(sink=sink).write(
        _snapshot("P", {"req-s": {"layer0": {0: _stat([0])}}}, is_prefill=True), layout
    )

    assert list(sink.files) == [f"{tmp_path}/rank0/req-s/step0.json"]
    body = sink.files[str(tmp_path) + "/rank0/req-s/step0.json"]
    assert isinstance(body, str) and body.startswith("{") and body.endswith("}")
    assert json.loads(body)["meta"]["role"] == "P"
    assert not (tmp_path / "rank0" / "req-s" / "step0.json").exists()


def test_monitor_runtime_hands_same_sink_to_both_policies(layout, tmp_path):
    sink = _RecordingSink()
    runtime = MonitorRuntime(None, lambda: str(tmp_path), sink=sink)
    assert runtime.producer_policy._sink is sink and runtime.consumer_policy._sink is sink

    runtime.write_snapshot(_snapshot("P", {"req-p": {"layer0": {0: _stat([0])}}}, is_prefill=True))
    assert list(sink.files) == [f"{tmp_path}/rank0/req-p/step0.json"]


def test_request_local_steps_increment():
    collector = KVStatsCollector(
        adapt_to(
            SimpleNamespace(vllm_config=SimpleNamespace(cache_config=SimpleNamespace(cache_dtype="auto")))
        ),
        kv_config.Config({}),
    )
    assert collector._advance_request_steps({"rid-1": []}) == {"rid-1": 0}
    assert collector._advance_request_steps({"rid-1": []}) == {"rid-1": 1}
    assert collector._request_local_steps["rid-1"] == 2
    assert collector._advance_request_steps({"rid-1": []}, {"rid-1"}) == {"rid-1": 0}
    assert collector._request_local_steps["rid-1"] == 1


def test_collector_port_reads_role_from_adapter():
    from msprobe.kv_monitor.engine.adapter import VllmEngineAdapter, adapt_to

    engine = SimpleNamespace(
        is_kv_producer=True,
        is_kv_consumer=False,
    )
    adapter = adapt_to(engine)
    from_adapter = KVStatsCollector(adapter, kv_config.Config({}))

    assert isinstance(adapter, VllmEngineAdapter)
    assert from_adapter._adapter is adapter
    assert from_adapter._role == "P"


def test_collector_uses_adapter_as_injected():

    class Stub:
        def role(self):
            return "P"

        def layers(self):
            return {"model.layers.0": None}

    stub = Stub()
    col = KVStatsCollector(stub, kv_config.Config({}))
    assert col._adapter is stub
    assert col._role == "P"
    assert col._adapter.layers() == {"model.layers.0": None}

    bare = SimpleNamespace(is_kv_producer=True, is_kv_consumer=False)
    try:
        KVStatsCollector(bare, kv_config.Config({}))
    except AttributeError:
        pass
    else:
        raise AssertionError("raw engine must not be accepted; adapt_to is the caller's job")


def test_collector_missing_block_table_raises_like_engine_access():
    from msprobe.kv_monitor.engine.adapter import adapt_to

    with pytest.raises(AttributeError):
        adapt_to(SimpleNamespace(input_batch=SimpleNamespace())).slot_values(3)


def test_window_segments_skips_uncovered_requests():
    mapping = RankSlotMapping(4)
    windows = {
        "r1": ([2, 3], 6),
        "r2": ([1], 6),
        "r3": (None, 6),
        "r4": ([5, 6], 6),
    }
    segments = window_segments(mapping, windows)

    assert [segment.request_id for segment in segments] == ["r1", "r4"]
    assert [segment.positions for segment in segments] == [list(range(6)), list(range(6))]
    assert segments[0].slots.tolist() == (
        [2 * 4 + i for i in range(4)] + [3 * 4 + i for i in range(2)]
    )
    assert segments[1].slots.tolist() == (
        [5 * 4 + i for i in range(4)] + [6 * 4 + i for i in range(2)]
    )


def test_decode_recompute_history_bit_identical_across_steps():
    torch = pytest.importorskip("torch")
    from _null_ports import _NullAdapter

    class _Runner(_NullAdapter):
        pass

    cache = torch.arange(8 * 4 * 2, dtype=torch.float32).reshape(8, 4, 2)
    collector = KVStatsCollector(_Runner(), kv_config.Config({}))
    blocks = [2, 3]

    entry_blocks = []
    for count in (6, 7):
        stats: dict = {}
        segments = window_segments(RankSlotMapping(4), {"rid": (blocks, count)})
        collector._collect("layer0", (cache, cache), segments, stats)
        assert len(stats["rid"]["layer0"]) == 2
        entry_blocks.append(stats["rid"]["layer0"][0])

    first, second = entry_blocks
    assert (len(first.positions), len(second.positions)) == (6, 7)
    assert first.positions == list(range(6)) and second.positions == list(range(7))
    assert list(zip(second.positions, second.slots, second.means, second.variances))[:6] == list(
        zip(first.positions, first.slots, first.means, first.variances)
    )
    assert second.slots[6] == 3 * 4 + 2


def test_decode_segments_reused_per_block_size_only_within_step():
    from _null_ports import _NullAdapter
    import msprobe.kv_monitor.collect.collector as collector_mod

    class _Runner(_NullAdapter):
        pass

    collector = KVStatsCollector(_Runner(), kv_config.Config({}))
    rows = [TokenRow(0, "rid", 5, 0)]
    calls: list[int] = []
    original = collector_mod.window_segments
    collector_mod.window_segments = lambda mapping, windows: (
        calls.append(mapping.block_size) or original(mapping, windows)
    )
    try:
        scan = _Scan(windows={"rid": ([2, 3, 4], 6)})
        collector._block_size = 4
        first = collector._decode_segments(None, rows, scan)
        second = collector._decode_segments(None, rows, scan)
        assert calls == [4]
        assert first is second
        collector._block_size = 2
        collector._decode_segments(None, rows, scan)
        assert calls == [4, 2]
    finally:
        collector_mod.window_segments = original


def _consumer_file(sink, index=-1):
    return json.loads(list(sink.files.values())[index])


def test_partial_fake_adapter_is_not_rewrapped_as_engine():
    from _null_ports import _NullAdapter

    from msprobe.kv_monitor.engine.adapter import (
        EngineAdapter,
        VllmEngineAdapter,
        adapt_to,
    )

    seen = []

    class Stub(_NullAdapter):
        def layers(self):
            seen.append("layers")
            return {}

        def positions(self, count):
            seen.append("positions")
            return list(range(count))

        def role(self):
            seen.append("role")
            return "P"

    stub = Stub()
    assert isinstance(stub, EngineAdapter)
    assert adapt_to(stub) is stub
    assert adapt_to(stub) is stub
    col = KVStatsCollector(stub, kv_config.Config({}))
    assert col._adapter is stub
    assert col._role == "P"
    assert col._adapter.layers() == {}
    assert seen == ["role", "layers"]

    assert adapt_to(stub).role() == "P"

    engine = SimpleNamespace(
        vllm_config=SimpleNamespace(cache_config=SimpleNamespace(cache_dtype="auto")),
        cache_config=SimpleNamespace(block_size=128),
        kv_cache_config=SimpleNamespace(kernel_block_size=128),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1, cp_kv_cache_interleave_size=1
        ),
    )
    wrapped = adapt_to(engine)
    assert isinstance(wrapped, VllmEngineAdapter)
    assert not isinstance(engine, EngineAdapter)
    assert adapt_to(engine) is adapt_to(engine) is wrapped

    class AdHoc:
        pass

    EngineAdapter.register(AdHoc)
    ad_hoc = AdHoc()
    assert adapt_to(ad_hoc) is ad_hoc
