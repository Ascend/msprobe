
import json
import os
import sys
from types import SimpleNamespace

import pytest

from msprobe.kv_monitor import config as kv_config
from msprobe.kv_monitor.collect.compute import TokenRow
from msprobe.kv_monitor.engine.adapter import RankSlotMapping, adapt_to
from msprobe.kv_monitor.utils import get_external_request_id


def test_cp_slot_mapping_follows_engine_rule():
    from msprobe.kv_monitor.engine.adapter import CpGeometry, RankSlotMapping

    blocks = [10, 11, 12, 13]
    window = 8

    def owned(cp, mapping):
        tokens = mapping.local_tokens(window, blocks)
        return tokens.positions if tokens else []

    cp0 = CpGeometry(total_cp=2, rank=0, interleave=1, physical_block_size=4, blocks_per_phys_block=1)
    cp1 = CpGeometry(total_cp=2, rank=1, interleave=1, physical_block_size=4, blocks_per_phys_block=1)
    assert owned(cp0, RankSlotMapping(4, cp0)) == [0, 2, 4, 6]
    assert owned(cp1, RankSlotMapping(4, cp1)) == [1, 3, 5, 7]

    tokens0 = RankSlotMapping(4, cp0).local_tokens(window, blocks)
    assert tokens0.slots.tolist() == [40, 41, 42, 43]

    cp0i2 = CpGeometry(total_cp=2, rank=0, interleave=2, physical_block_size=4, blocks_per_phys_block=1)
    cp1i2 = CpGeometry(total_cp=2, rank=1, interleave=2, physical_block_size=4, blocks_per_phys_block=1)
    assert owned(cp0i2, RankSlotMapping(4, cp0i2)) == [0, 1, 4, 5]
    assert owned(cp1i2, RankSlotMapping(4, cp1i2)) == [2, 3, 6, 7]

    plain = RankSlotMapping(4).local_tokens(window, blocks)
    assert plain.positions == list(range(window))
    assert plain.slots.tolist() == [40, 41, 42, 43, 44, 45, 46, 47]


def test_require_stats_dir_fails_fast():
    assert kv_config.Config({"kv_stats_dir": "/tmp/kv"}).require_stats_dir() == "/tmp/kv"
    for bad in (
        {"kv_stats_enable": True},
        {"kv_stats_dir": "   "},
        {"kv_stats_dir": 7},
        {"kv_stats_dir": "rel/out"},
        {"kv_stats_dir": "./kv"},
        {"kv_stats_dir": "~"},
    ):
        with pytest.raises(kv_config.WatchConfigError):
            kv_config.Config(bad).require_stats_dir()


def test_monitor_switch_and_limit_normalization():
    assert kv_config.Config({"kv_stats_enable": True}).enabled()
    assert not kv_config.Config({"kv_stats_enable": "true"}).enabled()
    assert kv_config.Config({"kv_stats_max_layers": -5}).max_layers() == 0
    assert kv_config.Config({"kv_stats_max_layers": "3"}).max_layers() is None
    assert kv_config.Config({"kv_stats_max_layers": True}).max_layers() is None
    assert kv_config.Config({}).max_layers() is None
    with pytest.raises(kv_config.WatchConfigError):
        kv_config.Config().enabled()


def test_config_freezes_first_success():
    config = kv_config.Config()

    assert config.load(None) is None
    values = {"kv_stats_enable": True, "kv_stats_dir": "/tmp/kv"}
    assert config.load(SimpleNamespace(additional_config={"kv_trace": values})) is config
    assert config.require_stats_dir() == "/tmp/kv"
    assert config.enabled()
    other = SimpleNamespace(additional_config={"kv_trace": {"kv_stats_enable": False}})
    assert config.load(other) is config
    assert config.enabled()

def test_external_request_id_single_strip():
    assert get_external_request_id("req-abc-1a2b3c4d") == "req-abc"
    assert get_external_request_id("req-abc") == "req-abc"
    assert get_external_request_id("req-deadbeef") == "req"


def test_vllm_adapter_structurally_satisfies_all_ports():
    import inspect

    from msprobe.kv_monitor.engine.adapter import (
        BatchPort,
        EngineAdapter,
        FactsPort,
        KvCachePort,
        StepPort,
        VllmEngineAdapter,
    )

    ports = (FactsPort, BatchPort, KvCachePort, StepPort)
    for port in ports:
        assert port not in VllmEngineAdapter.__mro__, "implementation must not inherit from the port"
        methods = {
            n: m for n, m in vars(port).items() if not n.startswith("_") and inspect.isfunction(m)
        }
        assert methods, f"{port.__name__} port must not be empty"
        for name, member in methods.items():
            assert not getattr(member, "__isabstractmethod__", False), (
                f"{port.__name__}.{name} must not be abstract"
            )
            assert hasattr(VllmEngineAdapter, name), f"{port.__name__}.{name} is not implemented"
            assert list(inspect.signature(getattr(VllmEngineAdapter, name)).parameters) == list(
                inspect.signature(member).parameters
            ), f"{port.__name__}.{name} parameter mismatch"
    assert issubclass(VllmEngineAdapter, EngineAdapter)
    assert not [
        n for n, m in vars(EngineAdapter).items() if not n.startswith("_") and callable(m)
    ], "EngineAdapter must be a marker with no methods"


def test_null_port_mixins_cover_their_ports():
    import inspect

    from _null_ports import _NullBatch, _NullFacts, _NullKvCache, _NullLoopState
    from msprobe.kv_monitor.engine.adapter import (
        BatchPort,
        FactsPort,
        KvCachePort,
        StepPort,
    )

    pairs = (
        (FactsPort, _NullFacts),
        (BatchPort, _NullBatch),
        (KvCachePort, _NullKvCache),
        (StepPort, _NullLoopState),
    )
    for port, mixin in pairs:
        assert port not in mixin.__mro__, f"{mixin.__name__} must not inherit from the port"
        methods = {
            n: m for n, m in vars(port).items() if not n.startswith("_") and inspect.isfunction(m)
        }
        assert methods, f"{port.__name__} has no public methods"
        for name, decl in methods.items():
            impl = getattr(mixin, name, None)
            assert impl is not None, f"{mixin.__name__} is missing {port.__name__}.{name}"
            assert tuple(inspect.signature(decl).parameters) == tuple(
                inspect.signature(impl).parameters
            ), f"{mixin.__name__}.{name} parameter names differ from {port.__name__} differ"


def test_engine_field_tokens_confined_to_adapter_module():
    import ast
    import pathlib
    import re

    import msprobe.kv_monitor as pkg

    banned_attrs = (
        "cache_config",
        "kv_cache_config",
        "input_batch",
        "static_forward_context",
        "arange_np",
        "_positions_np_buf",
        "scheduled_spec_decode_tokens",
    )
    num_tokens_cpu = re.compile(r"^num_.*_tokens_cpu$")
    banned_base = "vllm_config"
    allowed = {os.path.join("engine", "adapter.py")}

    def offenders(path):
        found = []
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Attribute):
                if node.attr in banned_attrs or num_tokens_cpu.match(node.attr):
                    found.append((node.lineno, node.attr))
                elif isinstance(node.value, ast.Attribute) and node.value.attr == banned_base:
                    found.append((node.lineno, f"{banned_base}.{node.attr}"))
            elif isinstance(node, ast.Name) and (
                node.id in banned_attrs or num_tokens_cpu.match(node.id)
            ):
                found.append((node.lineno, node.id))
        return found

    pkg_dir = pathlib.Path(pkg.__file__).parent
    bad = {}
    for path in sorted(pkg_dir.rglob("*.py")):
        if str(path.relative_to(pkg_dir)) in allowed:
            continue
        hits = offenders(path)
        if hits:
            bad[str(path.relative_to(pkg_dir))] = hits
    assert not bad, f"engine field access outside engine/adapter.py: {bad}"


def test_rank_slot_slots_for_positions_matches_kernel():
    mapping = RankSlotMapping(4)
    positions = list(range(8))
    assert mapping._slots_for_positions(positions, [10, 11]).tolist() == list(range(40, 48))


def test_rank_slot_slots_for_positions_rejects_uncovered():
    mapping = RankSlotMapping(4)
    assert mapping._slots_for_positions(list(range(8)), [10]) is None
    assert mapping._slots_for_positions(list(range(8)), None) is None
    assert mapping._slots_for_positions(list(range(8)), []) is None
    assert mapping._slots_for_positions([], [10, 11]) is None
    assert RankSlotMapping(0)._slots_for_positions([0], [10]) is None


@pytest.mark.parametrize(
    "block_size, count, want",
    [
        (4, 0, 0),
        (4, 1, 1),
        (4, 4, 1),
        (4, 5, 2),
        (0, 5, 0),
    ],
)
def test_rank_slot_blocks_needed(block_size, count, want):
    mapping = RankSlotMapping(block_size)
    assert mapping._blocks_needed(count) == want


def test_rank_slot_local_tokens_is_the_single_production_entry():
    plain = RankSlotMapping(4)
    tokens = plain.local_tokens(8, [10, 11])
    assert tokens.positions == list(range(8))
    assert tokens.slots.tolist() == list(range(40, 48))
    assert tokens.slots.tolist() == plain._slots_for_positions(list(range(8)), [10, 11]).tolist()
    assert plain.local_tokens(8, [10]) is None
    assert plain.local_tokens(8, None) is None
    assert plain.local_tokens(0, [10, 11]) is None
    assert RankSlotMapping(0).local_tokens(8, [10, 11]) is None
