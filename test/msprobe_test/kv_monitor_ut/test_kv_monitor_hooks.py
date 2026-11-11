
import json
import sys
from types import SimpleNamespace

from msprobe.kv_monitor import config as kv_config
from msprobe.kv_monitor.dump.ledger import WriteLedger
from msprobe.kv_monitor.hooks.monitor import KVMonitor


def test_reset_detaches_pending_post_import_finders():
    from msprobe.kv_monitor.hooks.installer import INSTALLER

    module_name = "kv_monitor_ut_never_imported_module"
    try:
        INSTALLER._register_post_import(module_name, lambda: None)
        finder = INSTALLER._post_import_finders[module_name]
        assert finder in sys.meta_path
        INSTALLER.reset()
        assert finder not in sys.meta_path
        assert module_name not in INSTALLER._post_import_finders
    finally:
        INSTALLER.reset()


def test_ledger_error_row_replaces_logging(tmp_path, monkeypatch):
    monkeypatch.setattr(
        kv_config,
        "DEFAULT_CONFIG",
        kv_config.Config({"kv_stats_enable": True, "kv_stats_dir": str(tmp_path)}),
    )
    ledger = WriteLedger()
    ledger.record(["not-a-mapping"], vllm_config=None)
    lines = [
        json.loads(line)
        for line in (tmp_path / "rank0" / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(lines) == 1, lines
    assert list(lines[0]) == ["write", "error", "tid", "time"], list(lines[0])
    assert lines[0]["write"] == "msprobe_error"
    assert "not a mapping" in lines[0]["error"]
    assert isinstance(lines[0]["tid"], int)
    assert isinstance(lines[0]["time"], float)


def test_ledger_error_row_never_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(
        kv_config,
        "DEFAULT_CONFIG",
        kv_config.Config({"kv_stats_enable": True, "kv_stats_dir": str(tmp_path)}),
    )
    ledger = WriteLedger()
    broken = tmp_path / "rank0"
    broken.write_text("placeholder file blocks dir creation", encoding="utf-8")
    ledger.record({"write": "decode", "req_id": "r1"}, vllm_config=None)
    ledger.record_error(RuntimeError("boom"), vllm_config=None)
    assert broken.read_text(encoding="utf-8") == "placeholder file blocks dir creation"


def test_consumers_attempt_load_in_their_own_process(tmp_path, monkeypatch):
    vllm_config = SimpleNamespace(
        additional_config={
            "kv_trace": {
                "kv_stats_enable": True,
                "kv_stats_dir": str(tmp_path),
                "kv_stats_max_layers": 2,
            }
        }
    )

    monkeypatch.setattr(kv_config, "DEFAULT_CONFIG", kv_config.Config())
    monitor = KVMonitor(SimpleNamespace(vllm_config=vllm_config))
    assert monitor._max_layers() == 2
    assert monitor._stats_dir() == str(tmp_path)

    monkeypatch.setattr(kv_config, "DEFAULT_CONFIG", kv_config.Config())
    WriteLedger().record({"write": "decode", "req_id": "r1"}, vllm_config=vllm_config)
    assert (tmp_path / "rank0" / "ledger.jsonl").exists()

    assert kv_config.DEFAULT_CONFIG.max_layers() == 2
