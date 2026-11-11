from msprobe.kv_monitor.dump.ledger import record_patched_call
from msprobe.kv_monitor.engine.adapter import adapt_to
from msprobe.kv_monitor.utils import wrap_class_method


def apply_patch(cls, method_name: str = "_dummy_run") -> None:
    wrap_class_method(
        cls, method_name=method_name, guard_attr="_kv_ledger_dummy_installed", wrapper_factory=_make_wrapper
    )


def _make_wrapper(original_dummy_run):
    def _wrapped_dummy_run(self, *args, **kwargs):
        result = original_dummy_run(self, *args, **kwargs)
        record_patched_call(self, _build_rows, args, kwargs)
        return result

    return _wrapped_dummy_run


def _probe_dummy_inputs(runner, args, kwargs) -> dict:
    num_tokens = kwargs.get("num_tokens", args[0] if args else None)
    try:
        slot_values = adapt_to(runner).slot_values(num_tokens) if num_tokens else None
    except Exception:
        slot_values = None
    row = {}
    if slot_values is not None:
        row["stale_slots"] = slot_values
    return row


def _valid_stale_slots(row: dict, cache_block_size) -> list[int]:
    min_slot = cache_block_size if isinstance(cache_block_size, int) and cache_block_size > 0 else 1
    return [s for s in (row.get("stale_slots") or []) if isinstance(s, int) and s >= min_slot]


def _build_rows(runner, args, kwargs) -> list[dict]:
    cache_block_size = adapt_to(runner).block_size()
    slots = _valid_stale_slots(_probe_dummy_inputs(runner, args, kwargs), cache_block_size)
    if not slots:
        return []
    return [{"write": "dummy", "slots": slots}]
