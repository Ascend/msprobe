from __future__ import annotations

from typing import Iterator

from msprobe.kv_monitor.dump.ledger import record_rows
from msprobe.kv_monitor.engine.adapter import adapt_to
from msprobe.kv_monitor.utils import (
    get_external_request_id,
    step_is_prefill,
    wrap_class_method,
)


def apply_patch(cls, method_name: str = "execute_model") -> None:
    wrap_class_method(
        cls, method_name=method_name, guard_attr="_kv_ledger_runner_installed", wrapper_factory=_make_wrapper
    )


def _make_wrapper(original_execute_model):
    def _wrapped_execute_model(self, scheduler_output, *args, **kwargs):
        result = original_execute_model(self, scheduler_output, *args, **kwargs)
        # 槽位必须等 block table 更新后才能读，所以放在执行之后；
        # 「本步是否 prefill」则用执行前的 computed 值判定（见 _slot_segments）。
        try:
            rows = _build_rows(self, scheduler_output)
        except Exception:
            rows = None
        if rows:
            record_rows(self, rows)
        return result

    return _wrapped_execute_model


def _slot_segments(runner, observation, total: int, scheduler_output=None) -> Iterator[tuple[str, list, bool]]:
    adapter = adapt_to(runner)
    request_ids = adapter.req_ids()
    num_reqs = len(request_ids)
    scheduled_counts = [0 if observation is None else observation.num_scheduled.get(rid, 0) for rid in request_ids]

    slot_values = adapter.slot_values(total)
    progress = adapter.request_progress(scheduler_output)
    computed_tokens_per_req = progress.computed_tokens[:num_reqs]
    prompt_tokens_per_req = progress.prompt_tokens[:num_reqs]

    offset = 0
    for index, (request_id, count) in enumerate(zip(request_ids, scheduled_counts)):
        if count <= 0:
            continue
        slots = slot_values[offset : offset + count] if slot_values is not None else []
        offset += count
        # request_progress 在执行后读取时 computed 已累加本步 token 数：
        # 回退得到「执行前」的值，否则把 prompt 一次算完的首个 prefill 步会误判成 decode。
        computed_before = int(computed_tokens_per_req[index]) - int(count)
        is_prefill = step_is_prefill(computed_before, int(prompt_tokens_per_req[index]))
        yield request_id, slots, is_prefill


def _build_rows(runner, scheduler_output) -> list[dict]:
    adapter = adapt_to(runner)
    observation = adapter.observe_batch(scheduler_output)
    total_scheduled_tokens = 0 if observation is None else observation.total_scheduled
    if total_scheduled_tokens <= 0:
        return []
    return [
        {
            "write": "prefill" if is_prefill else "decode",
            "req_id": get_external_request_id(request_id),
            "slots": slots,
        }
        for request_id, slots, is_prefill in _slot_segments(
            runner, observation, total_scheduled_tokens, scheduler_output
        )
    ]
