from msprobe.kv_monitor.dump.ledger import record_patched_call
from msprobe.kv_monitor.engine.adapter import adapt_to
from msprobe.kv_monitor.utils import get_external_request_id, get_field, wrap_class_method


def apply_patch(cls, method_name: str = "_transfer_kv_cache_all_groups") -> None:
    wrap_class_method(cls, method_name=method_name, guard_attr="_kv_ledger_transfer_kv", wrapper_factory=_make_wrapper)


def _build_rows(target, req_meta) -> list[dict]:
    """transfer_kv_cache 的账行：本 rank 的物理块号 + 块大小"""
    request_id = get_field(req_meta, "request_id")
    local_block_ids = get_field(req_meta, "local_block_ids")
    blocks = []
    for group in local_block_ids or []:
        if isinstance(group, (list, tuple)):
            blocks.extend(int(block) for block in group)
        else:
            blocks.append(int(group))
    row = {
        "write": "transfer_kv_cache",
        "req_id": get_external_request_id(str(request_id)),
        "blocks": blocks,
    }
    block_size = adapt_to(target).block_size()
    if block_size:
        row["block_size"] = block_size
    return [row]


def _make_wrapper(original_transfer):
    def _wrapped_transfer(self, req_meta, *args, **kwargs):
        result = original_transfer(self, req_meta, *args, **kwargs)
        record_patched_call(self, _build_rows, req_meta)
        return result

    return _wrapped_transfer
