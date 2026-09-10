# mypy: allow-untyped-defs
"""Lightweight tracing helpers for logic-buffer experiments.

This file is copied into torch/_inductor by patch.py.  It is intentionally
best-effort: failures should not change compiler behavior.
"""
# pylint: disable=duplicate-code, too-many-nested-blocks, chained-comparison

from __future__ import annotations

import hashlib
import inspect
import itertools
import json
import os
import re
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

TRACE_ENV = "INDUCTOR_LOGIC_BUFFER_TRACE"
DUMP_ENV = "INDUCTOR_LOGIC_BUFFER_DUMP_VALUES"
DIR_ENV = "INDUCTOR_LOGIC_BUFFER_DIR"
ACTIVE_DIR_ENV = "INDUCTOR_LOGIC_BUFFER_ACTIVE_DIR"
MAX_ELEMS_ENV = "INDUCTOR_LOGIC_BUFFER_MAX_ELEMS"

TRUE_VALUES = {"1", "true", "yes", "on"}

_KERNEL_CALL_COUNTER = itertools.count()


def _env_true(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in TRUE_VALUES


def trace_enabled() -> bool:
    return _env_true(TRACE_ENV)


def dump_enabled() -> bool:
    raw = os.environ.get(DUMP_ENV)
    if raw is not None and raw.strip():
        return raw.strip().lower() in TRUE_VALUES
    return trace_enabled()


def trace_dir() -> Path:
    explicit = os.environ.get(DIR_ENV, "").strip()
    if explicit:
        root = Path(explicit).expanduser()
    elif os.environ.get(ACTIVE_DIR_ENV, "").strip():
        root = Path(os.environ[ACTIVE_DIR_ENV]).expanduser()
    else:
        root = _default_trace_dir()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _default_trace_dir() -> Path:
    debug_path = _debug_context_path()
    if debug_path is not None:
        return debug_path / "logic_buffer_run"

    caller_path = _caller_model_dir()
    if caller_path is not None:
        return caller_path / "logic_buffer_run"

    return Path.cwd() / "logic_buffer_run"


def _debug_context_path() -> Optional[Path]:
    try:
        from torch._inductor.virtualized import V

        path = getattr(V.debug, "_path", None)
        if path:
            return Path(path).expanduser().resolve()
    except Exception:
        return None
    return None


def _caller_model_dir() -> Optional[Path]:
    interesting = {
        "output_code.py",
        "fx_graph_runnable.py",
        "fx_graph_transformed_runnable.py",
        "instrumented_fx_graph_runnable.py",
    }
    try:
        for frame in inspect.stack()[2:]:
            path = Path(frame.filename).expanduser()
            if path.name not in interesting:
                continue
            parent = path.parent.resolve()
            if parent.name == "logic_buffer_run":
                return parent.parent
            return parent
    except Exception:
        return None
    return None


def _json_default(obj: Any) -> str:
    try:
        return str(obj)
    except Exception:
        return f"<unprintable {type(obj).__name__}>"


def _append_jsonl(filename: str, payload: Dict[str, Any]) -> None:
    if not trace_enabled() and filename not in {"fx_values.jsonl", "kernel_values.jsonl"}:
        return
    path = trace_dir() / filename
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=_json_default, ensure_ascii=False) + "\n")


def _safe_name(text: str, limit: int = 120) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = text.strip("._") or "unnamed"
    if len(text) <= limit:
        return text
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"{text[: limit - len(digest) - 1]}_{digest}"


def _node_target(node: Any) -> str:
    try:
        return str(node.target)
    except Exception:
        return "<unknown>"


def _node_users(node: Any) -> List[str]:
    try:
        return [str(user.name) for user in node.users]
    except Exception:
        return []


def _node_name(node: Any) -> str:
    return str(getattr(node, "name", node))


def _origin_names(value: Any) -> List[str]:
    origins = getattr(value, "origins", None)
    if not origins:
        return []
    try:
        return [_node_name(origin) for origin in origins]
    except Exception:
        return []


def _origin_targets(value: Any) -> List[str]:
    origins = getattr(value, "origins", None)
    if not origins:
        return []
    targets: List[str] = []
    try:
        for origin in origins:
            targets.append(_node_target(origin))
    except Exception:
        return []
    return targets


def _origin_node_name(value: Any) -> Optional[str]:
    node = _safe_call(value, "get_origin_node", None)
    if node is None:
        node = getattr(value, "origin_node", None)
    if node is None:
        return None
    return _node_name(node)


def _safe_call(obj: Any, name: str, default: Any = None) -> Any:
    try:
        fn = getattr(obj, name)
        return fn()
    except Exception:
        return default


def _safe_list_call(obj: Any, name: str) -> Optional[List[str]]:
    value = _safe_call(obj, name, None)
    if value is None:
        return None
    try:
        return [str(x) for x in value]
    except Exception:
        return None


def _walk_result(value: Any, path: str = "") -> Iterable[Tuple[str, Any]]:
    if isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            next_path = f"{path}[{i}]" if path else f"[{i}]"
            yield from _walk_result(item, next_path)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            next_path = f"{path}[{key!r}]" if path else f"[{key!r}]"
            yield from _walk_result(item, next_path)
        return
    yield path, value


def _value_owner_alias_ids(value: Any) -> List[int]:
    """Object ids that represent the same lowered FX value.

    This is deliberately not based on origins.  It only follows wrapper boxes
    used by Inductor so copy_input(x) can map x back to the exact FX result
    object that GraphLowering.run_node returned earlier.
    """
    try:
        from torch._inductor import ir
    except Exception:
        return [id(value)]

    ids: List[int] = []
    seen = set()

    def add(obj: Any) -> None:
        oid = id(obj)
        if oid not in seen:
            seen.add(oid)
            ids.append(oid)

    cur = value
    add(cur)

    tensor_box = getattr(ir, "TensorBox", None)
    storage_box = getattr(ir, "StorageBox", None)
    base_view_cls = getattr(ir, "BaseView", None)

    if tensor_box is not None and isinstance(cur, tensor_box):
        cur = cur.data
        add(cur)

    if storage_box is not None and isinstance(cur, storage_box):
        cur = cur.data
        add(cur)
    elif base_view_cls is not None and isinstance(cur, base_view_cls):
        # Stop at the view object.  The underlying data is an input to the view,
        # not the same FX result.
        pass

    return ids


def _remember_fx_value_owner(graph: Any, node: Any, path: str, item: Any) -> None:
    owners = getattr(graph, "_logic_buffer_value_owners", None)
    if owners is None:
        owners = {}
        setattr(graph, "_logic_buffer_value_owners", owners)

    info = _classify_inductor_value(item) or {}
    owner = {
        "fx_node": _node_name(node),
        "fx_op": str(getattr(node, "op", "<unknown>")),
        "fx_target": _node_target(node),
        "fx_users": _node_users(node),
        "result_path": path,
        "value_kind_at_lowering": info.get("kind"),
        "buffer_type_at_lowering": info.get("buffer_type"),
    }
    for oid in _value_owner_alias_ids(item):
        bucket = owners.setdefault(oid, [])
        key = (owner["fx_node"], owner["result_path"])
        if all((row.get("fx_node"), row.get("result_path")) != key for row in bucket):
            bucket.append(dict(owner))


def _lookup_fx_value_owners(graph: Any, value: Any) -> List[Dict[str, Any]]:
    owners = getattr(graph, "_logic_buffer_value_owners", {})
    result: List[Dict[str, Any]] = []
    seen = set()
    for oid in _value_owner_alias_ids(value):
        for row in owners.get(oid, []):
            key = (row.get("fx_node"), row.get("result_path"))
            if key in seen:
                continue
            seen.add(key)
            result.append(dict(row))
    return result


def _classify_inductor_value(value: Any) -> Optional[Dict[str, Any]]:
    try:
        from torch._inductor import ir
    except Exception:
        return None

    cur = value
    wrappers: List[str] = []

    tensor_box = getattr(ir, "TensorBox", None)
    storage_box = getattr(ir, "StorageBox", None)
    if tensor_box is not None and isinstance(cur, tensor_box):
        wrappers.append(type(cur).__name__)
        cur = cur.data
    if storage_box is not None and isinstance(cur, storage_box):
        wrappers.append(type(cur).__name__)
        cur = cur.data

    buffer_cls = getattr(ir, "Buffer", None)
    base_view_cls = getattr(ir, "BaseView", None)
    loops_cls = getattr(ir, "Loops", None)

    if buffer_cls is not None and isinstance(cur, buffer_cls):
        return {
            "kind": "buffer",
            "buffer": _safe_call(cur, "get_name", None),
            "buffer_type": type(cur).__name__,
            "wrappers": wrappers,
            "device": str(_safe_call(cur, "get_device", None)),
            "dtype": str(_safe_call(cur, "get_dtype", None)),
            "size": _safe_list_call(cur, "get_size"),
            "stride": _safe_list_call(cur, "get_stride"),
        }

    if base_view_cls is not None and isinstance(cur, base_view_cls):
        return {
            "kind": "view",
            "buffer": _safe_call(cur, "get_name", None),
            "buffer_type": type(cur).__name__,
            "wrappers": wrappers,
            "device": str(_safe_call(cur, "get_device", None)),
            "dtype": str(_safe_call(cur, "get_dtype", None)),
            "size": _safe_list_call(cur, "get_size"),
            "stride": _safe_list_call(cur, "get_stride"),
        }

    if loops_cls is not None and isinstance(cur, loops_cls):
        return {
            "kind": "lazy",
            "buffer": None,
            "buffer_type": type(cur).__name__,
            "wrappers": wrappers,
        }

    return {
        "kind": "other",
        "buffer": None,
        "buffer_type": type(cur).__name__,
        "wrappers": wrappers,
    }


def _identity_record(value: Any) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "value_id": id(value),
        "value_type": type(value).__name__,
    }
    try:
        from torch._inductor import ir

        tensor_box = getattr(ir, "TensorBox", None)
        storage_box = getattr(ir, "StorageBox", None)
        base_view_cls = getattr(ir, "BaseView", None)

        cur = value
        if tensor_box is not None and isinstance(cur, tensor_box):
            record["tensorbox_id"] = id(cur)
            cur = cur.data
            record["tensorbox_data_id"] = id(cur)
            record["tensorbox_data_type"] = type(cur).__name__

        if storage_box is not None and isinstance(cur, storage_box):
            record["storagebox_id"] = id(cur)
            inner = cur.data
            record["storagebox_data_id"] = id(inner)
            record["storagebox_data_type"] = type(inner).__name__
            if hasattr(inner, "get_name"):
                record["storagebox_data_name"] = _safe_call(inner, "get_name", None)
        elif base_view_cls is not None and isinstance(cur, base_view_cls):
            record["view_id"] = id(cur)
            record["view_type"] = type(cur).__name__
            if hasattr(cur, "get_name"):
                record["view_name"] = _safe_call(cur, "get_name", None)
    except Exception:
        record["identity_error"] = traceback.format_exc()

    info = _classify_inductor_value(value)
    if info:
        record["value_kind"] = info.get("kind")
        record["buffer"] = info.get("buffer")
        record["buffer_type"] = info.get("buffer_type")
    return record


def _remember_identity_at_run_node(graph: Any, node: Any, path: str, item: Any) -> None:
    identities = getattr(graph, "_logic_buffer_run_node_identities", None)
    if identities is None:
        identities = {}
        setattr(graph, "_logic_buffer_run_node_identities", identities)
    identities[(_node_name(node), path)] = _identity_record(item)


def _record_identity_diff_at_env_delete(
    graph: Any,
    node: Any,
    path: str,
    item: Any,
) -> None:
    if os.environ.get("INDUCTOR_LOGIC_BUFFER_IDENTITY_TRACE", "").strip().lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return

    identities = getattr(graph, "_logic_buffer_run_node_identities", {})
    key = (_node_name(node), path)
    before = identities.get(key)
    after = _identity_record(item)
    if before is None:
        same_top = None
        same_storagebox = None
        same_storage_data = None
    else:
        same_top = before.get("value_id") == after.get("value_id")
        same_storagebox = before.get("storagebox_id") == after.get("storagebox_id")
        same_storage_data = before.get("storagebox_data_id") == after.get("storagebox_data_id")

    _append_jsonl(
        "lowering_env_identity_diff.jsonl",
        {
            "kind": "lowering_env_identity_diff",
            "time": time.time(),
            "graph_id": getattr(graph, "graph_id", None),
            "fx_node": _node_name(node),
            "fx_op": str(getattr(node, "op", "<unknown>")),
            "fx_target": _node_target(node),
            "result_path": path,
            "same_top_object": same_top,
            "same_storagebox": same_storagebox,
            "same_storagebox_data": same_storage_data,
            "run_node": before,
            "before_env_delete": after,
        },
    )


def _unwrap_inductor_value(value: Any) -> Any:
    try:
        from torch._inductor import ir
    except Exception:
        return value

    cur = value
    tensor_box = getattr(ir, "TensorBox", None)
    storage_box = getattr(ir, "StorageBox", None)
    if tensor_box is not None and isinstance(cur, tensor_box):
        cur = cur.data
    if storage_box is not None and isinstance(cur, storage_box):
        cur = cur.data
    return cur


def _buffer_debug_record(buffer: Any) -> Dict[str, Any]:
    return {
        "name": _safe_call(buffer, "get_name", None),
        "type": type(buffer).__name__,
        "device": str(_safe_call(buffer, "get_device", None)),
        "dtype": str(_safe_call(buffer, "get_dtype", None)),
        "size": _safe_list_call(buffer, "get_size"),
        "stride": _safe_list_call(buffer, "get_stride"),
        "origin_node": _origin_node_name(buffer),
        "origins": _origin_names(buffer),
        "origin_targets": _origin_targets(buffer),
        "repr": _safe_repr(buffer, 6000),
    }


def _safe_repr(value: Any, limit: int = 2000) -> str:
    try:
        text = str(value)
    except Exception:
        text = f"<unprintable {type(value).__name__}>"
    if len(text) > limit:
        return text[:limit] + f"... <truncated {len(text) - limit} chars>"
    return text


def _lazy_reads_and_body(value: Any) -> Optional[Dict[str, Any]]:
    cur = _unwrap_inductor_value(value)
    info = _classify_inductor_value(value)
    if not info or info.get("kind") != "lazy":
        return None

    reads: List[Dict[str, str]] = []
    try:
        for dep in cur.get_reads():
            reads.append(
                {
                    "name": str(getattr(dep, "name", "")),
                    "index": str(getattr(dep, "index", "")),
                    "type": type(dep).__name__,
                    "repr": _safe_repr(dep, 1000),
                }
            )
    except Exception as exc:
        reads.append({"name": "", "index": "", "type": "read_error", "repr": repr(exc)})

    try:
        body = cur.inner_fn_str()
    except Exception as exc:
        body = f"<inner_fn_str failed: {exc!r}>\n{_safe_repr(cur, 4000)}"

    return {
        "lazy_type": type(cur).__name__,
        "lazy_info": info,
        "reads": reads,
        "read_names": sorted({row["name"] for row in reads if row.get("name")}),
        "body": body,
        "repr": _safe_repr(cur, 6000),
    }


def _record_lazy_result_new_buffer_deps(
    graph: Any,
    node: Any,
    result: Any,
    buffer_watermark: Optional[int],
    buffer_end: Optional[int] = None,
) -> None:
    if buffer_watermark is None:
        ranges = getattr(graph, "_logic_buffer_node_buffer_ranges", {})
        saved_range = ranges.get(node)
        if saved_range is not None:
            buffer_watermark, buffer_end = saved_range
        else:
            last = getattr(graph, "_logic_buffer_last_buffer_count", 0)
            buffer_watermark = last
    try:
        setattr(graph, "_logic_buffer_last_buffer_count", len(graph.buffers))
    except Exception:  # nosec B110
        pass

    try:
        if buffer_end is None:
            new_buffers = list(graph.buffers[buffer_watermark:])
        else:
            new_buffers = list(graph.buffers[buffer_watermark:buffer_end])
    except Exception:
        return
    new_buffer_names = {
        str(_safe_call(buffer, "get_name", None)) for buffer in new_buffers if _safe_call(buffer, "get_name", None)
    }
    if not new_buffer_names:
        return

    for path, item in _walk_result(result):
        lazy = _lazy_reads_and_body(item)
        if not lazy:
            continue
        matched = sorted(new_buffer_names & set(lazy.get("read_names", [])))
        if not matched:
            continue

        record = {
            "kind": "lowering_lazy_depends_on_new_buffer",
            "time": time.time(),
            "graph_id": getattr(graph, "graph_id", None),
            "fx_node": _node_name(node),
            "fx_op": str(getattr(node, "op", "<unknown>")),
            "fx_target": _node_target(node),
            "fx_users": _node_users(node),
            "result_path": path,
            "buffer_watermark": buffer_watermark,
            "new_buffer_count": len(new_buffers),
            "new_buffer_names": sorted(new_buffer_names),
            "matched_new_buffers": matched,
            "new_buffers": [_buffer_debug_record(buffer) for buffer in new_buffers],
            **lazy,
        }
        _append_jsonl("lowering_lazy_depends_on_new_buffer.jsonl", record)


def remember_lowering_value_owner(
    graph: Any,
    node: Any,
    result: Any,
    buffer_watermark: Optional[int] = None,
    buffer_end: Optional[int] = None,
) -> None:
    """Remember object-identity ownership without writing trace rows.

    This is intentionally lighter than record_lowering_boundary().  It runs at
    GraphLowering.run_node return time so later copy_input/realize hooks can map
    an IR object back to the exact FX value owner.  The actual lowering map rows
    are emitted when the FX Interpreter is about to delete env[node].
    """
    if not trace_enabled():
        return

    if buffer_watermark is not None:
        try:
            if buffer_end is None:
                buffer_end = len(graph.buffers)
            ranges = getattr(graph, "_logic_buffer_node_buffer_ranges", None)
            if ranges is None:
                ranges = {}
                setattr(graph, "_logic_buffer_node_buffer_ranges", ranges)
            ranges[node] = (buffer_watermark, buffer_end)
        except Exception:  # nosec B110
            pass

    for path, item in _walk_result(result):
        try:
            _remember_fx_value_owner(graph, node, path, item)
            _remember_identity_at_run_node(graph, node, path, item)
        except Exception:  # nosec B110
            pass


def record_lowering_boundary(
    graph: Any,
    node: Any,
    result: Any,
    buffer_watermark: Optional[int] = None,
    buffer_end: Optional[int] = None,
    record_event: str = "before_env_delete",
) -> None:
    """Record FX nodes whose lowering result is a materialized Buffer.

    We deliberately do not realize anything here.  We only inspect the returned
    object currently stored in GraphLowering.env.
    """
    if not trace_enabled():
        return

    _record_lazy_result_new_buffer_deps(graph, node, result, buffer_watermark, buffer_end)

    records: List[Dict[str, Any]] = []
    for path, item in _walk_result(result):
        try:
            _record_identity_diff_at_env_delete(graph, node, path, item)
        except Exception:  # nosec B110
            pass
        try:
            _remember_fx_value_owner(graph, node, path, item)
            saved_values = getattr(graph, "_logic_buffer_fx_values", None)
            if saved_values is None:
                saved_values = []
                setattr(graph, "_logic_buffer_fx_values", saved_values)
            saved_values.append((node, path, item))
        except Exception:  # nosec B110
            pass

        info = _classify_inductor_value(item)
        if info:
            record = _fx_record(graph, node, path, item, "lowering_fx_value")
            record["record_event"] = record_event
            _append_jsonl("lowering_fx_value_map.jsonl", record)
        if not info or info.get("kind") != "buffer" or not info.get("buffer"):
            continue
        record = dict(info)
        record.update(
            {
                "kind": "lowering_fx_buffer",
                "value_kind": info.get("kind"),
                "time": time.time(),
                "graph_id": getattr(graph, "graph_id", None),
                "fx_node": str(getattr(node, "name", "<unknown>")),
                "fx_op": str(getattr(node, "op", "<unknown>")),
                "fx_target": _node_target(node),
                "fx_users": _node_users(node),
                "result_path": path,
                "record_event": record_event,
            }
        )
        records.append(record)

    for record in records:
        _append_jsonl("lowering_fx_buffer_map.jsonl", record)


def record_copy_input_materialization(source: Any, result: Any) -> None:
    """Record copy_input-created buffers and map them to exact FX value owners.

    Mapping is based on object identity saved in record_lowering_boundary, not
    on IR origins.  This avoids the severe one-to-many expansion from origins.
    """
    if not trace_enabled():
        return

    try:
        from torch._inductor.virtualized import V

        graph = V.graph
        source_info = _classify_inductor_value(source) or {
            "kind": "unclassified",
            "buffer": None,
            "buffer_type": type(source).__name__,
            "wrappers": [],
        }
        result_info = _classify_inductor_value(result) or {
            "kind": "unclassified",
            "buffer": None,
            "buffer_type": type(result).__name__,
            "wrappers": [],
        }
        owners = _lookup_fx_value_owners(graph, source)
        result_buffer = result_info.get("buffer")

        record = {
            "kind": "copy_input_materialization",
            "time": time.time(),
            "graph_id": getattr(graph, "graph_id", None),
            "source_value_kind": source_info.get("kind"),
            "source_buffer": source_info.get("buffer"),
            "source_buffer_type": source_info.get("buffer_type"),
            "source_wrappers": source_info.get("wrappers"),
            "result_value_kind": result_info.get("kind"),
            "result_buffer": result_buffer,
            "result_buffer_type": result_info.get("buffer_type"),
            "result_wrappers": result_info.get("wrappers"),
            "exact_fx_owners": owners,
            "exact_fx_owner_count": len(owners),
        }
        _append_jsonl("copy_input_materializations.jsonl", record)

        if result_buffer and owners:
            saved_mappings = getattr(graph, "_logic_buffer_copy_input_mappings", None)
            if saved_mappings is None:
                saved_mappings = []
                setattr(graph, "_logic_buffer_copy_input_mappings", saved_mappings)

            for owner in owners:
                map_record = {
                    "kind": "copy_input_fx_buffer",
                    "value_kind": "buffer",
                    "time": time.time(),
                    "graph_id": getattr(graph, "graph_id", None),
                    "buffer": str(result_buffer),
                    "buffer_type": result_info.get("buffer_type"),
                    "fx_node": owner.get("fx_node"),
                    "fx_op": owner.get("fx_op"),
                    "fx_target": owner.get("fx_target"),
                    "fx_users": owner.get("fx_users", []),
                    "result_path": owner.get("result_path", ""),
                    "source_value_kind": source_info.get("kind"),
                    "source_buffer": source_info.get("buffer"),
                    "source_buffer_type": source_info.get("buffer_type"),
                    "source_value_kind_at_lowering": owner.get("value_kind_at_lowering"),
                    "source_buffer_type_at_lowering": owner.get("buffer_type_at_lowering"),
                    "mapping_source": "copy_input_object_identity",
                }
                saved_mappings.append(dict(map_record))
                _append_jsonl("copy_input_fx_buffer_map.jsonl", map_record)
                _append_jsonl("lowering_fx_buffer_map.jsonl", map_record)
    except Exception:
        _append_jsonl(
            "logic_buffer_trace_errors.jsonl",
            {
                "kind": "copy_input_materialization_error",
                "time": time.time(),
                "error": traceback.format_exc(),
            },
        )


def _fx_record(graph: Any, node: Any, path: str, item: Any, kind: str) -> Dict[str, Any]:
    info = _classify_inductor_value(item) or {
        "kind": "unclassified",
        "buffer": None,
        "buffer_type": type(item).__name__,
        "wrappers": [],
    }
    record = dict(info)
    record.update(
        {
            "kind": kind,
            "value_kind": info.get("kind"),
            "time": time.time(),
            "graph_id": getattr(graph, "graph_id", None),
            "fx_node": _node_name(node),
            "fx_op": str(getattr(node, "op", "<unknown>")),
            "fx_target": _node_target(node),
            "fx_users": _node_users(node),
            "result_path": path,
            "origin_node": _origin_node_name(item),
            "origins": _origin_names(item),
            "origin_targets": _origin_targets(item),
        }
    )
    return record


def record_live_fx_buffer_map_after_dce(scheduler: Any) -> None:
    """Record the strict FX-node to live-buffer map after scheduler DCE.

    The FX Interpreter env is mostly garbage-collected by this point, so
    record_lowering_boundary stores object references during run_node.  Some of
    those TensorBoxes are later realized into ComputedBuffers.  Here we inspect
    the saved objects after scheduler.dead_node_elimination() and keep only
    entries whose current value is a direct Buffer that survived DCE.
    """
    if not trace_enabled():
        return

    try:
        from torch._inductor.virtualized import V

        graph = V.graph
        saved_values = list(getattr(graph, "_logic_buffer_fx_values", []))
        # SchedulerNode.get_name() returns "op<N>", not the buffer name, so use
        # the scheduler's name->SchedulerBuffer map instead: its keys are the
        # actual live buffer names (e.g. "buf2").
        name_to_buf = getattr(scheduler, "name_to_buf", None)
        if name_to_buf:
            live_node_names = set(name_to_buf.keys())
        else:
            live_node_names = {node.get_name() for node in getattr(scheduler, "nodes", [])}
        removed = set(getattr(graph, "removed_buffers", set()))
        emitted = set()

        for node, path, item in saved_values:
            info = _classify_inductor_value(item)
            if not info or info.get("kind") != "buffer" or not info.get("buffer"):
                continue
            buffer_name = str(info["buffer"])
            if buffer_name not in live_node_names or buffer_name in removed:
                continue

            key = (_node_name(node), path, buffer_name)
            if key in emitted:
                continue
            emitted.add(key)

            record = _fx_record(
                graph,
                node,
                path,
                item,
                "lowering_live_fx_buffer_after_dce",
            )
            record.update(
                {
                    "buffer": buffer_name,
                    "live_after_dce": True,
                    "scheduler_node_names_count": len(live_node_names),
                }
            )
            _append_jsonl("lowering_live_fx_buffer_map.jsonl", record)

        for row in list(getattr(graph, "_logic_buffer_copy_input_mappings", [])):
            buffer_name = str(row.get("buffer"))
            if not buffer_name or buffer_name not in live_node_names or buffer_name in removed:
                continue
            key = ("copy_input", row.get("fx_node"), row.get("result_path"), buffer_name)
            if key in emitted:
                continue
            emitted.add(key)
            live_row = dict(row)
            live_row.update(
                {
                    "kind": "lowering_live_fx_buffer_after_dce",
                    "live_after_dce": True,
                    "scheduler_node_names_count": len(live_node_names),
                }
            )
            _append_jsonl("lowering_live_fx_buffer_map.jsonl", live_row)

        # Aliases/views of a recorded live buffer map to the same FX node.
        # Kernels often write a reinterpret/alias of a fused output (e.g.
        # `buf36 = reinterpret_tensor(buf23, ...)`); those names never appear
        # at lowering time, so without this pass the comparison reports
        # no_fx_for_logical_buffer even though the value is exactly the FX node.
        try:
            from torch._inductor import ir as _ir

            base_to_fx = {}
            for _node, _path, _item in saved_values:
                _info = _classify_inductor_value(_item)
                if _info and _info.get("kind") == "buffer" and _info.get("buffer"):
                    _b = str(_info["buffer"])
                    _nm = str(getattr(_node, "name", ""))
                    if _b not in base_to_fx and _b in live_node_names and _b not in removed:
                        base_to_fx[_b] = _nm
            _ntb = getattr(graph, "name_to_buffer", None) or {}
            _ntb0 = getattr(graph, "name_to_buffer", None) or {}
            _ntb = getattr(graph, "name_to_buffer", None) or {}
            for _name, _buf in _ntb.items():
                _bases = set()
                try:
                    for _al in _buf.get_aliases():
                        try:
                            _bases.add(_al.get_name())
                        except Exception:  # nosec B110
                            pass
                except Exception:  # nosec B110
                    pass
                try:
                    _lay = _buf.get_layout()
                    _d = getattr(_lay, "data", None)
                    if _d is not None:
                        try:
                            _bases.add(_d.get_name())
                        except Exception:  # nosec B110
                            pass
                except Exception:  # nosec B110
                    pass
                _cur = _buf
                try:
                    while isinstance(_cur, _ir.BaseView) and getattr(_cur, "data", None) is not None:
                        _cur = _cur.data
                    _bases.add(_cur.get_name())
                except Exception:  # nosec B110
                    pass
                for _base_name in _bases:
                    if _base_name not in base_to_fx or str(_name) == _base_name:
                        continue
                    _key = ("alias", _base_name, str(_name), base_to_fx[_base_name])
                    if _key in emitted:
                        continue
                    emitted.add(_key)
                    _append_jsonl(
                        "lowering_live_fx_buffer_map.jsonl",
                        {
                            "kind": "lowering_live_fx_buffer_after_dce",
                            "buffer": str(_name),
                            "buffer_type": type(_buf).__name__,
                            "alias_of": _base_name,
                            "fx_node": base_to_fx[_base_name],
                            "live_after_dce": True,
                            "scheduler_node_names_count": len(live_node_names),
                        },
                    )
        except Exception:  # nosec B110
            pass
    except Exception:
        _append_jsonl(
            "logic_buffer_trace_errors.jsonl",
            {
                "kind": "lowering_live_fx_buffer_after_dce_error",
                "time": time.time(),
                "error": traceback.format_exc(),
            },
        )


def _is_removed(kernel_args: Any, value: Any) -> bool:
    try:
        return kernel_args._buffer_is_marked_removed(value)
    except Exception:
        return False


def _dedup_inplaced(kernel_args: Any) -> Dict[str, Dict[str, List[str]]]:
    by_formal: Dict[str, Dict[str, List[str]]] = {}
    seen = set()
    for inplaced in getattr(kernel_args, "inplace_buffers", {}).values():
        if id(inplaced) in seen or _is_removed(kernel_args, inplaced):
            continue
        seen.add(id(inplaced))
        logical_names = list(getattr(inplaced, "other_names", []))
        by_formal[getattr(inplaced, "inner_name", "")] = {
            "logical_names": logical_names,
            "input_logical_names": logical_names[:-1],
            "output_logical_names": logical_names[-1:],
        }
    return by_formal


def _is_tensor_arg(arg: Any) -> bool:
    return hasattr(arg, "buffer") and hasattr(arg, "dtype") and hasattr(arg, "name")


def _is_simple_expr(expr: Any) -> bool:
    return isinstance(expr, str) and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", expr) is not None


def collect_triton_kernel_call(
    kernel_name: str, kernel: Any, call_args: List[Any], precompile_args: List[Any]
) -> Tuple[int, List[Dict[str, Any]]]:
    """Collect logical-buffer to final wrapper call-arg mapping for a Triton call."""
    call_id = next(_KERNEL_CALL_COUNTER)
    if not trace_enabled():
        return call_id, []

    kernel_args = kernel.args
    live_outputs = set(kernel_args.live_output_buffers())
    input_buffers = set(getattr(kernel_args, "input_buffers", {}).keys())
    output_buffers = getattr(kernel_args, "output_buffers", {})
    inplaced_by_formal = _dedup_inplaced(kernel_args)

    entries: List[Dict[str, Any]] = []
    dump_entries: List[Dict[str, Any]] = []
    for index, (call_arg, meta) in enumerate(zip(call_args, precompile_args)):
        if not _is_tensor_arg(meta):
            continue

        formal_arg = str(meta.name)
        call_arg_s = str(call_arg)
        if formal_arg in inplaced_by_formal:
            inplace_info = inplaced_by_formal[formal_arg]
            logical_names = list(inplace_info["logical_names"])
            input_logical_names = list(inplace_info["input_logical_names"])
            output_logical_names = list(inplace_info["output_logical_names"])
            role = "input_output"
        else:
            logical_names = [str(meta.buffer)]
            input_logical_names = []
            output_logical_names = []
            logical = logical_names[0]
            if logical in live_outputs:
                role = "output"
                output_logical_names = [logical]
            elif logical in output_buffers and not _is_removed(kernel_args, output_buffers[logical]):
                role = "output"
                output_logical_names = [logical]
            elif logical in input_buffers:
                role = "input"
                input_logical_names = [logical]
            else:
                role = "unknown"

        entry = {
            "index": index,
            "formal_arg": formal_arg,
            "call_arg": call_arg_s,
            "logical_names": logical_names,
            "input_logical_names": input_logical_names,
            "output_logical_names": output_logical_names,
            "role": role,
            "dtype": str(getattr(meta, "dtype", "")),
        }
        entries.append(entry)
        if role in {"output", "input_output"} and _is_simple_expr(call_arg_s):
            dump_entries.append(entry)

    _append_jsonl(
        "kernel_arg_map.jsonl",
        {
            "kind": "triton_kernel_call",
            "time": time.time(),
            "call_id": call_id,
            "kernel_name": kernel_name,
            "entries": entries,
            "dump_entries": dump_entries,
        },
    )
    return call_id, dump_entries


def _buffer_name(node: Any) -> Optional[str]:
    return _safe_call(node, "get_name", None)


def _node_dtype(node: Any) -> str:
    return str(_safe_call(node, "get_dtype", ""))


def _is_multi_output_parent(node: Any) -> bool:
    return type(getattr(node, "layout", None)).__name__ == "MultiOutputLayout"


def collect_extern_kernel_call(
    kernel_name: str,
    node: Any,
    call_arg: Any = None,
    *,
    logical_name: Optional[str] = None,
    kind: str = "extern_kernel_call",
) -> Tuple[int, List[Dict[str, Any]]]:
    """Collect a single-output extern/fallback call.

    Multi-output fallback parents are intentionally ignored here.  Their tuple
    elements are represented by MultiOutput nodes; only the single-output case
    is handled by collect_single_output_multi_output_call().
    """
    call_id = next(_KERNEL_CALL_COUNTER)
    if not trace_enabled():
        return call_id, []

    logical = str(logical_name or _buffer_name(node) or "")
    call_arg_s = str(call_arg if call_arg is not None else logical)
    entries: List[Dict[str, Any]] = []
    dump_entries: List[Dict[str, Any]] = []

    if logical and not _is_multi_output_parent(node):
        entry = {
            "index": 0,
            "formal_arg": "out",
            "call_arg": call_arg_s,
            "logical_names": [logical],
            "input_logical_names": [],
            "output_logical_names": [logical],
            "role": "output",
            "dtype": _node_dtype(node),
        }
        entries.append(entry)
        if _is_simple_expr(call_arg_s):
            dump_entries.append(entry)

    _append_jsonl(
        "kernel_arg_map.jsonl",
        {
            "kind": kind,
            "time": time.time(),
            "call_id": call_id,
            "kernel_name": kernel_name,
            "entries": entries,
            "dump_entries": dump_entries,
            "single_output_only": True,
        },
    )
    return call_id, dump_entries


def collect_single_output_multi_output_call(node: Any, call_arg: Any = None) -> Tuple[int, List[Dict[str, Any]]]:
    """Collect the alias produced for a single-output FallbackKernel.

    FallbackKernel.create() represents tensor returns through a MultiOutput
    wrapper even when the fallback has exactly one tensor output.  Tuple/list
    fallbacks have multiple outputs and are ignored by design for this
    experiment.
    """
    parent = None
    try:
        inputs = list(getattr(node, "inputs", []))
        parent = inputs[0] if inputs else None
    except Exception:
        parent = None

    outputs = list(getattr(parent, "outputs", []) or [])
    if len(outputs) != 1 or outputs[0] is not node:
        # Multi-output parent (e.g. max_pool2d returns values + indices):
        # still record the parent kernel producing this tuple element so that
        # downstream consumers can be connected in the call graph.
        kernel_name = str(_safe_call(parent, "get_kernel_name", type(parent).__name__))
        return collect_extern_kernel_call(
            kernel_name,
            node,
            None,
            logical_name=_buffer_name(node),
            kind="extern_single_multi_output_alias_call",
        )

    kernel_name = str(_safe_call(parent, "get_kernel_name", type(parent).__name__))
    return collect_extern_kernel_call(
        kernel_name,
        node,
        call_arg,
        logical_name=_buffer_name(node),
        kind="extern_single_multi_output_alias_call",
    )


def codegen_kernel_runtime_dump(wrapper: Any, kernel_name: str, call_id: int, entries: List[Dict[str, Any]]) -> None:
    if not entries:
        return

    item_srcs: List[str] = []
    for entry in entries:
        call_arg = entry["call_arg"]
        if not _is_simple_expr(call_arg):
            continue
        item_srcs.append(
            "{"
            f"'logical_names': {entry['logical_names']!r}, "
            f"'input_logical_names': {entry.get('input_logical_names', [])!r}, "
            f"'output_logical_names': {entry.get('output_logical_names', [])!r}, "
            f"'call_arg': {call_arg!r}, "
            f"'role': {entry['role']!r}, "
            f"'tensor': {call_arg}"
            "}"
        )

    if not item_srcs:
        return

    wrapper.writeline(
        f"if os.environ.get({DUMP_ENV!r}, os.environ.get({TRACE_ENV!r}, '')).strip().lower() in ('1', 'true', 'yes', 'on'):"
    )
    wrapper.writeline(
        "    from torch._inductor.logic_buffer_trace import dump_kernel_tensors as _inductor_logic_buffer_dump_kernel_tensors"
    )
    wrapper.writeline(
        f"    _inductor_logic_buffer_dump_kernel_tensors({kernel_name!r}, {call_id!r}, [{', '.join(item_srcs)}])"
    )


def _max_elems() -> int:
    raw = os.environ.get(MAX_ELEMS_ENV, "1048576").strip()
    try:
        return int(raw)
    except ValueError:
        return 1048576


def _tensor_payload(tensor: Any):
    import torch

    if not isinstance(tensor, torch.Tensor):
        return None, {
            "is_tensor": False,
            "type": type(tensor).__name__,
            "repr": repr(tensor),
        }

    meta = {
        "is_tensor": True,
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "numel": int(tensor.numel()),
        "truncated": False,
    }
    with torch.no_grad():
        detached = tensor.detach()
        max_elems = _max_elems()
        if max_elems > 0 and detached.numel() > max_elems:
            detached = detached.reshape(-1)[:max_elems]
            meta["truncated"] = True
            meta["saved_numel"] = int(max_elems)
        else:
            meta["saved_numel"] = int(detached.numel())
        payload = detached.cpu().clone()
    return payload, meta


def _save_tensor(kind: str, stem: str, tensor: Any) -> Dict[str, Any]:
    root = trace_dir()
    value_dir = root / f"{kind}_tensors"
    value_dir.mkdir(parents=True, exist_ok=True)
    payload, meta = _tensor_payload(tensor)
    if payload is None:
        return {**meta, "path": None}

    path = value_dir / f"{_safe_name(stem)}.pt"
    import torch

    torch.save(payload, path)
    return {**meta, "path": str(path)}


def _clone_replay_arg(arg: Any) -> Any:
    import torch

    if isinstance(arg, torch.Tensor):
        with torch.no_grad():
            out = arg.detach().clone(memory_format=torch.preserve_format)
            out.requires_grad_(arg.requires_grad)
            return out
    return arg


def save_replay_args(args: List[Any], graph_key: str = "graph", arg_names: Optional[List[str]] = None) -> None:
    if not trace_enabled():
        return

    try:
        import torch

        root = trace_dir()
        payload = [_clone_replay_arg(arg) for arg in args]
        names = [str(name) for name in (arg_names or [])]
        identity = json.dumps(
            {"graph_key": str(graph_key), "arg_names": names},
            ensure_ascii=True,
            sort_keys=True,
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
        path = root / f"replay_args_{_safe_name(str(graph_key), 48)}_{digest}.pt"
        torch.save(payload, path)
        model_dir = _caller_model_dir()
        _append_jsonl(
            "replay_args.jsonl",
            {
                "kind": "replay_args",
                "time": time.time(),
                "path": str(path),
                "arg_count": len(payload),
                "arg_names": names,
                "graph_key": str(graph_key),
                "model_dir": str(model_dir) if model_dir is not None else None,
            },
        )
    except Exception:
        _append_jsonl(
            "logic_buffer_trace_errors.jsonl",
            {
                "kind": "save_replay_args_error",
                "time": time.time(),
                "error": traceback.format_exc(),
            },
        )


def dump_kernel_tensors(kernel_name: str, call_id: int, entries: List[Dict[str, Any]]) -> None:
    if not dump_enabled():
        return

    for i, entry in enumerate(entries):
        try:
            tensor = entry.get("tensor")
            logical_names = list(entry.get("logical_names", []))
            output_logical_names = list(entry.get("output_logical_names", []))
            input_logical_names = list(entry.get("input_logical_names", []))
            if not output_logical_names:
                if entry.get("role") == "input_output" and logical_names:
                    output_logical_names = logical_names[-1:]
                    input_logical_names = logical_names[:-1]
                elif entry.get("role") == "output":
                    output_logical_names = logical_names
            stem = f"{call_id:06d}_{i:02d}_{kernel_name}_{entry.get('call_arg', 'arg')}"
            meta = _save_tensor("kernel", stem, tensor)
            record = {
                "kind": "kernel_tensor",
                "time": time.time(),
                "call_id": call_id,
                "kernel_name": kernel_name,
                "entry_index": i,
                "logical_names": logical_names,
                "input_logical_names": input_logical_names,
                "output_logical_names": output_logical_names,
                "call_arg": str(entry.get("call_arg")),
                "role": str(entry.get("role")),
                **meta,
            }
            _append_jsonl("kernel_values.jsonl", record)
        except Exception:
            _append_jsonl(
                "kernel_values.jsonl",
                {
                    "kind": "kernel_tensor_error",
                    "time": time.time(),
                    "call_id": call_id,
                    "kernel_name": kernel_name,
                    "entry_index": i,
                    "error": traceback.format_exc(),
                },
            )


def dump_fx_tensor(fx_node: str, tensor: Any) -> None:
    if not dump_enabled():
        return
    try:
        meta = _save_tensor("fx", fx_node, tensor)
        _append_jsonl(
            "fx_values.jsonl",
            {
                "kind": "fx_tensor",
                "time": time.time(),
                "fx_node": fx_node,
                **meta,
            },
        )
    except Exception:
        _append_jsonl(
            "fx_values.jsonl",
            {
                "kind": "fx_tensor_error",
                "time": time.time(),
                "fx_node": fx_node,
                "error": traceback.format_exc(),
            },
        )
