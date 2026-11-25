# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Project public connector plans; never poll connector completion queues."""

from collections.abc import Mapping
from contextvars import ContextVar
from functools import partial, partialmethod
import numpy as np

from msprobe.kv_monitor.utils import wrap_class_method
from msprobe.vllm_kv_guard.events import emit


def record_plan(metadata, trace_id, boundary, rank=None):
    if metadata is None:
        return
    fields = dict(
        trace_id=trace_id,
        boundary=boundary,
        evidence="connector_plan",
        metadata_type=f"{type(metadata).__module__}.{type(metadata).__qualname__}",
        coverage="not_expanded",
    )
    if rank is not None:
        fields["rank"] = rank
    requests = getattr(metadata, "requests", None)
    if isinstance(requests, Mapping):
        plans, unexpanded = {}, []
        for req_id, request in requests.items():
            plan = {}
            for name in (
                "local_block_ids",
                "remote_block_ids",
                "num_computed_tokens",
                "num_external_tokens",
                "remote_engine_id",
                "remote_request_id",
            ):
                value = request.get(name) if isinstance(request, Mapping) else getattr(request, name, None)
                if value is not None:
                    plan[name] = value
            if plan:
                plans[req_id] = plan
            else:
                unexpanded.append(req_id)
        fields["requests"] = plans
        if plans:
            fields["coverage"] = "selected_public_fields"
        if unexpanded:
            fields["unexpanded_req_ids"] = unexpanded
    sending = getattr(metadata, "requests_to_send", None)
    if isinstance(sending, Mapping):
        fields["send_req_ids"] = list(sending)
        fields["coverage"] = "selected_public_fields"
    emit("boundary", **fields)


def patch(target, name, callback):
    factory = partialmethod if isinstance(target, type) else partial
    wrap_class_method(target, name, '_msprobe_slot_' + name, lambda native: factory(callback, native))


_reads = ContextVar("msprobe_slot_reads", default=None)


def prepare(guard, output, pd):
    metadata = output.kv_connector_metadata
    if pd:
        from vllm.distributed.kv_transfer import get_kv_transfer_group

        connector = get_kv_transfer_group().connector_worker
        spec, layers = connector.kv_group2layeridx[0]
        retired = set(output.finished_req_ids) | set(output.preempted_req_ids or ())
        for req in retired:
            guard.plans.pop(req, None)
        for req, meta in getattr(metadata, "requests", {}).items():
            guard.plans[req] = tuple(map(tuple, connector._get_kernel_block_ids(layers, meta, 0, spec)))
    else:
        for store, name, src, dst in ((True, "store", "gpu", "cpu"), (False, "load", "cpu", "gpu")):
            event = getattr(metadata, f"preempt_{name}_event", -1)
            if event >= 0:
                pair = [getattr(metadata, f"preempt_{name}_{side}_blocks") for side in (src, dst)]
                guard.plans[store, event] = tuple(map(tuple, pair))


def lifecycle(scheduler, guard, options, pending):
    manager = scheduler.kv_cache_manager

    def free(native, request, *args, **kwargs):
        request_id = request.request_id
        owned = any(manager.get_block_ids(request_id))
        guard.check(request_id not in pending or not owned, "slot_release_inflight", request_id=request_id)
        native(request, *args, **kwargs)
        guard.check(not any(manager.get_block_ids(request_id)), "slot_release_binding_stale", request_id=request_id)

    patch(manager, "free", free)
    if options.get('pd_slot'):
        connector = scheduler.connector.connector_scheduler

        def export(native, request, *args, **kwargs):
            result = native(request, *args, **kwargs)
            params = result[1]
            if params is not None and 'remote_block_ids' in params:
                request_id = request.request_id
                actual = connector._get_transfer_block_ids(manager.get_block_ids(request_id), request.num_prompt_tokens)
                valid = (params['remote_request_id'], params['remote_block_ids']) == (request_id, actual)
                guard.check(valid, "pd_slot_source_plan_mismatch", request_id=request_id)
            return result

        patch(connector, "request_finished", export)


def install(guard, pd):
    if pd:
        from vllm_ascend.distributed.kv_transfer.kv_p2p import mooncake_connector as mc

        def initialize(receiver, native, *args, **kwargs):
            native(receiver, *args, **kwargs)
            groups = receiver.kv_group2layeridx
            if set(groups) != {0} or groups[0][0]['kv_cache_spec_type'] != 'FullAttentionSpec':
                raise ValueError("kv_trace.slot requires a single full-attention Mooncake cache group")

        def read(engine, native, session, local, remote, lengths):
            scope = _reads.get()
            if scope is not None:
                scope.append(tuple(map(tuple, (local, remote, lengths))))
            return native(engine, session, local, remote, lengths)

        def transfer(receiver, native, req_meta):
            request = req_meta
            if not any(request['local_block_ids']):
                return native(receiver, request)
            request_id = request['request_id']
            check = partial(guard.check, request_id=request_id, evidence="transfer_descriptor")
            plan = guard.plans.get(request_id)
            rows = []
            token = _reads.set(rows)
            try:
                result = native(receiver, request)
            finally:
                _reads.reset(token)
            check(guard.plans.get(request_id) is plan, "pd_slot_stale_transfer")
            try:
                engine, port = request['remote_engine_id'], request['remote_handshake_port']
                with receiver.remote_metadata_lock:
                    local = receiver.kv_caches_base_addr[receiver.local_engine_id][receiver.local_handshake_port]
                    remote = receiver.kv_caches_base_addr[engine][port]
                    strides = receiver.remote_block_stride_per_addr[engine][port]
                    groups = receiver.remote_kv_group2layeridx.get(engine, {}).get(port, receiver.kv_group2layeridx)
                spec, layers = receiver.kv_group2layeridx[0]
                names = mc.build_layer_name_to_metadata_idx(groups)
                peers = [
                    mc.resolve_remote_layer_idx(i, spec, layers, names) if names and spec.get('layer_names') else i
                    for i in layers
                ]
                indices = layers, peers, layers, layers, peers
                tables = local, remote, receiver.block_len_per_addr, receiver.block_stride_per_addr, strides
                planes = [np.concatenate([data[i] for i in take]) for data, take in zip(tables, indices)]
                planes = np.asarray(planes, dtype='i8')
                targets = planes[[0, 1], :, None] + planes[[3, 4], :, None] * np.asarray(plan)[:, None, :]
                lengths = np.broadcast_to(planes[2, :, None], targets[0].shape)[None]
                expected = np.concatenate((targets, lengths)).reshape(3, -1).T
                actual = np.concatenate([np.asarray(row).T for row in rows] or [np.empty((0, 3), dtype='i8')])
                ends = expected[:, :2] + expected[:, 2, None], actual[:, :2] + actual[:, 2, None]
                # Paired endpoint multiplicities preserve coverage across merged and split ranges.
                points = np.concatenate((expected[:, :2], ends[0], actual[:, :2], ends[1]))
                weights = np.repeat((1, -1, -1, 1), (len(expected), len(expected), len(actual), len(actual)))
                _, index = np.unique(points, axis=0, return_inverse=True)
                valid = bool(rows) and np.all(actual[:, 2] > 0) and not np.any(np.bincount(index, weights=weights))
            except (AttributeError, KeyError, IndexError, TypeError, ValueError, RuntimeError):
                valid = False
            check(valid, "pd_slot_transfer_mismatch")
            return result

        patch(mc.MooncakeConnectorWorker, "register_kv_caches", initialize)
        patch(mc.KVCacheRecvingThread, "_transfer_kv_cache_all_groups", transfer)
        patch(mc.TransferEngine, "batch_transfer_sync_read", read)
    else:
        from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.preempt_offload.worker import PreemptOffloadWorker

        # pylint: disable-next=keyword-arg-before-vararg
        def submit(worker, native, src_block_ids, dst_block_ids, event_idx, is_store, sync=False, *args, **kwargs):
            key = is_store, event_idx
            if event_idx < 0 or (not is_store and event_idx in worker._submitted_load_event_indices):
                if worker._load_hwm >= event_idx:
                    guard.plans.pop(key, None)
                return native(worker, src_block_ids, dst_block_ids, event_idx, is_store, sync, *args, **kwargs)
            event = dict(event_id=event_idx, direction="D2H" if is_store else "H2D")
            capacities = worker.num_cpu_blocks, worker.num_gpu_blocks
            limits = capacities[::-1] if is_store else capacities
            actual = tuple(src_block_ids), tuple(dst_block_ids)
            valid = all(
                np.all((np.asarray(ids) >= 0) & (np.asarray(ids) < limit)) for ids, limit in zip(actual, limits)
            )
            guard.check(valid and actual == guard.plans.get(key) and sync, "swap_slot_plan_mismatch", **event)
            native(worker, src_block_ids, dst_block_ids, event_idx, is_store, sync, *args, **kwargs)
            finished = worker._completed_store_events.get(event_idx) == 1 if is_store else worker._load_hwm >= event_idx
            guard.plans.pop(key, None)
            return guard.check(finished, "swap_slot_completion_missing", **event)

        patch(PreemptOffloadWorker, "_submit_transfer", submit)
