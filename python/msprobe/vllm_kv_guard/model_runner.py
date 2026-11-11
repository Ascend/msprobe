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

"""Record cache-group address descriptions at the Model Runner V2 boundary."""

from functools import wraps
from threading import Lock
from uuid import uuid4

import torch
from torch.utils._pytree import tree_map

from msprobe.vllm_kv_guard.events import emit


class PlanRecorder:
    def __init__(self, rank, guard):
        self.rank = rank
        self.guard = guard
        self.trace_id = None
        self.plan = ()
        self.buffers = []
        self.event = torch.npu.Event()
        self.pending = lambda: None

    def capture(self, batch, block_tables, slot_mappings, layout, execution_mode):
        parallel = {key: value for key, value in layout.items() if key not in ("groups", "dynamic")}
        rows, tokens = batch.num_reqs, batch.num_tokens
        payload = {
            "execution_mode": execution_mode,
            "req_ids": list(batch.req_ids[:rows]),
            "num_tokens": tokens,
            "query_start_loc": batch.query_start_loc[: rows + 1],
            "positions": batch.positions[:tokens],
            "seq_lens": batch.seq_lens[:rows],
            "layout": layout,
            "dcp_local_seq_lens": batch.dcp_local_seq_lens[:rows]
            if parallel["dcp_size"] > 1 and batch.dcp_local_seq_lens is not None
            else None,
            "groups": [
                {
                    "block_table": table[:rows],
                    "slot_mapping": slot_mappings[group] if parallel["pcp_size"] > 1 else slot_mappings[group, :tokens],
                }
                for group, table in enumerate(block_tables)
            ],
        }

        plan = self.plan

        def record(trace_id, values):
            values["trace_id"] = trace_id
            self.guard.audit(values, plan, values["layout"])
            common = {
                "trace_id": trace_id,
                "rank": self.rank,
                "req_ids": values["req_ids"],
                "boundary": "model_state.prepare_attn.input",
                "evidence": "model_runner_plan",
                "execution_mode": values["execution_mode"],
                **parallel,
            }
            for group, data in enumerate(values["groups"]):
                common.update(
                    {key: values["layout"]["groups"][group][key] for key in ("block_size", "kernel_block_size")}
                )
                emit(
                    "kv.write",
                    **common,
                    group_id=group,
                    num_tokens=values["num_tokens"],
                    query_start_loc=values["query_start_loc"],
                    positions=values["positions"],
                    slot_mapping_scope="pcp_frame" if parallel["pcp_size"] > 1 else "local_tokens",
                    slot_mapping=data["slot_mapping"],
                )
                emit(
                    "kv.read",
                    **common,
                    group_id=group,
                    seq_lens=values["seq_lens"],
                    dcp_local_seq_lens=values["dcp_local_seq_lens"],
                    block_table=data["block_table"],
                )

        self.snapshot(payload, record)

    def capture_dummy(self, slot_mappings, persistent_slots, layout, execution_mode):
        def record(trace_id, values):
            for scope, groups in values.items():
                for group, slots in enumerate(groups):
                    size = layout["groups"][group]["block_size"]
                    targets = [(index, slot) for index, slot in enumerate(slots) if slot != -1]
                    emit(
                        "kv.write",
                        trace_id=trace_id,
                        rank=self.rank,
                        dp_rank=layout["dp_rank"],
                        group_id=group,
                        boundary="model_state.prepare_attn.input",
                        evidence="dummy_plan",
                        phase="dummy",
                        execution_mode=execution_mode,
                        block_size=size,
                        allow_null_block=layout["allow_null_block"],
                        slot_mapping_scope=scope,
                        slot_count=len(slots),
                        non_pad_slots=targets,
                    )
                    for index, slot in targets:
                        if not (layout["allow_null_block"] and 0 <= slot < size):
                            self.guard.fail(
                                "dummy_write_target",
                                trace_id=trace_id,
                                dp_rank=layout["dp_rank"],
                                group_id=group,
                                token_index=index,
                                actual_slot=slot,
                                slot_mapping_scope=scope,
                                boundary="model_state.prepare_attn.input",
                            )
                            break

        self.snapshot({"input": slot_mappings, "persistent": persistent_slots}, record)

    def capture_parallel(self, context, layout, input_batch, block_tables, slot_mappings):
        parallel = {key: value for key, value in layout.items() if key not in ("groups", "dynamic")}
        batch = context.global_batch
        rows, tokens = batch.num_reqs, batch.num_tokens
        local = None
        if input_batch is not None:
            local_rows, local_tokens = input_batch.num_reqs, input_batch.num_tokens
            local = {
                "req_ids": list(input_batch.req_ids[:local_rows]),
                "num_tokens": local_tokens,
                "query_start_loc": input_batch.query_start_loc[: local_rows + 1],
                "positions": input_batch.positions[:local_tokens],
                "seq_lens": input_batch.seq_lens[:local_rows],
                "dcp_local_seq_lens": None
                if input_batch.dcp_local_seq_lens is None
                else input_batch.dcp_local_seq_lens[:local_rows],
                "groups": None
                if block_tables is None
                else [
                    {
                        "block_table": table[:local_rows],
                        "slot_mapping": None if slot_mappings is None else slot_mappings[group],
                    }
                    for group, table in enumerate(block_tables)
                ],
            }
        payload = {
            "layout": layout,
            "local": local,
            "global_req_ids": list(batch.req_ids[:rows]),
            "global_num_tokens": tokens,
            "global_num_tokens_after_padding": batch.num_tokens_after_padding,
            "global_query_start_loc": batch.query_start_loc[: rows + 1],
            "global_positions": batch.positions[:tokens],
            "global_seq_lens": batch.seq_lens[:rows],
            "global_block_tables": tuple(table[:rows] for table in context.global_block_tables),
            "global_slot_mappings": context.global_slot_mappings,
            "local_to_global_req_indices": context.local_to_global_req_indices,
            "local_token_indices": context.local_token_indices,
            "padded_gather_idx": context.padded_gather_idx,
            "gathered_kv_write_mask": context.gathered_kv_write_mask,
            "global_block_table_num_blocks": context.global_block_table_num_blocks,
        }

        plan = self.plan

        def record(trace_id, values):
            values["trace_id"] = trace_id
            self.guard.parallel(values, plan, values["layout"])
            values.pop("layout")
            values.pop("local")
            emit(
                "boundary",
                rank=self.rank,
                **parallel,
                boundary="pcp_manager.build_attention_context.return",
                evidence="parallel_context",
                slot_mapping_scope="global_padded_tokens",
                **values,
            )

        self.snapshot(payload, record)

    def snapshot(self, payload, record):
        self.pending()
        sources, index = [], 0

        def freeze(value):
            nonlocal index
            if not isinstance(value, torch.Tensor):
                return value.copy() if hasattr(value, "copy") else value
            if value.device.type == "cpu":
                return value.clone()
            if index == len(self.buffers):
                self.buffers.append(None)
            host = self.buffers[index]
            if host is None or host.shape != value.shape or host.dtype != value.dtype:
                host = self.buffers[index] = torch.empty_like(value, device="cpu", pin_memory=True)
            index += 1
            sources.append(value)
            host.copy_(value, non_blocking=True)
            return host

        try:
            frozen = tree_map(freeze, payload)
        except BaseException:
            # A partial copy must also finish before its source or host buffer is released.
            self.event.record(torch.npu.current_stream())
            self.event.synchronize()
            raise
        self.event.record(torch.npu.current_stream())
        trace_id, lock, done, failure = self.trace_id, Lock(), False, None

        def finish():
            nonlocal done, failure
            with lock:
                if not done:
                    try:
                        self.event.synchronize()
                        sources.clear()
                        values = tree_map(lambda value: value.tolist() if hasattr(value, "tolist") else value, frozen)
                        record(trace_id, values)
                    except Exception as exception:
                        failure = exception
                    done = True
                if failure is not None:
                    raise failure

        self.pending = finish


def install(runner, rank, guard):
    recorder = PlanRecorder(rank, guard)
    native_initialize = runner.initialize_kv_cache
    native_postprocess = runner.postprocess_sampled
    native_dummy = runner._dummy_run

    @wraps(native_dummy)
    def dummy_run(*args, **kwargs):
        recorder.pending()
        previous_trace = recorder.trace_id
        recorder.trace_id = uuid4().hex
        context = dict(
            trace_id=recorder.trace_id, rank=rank, dp_rank=runner.vllm_config.parallel_config.data_parallel_rank
        )
        try:
            emit("boundary", **context, boundary="runner._dummy_run.input")
            result = native_dummy(*args, **kwargs)
            recorder.pending()
            emit("boundary", **context, boundary="runner._dummy_run.return", phase="observed")
            guard.check()
            return result
        finally:
            recorder.trace_id = previous_trace

    @wraps(native_postprocess)
    def postprocess_sampled(
        idx_mapping, sampled_tokens, num_sampled, num_rejected, query_start_loc=None, draft_tokens=None
    ):
        observed_trace_id = recorder.trace_id
        last_rank = runner.is_last_pp_rank
        active = observed_trace_id is not None and (
            runner.speculator is not None or runner.sync_spec_pp_cpu_counts or runner.num_speculative_steps > 0
        )
        rows = None
        if active and last_rank:
            budgets = {req: count for req, _computed, count, _groups in recorder.plan}
            rows = {
                index: (req, budgets[req]) for req, index in runner.req_states.req_id_to_index.items() if req in budgets
            }
        result = native_postprocess(
            idx_mapping, sampled_tokens, num_sampled, num_rejected, query_start_loc, draft_tokens
        )
        if active:

            def record(_trace_id, values):
                values["trace_id"] = observed_trace_id if last_rank else None
                guard.spec(values, rows)
                emit(
                    "boundary",
                    observed_trace_id=observed_trace_id,
                    rank=rank,
                    association="current_step" if last_rank else "observed_call_only",
                    boundary="runner.postprocess_sampled.return",
                    evidence="sample_postprocess",
                    native_returned=True,
                    **values,
                )

            recorder.snapshot(
                {
                    "idx_mapping": idx_mapping,
                    "num_sampled": num_sampled,
                    "num_rejected": num_rejected,
                    "query_start_loc": query_start_loc,
                    "sample_capacity": sampled_tokens.shape[-1],
                },
                record,
            )
        return result

    @wraps(native_initialize)
    def initialize_kv_cache(*args, **kwargs):
        recorder.pending()
        result = native_initialize(*args, **kwargs)
        native_prepare = runner.model_state.prepare_attn
        native_tables = runner.block_tables
        pcp = runner.pcp_manager
        layout = {
            "groups": [
                {
                    "block_size": size,
                    "kernel_block_size": native_tables.kernel_block_sizes[group],
                    "slot_mapping_enabled": native_tables.slot_mapping_enabled[group],
                }
                for group, size in enumerate(native_tables.block_sizes)
            ],
            "pcp_size": pcp.pcp_world_size if pcp is not None else 1,
            "pcp_rank": pcp.pcp_rank if pcp is not None else 0,
            "dcp_size": runner.dcp_size,
            "dcp_rank": runner.dcp_rank,
            "cp_size": native_tables.cp_size,
            "cp_rank": native_tables.cp_rank,
            "cp_interleave": native_tables.cp_interleave,
            "dynamic": runner.speculator is not None or runner.num_speculative_steps > 0,
            "num_blocks": runner.kv_cache_config.num_blocks,
            "dp_rank": runner.vllm_config.parallel_config.data_parallel_rank,
        }
        if pcp is not None and pcp.pcp_world_size > 1:
            native_context = pcp.build_attention_context

            @wraps(native_context)
            def build_attention_context(input_batch=None, block_tables=None, slot_mappings=None):
                context = native_context(input_batch, block_tables, slot_mappings)
                batch = context.global_batch
                if recorder.trace_id is not None and not (batch.is_dummy or batch.is_cpp_profile):
                    recorder.capture_parallel(
                        context,
                        {
                            **layout,
                            "allow_null_block": native_tables.redirect_writes_to_null_block,
                        },
                        input_batch,
                        block_tables,
                        slot_mappings,
                    )
                return context

            pcp.build_attention_context = build_attention_context

        @wraps(native_prepare)
        # Match the native positional/keyword calling convention.
        # pylint: disable-next=keyword-arg-before-vararg
        def prepare_attn(
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture=False,
            *args,
            **kwargs,
        ):
            current_layout = {**layout, "allow_null_block": native_tables.redirect_writes_to_null_block}
            if input_batch.is_dummy and not getattr(input_batch, "is_cpp_profile", False):
                persistent = pcp._gathered_kv_slot_mappings if pcp is not None else native_tables.slot_mappings
                recorder.capture_dummy(slot_mappings, persistent, current_layout, cudagraph_mode.name)
            elif recorder.trace_id is not None and not (for_capture or getattr(input_batch, "is_cpp_profile", False)):
                recorder.capture(input_batch, block_tables, slot_mappings, current_layout, cudagraph_mode.name)
            else:
                recorder.pending()
            return native_prepare(
                input_batch,
                cudagraph_mode,
                block_tables,
                slot_mappings,
                attn_groups,
                kv_cache_config,
                for_capture,
                *args,
                **kwargs,
            )

        runner.model_state.prepare_attn = prepare_attn
        return result

    runner.initialize_kv_cache, runner.postprocess_sampled = initialize_kv_cache, postprocess_sampled
    runner._dummy_run = dummy_run
    return recorder
