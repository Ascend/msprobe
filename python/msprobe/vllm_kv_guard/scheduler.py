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

"""Trace native scheduling boundaries without reconstructing allocator state.

Bindings are native deltas; processing a result does not prove device KV writes.
"""

from functools import wraps
from uuid import uuid4

from .checker import Guard, bind
from .config import get_options
from .events import emit, result_fields
from .transfer import record_plan


def _observe(scheduler, options):
    native_schedule = scheduler.schedule
    native_update = scheduler.update_from_output
    guard, bindings = Guard(options["action"]), {}
    emit("kv.enabled", boundary="scheduler", action=options["action"])

    @wraps(native_schedule)
    def schedule(*args, **kwargs):
        trace_id = uuid4().hex
        try:
            output = native_schedule(*args, **kwargs)
        except Exception as error:
            emit("kv.error", boundary="scheduler.schedule", trace_id=trace_id, error=repr(error))
            raise

        output._msprobe_kv_trace_id = trace_id
        bind(output, bindings, guard)
        plan = dict(boundary="scheduler.schedule", trace_id=trace_id, evidence="scheduler_plan")
        cached = output.scheduled_cached_reqs
        if scheduler.vllm_config.speculative_config is not None:
            plan["speculative"] = dict(
                draft_counts={req_id: len(tokens) for req_id, tokens in output.scheduled_spec_decode_tokens.items()},
                num_spec_tokens_to_schedule=output.num_spec_tokens_to_schedule,
                skip_drafts=output.skip_drafts,
                num_invalid_spec_tokens=output.num_invalid_spec_tokens,
            )
        emit(
            "kv.bind",
            **plan,
            num_scheduled_tokens=output.num_scheduled_tokens,
            total_num_scheduled_tokens=output.total_num_scheduled_tokens,
            finished_req_ids=sorted(output.finished_req_ids),
            preempted_req_ids=None if output.preempted_req_ids is None else sorted(output.preempted_req_ids),
            new_block_ids_to_zero=output.new_block_ids_to_zero,
            kv_cache_block_copies=[copy._asdict() for copy in output.kv_cache_block_copies]
            if output.kv_cache_block_copies is not None
            else None,
            scheduled_new_reqs=[
                dict(
                    req_id=req.req_id,
                    block_ids=req.block_ids,
                    num_computed_tokens=req.num_computed_tokens,
                    replay_start=req.replay_start,
                )
                for req in output.scheduled_new_reqs
            ],
            scheduled_cached_reqs=dict(
                req_ids=cached.req_ids,
                new_block_ids=cached.new_block_ids,
                num_computed_tokens=cached.num_computed_tokens,
                resumed_req_ids=sorted(cached.resumed_req_ids),
            ),
        )
        guard.check()
        record_plan(output.kv_connector_metadata, trace_id, "scheduler.schedule")
        return output

    @wraps(native_update)
    def update_from_output(scheduler_output, model_runner_output, *args, **kwargs):
        ids = list(model_runner_output.req_ids)
        common = dict(trace_id=scheduler_output._msprobe_kv_trace_id, boundary="scheduler.update_from_output")
        if len(ids) != len(set(ids)) or model_runner_output.req_id_to_index != dict(zip(ids, range(len(ids)))):
            guard.fail("result_request_index_mismatch", **common)
        if not set(ids) <= scheduler_output.num_scheduled_tokens.keys():
            guard.fail("result_request_not_scheduled", **common)
        fields = dict(
            boundary="scheduler.update_from_output.input",
            phase="input",
            evidence="native_result",
            trace_id=getattr(scheduler_output, "_msprobe_kv_trace_id", None),
            **result_fields(model_runner_output),
        )
        emit("boundary", **fields)
        guard.check()
        try:
            result = native_update(scheduler_output, model_runner_output, *args, **kwargs)
        except Exception as error:
            emit("kv.error", error=repr(error), **fields)
            raise
        return result

    scheduler.schedule = schedule
    scheduler.update_from_output = update_from_output


def install():
    from vllm.v1.core.sched.scheduler import Scheduler

    native_init = Scheduler.__init__

    @wraps(native_init)
    def initialize(self, *args, **kwargs):
        native_init(self, *args, **kwargs)
        options = get_options(self.vllm_config)
        if options is not None:
            _observe(self, options)

    Scheduler.__init__ = initialize
