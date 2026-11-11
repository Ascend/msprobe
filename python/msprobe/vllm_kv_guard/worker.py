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

"""Observe native worker calls and results without unpacking asynchronous outputs."""

from functools import wraps

from .config import get_options
from .events import emit, result_fields
from .checker import Guard
from .transfer import record_plan


def _observe(worker, options):
    from .model_runner import install as install_runner

    if not worker.use_v2_model_runner:
        raise ValueError("kv_trace requires Model Runner V2")
    guard = Guard(options["action"], worker.rank)
    recorder = install_runner(worker.model_runner, worker.rank, guard)
    emit("kv.enabled", rank=worker.rank, action=options["action"], boundary="worker.init_device.return")

    def error(boundary, trace_id, exception):
        emit(
            "kv.error",
            trace_id=trace_id,
            rank=worker.rank,
            boundary=boundary,
            error_type=type(exception).__name__,
            error=str(exception),
        )

    def observe_output(output, trace_id, pending):
        if output is None:
            return None

        def record(result):
            pending()
            emit(
                "boundary",
                trace_id=trace_id,
                rank=worker.rank,
                boundary="worker.output",
                phase="observed",
                evidence="native_result",
                **result_fields(result),
            )
            guard.check()
            return result

        native_get = getattr(output, "get_output", None)
        if native_get is None:
            record(output)
        else:

            @wraps(native_get)
            def get_output(*args, **kwargs):
                try:
                    result = native_get(*args, **kwargs)
                except Exception as exception:
                    error("worker.output", trace_id, exception)
                    raise
                return record(result)

            output.get_output = get_output
        return output

    native_execute = worker.execute_model

    @wraps(native_execute)
    def execute_model(scheduler_output, *args, **kwargs):
        trace_id = scheduler_output._msprobe_kv_trace_id
        recorder.trace_id = trace_id
        recorder.plan = scheduler_output._msprobe_kv_plan
        emit(
            "boundary",
            trace_id=trace_id,
            rank=worker.rank,
            boundary="worker.execute_model.input",
            evidence="scheduler_plan",
            num_scheduled_tokens=dict(scheduler_output.num_scheduled_tokens),
            total_num_scheduled_tokens=scheduler_output.total_num_scheduled_tokens,
        )
        record_plan(scheduler_output.kv_connector_metadata, trace_id, "worker.execute_model.input", worker.rank)
        try:
            output = native_execute(scheduler_output, *args, **kwargs)
        except Exception as exception:
            recorder.trace_id = None
            error("worker.execute_model", trace_id, exception)
            raise
        if output is not None:
            recorder.trace_id = None
        return observe_output(output, trace_id, recorder.pending)

    native_sample = worker.sample_tokens

    @wraps(native_sample)
    def sample_tokens(*args, **kwargs):
        trace_id = recorder.trace_id
        try:
            output = native_sample(*args, **kwargs)
        except Exception as exception:
            recorder.trace_id = None
            error("worker.sample_tokens", trace_id, exception)
            raise
        recorder.trace_id = None
        return observe_output(output, trace_id, recorder.pending)

    worker.execute_model, worker.sample_tokens = execute_model, sample_tokens


def install():
    from vllm.v1.worker.worker_base import WorkerWrapperBase

    native_init = WorkerWrapperBase.init_device

    @wraps(native_init)
    def init_device(wrapper, *args, **kwargs):
        result = native_init(wrapper, *args, **kwargs)
        options = get_options(wrapper.vllm_config)
        if options is not None:
            _observe(wrapper.worker, options)
        return result

    WorkerWrapperBase.init_device = init_device
