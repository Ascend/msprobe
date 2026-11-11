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
