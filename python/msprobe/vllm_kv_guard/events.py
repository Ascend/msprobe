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

"""Write native boundary facts without reconstructing KV internals."""

import json

from msprobe.core.common.log import logger


def emit(event, **fields):
    try:
        message = json.dumps(fields, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return
    log = logger.error if event in {"kv.error", "kv.violation"} else logger.info
    log(f"[KV_GUARD] {event} {message}")


def result_fields(output):
    requests = list(output.req_ids)
    sampled = output.sampled_token_ids
    fields = {
        "req_ids": requests,
        "req_id_to_index": dict(output.req_id_to_index),
        "num_output_tokens": {}
        if sampled is None or len(sampled) == 0
        else {
            request: len(sampled[index]) if 0 <= index < len(sampled) else None
            for request, index in output.req_id_to_index.items()
        },
    }
    connector = output.kv_connector_output
    if connector is not None:
        fields["kv_connector_output"] = {
            name: None if (value := getattr(connector, name)) is None else sorted(value)
            for name in ("finished_sending", "finished_recving", "failed_recving", "invalid_block_ids")
        }
    return fields
