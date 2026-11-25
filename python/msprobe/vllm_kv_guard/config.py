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

"""The additional_config.kv_trace switch shared by all vLLM processes."""


def parse_options(options):
    if options is None:
        return None
    if not isinstance(options, dict):
        raise TypeError("additional_config.kv_trace must be a dict")
    if "slot_audit" in options:
        raise ValueError("use kv_trace.slot instead of slot_audit")
    enabled = options.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("kv_trace.enabled must be a boolean")
    action = options.get("action", "log")
    if action not in ("log", "raise"):
        raise ValueError("kv_trace.action must be log or raise")
    slot = options.get("slot", False)
    if not isinstance(slot, bool):
        raise ValueError("kv_trace.slot must be a boolean")
    if slot and not enabled:
        raise ValueError("kv_trace.slot requires enabled=true")
    if not enabled:
        return None
    return {"enabled": True, "action": action, **({"slot": True} if slot else {})}


def get_options(vllm_config):
    options = parse_options((vllm_config.additional_config or {}).get("kv_trace"))
    if options is not None and options.get("slot", False):
        parallel = vllm_config.parallel_config
        model = vllm_config.model_config
        transfer = vllm_config.kv_transfer_config
        offload = (vllm_config.additional_config or {}).get("sparse_kv_offload_config")
        unsupported = {
            "DBO": getattr(parallel, "enable_dbo", False),
            "speculative decoding": vllm_config.speculative_config is not None,
            "encoder-decoder": getattr(model, "is_encoder_decoder", False),
            "sparse_kv_offload_config": offload is not None
            and (not isinstance(offload, dict) or offload.get("enabled", False)),
        }
        for path, active in unsupported.items():
            if active:
                raise ValueError(f"kv_trace.slot does not support {path}")
        if transfer is not None:
            roles = {"MooncakeConnectorV1": ("kv_producer", "kv_consumer"), "PreemptOffloadConnector": ("kv_consumer",)}
            name = getattr(transfer, "kv_connector", None)
            shards = parallel.tensor_parallel_size * parallel.pipeline_parallel_size
            shards *= parallel.prefill_context_parallel_size * parallel.decode_context_parallel_size
            if (
                getattr(transfer, "kv_role", None) not in roles.get(name, ())
                or getattr(transfer, "kv_connector_module_path", None) is not None
                or shards != 1
            ):
                raise ValueError("kv_trace.slot does not support this KV transfer configuration")
            options["pd_slot" if name == "MooncakeConnectorV1" else "swap_slot"] = True
    return options
