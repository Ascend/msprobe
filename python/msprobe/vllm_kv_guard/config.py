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
    if set(options) - {"enabled", "action"} or not isinstance(options.get("enabled", True), bool):
        raise ValueError("kv_trace supports enabled (boolean) and action (log or raise)")
    action = options.get("action", "log")
    if action not in ("log", "raise"):
        raise ValueError("kv_trace.action must be log or raise")
    return {"enabled": True, "action": action} if options.get("enabled", True) else None


def get_options(vllm_config):
    return parse_options((vllm_config.additional_config or {}).get("kv_trace"))
