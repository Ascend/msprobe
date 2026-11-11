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

"""Load KV boundary observers through vLLM's general plugin entry point."""

from functools import partial, wraps


def _run_engine_core(target, *args, **kwargs):
    from vllm.plugins import load_general_plugins

    load_general_plugins()
    return target(*args, **kwargs)


def register():
    from vllm.platforms import current_platform

    if current_platform.device_type != "npu":
        return

    from vllm_ascend import _ensure_global_patch
    from vllm_ascend.ascend_config import AscendConfig

    from . import scheduler, worker
    from .config import parse_options

    _ensure_global_patch()
    from vllm.v1.engine.core import EngineCoreProc

    native_init = AscendConfig.__init__

    @wraps(native_init)
    def initialize(self, *args, **kwargs):
        if "kv_trace" in kwargs:
            parse_options(kwargs.pop("kv_trace"))
        native_init(self, *args, **kwargs)

    # Ascend rejects tool keys; filter only its constructor arguments.
    AscendConfig.__init__ = initialize
    # Spawned Ascend engines validate config before EngineCore loads plugins.
    EngineCoreProc.run_engine_core = staticmethod(partial(_run_engine_core, EngineCoreProc.run_engine_core))
    scheduler.install()
    worker.install()
