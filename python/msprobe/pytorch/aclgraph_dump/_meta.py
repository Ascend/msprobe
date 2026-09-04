# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
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


import torch
from msprobe.core.common.log import logger


def _register_meta():
    try:

        @torch.library.register_fake("my_ns::acl_save")
        def _fake_acl_save(x: torch.Tensor, path: str):
            return torch.empty_strided(x.size(), x.stride(), dtype=x.dtype, device="meta")

        @torch.library.register_fake("my_ns::acl_tensor_save")
        def _fake_acl_tensor_save(
            x: torch.Tensor, path: str, api_name: str, is_call_start: bool = False, switch: torch.Tensor = None
        ):
            return x

        @torch.library.register_fake("my_ns::acl_stat")
        def _fake_acl_stat(x: torch.Tensor, stats: torch.Tensor, tag: str, switch: torch.Tensor = None):
            return x
    except Exception as exc:  # duplicate fake registration is safe to ignore, other failures should be visible
        logger.warning(
            f"Failed to register fake implementations for aclgraph dump ops, meta tensor tracing may be "
            f"incorrect. Error: {exc}"
        )
