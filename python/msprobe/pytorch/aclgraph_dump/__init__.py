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
import torch_npu  # noqa: F401 - registers the PrivateUse1 backend
from torch.fx.node import has_side_effect

# Import the C++ extension to register TORCH_LIBRARY implementations.
try:
    from msprobe.lib import aclgraph_dump_ext  # pylint: disable=no-name-in-module
except Exception as exc:
    raise RuntimeError(f"Failed to import msprobe.lib.aclgraph_dump_ext: {exc}")

# Register Python fake implementation for meta tensors.
from ._meta import _register_meta  # noqa: E402

_register_meta()

has_side_effect(torch.ops.my_ns.acl_save.default)
has_side_effect(torch.ops.my_ns.acl_tensor_save.default)
has_side_effect(torch.ops.my_ns.acl_stat.default)


def acl_save(x: torch.Tensor, path: str) -> torch.Tensor:
    """
    acl_save(tensor, path) -> tensor

    Copy tensor to CPU and save to a .pt file.
    The file name is generated as {base}_{seq}.pt in the same directory.
    For NPU input, the save runs on the current NPU stream; synchronize if needed.
    """
    tensor_to_save = x if x.is_contiguous() else x.contiguous()
    return torch.ops.my_ns.acl_save(tensor_to_save, path)


def acl_tensor_save(
    x: torch.Tensor, path: str, api_name: str, is_call_start: bool = False, switch: torch.Tensor = None
) -> torch.Tensor:
    """Save a whole-network tensor, grouped by its replay-time API call index."""
    tensor_to_save = x if x.is_contiguous() else x.contiguous()
    return torch.ops.my_ns.acl_tensor_save(tensor_to_save, path, api_name, is_call_start, switch)


def acl_stat(x: torch.Tensor, tag: str, switch: torch.Tensor = None) -> torch.Tensor:
    """
    acl_stat(tensor, tag) -> tensor

    Collect min/max/mean/norm on device, then stash the statistics plus dtype
    and shape into the host-side dictionary.
    """
    return torch.ops.my_ns.acl_stat(x, tag, switch)


def get_acl_stat_dict(clear: bool = False):
    return aclgraph_dump_ext.get_acl_stat_dict(clear)


__all__ = ["acl_save", "acl_tensor_save", "acl_stat", "get_acl_stat_dict"]
