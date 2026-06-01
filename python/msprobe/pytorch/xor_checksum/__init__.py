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

try:
    from msprobe.lib import xor_checksum_ext as _xor_checksum_ext  # noqa: F401  # pylint: disable=no-name-in-module
except Exception:
    _xor_checksum_ext = None


def _xor_checksum_fallback(tensor: torch.Tensor) -> torch.Tensor:
    bytes_tensor = tensor.view(torch.uint8).flatten()
    numel = bytes_tensor.numel()

    if numel % 8 != 0:
        bytes_tensor = torch.nn.functional.pad(bytes_tensor, (0, 8 - numel % 8), "constant", 0)
    if bytes_tensor.storage_offset() % 8 != 0:
        bytes_tensor = bytes_tensor.clone()

    words = bytes_tensor.view(torch.int64)
    numel = words.numel()
    while numel > 1:
        if numel % 2 != 0:
            words = torch.nn.functional.pad(words, (0, 1), "constant", 0)
        words = words.view(2, -1)
        words = torch.bitwise_xor(words[0, :], words[1, :])
        numel = words.numel()
    return words[0]


def xor_checksum(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 0:
        return tensor.clone()
    if tensor.numel() == 0:
        return torch.zeros((), dtype=torch.int64, device=tensor.device)

    if tensor.device.type == "npu" and _xor_checksum_ext is not None:
        bytes_tensor = tensor.view(torch.uint8)
        try:
            return torch.ops.msprobe.xor_checksum(bytes_tensor)
        except Exception:
            return _xor_checksum_fallback(tensor)
    return _xor_checksum_fallback(tensor)


__all__ = ["xor_checksum"]
