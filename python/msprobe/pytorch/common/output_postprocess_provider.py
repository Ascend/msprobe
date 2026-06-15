#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
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

import torch

import msprobe.core.common.output_postprocess.processor as output_postprocess_processor


_SUPPORTED_GROUP_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool)


def _is_torch_tensor(obj):
    return isinstance(obj, torch.Tensor)


def _extract_valid_len(group_tensor):
    if group_tensor.dtype not in _SUPPORTED_GROUP_DTYPES:
        raise TypeError(f"unsupported dtype: {group_tensor.dtype}")

    return int(group_tensor.sum().item())


def _clean_torch_tensor(tensor, valid_len: int):
    if tensor is None or not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
        return tensor

    if tensor.dim() == 0:
        return tensor

    clean_tensor = torch.zeros_like(tensor)
    safe_len = max(0, min(valid_len, tensor.shape[0]))
    if tensor.dim() == 1:
        clean_tensor[:safe_len] = tensor[:safe_len]
    else:
        clean_tensor[:safe_len, ...] = tensor[:safe_len, ...]
    return clean_tensor


output_postprocess_processor.register_tensor_postprocess_impl(  # pyright: ignore[reportAttributeAccessIssue]
    _is_torch_tensor, _extract_valid_len, _clean_torch_tensor
)
