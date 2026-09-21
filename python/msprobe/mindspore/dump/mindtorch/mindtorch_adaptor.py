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

# From PyTorch:

# Copyright (c) 2025      Huawei Technologies Co., Ltd
# Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
# Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
# Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
# Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
# Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
# Copyright (c) 2011-2013 NYU                      (Clement Farabet)
# Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
# Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
# Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

# From Caffe2:

# Copyright (c) 2016-present, Facebook Inc. All rights reserved.

# All contributions by Facebook:
# Copyright (c) 2016 Facebook Inc.

# All contributions by Google:
# Copyright (c) 2015 Google Inc.
# All rights reserved.

# All contributions by Yangqing Jia:
# Copyright (c) 2015 Yangqing Jia
# All rights reserved.

# All contributions by Kakao Brain:
# Copyright 2019-2020 Kakao Brain

# All contributions by Cruise LLC:
# Copyright (c) 2022 Cruise LLC.
# All rights reserved.

# All contributions by Tri Dao:
# Copyright (c) 2024 Tri Dao.
# All rights reserved.

# All contributions by Arm:
# Copyright (c) 2021, 2023-2024 Arm Limited and/or its affiliates

# All contributions from Caffe:
# Copyright(c) 2013, 2014, 2015, the respective contributors
# All rights reserved.

# All other contributions:
# Copyright(c) 2015, 2016 the respective contributors
# All rights reserved.

# Caffe2 uses a copyright model similar to Caffe: each contributor holds
# copyright over their contributions to Caffe2. The project versioning records
# all such contribution and copyright details. If a contributor wants to further
# mark their specific copyright on a particular contribution, they should
# indicate their copyright solely in the commit message of the change when it is
# committed.

# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.

# 3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories
#    America, IDIAP Research Institute and Huawei nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import warnings

import mindspore as ms
from mindspore.ops.operations import _inner_ops as inner
from torch.nn.modules.module import (
    _global_backward_pre_hooks,
    _global_backward_hooks,
    _global_is_full_backward_hook,
    _global_forward_pre_hooks,
    _global_forward_hooks,
    _global_forward_hooks_always_called,
)
from torch.utils.hooks import RemovableHandle

from msprobe.mindspore.common.utils import is_backward_hook_output_a_view


def _call_impl(self, *args, **kwargs):
    forward_call = self.forward
    if self.__ms_class__:
        return forward_call(*args, **kwargs)

    # If we don't have any hooks, we want to skip the rest of the logic in
    # this function, and just call forward.
    if not (
        self._backward_hooks
        or self._backward_pre_hooks
        or self._forward_hooks
        or self._forward_pre_hooks
        or _global_backward_pre_hooks
        or _global_backward_hooks
        or _global_forward_hooks
        or _global_forward_pre_hooks
    ):
        return forward_call(*args, **kwargs)

    try:  # pylint: disable=too-many-nested-blocks
        result = None
        called_always_called_hooks = set()

        if self._backward_pre_hooks or _global_backward_pre_hooks:
            _get_backward_pre_hooks(self)

        if self._backward_hooks or _global_backward_hooks:
            _get_backward_hooks(self)

        if _global_forward_pre_hooks or self._forward_pre_hooks:
            for hook_id, hook in (
                *_global_forward_pre_hooks.items(),
                *self._forward_pre_hooks.items(),
            ):
                if hook_id in self._forward_pre_hooks_with_kwargs:
                    args_kwargs_result = hook(self, args, kwargs)  # type: ignore[misc]
                    if args_kwargs_result is not None:
                        if isinstance(args_kwargs_result, tuple) and len(args_kwargs_result) == 2:
                            args, kwargs = args_kwargs_result
                        else:
                            raise RuntimeError(
                                "forward pre-hook must return None or a tuple "
                                f"of (new_args, new_kwargs), but got {args_kwargs_result}."
                            )
                else:
                    args_result = hook(self, args)
                    if args_result is not None:
                        if not isinstance(args_result, tuple):
                            args_result = (args_result,)
                        args = args_result

        bw_hook = None
        if self._backward_hooks:
            bw_hook = inner.CellBackwardHook(
                self.__class__.__name__ + "(" + str(id(self)) + ")", self, self._backward_hooks
            )
            bw_hook.register_backward_hook()
            args = apply_backward_hook_on_tensors(bw_hook, args)

        result = forward_call(*args, **kwargs)
        if _global_forward_hooks or self._forward_hooks:
            for hook_id, hook in (
                *_global_forward_hooks.items(),
                *self._forward_hooks.items(),
            ):
                # mark that always called hook is run
                if hook_id in self._forward_hooks_always_called or hook_id in _global_forward_hooks_always_called:
                    called_always_called_hooks.add(hook_id)

                if hook_id in self._forward_hooks_with_kwargs:
                    hook_result = hook(self, args, kwargs, result)
                else:
                    hook_result = hook(self, args, result)

                if hook_result is not None:
                    result = hook_result

        if bw_hook:
            if not isinstance(result, (ms.Tensor, tuple)):
                warnings.warn(
                    "For backward hooks to be called,"
                    " module output should be a Tensor or a tuple of Tensors"
                    f" but received {type(result)}"
                )
            result = apply_backward_hook_on_tensors(bw_hook, result)

        if self._backward_pre_hooks:
            bw_pre_hook = inner.CellBackwardHook(
                self.__class__.__name__ + "(" + str(id(self)) + ")", self, self._backward_pre_hooks
            )
            bw_pre_hook.register_backward_pre_hook()
            result = apply_backward_hook_on_tensors(bw_pre_hook, result)

        return result
    except Exception:
        # run always called hooks if they have not already been run
        # For now only forward hooks have the always_call option but perhaps
        # this functionality should be added to full backward hooks as well.
        for hook_id, hook in _global_forward_hooks.items():
            # type: ignore[possibly-undefined]
            if hook_id in _global_forward_hooks_always_called and hook_id not in called_always_called_hooks:
                try:
                    hook_result = hook(self, args, result)  # type: ignore[possibly-undefined]
                    if hook_result is not None:
                        result = hook_result
                except Exception as e:
                    warnings.warn(
                        "global module forward hook with ``always_call=True`` raised an exception "
                        f"that was silenced as another error was raised in forward: {str(e)}"
                    )
                    continue

        for hook_id, hook in self._forward_hooks.items():
            # type: ignore[possibly-undefined]
            if hook_id in self._forward_hooks_always_called and hook_id not in called_always_called_hooks:
                try:
                    if hook_id in self._forward_hooks_with_kwargs:
                        hook_result = hook(self, args, kwargs, result)  # type: ignore[possibly-undefined]
                    else:
                        hook_result = hook(self, args, result)  # type: ignore[possibly-undefined]
                    if hook_result is not None:
                        result = hook_result
                except Exception as e:
                    warnings.warn(
                        "module forward hook with ``always_call=True`` raised an exception "
                        f"that was silenced as another error was raised in forward: {str(e)}"
                    )
                    continue
        # raise exception raised in try block
        raise


def register_full_backward_pre_hook(self, hook, prepend: bool = False) -> RemovableHandle:
    handle = RemovableHandle(self._backward_pre_hooks)
    self._backward_pre_hooks[handle.id] = hook
    if prepend:
        self._backward_pre_hooks.move_to_end(handle.id, last=False)  # type: ignore[attr-defined]
    return handle


def register_full_backward_hook(self, hook, prepend: bool = False) -> RemovableHandle:
    if self._is_full_backward_hook is False:
        raise RuntimeError(
            "Cannot use both regular backward hooks and full backward hooks on a "
            "single Module. Please use only one of them."
        )

    self._is_full_backward_hook = True

    handle = RemovableHandle(self._backward_hooks)
    self._backward_hooks[handle.id] = hook
    if prepend:
        self._backward_hooks.move_to_end(handle.id, last=False)  # type: ignore[attr-defined]
    return handle


def _get_backward_pre_hooks(self):
    self._backward_pre_hooks.update(_global_backward_pre_hooks)


def _get_backward_hooks(self):
    if _global_is_full_backward_hook is True:
        self._backward_hooks.update(_global_backward_hooks)


def apply_backward_hook_on_tensors(cell_backward_hook, args):
    if is_backward_hook_output_a_view():
        hooked_args = cell_backward_hook(args)
    else:
        is_tuple = True
        if not isinstance(args, tuple):
            args = (args,)
            is_tuple = False
        hooked_args = cell_backward_hook(*args)
        if is_tuple and len(args) == 1:
            hooked_args = (hooked_args,)
    return hooked_args
