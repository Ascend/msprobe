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

import threading
from collections import defaultdict

from packaging.version import parse as version_parse

import mindspore as ms
from mindspore import nn

from msprobe.mindspore.common.utils import is_mindtorch

ms_version = version_parse(ms.__version__)


def add_cell_count(name):
    HOOKCell.cell_count[name] += 1


def get_cell_count(name):
    return HOOKCell.cell_count[name]


def __init__(self, hook_build_func) -> None:
    super(HOOKCell, self).__init__()
    self.msprobe_input_kwargs = {}
    prefix = self.prefix_api_name if hasattr(self, "prefix_api_name") else ""
    if callable(hook_build_func):
        hook_set = hook_build_func(prefix)
        if ms_version < version_parse("2.6.0") and not is_mindtorch():
            getattr(self, "_forward_pre_hook", {})[id(self)] = hook_set.forward_pre_hook
            if hook_set.forward_hook:
                getattr(self, "_forward_hook", {})[id(self)] = hook_set.forward_hook
        else:
            self.register_forward_pre_hook(hook_set.forward_pre_hook)
            if hook_set.forward_hook:
                self.register_forward_hook(hook_set.forward_hook)


def __call__(self, *args, **kwargs):
    tid = threading.get_ident()
    self.msprobe_input_kwargs[tid] = kwargs
    out = super(HOOKCell, self).__call__(*args, **kwargs)
    return out


hook_cell_dict = {
    "cell_count": defaultdict(int),
    "add_cell_count": staticmethod(add_cell_count),
    "get_cell_count": staticmethod(get_cell_count),
    "__init__": __init__,
    "__call__": __call__,
}

if is_mindtorch():
    import torch

    HOOKCell = type("HOOKCell", (torch.nn.Module,), hook_cell_dict)
else:
    HOOKCell = type("HOOKCell", (nn.Cell,), hook_cell_dict)
