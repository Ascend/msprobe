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
import importlib
from unittest import TestCase
from unittest.mock import MagicMock

import mindspore as ms
from mindspore import mint

try:
    from mint import distributed
except ImportError:
    distributed = MagicMock()
    setattr(mint, 'distributed', distributed)

from mindspore.common.api import _pynative_executor
mock_requires_grad = MagicMock(return_value=True)
setattr(_pynative_executor, "requires_grad", mock_requires_grad)

from mindspore import ops
if not hasattr(ops, 'DumpGradient'):
    DumpGradient = MagicMock()
    setattr(ops, 'DumpGradient', DumpGradient)

import sys
try:
    from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor, KeyedTensor
except Exception:
    fake_module = MagicMock()
    sys.modules["torchrec.sparse.jagged_tensor"] = fake_module
    fake_module.JaggedTensor = MagicMock
    fake_module.KeyedJaggedTensor = MagicMock
    fake_module.KeyedTensor = MagicMock

# ensure not to import torch_npu
from msprobe.mindspore.dump import mindspore_service
from msprobe.mindspore.monitor import common_func

from .mindtorch import reset_torch_tensor
from msprobe.mindspore.common import utils
from msprobe.mindspore.common.utils import is_mindtorch, register_backward_hook_functions

utils.mindtorch_check_result = None
importlib.reload(mindspore_service)
importlib.reload(common_func)
reset_torch_tensor()

def register_backward_pre_hook(*args, **kwargs):
    pass


register_backward_hook_functions['full'] = ms.nn.Cell.register_backward_hook
register_backward_hook_functions["pre"] = register_backward_pre_hook


class SetUp(TestCase):
    def test_case(self):
        self.assertTrue(hasattr(mint, 'distributed'))
        self.assertTrue(hasattr(_pynative_executor, 'requires_grad'))
        self.assertTrue(is_mindtorch())
        utils.mindtorch_check_result = None
