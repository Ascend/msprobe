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

from __future__ import absolute_import
import os, sys
import functools
import random
import torch_npu

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn import init

class GANLoss(nn.Module):
    def __init__(self, smooth=False):
        super(GANLoss, self).__init__()
        self.smooth = smooth

    def get_target_tensor(self, input, target_is_real):
        real_label = 1.0
        fake_label = 0.0
        if self.smooth:
            real_label = random.uniform(0.7,1.0)
            fake_label = random.uniform(0.0,0.3)
        if target_is_real:
            target_tensor = torch.ones_like(input).fill_(real_label)
        else:
            target_tensor = torch.zeros_like(input).fill_(fake_label)
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        input = F.sigmoid(input)
        return F.binary_cross_entropy(input, target_tensor)
