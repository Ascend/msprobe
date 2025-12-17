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

import torch_npu
import torch
import torch.nn as nn

m=nn.Conv3d(1, 16, kernel_size=(1, 1, 1), stride=(1, 1, 1))
input = torch.randn(8, 1, 16, 96, 96).float()
output = m(input)
print(output.size())


m=nn.Conv3d(1, 16, kernel_size=(1, 1, 1), stride=(1, 1, 1)).npu()
input = torch.randn(8, 1, 16, 96, 96).float().npu()
output = m(input)
print(output)
print(output.size())




