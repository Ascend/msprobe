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

"""Test for SegNet"""

from __future__ import print_function
from model import SegNet
from dataset import NUM_CLASSES
import matplotlib.pyplot as plt
import numpy as np
import torch_npu
import torch


if __name__ == "__main__":
    # RGB input
    input_channels = 3
    # RGB output
    output_channels = NUM_CLASSES

    # Model
    model = SegNet(input_channels=input_channels, output_channels=output_channels)

    print(model)

    img = torch.randn([4, 3, 224, 224])

    # plt.imshow(np.transpose(img.numpy()[0,:,:,:],
    #                         (1, 2, 0)))
    # plt.show()

    output, softmaxed_output = model(img)


    # plt.imshow(np.transpose(output.detach().numpy()[0,:,:,:],
    #                         (1, 2, 0)))
    # plt.show()


    print(output.size())
    print(softmaxed_output.size())

    print(output[0,:,0,0])
    print(softmaxed_output[0,:,0,0].sum())
