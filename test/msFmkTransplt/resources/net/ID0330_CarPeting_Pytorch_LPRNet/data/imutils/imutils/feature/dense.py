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

import cv2

class DENSE:
    def __init__(self, step=6, radius=.5):
        self.step = step
        self.radius = radius

    def detect(self, img):
        # initialize our list of keypoints
        kps = []

        # loop over the height and with of the image, taking a `step`
        # in each direction
        for x in range(0, img.shape[1], self.step):
            for y in range(0, img.shape[0], self.step):
                # create a keypoint and add it to the keypoints list
                kps.append(cv2.KeyPoint(x, y, self.radius))

        # return the dense keypoints
        return kps

    def setInt(self, var, val):
        if var == "initXyStep":
            self.step = val
