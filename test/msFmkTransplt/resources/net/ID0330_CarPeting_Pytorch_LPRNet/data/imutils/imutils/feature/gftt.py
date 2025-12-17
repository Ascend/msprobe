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
from .helpers import corners_to_keypoints


class GFTT:
    def __init__(self, maxCorners=0, qualityLevel=0.01, minDistance=1,
                 mask=None, blockSize=3, useHarrisDetector=False, k=0.04):
        self.maxCorners = maxCorners
        self.qualityLevel = qualityLevel
        self.minDistance = minDistance
        self.mask = mask
        self.blockSize = blockSize
        self.useHarrisDetector = useHarrisDetector
        self.k = k

    def detect(self, img):
        cnrs = cv2.goodFeaturesToTrack(img, self.maxCorners, self.qualityLevel, self.minDistance,
                                       mask=self.mask, blockSize=self.blockSize,
                                       useHarrisDetector=self.useHarrisDetector, k=self.k)

        return corners_to_keypoints(cnrs)

