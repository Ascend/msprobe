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

import os
from scipy.io import loadmat

class DATA:
    def __init__(self, image_name, bboxes):
        self.image_name = image_name
        self.bboxes = bboxes


class WIDER(object):
    def __init__(self, file_to_label, path_to_image=None):
        self.file_to_label = file_to_label
        self.path_to_image = path_to_image

        self.f = loadmat(file_to_label)
        self.event_list = self.f['event_list']
        self.file_list = self.f['file_list']
        self.face_bbx_list = self.f['face_bbx_list']

    def next(self):
        for event_idx, event in enumerate(self.event_list):
            # fix error of "can't not .. bytes and strings"
            e = str(event[0][0].encode('utf-8'))[2:-1]
            for file, bbx in zip(self.file_list[event_idx][0],
                                 self.face_bbx_list[event_idx][0]):
                f = file[0][0].encode('utf-8')
                # print(type(e), type(f))  # bytes, bytes
                # fix error of "can't not .. bytes and strings"
                f = str(f)[2:-1]
                # path_of_image = os.path.join(self.path_to_image, str(e), str(f)) + ".jpg"
                path_of_image = self.path_to_image + '/' + e + '/' + f + ".jpg"
                # print(path_of_image)

                bboxes = []
                bbx0 = bbx[0]
                for i in range(bbx0.shape[0]):
                    xmin, ymin, xmax, ymax = bbx0[i]
                    bboxes.append((int(xmin), int(ymin), int(xmax), int(ymax)))
                yield DATA(path_of_image, bboxes)
                    
