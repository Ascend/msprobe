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
import sys
sys.path.append(os.getcwd())
from wider_loader import WIDER
import cv2
import time

"""
 modify .mat to .txt 
"""

#wider face original images path
path_to_image = './data_set/face_detection/WIDERFACE/WIDER_train/WIDER_train/images'

#matlab file path
file_to_label = './data_set/face_detection/WIDERFACE/wider_face_split/wider_face_split/wider_face_train.mat'

#target file path
target_file = './anno_store/anno_train.txt'

wider = WIDER(file_to_label, path_to_image)

line_count = 0
box_count = 0

print('start transforming....')
t = time.time()

with open(target_file, 'w+') as f:
    # press ctrl-C to stop the process
    for data in wider.next():
        line = []
        line.append(str(data.image_name))
        line_count += 1
        for i,box in enumerate(data.bboxes):
            box_count += 1
            for j,bvalue in enumerate(box):
                line.append(str(bvalue))

        line.append('\n')

        line_str = ' '.join(line)
        f.write(line_str)

st = time.time()-t
print('end transforming')

print('spend time:%d'%st)
print('total line(images):%d'%line_count)
print('total boxes(faces):%d'%box_count)
print("执行成功")
