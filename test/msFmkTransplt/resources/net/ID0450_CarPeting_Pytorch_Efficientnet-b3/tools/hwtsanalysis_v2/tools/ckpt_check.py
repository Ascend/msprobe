# -*- coding: UTF-8 -*-
import os
import numpy as np
from tensorflow.python import pywrap_tensorflow


"""
@function:
    check ckpt consistency.
@author: 
    w00282991
@email:
    wangbei5@huawei.com
"""


def get_ckpt_list(input_path):
    ckpt_path_list = list()
    for root, dirs, files in os.walk(input_path):
        for ckpt_dir in dirs:
            if 'ckpt' in ckpt_dir and 'npu' not in root:
                ckpt_path_list.append(os.path.join(root, ckpt_dir))
    return ckpt_path_list


def get_file_list(input_path):
    ckpt_file_list = list()
    for root, dirs, files in os.walk(input_path):
        for ckpt_file in files:
            if 'index' in ckpt_file:
                ckpt_file_list.append(os.path.join(root, ckpt_file).split('.index')[0])
    return ckpt_file_list


def ckpt_check(newest_result):
    ckpt_path_list = get_ckpt_list(newest_result)
    ckpt_file_list = list()
    for ckpt_path in ckpt_path_list:
        ckpt_file_list.append(get_file_list(ckpt_path))
    for i in range(len(ckpt_file_list) - 1):
        if len(ckpt_file_list[i]) != len(ckpt_file_list[i + 1]):
            return 1
        for j in range(len(ckpt_file_list[i])):
            reader_1st = pywrap_tensorflow.NewCheckpointReader(ckpt_file_list[i][j])
            var_1st = reader_1st.get_variable_to_shape_map()
            reader_2nd = pywrap_tensorflow.NewCheckpointReader(ckpt_file_list[i + 1][j])
            var_2nd = reader_2nd.get_variable_to_shape_map()
            error_flag = False
            for key in var_1st:
                if 'moving_mean' not in key and 'moving_variance' not in key:
                    if var_1st[key] != var_2nd[key]:
                        error_flag = True
                    else:
                        value_1st = reader_1st.get_tensor(key)
                        value_2nd = reader_2nd.get_tensor(key)
                        if np.allclose(value_1st, value_2nd, 0, 0):
                            pass
                        else:
                            error_flag = True
            if error_flag:
                return 2
    return 0
