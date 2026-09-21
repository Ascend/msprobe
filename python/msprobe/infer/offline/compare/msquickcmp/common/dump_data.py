# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
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

"""
Function:
This class mainly involves generate dump data function.
"""

import os
import time

import numpy as np

from msprobe.core.common.file_utils import save_numpy_to_bin
from msprobe.infer.offline.compare.msquickcmp.common import utils
from msprobe.core.common.log import logger
from msprobe.infer.offline.compare.msquickcmp.common.utils import AccuracyCompareException
from msprobe.infer.utils.check.rule import Rule


class DumpData(object):  # pylint: disable=useless-object-inheritance
    """
    Class for generate dump data.
    """

    def __init__(self):
        self.net_output = {}
        pass

    @staticmethod
    def _to_valid_name(name_str):
        return name_str.replace('.', '_').replace('/', '_')

    @staticmethod
    def _check_path_exists(input_path, extentions=None):
        input_path = os.path.realpath(input_path)
        if not os.path.exists(input_path):
            logger.error(f"path '{input_path}' not exists")
            raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PATH_ERROR)

        if extentions and not any([input_path.endswith(extention) for extention in extentions]):  # pylint: disable=use-a-generator
            logger.error(f"path '{input_path}' not ends with extention {extentions}")
            raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PATH_ERROR)

        if not os.access(input_path, os.R_OK):
            logger.error(f"user doesn't have read permission to the file {input_path}.")
            raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PATH_ERROR)

    @staticmethod
    def _check_input_data_path(input_path, inputs_tensor_info):
        if len(inputs_tensor_info) != len(input_path):
            logger.error(
                f"the number of model inputs tensor_info is not equal the number of inputs data, "
                f"inputs tensor_info is: {len(inputs_tensor_info)}, "
                f"inputs data is: {len(input_path)}"
            )
            raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_DATA_ERROR)

        for cur_path in input_path:
            if not os.path.exists(cur_path):
                logger.error(f"input data path '{cur_path}' not exists")
                raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PATH_ERROR)

    @staticmethod
    def _generate_random_input_data(save_dir, names, shapes, dtypes):
        inputs_map = {}
        for index, (tensor_name, tensor_shape, tensor_dtype) in enumerate(zip(names, shapes, dtypes)):
            input_data = np.random.random(tensor_shape).astype(tensor_dtype)
            inputs_map[tensor_name] = input_data
            file_name = "input_" + str(index) + ".bin"
            input_data_path = os.path.join(save_dir, file_name)
            save_numpy_to_bin(input_data, input_data_path)
            logger.info(f"save input file name: {file_name}, shape: {input_data.shape}, dtype: {input_data.dtype}")
        return inputs_map

    def generate_dump_data(self):
        pass

    def get_net_output_info(self):
        return self.net_output

    def generate_inputs_data(self):
        pass

    def _generate_dump_data_file_name(self, name_str, node_id):
        return ".".join([self._to_valid_name(name_str), str(node_id), str(round(time.time() * 1e6)), "npy"])

    def _read_input_data(self, input_pathes, names, shapes, dtypes):
        inputs_map = {}
        for input_path, name, shape, dtype in zip(input_pathes, names, shapes, dtypes):
            if dtype == np.float32 and os.path.getsize(input_path) == np.prod(shape) * 2:
                if Rule.input_file().check(input_path, will_raise=True):
                    input_data = np.fromfile(input_path, dtype=np.float16).astype(np.float32)
            else:
                if Rule.input_file().check(input_path, will_raise=True):
                    input_data = np.fromfile(input_path, dtype=dtype)
            if np.prod(input_data.shape) != np.prod(shape):
                cur = input_data.shape
                logger.error(f"input data shape not match, input_path: {input_path}, shape: {cur}, target: {shape}")
                raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_DATA_ERROR)
            input_data = input_data.reshape(shape)
            inputs_map[name] = input_data
            logger.info(
                f"load input file name: {os.path.basename(input_path)}, "
                f"shape: {input_data.shape}, "
                f"dtype: {input_data.dtype}"
            )
        return inputs_map
