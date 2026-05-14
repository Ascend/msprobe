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

import numpy as np
import torch

from msprobe.core.common.utils import logger, CompareException
from msprobe.core.common.file_utils import FileChecker, FileCheckConst
from msprobe.pytorch.common.utils import load_pt
from msprobe.pytorch.common.utils import is_float8_tensor, load_pt


def _prepare_tensor_for_numpy(tensor):
    # here not concerned about float4 tensor
    if tensor.dtype == torch.bfloat16 or is_float8_tensor(tensor):
        return tensor.to(torch.float32)
    return tensor

def read_pt_data(dir_path, file_name):
    if not file_name:
        return None

    data_path = os.path.join(dir_path, file_name)
    path_checker = FileChecker(data_path, FileCheckConst.FILE, FileCheckConst.READ_ABLE, FileCheckConst.PT_SUFFIX)
    data_path = path_checker.common_check()
    try:
        # detach because numpy can not process gradient information
        data_value = load_pt(data_path, to_cpu=True).detach()
    except RuntimeError as e:
        # 这里捕获 load_pt 中抛出的异常
        data_path_file_name = os.path.basename(data_path)
        logger.error(f"Failed to load the .pt file at {data_path_file_name}.")
        raise CompareException(CompareException.INVALID_FILE_ERROR) from e
    except AttributeError as e:
        # 这里捕获 detach 方法抛出的异常
        logger.error(f"Failed to detach the loaded tensor.")
        raise CompareException(CompareException.DETACH_ERROR) from e
    data_value = _prepare_tensor_for_numpy(data_value)
    data_value = data_value.numpy()
    data_value = np.squeeze(data_value)
    return data_value
