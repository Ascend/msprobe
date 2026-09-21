# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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
import random
import re
import time

import numpy as np
import torch

from msprobe.core.common.exceptions import MsprobeException
from msprobe.core.common.file_utils import create_directory, write_csv
from msprobe.core.common.utils import is_int
from msprobe.pytorch.common.log import logger


class Const:
    METHOD_LIST = ["tensor_random"]
    API_MAPPING = {
        "python_random": random,
        "numpy_random": np.random,
        "torch_random": torch,
        "tensor_random": torch.Tensor,
    }

    CSV_HEADER = [['api_name', 'stack']]


def _check_torch_gpu_available() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def _check_torch_npu_available() -> bool:
    try:
        return torch.npu.is_available()
    except Exception:
        return False


gpu_available = _check_torch_gpu_available()
npu_available = _check_torch_npu_available()


def get_rank_id():
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return None


def create_csv(output_path, rank_id):
    create_directory(output_path)
    if rank_id is not None:
        infix = f"rank{rank_id}"
    else:
        infix = f"proc{os.getpid()}"
    time_suffix = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
    csv_name = f'random_{infix}_{time_suffix}.csv'

    csv_path = os.path.join(output_path, csv_name)
    write_csv(Const.CSV_HEADER, csv_path, mode='w')
    logger.info(f"The {csv_name} file created.")
    logger.info(f"The random info will be saved in {csv_path}.")
    return csv_path


def rename_csv(old_csv_path, rank_id):
    old_name = os.path.basename(old_csv_path)
    new_name = re.sub(r'proc(\d+)', f'rank{rank_id}', old_name)
    new_csv_path = os.path.join(os.path.dirname(old_csv_path), new_name)
    os.rename(old_csv_path, new_csv_path)
    logger.info(f"Detected rank info, renamed CSV file: '{old_name}' -> '{new_name}'.")
    return new_csv_path


def check_arguments(seed, is_deterministic, is_enhanced):
    if is_int(seed):
        if seed < 0 or seed > 2**32 - 1:
            logger.error("The seed must be between 0 and 2**32 - 1.")
            raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR, "the seed must be between 0 and 2**32 - 1.")
    else:
        logger.error("The seed must be integer.")
        raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR, "the seed must be integer.")
    if not isinstance(is_deterministic, bool):
        logger.error("The is_deterministic must be bool.")
        raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR, "the is_deterministic must be bool.")
    if not isinstance(is_enhanced, bool):
        logger.error("The is_enhanced must be bool.")
        raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR, "the is_enhanced must be bool.")
