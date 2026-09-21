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

import numpy as np
import torch
from packaging import version

from msprobe.core.common.exceptions import MsprobeException
from msprobe.pytorch.common.log import logger
from msprobe.pytorch.reproducibility.common import gpu_available, npu_available, check_arguments
from msprobe.pytorch.reproducibility.random_api_processor import GlobalRandomApiProcessor


def set_reproducibility(seed=1234, is_deterministic=False, is_enhanced=False):
    check_arguments(seed, is_deterministic, is_enhanced)

    # python package
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # numpy package
    np.random.seed(seed)

    # torch package
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(is_deterministic)

    # torch npu package
    if npu_available:
        torch.npu.manual_seed_all(seed)
        os.environ.update(
            {
                'HCCL_DETERMINISTIC': "True",
                'LCCL_DETERMINISTIC': "1",
                'CLOSE_MATMUL_K_SHIFT': "1",
                'ATB_LLM_LCOC_ENABLE': "0",
            }
        )

    # torch gpu package
    if gpu_available:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        cuda_version = torch.version.cuda
        if cuda_version and version.parse(cuda_version) >= version.parse("10.2"):
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

    # fix random state
    if is_enhanced:
        logger.info("The random state fixation function is enabled.")
        processor = GlobalRandomApiProcessor()
        processor.fix_random_state()


def random_save(output_path="./output"):
    logger.info("The random save function is enabled.")
    if not isinstance(output_path, str):
        logger.error(f"The output_path must be str, not {type(output_path)}")
        raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR, "the output_path must be str.")
    processor = GlobalRandomApiProcessor()
    processor.save_random_api(output_path)
