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

import numpy as np

from msprobe.core.common.log import logger


def in_different_shape(a, b):
    if a.shape != b.shape:
        logger.warning(f"a, b are in different shape. a: {a.shape}, b: {b.shape}")
        return True
    return False


def l2_distance(a, b):
    if a is None or b is None:
        return None
    if in_different_shape(a, b):
        return None
    return np.linalg.norm(a - b).item()


def cos_sim(a, b):
    if a is None or b is None:
        return None

    if in_different_shape(a, b):
        return None
    if a.ndim > 0:
        a = a.flatten().squeeze()
        b = b.flatten().squeeze()

    num = a.dot(b)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if a_norm == 0 and b_norm == 0:
        return 1.0
    if a_norm == 0 or b_norm == 0:
        logger.warning('One tensor norm is zero.')
        return None

    sim = num / (a_norm * b_norm)

    return sim.item()


def numel(a, b):
    n1 = a.size
    n2 = b.size
    if n1 != n2:
        logger.warning('parameters have different number of element')
        return (n1, n2)
    return n1


def shape(a, b):
    if in_different_shape(a, b):
        return [list(a.shape), list(b.shape)]
    return list(a.shape)


METRIC_FUNC = {'l2': l2_distance, 'cos': cos_sim, 'numel': numel, 'shape': shape}
