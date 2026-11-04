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

import torch

from msprobe.pytorch.dump.debugger.precision_debugger import PrecisionDebugger, module_dump, module_dump_end  # noqa: F401
from msprobe.pytorch.reproducibility.random_reproducibility import random_save  # noqa: F401
from msprobe.pytorch.tag import tag  # noqa: F401
from .common.utils import seed_all  # noqa: F401
from .torchair_dump import set_fx_dump_config, set_ge_dump_config  # noqa: F401

torch_version_above_or_equal_2 = torch.__version__.split('+')[0] >= '2.0'
if torch_version_above_or_equal_2:
    from msprobe.pytorch.monitor.module_hook import TrainerMon  # noqa: F401

try:
    from msprobe.pytorch.aclgraph_dump import acl_save  # noqa: F401
    from msprobe.pytorch.aclgraph_dumper import AclGraphDumper  # noqa: F401
except Exception:
    acl_save = None
    AclGraphDumper = None
