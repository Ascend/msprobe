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

"""
gpu and cpu not implement benchmark function, supplementary benchmarking function implementation
"""

import os
from pkgutil import iter_modules
from importlib import import_module

package_path = os.path.dirname(os.path.realpath(__file__))
for _, module_name, _ in iter_modules([package_path]):
    module = import_module(f"{__name__}.{module_name}")
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if callable(attr) and "npu_custom" not in attr_name and "template_" not in attr_name:
            globals()[attr_name] = attr
