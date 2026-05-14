#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
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

_loader = None  # 缓存 load_pt 函数


def load_pt_file(pt_path: str, to_cpu=False, weights_only=True):
    """延迟 import load_pt 函数，避免循环依赖。"""
    global _loader

    if _loader is None:
        from msprobe.pytorch.common.utils import load_pt
        _loader = load_pt
    
    return _loader(pt_path, to_cpu=to_cpu, weights_only=weights_only)
