#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

PATH_MAPPING_CONFIG = {
    'input': {
        # Add path mapping here for downloading data before training
        # format: <local path>: <obs/s3 path>
        # For example: '/data/dataset/imagenet': 'obs://dataset/imagenet',

    },
    'output': {
        # Add path mapping here for uploading output after training
        # format: <local path>: <obs/s3 path>
        # For example: './checkpoints': 'obs://outputs/',

    }
}
