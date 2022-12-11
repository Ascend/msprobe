#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

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
