#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
"""
Function:
This file mainly involves the reg const.
"""

import re

from const_manager import ConstManager


class RegManager:
    """
    The class for reg manager
    """
    NUMBER_PATTERN = r"^[0-9]+$"

    # mapping of built in algorithms to numbers, the indexes correspond one-to-one to the list above
    BUILTIN_ALGORITHM_INDEX_PATTERN = r"^([0-" + str(len(ConstManager.BUILT_IN_ALGORITHM) - 1) + "])$"

    # Standard
    STANDARD_DUMP_PATTERN = r"^([A-Za-z0-9_-]+\.[0-9]+)\.[0-9]{1,255}\.pb$"

    # Qunat
    QUANT_DUMP_PATTERN = r"^([A-Za-z0-9_-]+\.[0-9]+)\.[0-9]{1,255}\.quant$"

    # Offline
    OFFLINE_DUMP_PATTERN = r"^[A-Za-z0-9_-]+\.([A-Za-z0-9_-]+)\.[0-9]+" \
                           r"(\.[0-9]+)?\.[0-9]{1,255}(\.[0-9]+\.[0-9]+\.[0-9]+)?(\.[0-9]+)?"
    OFFLINE_NUMPY_PATTERN = r"^([A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[0-9]+" \
                            r"\.[0-9]{1,255})\.[0-9]+\.\b(npy|data|bin|txt)\b$"
    OFFLINE_FFTS_DUMP_PATTERN = r"^[A-Za-z0-9_-]+\.([A-Za-z0-9_-]+)\.[0-9]+" \
                                r"(\.[0-9]+)?\.[0-9]{1,255}\.[0-9]+\.[0-9]+\.[0-9]+"

    # Standard
    NUMPY_DUMP_PATTERN = r"^([A-Za-z0-9_-]+\.[0-9]+)\.[0-9]{1,255}\.npy$"
    STANDARD_NUMPY_PATTERN = r"^([A-Za-z0-9_-]+\.[0-9]+\.[0-9]{1,255})\.npy$"

    SUPPORT_SHAPE_PATTERN = r"^([0-9]+,)+[0-9]+$"

    FORMAT_CONVERT_FILE_NAME_PATTERN = r"^(convert_[A-Za-z0-9_]+_to_[A-Za-z0-9_]+)\.py[c]?$"

    SUPPORT_PATH_PATTERN = r"^[A-Za-z0-9_\./:()=\\-]+$"

    @staticmethod
    def match_pattern(pattern: str, value: any) -> bool:
        """
        The value match pattern or not
        :param pattern: the pattern
        :param value: the value to match
        :return bool
        """
        re_pattern = re.compile(pattern)
        match = re_pattern.match(value)
        return match is not None

    @staticmethod
    def match_group(pattern: str, value: any) -> (bool, any):
        """
        The value match pattern or not
        :param pattern: the pattern
        :param value: the value to match
        :return bool, match
        """
        re_pattern = re.compile(pattern)
        match = re_pattern.match(value)
        if match is not None:
            return True, match
        return False, match
