
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
"""
Function:
This file mainly involves the reg const.
"""

import re

from cmp_utils.constant.const_manager import ConstManager


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
    NUMPY_DUMP_PATTERN = r"^([\.A-Za-z0-9_-]+\.[0-9]+)\.[0-9]{1,255}\.npy$"

    STANDARD_NUMPY_PATTERN = r"^([\.A-Za-z0-9_-]+\.[0-9]+\.[0-9]{1,255})\.npy$"

    SUPPORT_SHAPE_PATTERN = r"^([0-9]+,)+[0-9]+$"

    # matching `convert_xxx_to_yyy.py`, the `{1,128}` is used for limiting string length, bypassing ReDoS attack.
    FORMAT_CONVERT_FILE_NAME_PATTERN = r"^(convert_[A-Za-z0-9_]{1,128}_to_[A-Za-z0-9_]{1,128})\.py[c]?$"

    SUPPORT_PATH_PATTERN = r"^[A-Za-z0-9_\./:()=\\-]+$"

    FFTS_MANUAL_FIELD_PATTERN = r"lxslice[0-9]+"

    LXSLICE_PATTERN = r"_lxslice[0-9]+"

    SGT_FLIED_PATTERN = r"sgt_graph_[0-9]+"

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

    @staticmethod
    def get_matchs(pattern: str, value: any) -> list:
        re_pattern = re.compile(pattern)
        return [match for match in re.finditer(re_pattern, value)]
