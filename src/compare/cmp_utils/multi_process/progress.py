# coding=utf-8
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
Function:
Progress class. This class mainly involves the print_progress function.
"""
import time
import math

from cmp_utils import log
from cmp_utils.constant.const_manager import ConstManager
from cmp_utils.constant.compare_error import CompareError


class Progress:
    """
    The class for progress
    """
    PROGRESS_GREATER_THAN_COUNT = 50
    PROGRESS_GREATER_THAN = '>'
    INTERVAL_TIME_SECOND = 1

    def __init__(self: any, total_count: int) -> None:
        self.total_count = total_count
        self.last_progress_time = 0
        self.current_count = 0

    def update_progress(self: any, update_count: int = 1) -> None:
        """
        Update the progress
        """
        self.current_count += update_count

    def is_done(self: any) -> bool:
        """
        check if the process is done
        """
        return self.current_count == self.total_count

    def update_and_print_progress(self: any, progress: int = None) -> None:
        """
        Print the progress realtime
        :param progress: the progress
        """
        if progress is None:
            if self.total_count != 0:
                progress = round(self.current_count * 100.0 / self.total_count, 2)
            else:
                progress = 0
                log.print_error_log('Can not divide zero.')
        current_time = time.time()
        denominator = 1
        greater_than_count = 0
        if self.PROGRESS_GREATER_THAN_COUNT != 0:
            denominator = ConstManager.MAX_PROGRESS // self.PROGRESS_GREATER_THAN_COUNT
        if denominator != 0:
            greater_than_count = math.floor(progress / denominator)
        progress_info = '%s%s' % (self.PROGRESS_GREATER_THAN * greater_than_count,
                                  ' ' * (self.PROGRESS_GREATER_THAN_COUNT - greater_than_count))
        if current_time - self.last_progress_time >= self.INTERVAL_TIME_SECOND \
                or progress == ConstManager.MAX_PROGRESS:
            log.print_info_log('[ %s %d%%]' % (progress_info, progress))
            self.last_progress_time = current_time
