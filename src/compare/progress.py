#!/usr/bin/env python
# coding=utf-8
"""
Function:
Progress class. This class mainly involves the print_progress function.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2019-2021
"""
import time
import math

import log
from const_manager import ConstManager


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

    def print_progress(self: any, progress: int = None) -> None:
        """
        Print the progress
        :param progress: the progress
        """
        if progress is None:
            progress = round(self.current_count * 100.0 / self.total_count, 2) if self.total_count != 0 else 0
            if self.total_count == 0:
                log.print_error_log('Can not divide zero.')
        current_time = time.time()
        greater_than_count = math.floor(
            progress / (ConstManager.MAX_PROGRESS / self.PROGRESS_GREATER_THAN_COUNT)) \
            if ConstManager.MAX_PROGRESS != 0 and self.PROGRESS_GREATER_THAN_COUNT != 0 else 0
        progress_info = '%s%s' % (self.PROGRESS_GREATER_THAN * greater_than_count,
                                  ' ' * (self.PROGRESS_GREATER_THAN_COUNT - greater_than_count))
        if current_time - self.last_progress_time >= self.INTERVAL_TIME_SECOND \
                or progress == ConstManager.MAX_PROGRESS:
            log.print_info_log('[ %s %d%%]' % (progress_info, progress))
            self.last_progress_time = current_time
