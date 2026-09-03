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

import os
import time
import sys
from functools import wraps
from msprobe.core.common.const import MsgConst

# 模块导入时的进程号。fork出的子进程会继承该值但不等于自身getpid()，
# 用于识别子进程（子进程中tqdm进度条实例是继承的过期副本，不应重绘）
_INIT_PID = os.getpid()


def filter_special_chars(func):
    @wraps(func)
    def func_level(self, msg, **kwargs):
        for char in MsgConst.SPECIAL_CHAR:
            msg = msg.replace(char, '_')
        return func(self, msg, **kwargs)

    return func_level


class BaseLogger:
    def __init__(self):
        self.rank = None
        self.level = self.get_level()

    @staticmethod
    def get_level():
        input_level = os.environ.get(MsgConst.MSPROBE_LOG_LEVEL)
        if input_level not in MsgConst.LOG_LEVEL_ENUM:
            return MsgConst.LogLevel.INFO.value
        else:
            return int(input_level)

    @filter_special_chars
    def raw(self, msg, **kwargs):
        """
        直接输出原始内容，不添加日志前缀
        用于输出子进程的原始日志，避免双份前缀
        """
        msg = msg.rstrip("_")
        self._output(msg)

    def get_rank(self):
        return self.rank

    @filter_special_chars
    def error(self, msg):
        if self.level <= MsgConst.LogLevel.ERROR.value:
            self._print_log(MsgConst.LOG_LEVEL[3], msg)

    @filter_special_chars
    def warning(self, msg):
        if self.level <= MsgConst.LogLevel.WARNING.value:
            self._print_log(MsgConst.LOG_LEVEL[2], msg)

    @filter_special_chars
    def info(self, msg):
        if self.level <= MsgConst.LogLevel.INFO.value:
            self._print_log(MsgConst.LOG_LEVEL[1], msg)

    @filter_special_chars
    def debug(self, msg):
        if self.level <= MsgConst.LogLevel.DEBUG.value:
            self._print_log(MsgConst.LOG_LEVEL[0], msg)

    def on_rank_0(self, func):
        def func_rank_0(*args, **kwargs):
            current_rank = self.get_rank()
            if current_rank is None or current_rank == 0:
                return func(*args, **kwargs)
            else:
                return None

        return func_rank_0

    def info_on_rank_0(self, msg):
        return self.on_rank_0(self.info)(msg)

    def error_on_rank_0(self, msg):
        return self.on_rank_0(self.error)(msg)

    def warning_on_rank_0(self, msg):
        return self.on_rank_0(self.warning)(msg)

    @filter_special_chars
    def info_on_rank_0_without_rank_prefix(self, msg):
        if self.level <= MsgConst.LogLevel.INFO.value:
            self.on_rank_0(lambda m: self._print_log(MsgConst.LOG_LEVEL[1], m, show_rank=False))(msg)

    @filter_special_chars
    def warning_on_rank_0_without_rank_prefix(self, msg):
        if self.level <= MsgConst.LogLevel.WARNING.value:
            self.on_rank_0(lambda m: self._print_log(MsgConst.LOG_LEVEL[2], m, show_rank=False))(msg)

    def error_log_with_exp(self, msg, exception):
        self.error(msg)
        raise exception

    def warning_log_with_exp(self, msg, exception):
        """
        打印警告日志并抛出指定异常
        """
        self.warning(msg)
        raise exception

    @staticmethod
    def _output(msg, end='\n'):
        """
        输出日志。存在活动的tqdm进度条时避免日志与进度条显示在同一行。
        - 本进程创建的进度条（主进程）：tqdm.write 先清行、打印日志、再重绘当前进度
        - fork子进程：继承的进度条实例进度值停留在fork时刻（通常为0），重绘会闪现旧进度，
          因此只清除进度条行、不重绘，由主进程下次update时重绘真实进度
        """
        try:
            from tqdm import tqdm

            instances = getattr(tqdm, '_instances', None)
            if instances:
                if os.getpid() == _INIT_PID:
                    tqdm.write(msg, end=end)
                    sys.stdout.flush()
                    return
                # fork子进程：仅清除继承的过期进度条行，不重绘
                for inst in list(instances):
                    try:
                        inst.clear(nolock=True)
                    except Exception:  # nosec B110
                        pass
        except Exception:  # nosec B110
            pass
        print(msg, end=end)
        sys.stdout.flush()

    def _print_log(self, level, msg, end='\n', show_rank=True):
        current_rank = self.get_rank()
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        pid = os.getpid()
        if show_rank and current_rank is not None:
            full_msg = f"{current_time} ({pid}) [rank {current_rank}] [{level}] {msg}"
        else:
            full_msg = f"{current_time} ({pid}) [{level}] {msg}"
        self._output(full_msg, end=end)


logger = BaseLogger()
