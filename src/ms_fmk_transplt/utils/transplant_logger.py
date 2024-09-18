#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.

import logging as logger
from logging.handlers import RotatingFileHandler
import os
import re
from utils import trans_utils as utils

LOG_FORMAT = '%(asctime)s [%(levelname)s] %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
BACKUP_COUNT = 10
MAX_BYTES = 1024 ** 2
progress_info = ''
pattern_nblank = re.compile('[\r\n\f\v\t\b\u007F]')
pattern_blank = re.compile(' {2,}')

logger.basicConfig(level=logger.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)


class RotatingFileHandlerWithPermission(RotatingFileHandler):
    def doRollover(self):
        os.chmod(self.baseFilename, 0o440)
        super().doRollover()
        os.chmod(self.baseFilename, 0o640)


def init_logging_file(filename):
    file_path = os.path.split(filename)[0]
    if not os.path.exists(file_path):
        utils.make_dir_safety(file_path)

    formatter = logger.Formatter(LOG_FORMAT, DATE_FORMAT)
    file_handler = RotatingFileHandlerWithPermission(filename=filename, encoding="utf-8", maxBytes=MAX_BYTES,
                                                     backupCount=BACKUP_COUNT)
    file_handler.setFormatter(formatter)
    logger.getLogger().addHandler(file_handler)
    os.chmod(filename, 0o640)


def set_progress_info(progress):
    global progress_info
    progress_info = progress


def log_format(sep, msg):
    msg = pattern_nblank.sub('', str(msg))
    msg = pattern_blank.sub(' ', str(msg))
    if progress_info:
        return ' ' * sep + f'{progress_info:20s}' + str(msg)
    else:
        return ' ' * sep + str(msg)


def debug(msg):
    logger.debug(log_format(2, msg))


def info(msg):
    logger.info(log_format(3, msg))


def warning(msg):
    logger.warning(log_format(0, msg))


def error(msg):
    logger.error(log_format(2, msg))


def info_without_format(msg):
    logger.info(str(msg))
