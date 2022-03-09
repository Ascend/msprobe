#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.

import logging as logger
from logging.handlers import RotatingFileHandler
import os

LOG_FORMAT = '%(asctime)s [%(levelname)s] %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
progress_info = ''

logger.basicConfig(level=logger.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)


def init_logging_file(filename):
    file_path = os.path.split(filename)[0]
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    formatter = logger.Formatter(LOG_FORMAT, DATE_FORMAT)
    file_handler = RotatingFileHandler(filename=filename, encoding="utf-8", maxBytes=1024 ** 2, backupCount=10)
    file_handler.setFormatter(formatter)
    logger.getLogger().addHandler(file_handler)


def set_progress_info(progress):
    global progress_info
    progress_info = progress


def log_format(sep, msg):
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