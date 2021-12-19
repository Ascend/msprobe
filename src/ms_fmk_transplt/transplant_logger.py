#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.

import logging as logger
import os

LOG_FORMAT = '%(asctime)s [%(levelname)s] %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
process_info = ''

logger.basicConfig(level=logger.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)


def init_logging_file(filename):
    file_path = os.path.split(filename)[0]
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    formatter = logger.Formatter(LOG_FORMAT, DATE_FORMAT)
    file_handler = logger.FileHandler(filename=filename, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.getLogger().addHandler(file_handler)


def set_process_info(process):
    global process_info
    process_info = process


def debug(msg):
    logger.debug("%-18s%s" % (process_info, msg) if process_info else msg)


def info(msg):
    logger.info("%-18s%s" % (process_info, msg) if process_info else msg)


def warning(msg):
    logger.warning("%-18s%s" % (process_info, msg) if process_info else msg)


def error(msg):
    logger.error("%-18s%s" % (process_info, msg) if process_info else msg)