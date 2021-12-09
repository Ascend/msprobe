#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.

import os
import shutil
import sys
import logging

LOG_FORMAT = '%(asctime)s [%(levelname)s] %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)

ALL_MODULES = {
    'ms_fmk_transplt': 'ms_fmk_transplt',
}


def clear_output(output_path):
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
        logging.info('Clean %s' % output_path)


def main():
    build_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(build_dir, 'output')
    code_src_dir = os.path.join(build_dir, '..', 'src')

    clear_output(output_dir)
    os.mkdir(output_dir)

    for mod, mod_out in ALL_MODULES.items():
        mod_dir = os.path.join(code_src_dir, mod)
        if not os.path.exists(mod_dir):
            logging.warning('%s does not exist' % mod_dir)
            continue

        mod_output_path = os.path.join(output_dir, mod_out)
        logging.info('Copy from %s to %s' % (mod_dir, mod_output_path))
        shutil.copytree(mod_dir, mod_output_path)

    return 0


if __name__ == '__main__':
    ret = main()
    if ret != 0:
        sys.exit(ret)
