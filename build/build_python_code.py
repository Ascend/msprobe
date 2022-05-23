#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.

import os
import shutil
import sys
import logging
import subprocess

LOG_FORMAT = '%(asctime)s [%(levelname)s] %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)

ALL_MODULES = {
    # <module_dir>: <module_output_dir>
    'src': 'operator_cmp',
}


def clear_output(output_path):
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
        logging.info('Clean %s', output_path)


def generate_dump_data_api():
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    top_dir = os.path.abspath(os.path.dirname(cur_dir))
    dump_proto_dir = os.path.join(top_dir, 'resource/')
    dump_proto_path = os.path.join(dump_proto_dir, 'dump_data.proto')
    src_compare_path = os.path.join(top_dir, 'src/compare')
    protoc = os.path.join(top_dir, "opensource/cmake/protoc")

    cmd = [protoc, '-I=' + dump_proto_dir,
           '--python_out=' + src_compare_path, dump_proto_path]

    gen_api = subprocess.Popen(cmd, shell=False,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)

    while gen_api.poll():
        line = gen_api.stdout.readline()
        if line:
            logging.info("Failed to generate dump_data_pb2.py")
            break

    api_path = os.path.join(src_compare_path, 'dump_data_pb2.py')
    if os.path.exists(api_path):
        logging.info('dump_data_pb2.py is correctly generated to %s', src_compare_path)


def main():
    build_dir = os.path.dirname(os.path.realpath(__file__))
    output_dir = os.path.join(build_dir, 'output')
    code_src_dir = os.path.join(build_dir, '..')

    clear_output(output_dir)
    os.mkdir(output_dir)

    generate_dump_data_api()

    for mod, mod_out in ALL_MODULES.items():
        mod_dir = os.path.join(code_src_dir, mod)
        if not os.path.exists(mod_dir):
            logging.warning('%s does not exist', mod_dir)
            continue

        mod_output_path = os.path.join(output_dir, mod_out)
        logging.info('Copy from %s to %s', mod_dir, mod_output_path)
        shutil.copytree(mod_dir, mod_output_path)

    return 0


if __name__ == '__main__':
    main()
