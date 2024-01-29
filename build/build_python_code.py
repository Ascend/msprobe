#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.

import os
import shutil
import sys
import stat
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


def prepare_third_party_tool():
    cur_dir = os.path.realpath(os.path.dirname(__file__))
    prepare_shell = os.path.join(cur_dir, "prepare_thirdparty_tool.sh")

    os.chmod(prepare_shell, stat.S_IRUSR | stat.S_IXGRP | stat.S_IXUSR | stat.S_IRGRP)
    cmd = [prepare_shell]
    logging.info("--------------------start compile protobuf"
                 + "--------------------")
    prepare_protoc = subprocess.Popen(cmd, shell=False,
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT)

    while prepare_protoc.poll() is None:
        line = prepare_protoc.stdout.readline()
        line = line.strip()
        if line:
            logging.info(line)

    top_dir = os.path.join(os.path.dirname(cur_dir))
    protoc_dir = os.path.join(top_dir, "opensource/protobuf/cmake/protoc")
    if os.path.exists(protoc_dir):
        result = "Compile protobuf success."
    else:
        result = "Compile protobuf failed."
    logging.info("--------------------" + result + "--------------------")


def prepare_ait_backend():
    cur_dir = os.path.realpath(os.path.dirname(__file__))
    prepare_shell = os.path.join(cur_dir, "prepare_ait_backend.sh")

    os.chmod(prepare_shell, stat.S_IRUSR | stat.S_IXGRP | stat.S_IXUSR | stat.S_IRGRP)
    cmd = [prepare_shell]
    logging.info("--------------------start compile ait_backend"
                 + "--------------------")
    prepare_protoc = subprocess.Popen(cmd, shell=False,
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT)

    while prepare_protoc.poll() is None:
        line = prepare_protoc.stdout.readline()
        line = line.strip()
        if line:
            logging.info(line)


def generate_dump_data_api():
    cur_dir = os.path.realpath(os.path.dirname(__file__))
    top_dir = os.path.realpath(os.path.dirname(cur_dir))
    dump_proto_dir = os.path.join(top_dir, 'resource/')
    dump_proto_path = os.path.join(dump_proto_dir, 'dump_data.proto')
    src_compare_path = os.path.join(top_dir, 'src/compare')
    protoc_dir = os.path.join(top_dir, "opensource/protobuf/cmake/protoc")

    if not os.path.exists(protoc_dir):
        logging.info(protoc_dir, "is not exist.")

    cmd = [protoc_dir, '-I=' + dump_proto_dir,
           '--python_out=' + src_compare_path, dump_proto_path]

    gen_api = subprocess.Popen(cmd, shell=False,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)

    while gen_api.poll() is None:
        line = gen_api.stdout.readline()
        line = line.strip()
        if line:
            break

    api_path = os.path.join(src_compare_path, 'dump_data_pb2.py')
    if os.path.exists(api_path):
        logging.info('dump_data_pb2.py is correctly generated to %s',
                     src_compare_path)
    else:
        logging.error("Failed to generate 'dump_data_pb2.py'.")
        sys.exit(1)


def main():
    build_dir = os.path.dirname(os.path.realpath(__file__))
    output_dir = os.path.join(build_dir, 'output')
    code_src_dir = os.path.join(build_dir, '..')

    clear_output(output_dir)
    os.mkdir(output_dir)

    prepare_third_party_tool()
    prepare_ait_backend()
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
