#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
import shutil
import sys
import stat
import logging
import subprocess

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)

ALL_MODULES = {
    # <module_dir>: <module_output_dir>
    "src": "operator_cmp",
}


def clear_output(output_path):
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
        logging.info("Clean %s", output_path)


def prepare_ait_backend():
    cur_dir = os.path.realpath(os.path.dirname(__file__))
    prepare_shell = os.path.join(cur_dir, "prepare_ait_backend.sh")

    os.chmod(prepare_shell, stat.S_IRUSR | stat.S_IXGRP | stat.S_IXUSR | stat.S_IRGRP)
    cmd = [prepare_shell]
    logging.info("--------------------start compiling ait_backend--------------------")
    prepare_ait_backend = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    while prepare_ait_backend.poll() is None:
        line = prepare_ait_backend.stdout.readline()
        line = line.strip()
        if line:
            logging.info(line)
    return prepare_ait_backend.returncode


def ignore_function(dir, contents):
    if os.path.basename(dir) == "compare":
        return []
    elif os.path.basename(dir) == "load_balancing":
        return [f for f in contents if not (f.endswith(".so") and os.path.isfile(os.path.join(dir, f)))]
    else:
        return []


def main():
    build_dir = os.path.dirname(os.path.realpath(__file__))
    output_dir = os.path.join(build_dir, "output")
    code_src_dir = os.path.join(build_dir, "..")

    clear_output(output_dir)
    os.mkdir(output_dir)

    returncode = prepare_ait_backend()
    if returncode != 0:
        return returncode

    for mod, mod_out in ALL_MODULES.items():
        mod_dir = os.path.join(code_src_dir, mod)
        if not os.path.exists(mod_dir):
            logging.warning("%s does not exist", mod_dir)
            continue

        mod_output_path = os.path.join(output_dir, mod_out)
        logging.info("Copy from %s to %s", mod_dir, mod_output_path)
        shutil.copytree(mod_dir, mod_output_path, ignore=ignore_function)

    return 0


if __name__ == "__main__":
    sys.exit(main())
