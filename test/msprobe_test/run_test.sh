# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

#!/bin/bash
CUR_DIR=$(dirname $(readlink -f $0))
TOP_DIR=${CUR_DIR}/..
TEST_DIR=${TOP_DIR}/"msprobe_test"
SRC_DIR=${TOP_DIR}/../python/

install_deps() {
    # Install the project and its dependencies (numpy, pandas, h5py, protobuf, etc.)
    PROJECT_ROOT=${TOP_DIR}/..
    echo "Installing project dependencies from ${PROJECT_ROOT}..."
    pip install "${PROJECT_ROOT}"

    if ! pip show pytest &> /dev/null; then
        echo "pytest not found, trying to install..."
        pip install pytest
    fi

    if ! pip show pytest-cov &> /dev/null; then
        echo "pytest-cov not found, trying to install..."
        pip install pytest-cov
    fi
}

run_ut() {
    install_deps

    python3 run_ut.py
}

main() {
    cd ${TEST_DIR} && run_ut
}

main $@
