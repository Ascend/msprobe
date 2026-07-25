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
