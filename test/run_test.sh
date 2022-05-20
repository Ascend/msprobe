#!/bin/bash
# This script is used to run ut and st testcase.
# Copyright Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
TOP_DIR=$(dirname $(pwd))
TEST_DIR=${TOP_DIR}/"test"
SRC_DIR=${TOP_DIR}/"src"

clean() {
  cd ${TEST_DIR}
  if [ -e st_report.xml ]; then
    rm st_report.xml
    echo "remove last st_report success"
  fi

  if [ -e ut_report.xml ]; then
    rm -r ut_report.xml
    echo "remove last ut_report success"
  fi

  if [ -e compare ]; then
    rm -r compare
    echo "remove compare success"
  fi
}

copy_src() {
  local src_pkg="compare.tar.gz"
  cd ${SRC_DIR}
  tar -zcf src_pkg compare
  mv ${SRC_DIR}/src_pkg ${TEST_DIR}
  cd ${TEST_DIR}
  tar -zxf src_pkg && rm src_pkg
}

run_st() {
  export PYTHONPATH=PYTHONPATH:${TEST_DIR}/compare && python3 run_st.py
}

run_ut() {
  export PYTHONPATH=PYTHONPATH:${TEST_DIR}/compare && python3 run_ut.py
}

main() {
  clean
  copy_src

  if [[ $1 == "ut" ]] || [[ $1 == "st" ]]; then
    if [ $1 == "ut" ]; then
      echo "start run ut"
      run_ut
    elif [ $1 == "st" ]; then
      echo "start run st"
      run_st
    fi
  else
    echo "start run ut and st"
    run_ut
    run_st
  fi
}

main $@
