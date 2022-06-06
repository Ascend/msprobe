#!/bin/bash
# This script is used to run ut and st testcase.
# Copyright Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
CUR_DIR=$(dirname $(readlink -f $0))
TOP_DIR=${CUR_DIR}/..
TEST_DIR=${TOP_DIR}/"test"
SRC_DIR=${TOP_DIR}/"src"

clean() {
  cd ${TEST_DIR}
  if [ -e ${TEST_DIR}/st_report.xml ]; then
    rm st_report.xml
    echo "remove last st_report success"
  fi

  if [ -e ${TEST_DIR}/report ]; then
    rm -r ${TEST_DIR}/report
    echo "remove last ut_report success"
  fi
}

gen_dump_api() {
  cd ${CUR_DIR}
  local top_dir=$(dirname $(pwd))

  local protoc_path=${top_dir}/opensource/protobuf/cmake/protoc
  local make_proto_sh=${top_dir}/build/prepare_thirdparty_tool.sh
  if [ ! -e protoc_path ]; then
    bash $make_proto_sh
  fi

  local proto_path=${top_dir}/resource
  local output_path=${top_dir}/src/compare
  ${protoc_path} -I=${proto_path} --python_out=${output_path} ${proto_path}/dump_data.proto
}

run_st() {
  export PYTHONPATH=${SRC_DIR}/compare:PYTHONPATH && python3 run_st.py
}

run_ut() {
  export PYTHONPATH=${SRC_DIR}/compare:PYTHONPATH && python3 run_ut.py
}

main() {
  clean

  if [[ $1 == "ut" ]] || [[ $1 == "st" ]]; then
    [ $1 == "ut" ] && run_ut
    [ $1 == "st" ] && run_st
  else
    run_ut && run_st
  fi
}

main $@
