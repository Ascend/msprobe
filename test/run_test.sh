#!/bin/bash
# This script is used to run ut and st testcase.
# Copyright Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
CUR_DIR=$(dirname $(readlink -f $0))
TOP_DIR=${CUR_DIR}/..
TEST_DIR=${TOP_DIR}/"test"
SRC_DIR=${TOP_DIR}/"src"
COMPILE_FLAG=0

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

clean_dump_api() {
  local api_file=${SRC_DIR}/compare/dump_data_pb2.py
  if [ -e ${api_file} ] && [ "x"${COMPILE_FLAG} == "x"1 ]; then
    rm ${api_file}
  fi
}

gen_dump_api() {
  local api_file=${SRC_DIR}/compare/dump_data_pb2.py
  if [ -e ${api_file} ]; then
    return
  fi

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
  COMPILE_FLAG=1
}

run_st() {
  export PYTHONPATH=${SRC_DIR}/compare:${PYTHONPATH} && python3 run_st.py
}

run_ut() {
  export PYTHONPATH=${SRC_DIR}/compare:${PYTHONPATH} && python3 run_ut.py
}

main() {
  clean

  gen_dump_api

  local ret=1
  if [[ $1 == "ut" ]] || [[ $1 == "st" ]]; then
    [ $1 == "ut" ] && run_ut && ret=$?
    [ $1 == "st" ] && run_st && ret=$?
  else
    run_ut && ret=$?
    run_st && ret=$(($ret+$?))
  fi

  clean_dump_api

  if [ "x"$ret == "x"0 ]; then
    exit 0
  else
    exit 1;
  fi
}

main $@
