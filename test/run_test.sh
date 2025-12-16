#!/bin/bash
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

  local protoc_path=${top_dir}/opensource/protobuf/install/bin/protoc
  local make_proto_sh=${top_dir}/build/prepare_thirdparty_tool.sh
  if [ ! -e protoc_path ]; then
    bash $make_proto_sh
  fi

  local proto_path=${top_dir}/resource
  local output_path=${top_dir}/src/compare
  ${protoc_path} -I=${proto_path} --python_out=${output_path} ${proto_path}/dump_data.proto
  COMPILE_FLAG=1
}

run_ait() {
  cd ${TEST_DIR}/ut/ait_testcase
  chmod +x ./run_ait_cases.sh
  ./run_ait_cases.sh
}

run_st() {
  export PYTHONPATH=${SRC_DIR}/compare:${PYTHONPATH} && python3 run_st.py
}

run_ut() {
  export PYTHONPATH=${SRC_DIR}/compare:${PYTHONPATH} && python3 run_ut.py
  run_ait
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
