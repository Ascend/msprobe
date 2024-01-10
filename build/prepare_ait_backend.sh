#!/bin/bash

PROJECT_NAME="ait_backend"
CUR_DIR=$(dirname $(readlink -f $0))
AIT_DIR=${CUR_DIR}/"../"${PROJECT_NAME}

BUILD_TYPE=Debug
#BUILD_TYPE=Release

make_ait_backend() {
  cd ${AIT_DIR} && echo "Start building project \"${PROJECT_NAME}\""
  if [ ! -d "${BUILD_TYPE}" ]; then
      mkdir "${BUILD_TYPE}"
  fi
  cd ${BUILD_TYPE}
  cmake -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" .. && make
  if [ $? -ne 0 ]; then
      echo -e "Build ${PROJECT_NAME} Failed"
      return 1
  fi
  make install
}

make_ait_backend
