#!/bin/bash

CUR_DIR=$(dirname $(readlink -f $0))
TOP_DIR=${CUR_DIR}/..
OPENSOURCE_DIR=${TOP_DIR}/"opensource"
PROTOBUF_DIR=${OPENSOURCE_DIR}/"protobuf"

# download from opensource center
download_thirdparty_code() {
  cd ${OPENSOURCE_DIR}
  if [ ! -e protobuf ]; then
    git clone ssh://git@codehub-dg-y.huawei.com:2222/OpenSourceCenter/protobuf.git protobuf -b v3.13.0
  fi
}

make_protobuf() {
  cd ${OPENSOURCE_DIR}
  if [ ! -e protobuf ]; then
    download_thirdparty_code
  fi
  cmake . -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_SKIP_RPATH=TRUE && make -j16
}

make_protobuf
