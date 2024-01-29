#!/bin/bash

CUR_DIR=$(dirname $(readlink -f $0))
TOP_DIR=${CUR_DIR}/..
OPENSOURCE_DIR=${TOP_DIR}/"opensource"
PROTOBUF_DIR=${OPENSOURCE_DIR}/"protobuf"

make_protobuf() {
  cd ${OPENSOURCE_DIR}
  if [ -e protobuf ]; then
    cmake . -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_SKIP_RPATH=TRUE && make -j16
  else
    echo "Can't find "${PROTOBUF_DIR}
    return 1
  fi
}

make_protobuf
