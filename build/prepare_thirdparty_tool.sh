#!/bin/bash

CUR_DIR=$(dirname $(readlink -f $0))
TOP_DIR=${CUR_DIR}/..
OPENSOURCE_DIR="${TOP_DIR}/opensource"
PROTOBUF_DIR="${OPENSOURCE_DIR}/protobuf"

PROTOBUF_VERSION="25.1"
PROTOBUF_TAR="protobuf-all-${PROTOBUF_VERSION}.tar.gz"
PROTOBUF_SRC_DIR="protobuf-${PROTOBUF_VERSION}"
ABSEIL_DIR="${OPENSOURCE_DIR}/abseil-cpp"

function compile_protobuf() {
    mkdir -p "$OPENSOURCE_DIR"
    if [ -d "$ABSEIL_DIR" ]; then
        cd "$ABSEIL_DIR"
        if  [ -f "abseil-cpp-20230802.1.tar.gz" ]; then
            tar -xzvf abseil-cpp-20230802.1.tar.gz
            SRC_DIR=$(tar -tf "abseil-cpp-20230802.1.tar.gz" | head -n1 | cut -d/ -f1)
            for patch_file in *.patch; do
                if [ -f "$patch_file" ]; then
                    echo "$patch_file"
                    patch -d "$SRC_DIR" -p1 --forward < "$patch_file"
                fi
            done
            rsync -a "$SRC_DIR/" .       
            rm -rf "$SRC_DIR"            
            rm -f *.tar.gz *.patch   
        fi
    else
        echo "abseil-cpp does not exist."
    fi

    cd "$PROTOBUF_DIR"
    if [ ! -d "${PROTOBUF_SRC_DIR}" ]; then

        tar -zxvf "${PROTOBUF_TAR}"
    fi

    cd "$PROTOBUF_SRC_DIR"

    rsync -a --delete "$ABSEIL_DIR/" "third_party/abseil-cpp/"

    ulimit -n 8192

    patch -p1 --forward < ../backport-0001-add-secure-compile-option.patch
    patch -p1 --forward < ../backport-0002-Fix-CC-compiler-support.patch
    patch -p1 --forward < ../backport-0003-protobuf-add-coverage-compile-option.patch
    patch -p1 --forward < ../backport-0004-fix-CVE-2025-4565-1.patch
    patch -p1 --forward < ../backport-0005-fix-CVE-2025-4565-2.patch
    patch -p1 --forward < ../huawei-0001-add-secure-compile-fs-in-CMakeLists.patch
    patch -p1 --forward < ../huawei-0002-add-secure-compile-Stack-Protect-in-CMakeLists.patch

    mkdir -p build
    cd build

    cmake .. \
      -DCMAKE_BUILD_TYPE=Release \
      -Dprotobuf_BUILD_TESTS=OFF \
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DCMAKE_INSTALL_PREFIX=${PROTOBUF_DIR}/install && \

    make -j16 && \
    make install && \

    echo "[INFO] Protobuf v${PROTOBUF_VERSION} install successfully, path: ${PROTOBUF_DIR}/install"
}

compile_protobuf