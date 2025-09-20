#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e
SCRIPT_DIR=$(cd $(dirname $0); pwd)
CURRENT_DIR=$(pwd)
cd $SCRIPT_DIR
cd ../../
TEST_DIR=$(pwd)

CACHE_DIR=$SCRIPT_DIR/cache
THIRD_PARTY_DIR=$SCRIPT_DIR/3rdparty

function fn_build_atb_probe()
{
    chmod +x $TEST_DIR/../build/prepare_ait_backend.sh
    $TEST_DIR/../build/prepare_ait_backend.sh
}

function fn_build_libsecurec_if_not_exists()
{
    SECUREC_SOURCE_PATH=$SCRIPT_DIR/../../../platform/securec  # Use source code built lib if CANN toolkit not installed
    if [ "$ASCEND_HOME_PATH" = "" ] && [ -d "$SECUREC_SOURCE_PATH" ]; then
        echo "Build libsecurec.so from source"
        if [ ! -e "$SECUREC_SOURCE_PATH/lib/libsecurec.so" ]; then
            cd $SECUREC_SOURCE_PATH/src && make && cd -
        fi

        echo "copy securec lib and include from $SECUREC_SOURCE_PATH to $THIRD_PARTY_DIR/securec"
        mkdir -p $THIRD_PARTY_DIR/securec
        cp $SECUREC_SOURCE_PATH/include $SECUREC_SOURCE_PATH/lib $THIRD_PARTY_DIR/securec -r
    fi
}

function fn_build_googletest()
{
    if [ -d "$THIRD_PARTY_DIR/googletest/lib" -a -d "$THIRD_PARTY_DIR/googletest/include" ]; then
        return $?
    fi
    if [ ! -d "$CACHE_DIR" ]; then
        mkdir -p $CACHE_DIR
    fi
    cd $CACHE_DIR
    if [[ ! -f "googletest-release-1.12.1.tar.gz" ]]; then
        wget --no-check-certificate https://cmc-hgh-artifactory.cmc.tools.huawei.com/artifactory/opensource_general/googletest/1.12.1/package/googletest-release-1.12.1.tar.gz
    fi
    tar -xf googletest-release-1.12.1.tar.gz
    cd googletest-release-1.12.1
    mkdir gtest_build
    cd gtest_build
    cmake -DCMAKE_INSTALL_PREFIX=$THIRD_PARTY_DIR/googletest ..
    make -j20
    make install
}

function fn_build_stub()
{
    if [[ -f "$THIRD_PARTY_DIR/googletest/include/gtest/stub.h" ]]; then
        return $?
    fi
    if [ ! -d "$CACHE_DIR" ]; then
        mkdir -p $CACHE_DIR
    fi
    cd $CACHE_DIR
    if [[ ! -f "master.tar.gz" ]]; then
        wget --no-check-certificate https://github.com/coolxv/cpp-stub/archive/refs/heads/master.tar.gz
    fi
    tar -zxvf master.tar.gz
    cp $CACHE_DIR/cpp-stub-master/src/stub.h $THIRD_PARTY_DIR/googletest/include/gtest
}

function fn_build_cases()
{
    if [ ! -d "$SCRIPT_DIR/build" ]; then
        mkdir -p $SCRIPT_DIR/build
    fi
    cd $SCRIPT_DIR/build
    cmake ..
    make -j20
}

function fn_run_and_record()
{
    cd $SCRIPT_DIR/build
    ./ait_backend_ut > ut.log 2>&1
}

function fn_execute_cases()
{
    local ret=1
    fn_run_and_record && ret=$?
    cat ut.log
    if [ "x"$ret == "x"0 ]; then
        exit 0
    else
        exit 1;
    fi
}

function fn_main()
{
    umask 022 # 确保所有编译完的目录权限不会超过755
    chmod 750 $SCRIPT_DIR
    # 1、构建最新的atbprobe
    fn_build_libsecurec_if_not_exists
    fn_build_atb_probe

    # 2、构建测试工程
    fn_build_googletest
    fn_build_cases

    # 3、执行用例
    fn_execute_cases
}
fn_main "$@"

