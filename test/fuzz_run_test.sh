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

set -e
CUR_DIR=$(dirname $(readlink -f $0))
TOP_DIR=${CUR_DIR}/..
FUZZ_WORK_PATH=${TOP_DIR}/test/fuzz_test/build
FUZZ_RESULT_PATH=${TOP_DIR}/test/fuzz_test/result
LLM_FUZZ_REPORT_PATH=${TOP_DIR}/test/fuzz_test/report/msaccucmp
THIRD_PARTY_DIR=${TOP_DIR}/test/fuzz_test/3rdparty

 
declare -g FUZZ_RUN_TIMES=1000000
declare -g FUZZ_RUN_MODULE="all"
 
function fn_build_atb_probe()
{
    chmod +x $CUR_DIR/../build/prepare_ait_backend.sh
    $CUR_DIR/../build/prepare_ait_backend.sh
}

function fn_build_libsecurec_if_not_exists()
{
    SECUREC_SOURCE_PATH=$CUR_DIR/../platform/securec  # Use source code built lib if CANN toolkit not installed
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
    cd ${THIRD_PARTY_DIR}
    mkdir -p googletest && cd googletest
    if [[ ! -f "googletest-release-1.12.1.tar.gz" ]]; then
        wget --no-check-certificate https://cmc-hgh-artifactory.cmc.tools.huawei.com/artifactory/opensource_general/googletest/1.12.1/package/googletest-release-1.12.1.tar.gz
    fi
    tar -xf googletest-release-1.12.1.tar.gz
    cd googletest-release-1.12.1
    mkdir -p gtest_build
    cd gtest_build
    cmake -DCMAKE_INSTALL_PREFIX=$THIRD_PARTY_DIR/googletest ..
    make -j20
    make install
}

function download_thirdparty() {
    echo "*************[INFO] download thirdparty start..."

    mkdir -p ${THIRD_PARTY_DIR} 

    fn_build_atb_probe

    fn_build_libsecurec_if_not_exists

    fn_build_googletest

    mkdir -p ${THIRD_PARTY_DIR}/secodefuzz && cd ${THIRD_PARTY_DIR}/secodefuzz

    if [ ! -d "${THIRD_PARTY_DIR}/secodefuzz/SecTracy" ]; then
        git clone ssh://git@codehub-dg-y.huawei.com:2222/software-engineering-research-community/fuzz/SecTracy.git
    fi

    if [ ! -d "${THIRD_PARTY_DIR}/secodefuzz/secodefuzz" ]; then
        git clone https://szv-open.codehub.huawei.com/innersource/Fuzz/secodefuzz.git
    fi
    cd secodefuzz && bash build.sh gcc
    echo "*************[INFO] download thirdparty succ"
}
 
function check_gcc_version() {
    gcc_version=$(gcc -dumpversion)
    IFS=. read -r major minor patch <<< "$gcc_version"
    if [ $major -lt 7 ]; then
        echo "*************[INFO] gcc version is $gcc_version"
        echo "*************[INFO] please upgrade gcc version >= 8.0.0 before run fuzz testcase"
        exit 1
    else
        echo "*************[INFO] get gcc version: $gcc_version"
    fi
}
 
function build_fuzz_binary() {
    echo "*************[INFO] build fuzz binary start..."
    if [ -e "${TOP_DIR}/test/fuzz_test/build" ]; then
        rm -rf ${TOP_DIR}/test/fuzz_test/build
    fi
    mkdir -p ${TOP_DIR}/test/fuzz_test/build
    cd ${TOP_DIR}/test/fuzz_test/build
    cmake ../
    make -j$(nproc)
    if [ $? -ne 0 ]; then
        echo "*************[ERROR] build fuzz binary fail!!!"
        exit 1
    fi
    echo "*************[INFO] build fuzz binary succ"
}
 
function run_msaccucmp_fuzz_binary() {
    if [ -d ${LLM_FUZZ_REPORT_PATH} ]; then
        rm -rf ${LLM_FUZZ_REPORT_PATH}
    fi
    mkdir -p ${LLM_FUZZ_REPORT_PATH} && cd ${LLM_FUZZ_REPORT_PATH}
    mkdir -p log
    echo "*************[INFO] run llm fuzz binary start..."
    ${FUZZ_WORK_PATH}/msaccucmp_fuzz ${FUZZ_RUN_TIMES} > log/test_llm_${FUZZ_RUN_TIMES}.log 2>&1
    if [ $? != 0 ]; then
        echo "*************[ERROR] run llm fuzz binary fail!!!"
        exit 1
    fi
    echo "*************[INFO] run llm fuzz binary succ"
}
 
function run_fuzz_binary() {
    echo "*************[INFO] run fuzz binary start..."
    asan_path=$(find /usr/lib -name "libasan.so" | sort -r | head -n 1)
    ubsan_path=$(find /usr/lib -name "libubsan.so" | sort -r | head -n 1)
    export LD_PRELOAD=${asan_path}:${ubsan_path}
    if [ -e ${FUZZ_WORK_PATH} ]; then
        run_msaccucmp_fuzz_binary
    else
        echo "*************[ERROR] cannot find ${FUZZ_WORK_PATH}"
    fi
    echo "*************[INFO] run fuzz binary succ"
}
 
function package_result_retult() {
    echo "*************[INFO] package fuzz result start..."
    if [ -d ${FUZZ_RESULT_PATH} ]; then
        rm -rf ${FUZZ_RESULT_PATH}
    fi
    mkdir -p ${FUZZ_RESULT_PATH}
    tar -zcvf ${FUZZ_RESULT_PATH}/fuzz.tar.gz ${TOP_DIR}/test/fuzz_test/report
    echo "*************[INFO] package fuzz result succ"
}
 
while getopts "t:h" opt; do
    case ${opt} in
        t )
            if [[ $OPTARG =~ ^[0-9]+$ ]]; then
                FUZZ_RUN_TIMES=$OPTARG
            else
                echo "*************[WARNING] invalid fuzz execution times, please input valid number"
                exit 1
            fi
            ;;
        h )
            echo "[INFO] Description for msprof fuzz testcase shell..."
            echo "[INFO] Options:"
            echo "[INFO] -t------------Fuzz exection time, default time: 1000000"
            echo "[INFO] -h------------Fuzz exection shell help"
            exit 0
            ;;
        \? )
            echo "Invalid option: -$OPTARG" 1>&2
            exit 1
            ;;
        : )
            echo "Option -$OPTARG requires an argument." 1>&2
            exit 1
            ;;
    esac
done
 
echo "*************[INFO] run fuzz testcase for $FUZZ_RUN_TIMES..."
 
check_gcc_version
 
download_thirdparty
 
build_fuzz_binary
 
run_fuzz_binary
 
package_result_retult
 
echo "*************[INFO] run fuzz testcase succ"
echo "*************[INFO] you can see fuzz result in ${FUZZ_RESULT_PATH}"