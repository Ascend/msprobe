#!/usr/bin/env bash
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

CMC_URL_COMMON=https://cmc-szver-artifactory.cmc.tools.huawei.com/artifactory/cmc-software-release/Baize%20C/AscendTransformerBoost/1.0.0/asdops_dependency/common
CUR_PATH=$(dirname $(readlink -f $0))

function fn_build_nlohmann_json()
{
    if [ -d "${CUR_PATH}/nlohmannJson" ]; then
        return $?
    fi

    wget --no-check-certificate $CMC_URL_COMMON/nlohmannjson-v3.11.2.tar.gz
    tar -xf nlohmannjson-v3.11.2.tar.gz
    rm nlohmannjson-v3.11.2.tar.gz
    mv ./nlohmannJson ${CUR_PATH}
}

function fn_main()
{
    # 设置CMake构建目录
    build_dir="${CUR_PATH}/build"
    
    fn_build_nlohmann_json

    # 检查构建目录是否存在，如果不存在则创建
    if [ ! -d "$build_dir" ]; then
        mkdir "$build_dir"
        chmod 750 $build_dir
    fi

    # 进入CMake构建目录
    cd "$build_dir"

    # 调用CMake来构建项目
    cmake ..

    # 使用make来编译项目
    make -j20
}
fn_main "$@"