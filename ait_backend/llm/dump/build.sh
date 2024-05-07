#!/usr/bin/env bash
# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.
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