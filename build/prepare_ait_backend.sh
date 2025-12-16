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


CMC_URL_COMMON=https://cmc-szver-artifactory.cmc.tools.huawei.com/artifactory/cmc-software-release/Baize%20C/AscendTransformerBoost/1.0.0/asdops_dependency/common

PROJECT_NAME="ait_backend"
CUR_DIR=$(dirname $(readlink -f $0))
AIT_DIR=${CUR_DIR}/"../"${PROJECT_NAME}

BUILD_TYPE=Debug

function fn_clone_securec()
{
  PLATFORM_PATH=${CUR_DIR}/../platform
  if [ -d "${PLATFORM_PATH}/securec" ]; then
      return
  fi

  mkdir -p ${PLATFORM_PATH}
  cd ${PLATFORM_PATH}
  SECUREC_GIT_URL="https://codehub-dg-y.huawei.com/hwsecurec_group/huawei_secure_c.git"
  SECUREC_BRANCH="tag_Huawei_Secure_C_V100R001C01SPC012B002_00001"
  git clone $SECUREC_GIT_URL -b $SECUREC_BRANCH securec
  cd -
}

function fn_build_nlohmann_json()
{
  echo "Start building nolhmann_json"
  cd ${CUR_DIR}/"../ait_backend/llm/dump/"
  if [ -d "nlohmannJson" ]; then
      return $?
  fi

  wget --no-check-certificate $CMC_URL_COMMON/nlohmannjson-v3.11.2.tar.gz
  tar -xf nlohmannjson-v3.11.2.tar.gz
  rm nlohmannjson-v3.11.2.tar.gz
  echo "End building nolhmann_json"
}

make_ait_backend() {
  fn_build_nlohmann_json
  fn_clone_securec
  cd ${AIT_DIR} && echo "Start building project \"${PROJECT_NAME}\""
  if [ ! -d "${BUILD_TYPE}" ]; then
      mkdir "${BUILD_TYPE}"
  fi
  cd ${BUILD_TYPE}

  # Hi Test
  HI_TEST="off"
  if [ ! -z ${TOOLKIT_HITEST} ] && [ ${TOOLKIT_HITEST} == "on" ]; then
      HI_TEST=${TOOLKIT_HITEST}
  fi

  cmake -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" -DHITEST=${HI_TEST} .. && make
  if [ $? -ne 0 ]; then
      echo -e "Build ${PROJECT_NAME} Failed"
      return 1
  fi
  make install
}

make_ait_mindie_torch() {
    cd ${CUR_DIR}/"../ait_backend/llm/mindie_torch_dump/"
    rm -rf build
    mkdir build
    cd build
    cmake ..
    make -j
    cd -
}

make_load_balancing() {
    if [ -f /etc/profile ]; then    # 检查文件是否存在
        source /etc/profile         # 重新加载配置文件
    fi
    
    if [ -n "$CYTHON_PATH" ]; then
        export PYTHONPATH="$CYTHON_PATH:$PYTHONPATH"
    else
        echo "警告：CYTHON_PATH 未定义或为空，跳过路径添加。"
    fi

    cd "${CUR_DIR}/../src/load_balancing/"
    rm -f ./*.so ./*.c
    cp "./c2lb/computing_communication.py" "./c2lb.pyx"
    cp "./speculative_moe/speculative_moe.py" "./speculative_moe.pyx"
    cp "./c2lb_dynamic/c2lb_dynamic.py" "./c2lb_dynamic.pyx"
    cp "./c2lb_a3/c2lb_a3.py" "./c2lb_a3.pyx"

    python3 setup.py build_ext --inplace

    for module in c2lb speculative_moe c2lb_dynamic c2lb_a3; do
        # 匹配模式：<module>.cpython-<version>-<arch>.so
        so_file=$(find . -name "${module}.cpython-*.so" -print -quit)
        
        if [ -n "$so_file" ]; then
            mv "$so_file" "${module}.so"
            echo "Generated: $(pwd)/${module}.so"
        else
            echo "Error: Failed to build ${module}.so"
            exit 1
        fi
    done
    
    for module in c2lb speculative_moe c2lb_dynamic c2lb_a3; do
        pattern="${module}.cpython-*.so"
        if ls $pattern >/dev/null 2>&1; then
            echo "Deleting original files matching pattern: $pattern"
            rm -f $pattern
        else
            echo "No original files found for module: $module"
        fi
    done
    echo "===== 编译完成，恢复环境 ====="
    export PYTHONPATH="$OLD_PYTHONPATH"
    unset OLD_PYTHONPATH
}


# 编译AIT_LLM_ABI=0
export AIT_LLM_ABI=0
make_ait_backend
if [ $? -ne 0 ]; then
    exit 1
fi

# 编译AIT_LLM_ABI=1
export AIT_LLM_ABI=1
make_ait_backend

# 编译mindie-torch依赖
make_ait_mindie_torch

make_load_balancing
