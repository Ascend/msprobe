#!/bin/bash
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

