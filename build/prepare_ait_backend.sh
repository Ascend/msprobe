#!/bin/bash
CMC_URL_COMMON=https://cmc-szver-artifactory.cmc.tools.huawei.com/artifactory/cmc-software-release/Baize%20C/AscendTransformerBoost/1.0.0/asdops_dependency/common

PROJECT_NAME="ait_backend"
CUR_DIR=$(dirname $(readlink -f $0))
AIT_DIR=${CUR_DIR}/"../"${PROJECT_NAME}

BUILD_TYPE=Debug
#BUILD_TYPE=Release

function fn_build_nlohmann_json()
{
  cd ${CUR_DIR}/"../ait_backend/llm/dump/"
  if [ -d "nlohmannJson" ]; then
      return $?
  fi

  wget --no-check-certificate $CMC_URL_COMMON/nlohmannjson-v3.11.2.tar.gz
  tar -xf nlohmannjson-v3.11.2.tar.gz
  rm nlohmannjson-v3.11.2.tar.gz
}

function fn_clone_securec()
{
  cd ${CUR_DIR}/"../ait_backend/llm/dump/"
  if [ -d "${AIT_DIR}/dump/securec" ]; then
      return $?
  fi

  git clone https://codehub-dg-y.huawei.com/hwsecurec_group/huawei_secure_c.git securec -b tag_Huawei_Secure_C_V100R001C01SPC012B002_00001
}

make_ait_backend() {
  fn_build_nlohmann_json
  fn_clone_securec
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
# 编译AIT_LLM_ABI=0
export AIT_LLM_ABI=0
make_ait_backend

# 编译AIT_LLM_ABI=1
export AIT_LLM_ABI=1
make_ait_backend