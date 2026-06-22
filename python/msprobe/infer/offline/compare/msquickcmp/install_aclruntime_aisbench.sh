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

NO_CHECK_CERTIFICATE="$1"

declare -i ret_ok=0
declare -i ret_run_failed=1

# 读取 config.ini 中的 URL 配置
# 获取当前脚本的目录
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
# 构建config.ini的路径
CONFIG_FILE="$SCRIPT_DIR/config.ini"
WHL_BASE_URL=$(grep 'whl_base_url' "$CONFIG_FILE" | sed 's/whl_base_url=//')
ACLRUNTIME_SHA_BASE_URL=$(grep 'aclruntime_sha_base_url' "$CONFIG_FILE" | sed 's/aclruntime_sha_base_url=//')
AIS_BENCH_SHA_BASE_URL=$(grep 'ais_bench_sha_base_url' "$CONFIG_FILE" | sed 's/ais_bench_sha_base_url=//')
AIS_BENCH_WHL_DOWNLOAD_URL=$(grep 'ais_bench_whl_download_url' "$CONFIG_FILE" | sed 's/ais_bench_whl_download_url=//')


download_and_install_aclruntime() {
    ACLRUNTIME_VERSION=`pip3 show aclruntime | awk '/Version: /{print $2}'`

    if [ "$arg_force_reinstall" = "--force-reinstall" ]; then
        echo "Force reinstall aclruntime"
    elif [ "$ACLRUNTIME_VERSION" = "0.0.2" ]; then
        echo "aclruntime==0.0.2 already installed, skip"
        return
    fi

    echo "download and install aclruntime"
    PYTHON3_MINI_VERSION=`python3 --version | cut -d'.' -f 2`
    if [ "$PYTHON3_MINI_VERSION" = "7" ]; then
        SUB_SUFFIX="m"
    else
        SUB_SUFFIX=""
    fi
    echo "PYTHON3_MINI_VERSION=$PYTHON3_MINI_VERSION, SUB_SUFFIX=$SUB_SUFFIX"
    PLATFORM=$(uname -m)

    WHL_NAME="aclruntime-0.0.2-cp3${PYTHON3_MINI_VERSION}-cp3${PYTHON3_MINI_VERSION}${SUB_SUFFIX}-linux_${PLATFORM}.whl"
    echo "WHL_NAME=$WHL_NAME, URL=${WHL_BASE_URL}${WHL_NAME}"
    if [ "$NO_CHECK_CERTIFICATE" == "True" ]; then
        echo "[WARNING] --no-check will skip checking the certificate of the target website, posing security risk."
        wget --no-check-certificate -c "${WHL_BASE_URL}${WHL_NAME}"
    else
        wget -c "${WHL_BASE_URL}${WHL_NAME}"
    fi

    if [ $PYTHON3_MINI_VERSION -gt 12 ]; then
        echo "Unsupported python3 version"
        exit 1
    fi

    SHA_NAME="${WHL_NAME%.whl}.sha256"
    SHA_URL=${ACLRUNTIME_SHA_BASE_URL}${SHA_NAME}
    echo "Downloading aclruntime SHA from: ${SHA_URL}"
    if [ "$NO_CHECK_CERTIFICATE" == "True" ]; then
        sha256Value=$(curl -kfsSL "${SHA_URL}" | awk '{print $1}')
    else
        sha256Value=$(curl -fsSL "${SHA_URL}" | awk '{print $1}')
    fi


    sha256Data=$(sha256sum "$WHL_NAME" | cut -d' ' -f1)
    if [[ "${sha256Data}" != "${sha256Value}" ]]; then
        echo "Failed to verify sha256: $WHL_NAME"
        exit 1
    else
        echo "sha256 verification passed: $WHL_NAME"
    fi

    MAX_RETRIES=3  # 最大重试次数
    retry_count=0  # 当前重试计数

    while [ $retry_count -lt $MAX_RETRIES ]; do
        pip3 install $WHL_NAME $arg_force_reinstall && rm -f $WHL_NAME
        if [ $? -eq 0 ]; then
            echo "[INFO] Installation succeeded."
            break  # 成功安装，退出循环
        else
            retry_count=$((retry_count + 1))
            echo "[ERROR] Downloading or installing from whl failed. Attempt $retry_count of $MAX_RETRIES."
            echo "Please go to '$AIS_BENCH_WHL_DOWNLOAD_URL', download the whl file and install it manually if the problem persists."

            # Optional: 等待一段时间再重试
            sleep 1  # 等待 1 秒再重试
        fi
    done

    if [ $retry_count -eq $MAX_RETRIES ]; then
        echo "[ERROR] Reached maximum retry attempts. Exiting."
        exit 1  # 达到最大重试次数，退出
    fi
}

download_and_install_ais_bench() {
    AIS_BENCH_VERSION=`pip3 show ais_bench | awk '/Version: /{print $2}'`

    if [ "$arg_force_reinstall" = "--force-reinstall" ]; then
        echo "Force reinstall ais_bench"
    elif [ "$AIS_BENCH_VERSION" = "0.0.2" ]; then
        echo "ais_bench==0.0.2 already installed, skip"
        return
    fi

    WHL_NAME="ais_bench-0.0.2-py3-none-any.whl"
    echo "WHL_NAME=$WHL_NAME, URL=${WHL_BASE_URL}${WHL_NAME}"
    if [ "$NO_CHECK_CERTIFICATE" == "True" ]; then
        echo "[WARNING] --no-check will skip checking the certificate of the target website, posing security risk."
        wget --no-check-certificate -c "${WHL_BASE_URL}${WHL_NAME}"
    else
        wget -c "${WHL_BASE_URL}${WHL_NAME}"
    fi

    SHA_NAME="${WHL_NAME%.whl}.sha256"
    SHA_URL=${AIS_BENCH_SHA_BASE_URL}${SHA_NAME}
    echo "Downloading ais_bench SHA from: ${SHA_URL}"
    if [ "$NO_CHECK_CERTIFICATE" == "True" ]; then
        sha256Value=$(curl -kfsSL "${SHA_URL}" | awk '{print $1}')
    else
        sha256Value=$(curl -fsSL "${SHA_URL}" | awk '{print $1}')
    fi

    sha256Data=$(sha256sum "$WHL_NAME" | cut -d' ' -f1)
    if [[ "${sha256Data}" != "${sha256Value}" ]]; then
        echo "Failed to verify sha256: $WHL_NAME"
        exit 1
    else
        echo "sha256 verification passed: $WHL_NAME"
    fi

    MAX_RETRIES=3  # 最大重试次数
    retry_count=0  # 当前重试计数

    while [ $retry_count -lt $MAX_RETRIES ]; do
        pip3 install $WHL_NAME $arg_force_reinstall && rm -f $WHL_NAME
        if [ $? -eq 0 ]; then
            echo "[INFO] Installation succeeded."
            break  # 成功安装，退出循环
        else
            retry_count=$((retry_count + 1))
            echo "[ERROR] Downloading or installing from whl failed. Attempt $retry_count of $MAX_RETRIES."
            echo "Please go to '$AIS_BENCH_WHL_DOWNLOAD_URL', download the whl file and install it manually if the problem persists."

            # Optional: 等待一段时间再重试
            sleep 1  # 等待 1 秒再重试
        fi
    done

    if [ $retry_count -eq $MAX_RETRIES ]; then
        echo "[ERROR] Reached maximum retry attempts. Exiting."
        exit 1  # 达到最大重试次数，退出
    fi
}

ret=0
download_and_install_aclruntime
ret=$(( $ret + $? ))

download_and_install_ais_bench
ret=$(( $ret + $? ))

exit $ret
