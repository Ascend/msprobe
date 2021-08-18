#!/bin/bash
# Copyright Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.

set -e

BUILD_DIR=$(dirname $(readlink -f $0))
INPUT_PATH=$1
TOOLS_DST_DTR=${INPUT_PATH}/ascend-ide-backend

log_info(){
    echo "[$(date +%Y-%m-%d-%H:%M:%S)] [INFO] $1"
}

log_error(){
    echo "[$(date +%Y-%m-%d-%H:%M:%S)] [ERROR] $1"
}

log_warning(){
    echo "[$(date +%Y-%m-%d-%H:%M:%S)] [WARNING] $1"
}

merge_toolkit(){
    mode=$(basename ${INPUT_PATH})
    origin_xml=${INPUT_PATH}/build/release/config/module/ascend/OptTool.xml

    if [ "$mode" = "infer" ] || [ "$mode" = "train" ] || [ "$mode" = "florence" ]; then
        base_origin_xml=${BUILD_DIR}/toolkit_xml/toolkit_${mode}_origin.xml
        cp ${origin_xml} ${base_origin_xml}

        modified_xml=${BUILD_DIR}/toolkit_xml/toolkit_${mode}_modified.xml
        python3 ${BUILD_DIR}/merge_xml.py ${base_origin_xml} ${modified_xml}
        if [ $? != 0 ]; then
            log_error "Generate modified toolkit.xml failed."
            exit 1
        fi

        cp ${modified_xml} ${origin_xml}
        if [ $? != 0 ]; then
            log_error "Copy toolkit.xml failed."
            exit 1
        fi

        mkdir -p ${TOOLS_DST_DTR}
        for dir in "ms_fmk_transplt";
        do
        cp -rf ${BUILD_DIR}/output/${dir} ${TOOLS_DST_DTR}
        if [ $? != 0 ]; then
            log_error "Copy file failed."
            exit 1
        fi
        done
    else
        log_error "mode [${mode}] is not found."
        exit 1
    fi
}

main(){
    if [ $# -ne 1 ]; then
        log_error "Input parameter error"
        exit 1
    fi

    log_info "Prepare python files"
    python3 ${BUILD_DIR}/build_python_code.py
    if [ $? -ne 0 ]; then
        log_error "Prepare python files failed"
        exit 1
    fi
    log_info "Prepare success"

    log_info "Start to merge xml"
    merge_toolkit
    log_info "Merge success"
}

main $@


