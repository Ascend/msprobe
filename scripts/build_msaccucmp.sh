#!/bin/bash
# get real path of parents' dir of this file
# source ./download_thirdparty.sh
CUR_DIR=$(dirname $(readlink -f $0))
TOP_DIR=${CUR_DIR}/..


OPENSOURCE_DIR=${TOP_DIR}/opensource

THIRDPARTY_LIST="${OPENSOURCE_DIR}/makeself"

if [ -n "$1" ]; then
    if [ "$1" == "force" ]; then
        echo "force delete origin opensource files"
        echo ${THIRDPARTY_LIST}
        rm -rf ${THIRDPARTY_LIST}
    fi
fi

function patch_makeself() {
    cd ${OPENSOURCE_DIR}
    git clone -b v2.5.0.x https://gitcode.com/cann-src-third-party/makeself.git
    cd ${OPENSOURCE_DIR}/makeself
    tar -zxf makeself-release-2.5.0.tar.gz
    cd makeself-release-2.5.0
    ulimit -n 8192
    patch -p1 < ../makeself-2.5.0.patch
    cd ${OPENSOURCE_DIR}/makeself
    cp -r makeself-release-2.5.0 ${OPENSOURCE_DIR}
    cd ${OPENSOURCE_DIR}
    rm -rf makeself
    mv makeself-release-2.5.0 makeself
}

mkdir -p ${OPENSOURCE_DIR} && cd ${OPENSOURCE_DIR}

[ ! -d "makeself" ] && patch_makeself


# store product
CMP_TEMP_DIR=${TOP_DIR}/build/msaccucmp
rm -rf ${CMP_TEMP_DIR}
mkdir -p ${CMP_TEMP_DIR}

# makeself is tool for compiling run package
MAKESELF_DIR=${TOP_DIR}/opensource/makeself

# footnote for creating run package
CREATE_RUN_SCRIPT=${MAKESELF_DIR}/makeself.sh

# footnote for controling params
CONTROL_PARAM_SCRIPT=${MAKESELF_DIR}/makeself-header.sh

# store run package
OUTPUT_DIR=${TOP_DIR}/output
mkdir -p "${OUTPUT_DIR}"
RUN_SCRIPT_DIR=${TOP_DIR}/scripts/run_script
CONF_DIR=${TOP_DIR}/scripts/conf
FILTER_PARAM_SCRIPT=${CONF_DIR}/help.info
MAIN_SCRIPT=main_msaccucmp.sh
COMMON_SCRIPT=common.sh
INSTALL_SCRIPT=install.sh
UTILS_SCRIPT=utils.sh
UNINSTALL_SCRIPT=uninstall.sh
VERSION_INFO=version.info

CMP_RUN_NAME="mindstudio-accucmp"

PKG_LIMIT_SIZE=524288000 # 500M

function parse_script_args() {
    if [ $# -gt 2 ]; then
        echo "[ERROR] Too many arguments. Only one or two arguments are allowed."
        exit 1
    elif [ $# -eq 2 ]; then
        VERSION="$1"
        BUILD_MODE="$2"
    elif [ $# -eq 1 ]; then
        VERSION="$1"
    fi
}

# create temp dir for product
function create_temp_dir() {
    local temp_dir=${1}
    local cmp_dir=${TOP_DIR}/python/msprobe/msaccucmp

    cp -r ${cmp_dir} ${temp_dir}

    # run install scripts
    copy_script ${MAIN_SCRIPT} ${temp_dir}
    copy_script ${COMMON_SCRIPT} ${temp_dir}
    copy_script ${UNINSTALL_SCRIPT} ${temp_dir}
    copy_script ${VERSION_INFO} ${temp_dir}
}

# copy script
function copy_script() {
    local script_name=${1}
    local temp_dir=${2}

    if [ -f "${temp_dir}/${script_name}" ]; then
        rm -f "${temp_dir}/${script_name}"
    fi

    cp ${RUN_SCRIPT_DIR}/${script_name} ${temp_dir}/${script_name}
    chmod 500 "${temp_dir}/${script_name}"
}

function get_package_name() {

    CONFIG_FILE="${CUR_DIR}/run_script/version.info"
    NAME=$(grep -E '^Name=' "$CONFIG_FILE" | cut -d'=' -f2)
    # 如果VERSION未通过参数设置，则从配置文件读取
    if [ -z "${VERSION}" ]; then
        VERSION=$(grep -E '^Version=' "$CONFIG_FILE" | cut -d'=' -f2)
    fi
    local os_arch=$(arch)
    echo "${NAME}_${VERSION}_linux_${os_arch}.run"
}

function create_run_package() {
    local run_name=${1}
    local temp_dir=${2}
    local main_script=${MAIN_SCRIPT}
    local package_name=$(get_package_name)

    ${CREATE_RUN_SCRIPT} \
    --header ${CONTROL_PARAM_SCRIPT} \
    --help-header ${FILTER_PARAM_SCRIPT} \
    --pigz \
    --tar-quietly \
    --complevel 4 \
    --nomd5 \
    --sha256 \
    --chown \
    ${temp_dir} \
    ${OUTPUT_DIR}/${package_name} \
    ${run_name} \
    ./${main_script}
}


function check_file_exist() {
    local temp_dir=${1}

    check_package ${temp_dir} ${PKG_LIMIT_SIZE}

    check_package ${temp_dir}/${MAIN_SCRIPT} ${PKG_LIMIT_SIZE}
    check_package ${temp_dir}/${COMMON_SCRIPT} ${PKG_LIMIT_SIZE}
    check_package ${temp_dir}/${UNINSTALL_SCRIPT} ${PKG_LIMIT_SIZE}
    check_package ${temp_dir}/${VERSION_INFO} ${PKG_LIMIT_SIZE}
}

function check_package() {
    local _path="$1"
    local _limit_size=$2
    echo "check ${_path} exists"
    # 检查路径是否存在
    if [ ! -e "${_path}" ]; then
        echo "${_path} does not exist."
        exit 1
    fi

    # 检查路径是否为文件
    if [ -f "${_path}" ]; then
        local _file_size=$(stat -c%s "${_path}")
        # 检查文件大小是否超过限制
        if [ "${_file_size}" -gt "${_limit_size}" ] || [ "${_file_size}" -eq 0 ]; then
            echo "package size exceeds limit:${_limit_size}"
            exit 1
        fi
    fi
}

function main() {
	create_temp_dir ${CMP_TEMP_DIR}
	check_file_exist ${CMP_TEMP_DIR}
	create_run_package ${CMP_RUN_NAME} ${CMP_TEMP_DIR} ${main_script}
	check_package ${OUTPUT_DIR}/$(get_package_name) ${PKG_LIMIT_SIZE}
	# 清理makeself创建的临时文件
	if [ -f "${OUTPUT_DIR}/temp" ]; then
		rm -f "${OUTPUT_DIR}/temp"
	fi
}

# 解析脚本参数
parse_script_args "$@"

main

