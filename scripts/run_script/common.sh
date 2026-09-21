# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

#!/bin/bash
INSTALL_INFO_KEY_ARRAY=("UserName" "UserGroup" "Install_Path_Param")
MODULE_NAME="mindstudio-accucmp"
LEVEL_ERROR="ERROR"
LEVEL_WARN="WARNING"
LEVEL_INFO="INFO"
USERNAME=$(id -un)
USERGROUP=$(id -gn)
SHELL_DIR=$(cd "$(dirname "$0")" || exit; pwd)

MINDSTUDIO_ARRAY=(${MODULE_NAME})
LINK_ARRAY=("bin" "lib64")

# whl子包列表,待添加
SUBWHL=("")
# run子包列表,待添加
SUBRUN=("")

export log_file=""

function log() {
    local content=`echo "$@" | cut -d" " -f2-`
    local cur_date=`date +"%Y-%m-%d %H:%M:%S"`

    echo "[${MODULE_NAME}] [${cur_date}] [$1]: $content" >> "${log_file}"
}

function log_and_print() {
    local content=`echo "$@" | cut -d" " -f2-`
    local cur_date=`date +"%Y-%m-%d %H:%M:%S"`

    echo "[${MODULE_NAME}] [${cur_date}] [$1]: $content"
    echo "[${MODULE_NAME}] [${cur_date}] [$1]: $content" >> "${log_file}"
}

function print_log() {
    local content=`echo "$@" | cut -d" " -f2-`
    local cur_date=`date +"%Y-%m-%d %H:%M:%S"`

    echo "[${MODULE_NAME}] [${cur_date}] [$1]: $content"
}

function init_log() {
    local _log_path="/var/log/ascend_seclog"
    local _log_file="ascend_install.log"

    if [ $(id -u) -ne 0 ]; then
        local _home_path=`eval echo "~"`
        _log_path="${_home_path}${_log_path}"
    fi

    log_file="${_log_path}/${_log_file}"

    create_folder "${_log_path}" "${USERNAME}:${USERGROUP}" 750
    if [ $? -ne 0 ]; then
        print_log $LEVEL_WARN "Create ${_log_path} failed."
    fi

    if [ -L "${log_file}" ] || [ ! -f "${log_file}" ]; then
        rm -rf "${log_file}" >/dev/null 2>&1
    fi
    create_file "${log_file}" "${USERNAME}:${USERGROUP}" 640
    if [ $? -ne 0 ]; then
        print_log $LEVEL_WARN "Create ${log_file} failed."
    fi
}

function start_log() {
    free_log_space
    local cur_date=`date +"%Y-%m-%d %H:%M:%S"`

    echo "[${MODULE_NAME}] [${cur_date}] [INFO]: Start Time: $cur_date"
    echo "[${MODULE_NAME}] [${cur_date}] [INFO]: Start Time: $cur_date" >> "${log_file}"
}

function exit_log() {
    local cur_date=`date +"%Y-%m-%d %H:%M:%S"`

    echo "[${MODULE_NAME}] [${cur_date}] [INFO]: End Time: $cur_date"
    echo "[${MODULE_NAME}] [${cur_date}] [INFO]: End Time: $cur_date" >> "${log_file}"
    exit $1
}

function free_log_space() {
    local file_size=$(stat -c %s "${log_file}")
    # mindstudio install log file will be limited in 20M
    if [ "${file_size}" -gt $((1024 * 1024 * 20)) ]; then
        local ibs=512
        local delete_size=$((${file_size} / 3 / ${ibs}))
        dd if="$log_file" of="${log_file}_tmp" bs="${ibs}" skip="${delete_size}" > /dev/null 2>&1
        mv "${log_file}_tmp" "${log_file}"
        chmod 640 ${log_file}
    fi
}

function update_install_param() {
    local _key=$1
    local _val=$2
    local _file=$3
    local _param

    if [ ! -f "${_file}" ]; then
        exit 1
    fi

    for key_param in "${INSTALL_INFO_KEY_ARRAY[@]}"; do
        if [ ${key_param} != ${_key} ]; then
            continue
        fi
        _param=`grep -r "${_key}=" "${_file}"`
        if [ "x${_param}" = "x" ]; then
            echo "${_key}=${_val}" >> "${_file}"
        else
            sed -i "/^${_key}=/c ${_key}=${_val}" "${_file}"
        fi
        break
    done
}

function get_install_param() {
    local _key=$1
    local _file=$2
    local _param

    if [ ! -f "${_file}" ];then
        exit 1
    fi

    for key_param in "${INSTALL_INFO_KEY_ARRAY[@]}"; do
        if [ ${key_param} != ${_key} ]; then
            continue
        fi
        _param=`grep -r "${_key}=" "${_file}" | cut -d"=" -f2-`
        break
    done
    echo "${_param}"
}

function change_mode() {
    local _mode=$1
    local _path=$2
    local _type=$3

    if [ ! x"${install_for_all}" = "x" ] && [ ${install_for_all} = y ]; then
        _mode="$(expr substr ${_mode} 1 2)$(expr substr ${_mode} 2 1)"
    fi
    if [ ${_type} = "dir" ]; then
        find "${_path}" -type d -exec chmod ${_mode} {} \; 2> /dev/null
    elif [ ${_type} = "file" ]; then
        find "${_path}" -type f -exec chmod ${_mode} {} \; 2> /dev/null
    fi
}

function change_file_mode() {
    local _mode=$1
    local _path=$2
    change_mode ${_mode} "${_path}" file
}

function change_dir_mode() {
    local _mode=$1
    local _path=$2
    change_mode ${_mode} "${_path}" dir
}

function create_file() {
    local _file=$1

    if [ ! -f "${_file}" ]; then
        touch "${_file}"
        [ $? -ne 0 ] && return 1
    fi

    chown -hf "$2" "${_file}"
    [ $? -ne 0 ] && return 1
    change_file_mode "$3" "${_file}"
    [ $? -ne 0 ] && return 1
    return 0
}

function create_folder() {
    local _path=$1

    if [ -z ${_path} ]; then
        return 1
    fi

    if [ ! -d ${_path} ]; then
        mkdir -p ${_path} >/dev/null 2>&1
        [ $? -ne 0 ] && return 1
    fi

    chown -hf $2 ${_path}
    [ $? -ne 0 ] && return 1
    change_dir_mode $3 ${_path}
    [ $? -ne 0 ] && return 1
    return 0
}

function is_dir_empty() {
    local _path=$1
    local _file_num

    if [ -z ${_path} ]; then
        return 1
    fi

    if [ ! -d ${_path} ]; then
        return 1
    fi
    _file_num=`ls "${_path}" | wc -l`
    if [ ${_file_num} -eq 0 ]; then
        return 0
    fi
    return 1
}

function check_install_path_valid() {
    local install_path="$1"
    # 黑名单设置，不允许//，...这样的路径
    if echo "${install_path}" | grep -Eq '/{2,}|\.{3,}'; then
        return 1
    fi
    # 白名单设置，只允许常见字符
    if echo "${install_path}" | grep -Eq '^~?[a-zA-Z0-9./_-]*$'; then
        return 0
    else
        return 1
    fi
}

function check_dir_permission() {
    local _path=$1

    if [ -z ${_path} ]; then
        log_and_print $LEVEL_ERROR "The dir path is empty."
        exit 1
    fi

    if [ "$(id -u)" -eq 0 ]; then
        return 0
    fi

    if [ -d ${_path} ] && [ ! -w ${_path} ]; then
        return 1
    fi

    return 0
}

function create_relative_softlink() {
    local _src_path="$1"
    local _des_path="$2"

    local _des_dir_name=$(dirname $_des_path)
    _src_path=$(readlink -f ${_src_path})
    if [ ! -f "$_src_path" -a ! -d "$_src_path" -a ! -L "$_src_path" ]; then
        return
    fi
    _src_path=$(get_relative_path $_des_dir_name $_src_path)
    if [ -L "${_des_path}" ]; then
        delete_softlink "${_des_path}"
    fi
    ln -sf "${_src_path}" "${_des_path}"
    if [ $? -ne 0 ]; then
        print_log $LEVEL_ERROR "${_src_path} softlink to ${_des_path} failed!"
        return 1
    fi
}

function delete_softlink() {
    local _path="$1"
    # 如果目标路径是个软链接，则移除
    if [ -L "${_path}" ]; then
        local _parent_path=$(dirname ${_path})
        if [ ! -w ${_parent_path} ]; then
            chmod u+w ${_parent_path}
            rm -f "${_path}"
            if [ $? -ne 0 ]; then
                print_log $LEVEL_ERROR "remove softlink ${_path} failed!"
                exit 1
            fi
            chmod u-w ${_parent_path}
        else
            rm -f "${_path}"
            if [ $? -ne 0 ]; then
                print_log $LEVEL_ERROR "remove softlink ${_path} failed!"
                exit 1
            fi
        fi
    fi
}

function create_install_path() {
    local _install_path=$1

    if [ ! -d "${_install_path}" ]; then
        local _ppath=$(dirname ${_install_path})
        while [[ ! -d ${_ppath} ]];do
            _ppath=$(dirname ${_ppath})
        done

        check_dir_permission "${_ppath}"
        if [ $? -ne 0 ]; then
            chmod u+w -R ${_ppath}
            [ $? -ne 0 ] && exit_log 1
        fi

        create_folder "${_install_path}" $USERNAME:$USERGROUP 750
        [ $? -ne 0 ] && exit_log 1
    else
        check_dir_permission "${_install_path}"
        if [ $? -ne 0 ]; then
            chmod u+w -R ${_install_path}
        fi
    fi
}

function remove_empty_dir() {
    if [ -d "$1" ] && [ -z "$(ls -A $1 2>/dev/null)" ] && [[ ! "$1" =~ ^/+$ ]]; then
        if [ ! -w $(dirname $1) ]; then
            chmod u+w $(dirname $1)
            rm -rf "$1"
            if [ $? != 0 ]; then
                print_log $LEVEL_ERROR "delete directory $1 fail"
                exit 1
            fi
            chmod u-w $(dirname $1)
        else
            rm -rf "$1"
            if [ $? != 0 ]; then
                print_log $LEVEL_ERROR "delete directory $1 fail"
                exit 1
            fi
        fi
    fi
}

function remove_if_file_exist() {
    local _file=$1
    if [[ -f "${_file}" ]]; then
        rm -f ${_file}
    fi
}

function get_relative_path() {
    local _relative_to_path=$1
    local _des_path=$2
    echo $(realpath --relative-to=$_relative_to_path $_des_path)
}

# 删除uninstall.sh文件，如果已经没有uninstall_package调用
function remove_uninstall_file_if_no_content() {
    local _file="$1"
    local _num

    if [ ! -f "${_file}" ]; then
        return 0
    fi

    _num=$(grep "^uninstall_package " ${_file} | wc -l)
    if [ ${_num} -eq 0 ]; then
        rm -f "${_file}" > /dev/null 2>&1
        if [ $? -ne 0 ]; then
            log_and_print $LEVEL_WARN "Delete file:${_file} failed, please delete it by yourself."
        fi
    fi
}

# 删除uninstall.sh文件中的uninstall_package函数调用
function remove_uninstall_package() {
    local _file="$1"

    if [ -f "${_file}" ]; then
        sed -i "/uninstall_package \"share\/info\/${MODULE_NAME}\/script\"/d" "${_file}"
        if [ $? -ne 0 ]; then
            log_and_print $LEVEL_ERROR "remove ${_file} uninstall_package command failed!"
            exit 1
        fi
    fi
}


function installWhlPackage() {
    local _pylocal=$1
    local _package_path=$2
    local _pythonlocalpath=$3

    log_and_print ${LEVEL_INFO} "start to install whl package."
    # 一次性安装路径下的所有whl包，避免whl包安装中出现路径冲突的问题
    whl_files=()
    for whl in "${SUBWHL[@]}"; do
        # 如果目标安装路径是一个软连接，需要先删除再安装，否则会导致无法更新
        if [ -L "${_pythonlocalpath}/${whl}" ]; then
            delete_softlink ${_pythonlocalpath}/${whl}
        fi
        whl_files+=" $_package_path/${whl}*.whl"
    done
    if [ "-${_pylocal}" = "-y" ]; then
        pip3 install --upgrade --no-index --no-deps --force-reinstall ${whl_files} -t ${_pythonlocalpath}
    else
        if [ "$(id -u)" -ne 0 ]; then
            pip3 install --upgrade --no-index --no-deps --force-reinstall ${whl_files} --user
        else
            pip3 install --upgrade --no-index --no-deps --force-reinstall ${whl_files}
        fi
    fi
    if [ $? -ne 0 ]; then
        log_and_print ${LEVEL_ERROR} "Install whl package failed."
        return 1
    fi
    # 安装完成后删除whl包
    rm -rf $whl_files || return 1
    remove_empty_dir ${_package_path}
    log ${LEVEL_INFO} "install whl package succeed."
    return 0
}

### uninstall whl
function whlUninstallPackage() {
    local module_="$1"
    local python_path_="$2"

    log ${LEVEL_INFO} "start to uninstall ${module_}"
    export PYTHONPATH=${python_path_}
    pip3 show ${module_} > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        log ${LEVEL_WARN} "${module_} has not been installed."
        return 0
    fi
    pip3 uninstall -y "${module_}" > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        log_and_print ${LEVEL_ERROR} "uninstall ${module_} failed."
        return 1
    fi
    remove_if_file_exist ${python_path_}/bin/${module_}
    remove_if_file_exist ${python_path_}/bin/${module_}.ini
    remove_empty_dir ${python_path_}/bin
    log ${LEVEL_INFO} "uninstall ${module_} succeed."
    return 0
}

function install_subpackage() {
    local _package_path="$1"
    local _install_path="$2"
    # 遍历查找整包安装路径下的所有子run包，执行子包的安装过程
    find "${_package_path}" -type f -name "*.run" | while read -r run_file; do
        log_and_print $LEVEL_INFO "Installing ${run_file}"
        ${run_file} --install-path=${_install_path} --run --force || return 1
        # 安装完成后删除子run包
        rm -rf ${run_file} || return 1
    done

    # 安装所有whl子包
    installWhlPackage ${pylocal} ${_package_path} ${_package_path}/../python/site-packages || return 1
    log_and_print $LEVEL_INFO "all subpackage installed succeed"
    return 0
}

function uninstall_subpackage() {
    # 卸载SUBWHL中定义的whl包
    for whl in "${SUBWHL[@]}"; do
        whlUninstallPackage ${whl} ${install_path}/python/site-packages
    done
    # 卸载SUBRUN中定义的子run包
    for run in "${SUBRUN[@]}"; do
        local _uninstall_file=${install_path}/share/info/${run}/script/uninstall.sh
        if [ -e "${_uninstall_file}" ]; then
            bash ${_uninstall_file}
            if [ $? -ne 0 ]; then
                log_and_print $LEVEL_ERROR "Remove ${MODULE_NAME} run package failed in ${install_path}."
                return 1
            fi
        fi
    done
    log_and_print $LEVEL_INFO "all subpackage uninstalled succeed"
    return 0
}

function uninstall_tool() {
    # 卸载子包，包括子run包及whl包
    uninstall_subpackage || return 1
    # when normal user uninstall package, shell need to restore dir permission
    "$COMMON_PARSER_PATH" --restoremod --package=${MODULE_NAME} --username="unknown" --usergroup="unknown" \
        "${install_path}" "${FILELIST_CSV_PATH}"
    if [ $? -ne 0 ]; then
        log_and_print $LEVEL_ERROR "Restore directory written permission failed."
        return 1
    fi

    "$COMMON_PARSER_PATH" --remove --package=${MODULE_NAME} "${install_path}" "${FILELIST_CSV_PATH}"
    if [ $? -ne 0 ]; then
        log_and_print $LEVEL_ERROR "ERR_NO:0X0090;ERR_DES: Remove ${MODULE_NAME} files failed in ${install_path}."
        return 1
    fi
    log $LEVEL_INFO "Remove ${MODULE_NAME} files succeed in ${install_path}!"
    return 0
}

function uninstall() {
    ARCH=$(cat $SHELL_DIR/../scene.info | grep arch | cut -d"=" -f2)
    OS=$(cat $SHELL_DIR/../scene.info | grep os | cut -d"=" -f2)

    uninstall_tool
    if [ $? -ne 0 ]; then
        log_and_print ${LEVEL_ERROR} "${MODULE_NAME} uninstall failed."
        return 1
    fi
    rm -f ${install_file}

    unregister_uninstall ${install_path}
    remove_uninstall_file_if_no_content ${install_path}/cann_uninstall.sh
    remove_empty_dir ${install_path}/share/info/${MODULE_NAME}
    remove_empty_dir ${install_path}/share/info
    remove_empty_dir ${install_path}/share
    remove_empty_dir ${install_path}/${ARCH}-${OS}/include
    remove_empty_dir ${install_path}
    remove_empty_dir $(dirname ${install_path})
    log_and_print $LEVEL_INFO "${MODULE_NAME} uninstall success!"
}

init_log
