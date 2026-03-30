#!/bin/bash
# install.sh - 安装脚本
USERNAME=$(id -un)
USERGROUP=$(id -gn)

COMMON_SHELL_PATH="common.sh"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印彩色信息
info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 创建安装目录
create_install_dir() {
    info "Creating install directory: $INSTALL_DIR"

    # 创建目录并直接设置权限为777
    if ! mkdir -p -m 777 "$INSTALL_DIR"; then
        error "Failed to create directory: $INSTALL_DIR"
        error "Please check if you have sufficient permissions, or create the directory manually"
        exit 1
    fi
    
    # 设置目录权限为777
    chmod -R 777 "$INSTALL_DIR"
    info "Set directory permissions to 777: $INSTALL_DIR"
    if [ $? -ne 0 ]; then
        warn "Failed to set directory permissions to 777: $INSTALL_DIR"
    fi
}

function convert_install_path() {
    local _install_path="$1"

    # delete last "/" "/."，并将中间的"/./"替换成"/"
    _install_path=`echo "${_install_path}" | sed -r "s/((\/)|(\/\.))*$//g" | sed -r "s|/\./|/|g"`
    if [ -z "${_install_path}" ]; then
        _install_path="/"
    fi
    # covert relative path to absolute path
    # 处理以 "./" 或 "." 开头的路径
    if [[ "${_install_path}" =~ ^\./.* ]] || [[ "${_install_path}" =~ ^\.$ ]]; then
        # 移除开头的 "./" 或 "."
        _install_path=`echo "${_install_path}" | sed -r "s|^\./||" | sed -r "s|^\.$||"`
        if [ -z "${_install_path}" ]; then
            _install_path="${run_path}"
        else
            _install_path="${run_path}/${_install_path}"
        fi
    else
        local _prefix=`echo "${_install_path}" | cut -d"/" -f1`
        if [ ! -z "${_prefix}" ] && [ "~" != "${_prefix}" ]; then
            _install_path="${run_path}/${_install_path}"
        fi
    fi
    # covert '~' to home path
    local _suffix_path=`echo "${_install_path}" | cut -d"~" -f2`
    if [ "${_suffix_path}" != "${_install_path}" ]; then
        local _home_path=`eval echo "~" | sed "s/\/*$//g"`
        _install_path="${_home_path}${_suffix_path}"
    fi
    # 规范化路径：移除双斜杠，处理相对路径组件
    _install_path=`echo "${_install_path}" | sed -r "s|//+|/|g"`
    # 转换为绝对路径（如果可能）
    if command -v realpath >/dev/null 2>&1; then
        _install_path=$(realpath -m "${_install_path}" 2>/dev/null || echo "${_install_path}")
    fi
    echo "${_install_path}"
}

# 主安装函数
main_install() {
    # 设置安装目录
    INSTALL_DIR=$(get_install_path)

    if [[ "$SILENT_MODE" != "true" ]]; then
        echo "========================================="
        echo "  mindstudio-accucmp Installer"
        echo "========================================="
        echo "Install directory: $INSTALL_DIR"
        echo ""
    fi

    # 创建安装目录
    create_install_dir

    info "Installing files..."

    # 复制文件到安装目录
    if [[ "$SUDO_USED" == "true" ]]; then
        sudo cp -r "$PWD"/msaccucmp/* "$INSTALL_DIR"/
    else
        cp -r "$PWD"/msaccucmp/* "$INSTALL_DIR"/
    fi
    
    # 安装完成后，将目录权限改回安全权限（750）
    chmod -R 750 "$INSTALL_DIR"
    if [ $? -ne 0 ]; then
        warn "Failed to set directory permissions to 750: $INSTALL_DIR"
    fi
    info "Set directory permissions to 750: $INSTALL_DIR"
    info "Installation completed!"
    info "Install directory: $INSTALL_DIR"
    
    # 注册卸载脚本
    register_uninstall
}


function get_install_path() {
    if [ -z "${input_install_path}" ]; then
        local _install_path
        _install_path="/usr/local/Ascend/tools/operator_cmp/compare"
    else
        _install_path="${input_install_path}/tools/operator_cmp/compare"
    fi
    install_path=$(convert_install_path "${_install_path}")
    echo "${install_path}"
}

# 注册卸载脚本
register_uninstall() {
    info "Registering uninstall script..."
    local _install_path="$INSTALL_DIR"
    local _info_dir="${_install_path}/../../../share/info/operator_cmp"
    local _script_dir="${SHELL_DIR}"
    local _uninstall_source="${_script_dir}/uninstall.sh"
    local _cann_uninstall="${_install_path}/../../../cann_uninstall.sh"
    local _version_source="${_script_dir}/version.info"
 
    # 如果存在cann_uninstall.sh，则执行sed命令
    if [ -f "${_cann_uninstall}" ] && ! grep -q "uninstall_package \"share/info/operator_cmp\"" "${_cann_uninstall}"; then
        sed -i "/^exit /i uninstall_package \"share/info/operator_cmp\"" "${_cann_uninstall}"
    fi
    
    # 检查父目录是否存在和权限
    local _parent_dir=$(dirname "${_info_dir}")
    info "Checking parent directory: ${_parent_dir}"
    if [ ! -d "${_parent_dir}" ]; then
        error "Parent directory does not exist: ${_parent_dir}"
        error "Please ensure the installation directory structure is correct"
        return 1
    fi
    
    if [ ! -w "${_parent_dir}" ]; then
        error "No write permission to parent directory: ${_parent_dir}"
        error "Current user: $(whoami)"
        error "Please run with appropriate permissions"
        return 1
    fi
    
    # 创建目录
    info "Creating directory: ${_info_dir}"
    mkdir -p "${_info_dir}" 2>&1
    local _mkdir_result=$?
    if [ ${_mkdir_result} -ne 0 ]; then
        error "Failed to create directory: ${_info_dir}"
        error "Parent directory: ${_parent_dir}"
        error "Parent directory writable: $([ -w "${_parent_dir}" ] && echo "yes" || echo "no")"
        error "mkdir exit code: ${_mkdir_result}"
        return 1
    fi
    
    # 验证目录是否创建成功
    if [ ! -d "${_info_dir}" ]; then
        error "Directory was not created: ${_info_dir}"
        return 1
    fi
    
    if [ ! -w "${_info_dir}" ]; then
        error "Directory exists but is not writable: ${_info_dir}"
        return 1
    fi
    
    # 拷贝version.info
    if [ -f "${_version_source}" ]; then
        info "Copying version info from ${_version_source} to ${_info_dir}/version.info"
        cp "${_version_source}" "${_info_dir}/version.info" 2>&1
        local _cp_result=$?
        if [ ${_cp_result} -ne 0 ]; then
            error "Failed to copy version info"
            error "Source: ${_version_source}"
            error "Destination: ${_info_dir}/version.info"
            error "Destination directory writable: $([ -w "${_info_dir}" ] && echo "yes" || echo "no")"
            error "cp exit code: ${_cp_result}"
            return 1
        fi
        info "Version info registered successfully"
    else
        warn "Version info not found: ${_version_source}"
        warn "Script directory: ${_script_dir}"
        warn "Current working directory: $(pwd)"
        warn "Script path (\$0): $0"
        return 1
    fi

    # 拷贝uninstall.sh
    if [ -f "${_uninstall_source}" ]; then
        info "Copying uninstall script from ${_uninstall_source} to ${_info_dir}/uninstall.sh"
        cp "${_uninstall_source}" "${_info_dir}/uninstall.sh" 2>&1
        local _cp_result=$?
        if [ ${_cp_result} -ne 0 ]; then
            error "Failed to copy uninstall script"
            error "Source: ${_uninstall_source}"
            error "Destination: ${_info_dir}/uninstall.sh"
            error "Destination directory writable: $([ -w "${_info_dir}" ] && echo "yes" || echo "no")"
            error "cp exit code: ${_cp_result}"
            return 1
        fi
        chmod +x "${_info_dir}/uninstall.sh" 2>/dev/null || true
        info "Uninstall script registered successfully"
    else
        warn "Uninstall script not found: ${_uninstall_source}"
        return 1
    fi
    
    return 0
}

# 主卸载函数
main_uninstall() {
    # 获取operator_cmp目录路径
    OPERATOR_CMP_DIR=$(get_install_path)
    local _install_path="$INSTALL_DIR"
    local _cann_uninstall="${_install_path}/../../../cann_uninstall.sh"
    
    
    if [[ "$SILENT_MODE" != "true" ]]; then
        echo "========================================="
        echo "  mindstudio-accucmp Uninstaller"
        echo "========================================="
        echo "Target directory: $OPERATOR_CMP_DIR"
        echo ""
    fi
    
    # 检查目录是否存在
    if [[ ! -d "$OPERATOR_CMP_DIR" ]]; then
        warn "Directory does not exist: $OPERATOR_CMP_DIR"
        warn "Nothing to uninstall"
        return 0
    fi
    
    # 检查写权限
    if [[ ! -w "$(dirname "$OPERATOR_CMP_DIR")" ]]; then
        error "No permission to delete $OPERATOR_CMP_DIR"
        error "Current user: $(whoami)"
        error "Please run uninstall with a user that has permissions"
        error "or manually delete: rm -rf $OPERATOR_CMP_DIR"
        exit 1
    fi
    
    info "Uninstalling mindstudio-accucmp..."
    info "Removing directory: $OPERATOR_CMP_DIR"
    
    # 删除operator_cmp目录
    if rm -rf "$OPERATOR_CMP_DIR"; then
        info "mindstudio-accucmp has been successfully removed"
        return 0
    else
        error "Deletion failed, please check permissions"
        exit 1
    fi

    if [ -f "${_cann_uninstall}" ]; then
        sed -i "/uninstall_package \"share\/info\/operator_cmp\"/d" "${_cann_uninstall}"
    fi
}

# 主函数
main() {
    # 根据标志决定执行安装还是卸载
    if [[ "$uninstall_flag" == "y" ]]; then
        # 执行卸载
        main_uninstall
    else
        # 执行安装
        main_install
    fi

    # 清理（makeself 会自动清理临时目录）
    exit 0
}

source ${COMMON_SHELL_PATH}
#参数初始化
install_file=""
input_install_path=""
install_for_all=n
install_flag=n
uninstall_flag=n
upgrade_flag=n
quiet_flag=n
version_flag=n
check_flag=n
pylocal=y
force_flag=n

# get run package path
run_path=`echo "$2" | cut -d"-" -f3-`
if [ -z "${run_path}" ]; then
    run_path=`pwd`
else
    # delete last "/" "/."
    run_path=`echo "${run_path}" | sed "s/((\/)|(\/\.))*$//g"`
    [ -z "${run_path}" ] && run_path="/"
    if [ ! -d "${run_path}" ]; then
        log_and_print $LEVEL_ERROR "Run package path is invalid: $run_path"
        exit 1
    fi
    # Ensure run_path is an absolute path
    if [[ ! "${run_path}" =~ ^/ ]]; then
        run_path="$(cd "${run_path}" && pwd)"
    fi
fi

shift 2

start_log

for arg in "$@"; do
    case "$arg" in
    --force)
        force_flag=y
        ;;
    --check)
        check_flag=y
        ;;
    --quiet)
        quiet_flag=y
        ;;
    --install-path=*)
        input_install_path=`echo "$arg" | cut -d"=" -f2`
        ;;
    --install-for-all)
        install_for_all=y
        ;;
    --full)
        install_flag=y
        ;;
    --install)
        install_flag=y
        ;;
    --run)
        install_flag=y
        ;;
    --devel)
        install_flag=y
        ;;
    --uninstall)
        uninstall_flag=y
        ;;
    --upgrade)
        upgrade_flag=y
        ;;
    --version)
        version_flag=y
        ;;
    -*)
        echo "Unsupported parameters: $arg"
        ;;
    *)
        if [ ! -z "$arg" ]; then
            echo "Unsupported parameters: $arg"
        fi
        break
        ;;
    esac
done

# 执行主函数
main
