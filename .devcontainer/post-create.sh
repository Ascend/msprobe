#!/usr/bin/env bash

# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
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

set -u

readonly WORKSPACE_DIR="/workspace"
readonly HOST_GITCONFIG="/tmp/host-gitconfig"
readonly HINT_MARKER="# >>> msprobe devcontainer >>>"
readonly TORCH_VERSION="2.11.0"
readonly TORCH_NPU_VERSION="2.11.0"
readonly TORCH_CPU_INDEX="${MSPROBE_TORCH_CPU_INDEX:-https://mirrors.nju.edu.cn/pytorch/whl/cpu}"
readonly TORCH_CPU_FALLBACK_INDEX="https://download.pytorch.org/whl/cpu"
readonly CANN_ARCH_LIBRARY_PATH="/usr/local/Ascend/cann/aarch64-linux/lib64"

warn() {
    echo "[devcontainer] warning: $*" >&2
}

append_line_once() {
    local line="$1"
    local target="$2"
    touch "${target}" 2>/dev/null || return 0
    grep -Fqx "${line}" "${target}" 2>/dev/null || echo "${line}" >> "${target}"
}

configure_user_bin() {
    mkdir -p "${HOME}/.local/bin" || {
        warn "failed to create ${HOME}/.local/bin"
        return 0
    }

    if command -v npm >/dev/null 2>&1; then
        npm config set prefix "${HOME}/.local" >/dev/null 2>&1 || warn "failed to configure npm user prefix"
    else
        warn "npm is unavailable; npm prefix was not configured"
    fi

    append_line_once 'export PATH="$HOME/.local/bin:$PATH"' "${HOME}/.bashrc"
    append_line_once 'export PATH="$HOME/.local/bin:$PATH"' "${HOME}/.bash_profile"
    export PATH="${HOME}/.local/bin:${PATH}"
}

configure_python311() {
    if [[ -f /etc/profile.d/z_python311.sh ]]; then
        # shellcheck disable=SC1091
        source /etc/profile.d/z_python311.sh || warn "failed to activate Python 3.11"
    elif ! python3 -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 11))' >/dev/null 2>&1; then
        warn "Python 3.11 activation script is unavailable"
    fi
}
configure_cann_runtime() {
    if [[ ! -d "${CANN_ARCH_LIBRARY_PATH}" ]]; then
        warn "${CANN_ARCH_LIBRARY_PATH} is unavailable; TorchNPU may not find CANN runtime libraries"
        return 0
    fi

    case ":${LD_LIBRARY_PATH:-}:" in
        *:"${CANN_ARCH_LIBRARY_PATH}":*) ;;
        *) export LD_LIBRARY_PATH="${CANN_ARCH_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}" ;;
    esac
    append_line_once \
        'export LD_LIBRARY_PATH="/usr/local/Ascend/cann/aarch64-linux/lib64:${LD_LIBRARY_PATH:-}"' \
        "${HOME}/.bashrc"
}

install_python_devel() {
    if python3 -c 'import os, sysconfig; raise SystemExit(not os.path.isfile(os.path.join(sysconfig.get_path("include"), "Python.h")))' \
        >/dev/null 2>&1; then
        return 0
    fi

    warn "Python development headers are unavailable; attempting installation"
    if command -v sudo >/dev/null 2>&1 && command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y python3-devel >/dev/null 2>&1 || warn "python3-devel installation failed"
    elif command -v dnf >/dev/null 2>&1; then
        dnf install -y python3-devel >/dev/null 2>&1 || warn "python3-devel installation failed"
    else
        warn "dnf is unavailable; install Python development headers manually"
    fi
}

pytorch_stack_is_ready() {
    python3 - "${TORCH_VERSION}" "${TORCH_NPU_VERSION}" <<'PY' >/dev/null 2>&1
import importlib.metadata
import sys

import torch
import torch_npu

torch_version, torch_npu_version = sys.argv[1:]
installed_names = {
    name.lower()
    for dist in importlib.metadata.distributions()
    if (name := dist.metadata.get("Name"))
}
cuda_packages = {
    name for name in installed_names
    if name == "triton" or name.startswith("cuda-") or name.startswith("nvidia-")
}
valid = (
    torch.__version__.split("+")[0] == torch_version
    and torch.version.cuda is None
    and torch_npu.__version__.split("+")[0] == torch_npu_version
    and not cuda_packages
)
raise SystemExit(not valid)
PY
}

install_pytorch_npu() {
    if pytorch_stack_is_ready; then
        echo "[devcontainer] PyTorch ${TORCH_VERSION} CPU and TorchNPU ${TORCH_NPU_VERSION} are ready"
        return 0
    fi
    if ! python3 -m pip --version >/dev/null 2>&1; then
        warn "pip is unavailable; PyTorch and TorchNPU were not configured"
        return 0
    fi

    warn "installing the msprobe PyTorch/TorchNPU development stack"
    local -a conflicting_packages=()
    while IFS= read -r package_name; do
        [[ -z "${package_name}" ]] || conflicting_packages+=("${package_name}")
    done < <(
        python3 -m pip list --format=freeze 2>/dev/null | awk -F '==' '
            BEGIN { IGNORECASE = 1 }
            $1 == "torch" || $1 == "torch-npu" || $1 == "triton" ||
            $1 ~ /^cuda-/ || $1 ~ /^nvidia-/ { print $1 }
        '
    )

    if ((${#conflicting_packages[@]} > 0)); then
        python3 -m pip uninstall -y "${conflicting_packages[@]}" >/dev/null 2>&1 || \
            warn "failed to remove one or more conflicting CUDA/PyTorch packages"
    fi
    python3 -m pip install --user --progress-bar off attrs decorator numpy pyyaml setuptools || \
        warn "one or more TorchNPU prerequisites could not be installed"
    if ! python3 -m pip install --user --progress-bar off --index-url "${TORCH_CPU_INDEX}" \
        "torch==${TORCH_VERSION}"; then
        warn "primary PyTorch CPU index failed; retrying the official index"
        python3 -m pip install --user --progress-bar off --index-url "${TORCH_CPU_FALLBACK_INDEX}" \
            "torch==${TORCH_VERSION}" || {
            warn "PyTorch ${TORCH_VERSION} CPU installation failed"
            return 0
        }
    fi
    python3 -m pip install --user --progress-bar off "torch-npu==${TORCH_NPU_VERSION}" || {
        warn "TorchNPU ${TORCH_NPU_VERSION} installation failed"
        return 0
    }
    pytorch_stack_is_ready || warn "PyTorch/TorchNPU verification failed after installation"
}
install_project_dependencies() {
    if ! python3 -m pip --version >/dev/null 2>&1; then
        warn "pip is unavailable; msprobe runtime dependencies were not installed"
        return 0
    fi

    python3 -m pip install --user --progress-bar off --editable "${WORKSPACE_DIR}" || \
        warn "msprobe runtime dependency installation failed"
}

sync_git_identity() {
    if ! command -v git >/dev/null 2>&1; then
        warn "git is unavailable; Git identity was not synchronized"
        return 0
    fi
    if [[ ! -s "${HOST_GITCONFIG}" ]]; then
        warn "host Git identity is empty; configure user.name and user.email manually if needed"
        return 0
    fi

    local user_name
    local user_email
    user_name="$(git config --file "${HOST_GITCONFIG}" --get user.name 2>/dev/null || true)"
    user_email="$(git config --file "${HOST_GITCONFIG}" --get user.email 2>/dev/null || true)"

    [[ -z "${user_name}" ]] || git config --global user.name "${user_name}" || warn "failed to synchronize Git user.name"
    [[ -z "${user_email}" ]] || git config --global user.email "${user_email}" || warn "failed to synchronize Git user.email"

    if [[ -n "${user_name}" && -n "${user_email}" ]]; then
        echo "[devcontainer] synchronized host Git identity for ${user_name}"
    fi
}

append_dev_hint_once() {
    local target="${HOME}/.bashrc"
    touch "${target}" 2>/dev/null || return 0
    if grep -Fq "${HINT_MARKER}" "${target}" 2>/dev/null; then
        return 0
    fi

    {
        echo ""
        echo "${HINT_MARKER}"
        echo "echo 'msprobe development commands:'"
        echo "echo '  Release: python3 build.py -e include-mod=all'"
        echo "echo '  Debug:   python3 build.py -e include-mod=all -e build-type=debug'"
        echo "echo '  Tests:   python3 build.py test'"
        echo "# <<< msprobe devcontainer <<<"
    } >> "${target}"
}

install_pre_commit_hook() {
    if ! command -v pre-commit >/dev/null 2>&1; then
        warn "pre-commit is unavailable; hook installation was skipped"
        return 0
    fi
    if ! git -C "${WORKSPACE_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        warn "${WORKSPACE_DIR} is not a Git repository; hook installation was skipped"
        return 0
    fi

    (cd "${WORKSPACE_DIR}" && pre-commit install) || warn "pre-commit hook installation failed"
}

setup_clangd() {
    if command -v clangd >/dev/null 2>&1; then
        return 0
    fi

    warn "clangd is unavailable; attempting installation"
    if command -v sudo >/dev/null 2>&1 && command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y clang-tools-extra >/dev/null 2>&1 || warn "clangd installation failed"
    elif command -v dnf >/dev/null 2>&1; then
        dnf install -y clang-tools-extra >/dev/null 2>&1 || warn "clangd installation failed"
    else
        warn "dnf is unavailable; install clangd manually"
    fi
}

ignore_vscode_settings() {
    if [[ -f "${WORKSPACE_DIR}/.vscode/settings.json" ]]; then
        git -C "${WORKSPACE_DIR}" update-index --skip-worktree .vscode/settings.json 2>/dev/null || \
            warn "failed to mark .vscode/settings.json as skip-worktree"
    fi
}

main() {
    echo "[devcontainer] starting msprobe environment initialization"
    configure_user_bin
    configure_cann_runtime
    configure_python311
    install_python_devel
    install_pytorch_npu
    install_project_dependencies
    sync_git_identity
    append_dev_hint_once
    install_pre_commit_hook
    setup_clangd
    ignore_vscode_settings

    if [[ ! -e "${WORKSPACE_DIR}/build/compile_commands.json" ]]; then
        echo "[devcontainer] build/compile_commands.json will be created after the first C++ build."
    fi
    echo "[devcontainer] msprobe environment initialization completed"
}

main || warn "initialization completed with recoverable errors"
exit 0
