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

readonly SNAPSHOT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.host-gitconfig"
readonly SNAPSHOT_TEMP="${SNAPSHOT_PATH}.tmp.$$"

cleanup() {
    rm -f "${SNAPSHOT_TEMP}"
}
trap cleanup EXIT

for cache_dir in "${HOME}/.cache/uv" "${HOME}/.cache/pip" "${HOME}/.cache/pre-commit"; do
    mkdir -p "${cache_dir}" || {
        echo "[devcontainer] warning: failed to create ${cache_dir} on the Docker host." >&2
    }
done

: > "${SNAPSHOT_TEMP}"

if ! command -v git >/dev/null 2>&1; then
    cp "${SNAPSHOT_TEMP}" "${SNAPSHOT_PATH}"
    echo "[devcontainer] warning: git is unavailable on the Docker host; Git identity will not be synchronized."
    exit 0
fi

user_name="$(git config --global --get user.name 2>/dev/null || true)"
user_email="$(git config --global --get user.email 2>/dev/null || true)"

if [[ -n "${user_name}" ]]; then
    git config --file "${SNAPSHOT_TEMP}" user.name "${user_name}"
fi
if [[ -n "${user_email}" ]]; then
    git config --file "${SNAPSHOT_TEMP}" user.email "${user_email}"
fi

# Copying into the existing file preserves its inode, so a running container's
# single-file bind mount does not keep pointing at an obsolete empty snapshot.
cp "${SNAPSHOT_TEMP}" "${SNAPSHOT_PATH}"
chmod 600 "${SNAPSHOT_PATH}" 2>/dev/null || true
