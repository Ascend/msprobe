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

import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

CI_GLIBC_PATH = "/opt/gcc11-glibc2.17"
BUILD_INFO_PATH = Path("python") / "msprobe" / "_build_info.py"
REPOSITORY_URL = "https://gitcode.com/Ascend/MindStudio-Probe"


def _normalise_commit_id(commit_id):
    commit_id = commit_id.strip()
    if re.fullmatch(r"[0-9a-fA-F]{7,64}", commit_id):
        return commit_id[:7].lower()
    return "unknown"


def _get_commit_id(project_root):
    for variable in ("MSPROBE_BUILD_COMMIT", "CI_COMMIT_SHA", "GIT_COMMIT", "GITHUB_SHA"):
        commit_id = os.environ.get(variable)
        if commit_id:
            return _normalise_commit_id(commit_id)

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return _normalise_commit_id(result.stdout)


def _get_build_date():
    source_date_epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if source_date_epoch:
        try:
            build_time = datetime.fromtimestamp(int(source_date_epoch), tz=timezone.utc)
        except (ValueError, OverflowError, OSError):
            build_time = datetime.now(timezone.utc)
    else:
        build_time = datetime.now(timezone.utc)
    return build_time.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _render_build_info(version, commit_id, build_date):
    return (
        '"""Build metadata generated while creating the distribution package."""\n\n'
        f'VERSION = {version!r}\n'
        f'COMMIT_ID = {commit_id!r}\n'
        f'BUILD_DATE = {build_date!r}\n'
        f'REPOSITORY_URL = {REPOSITORY_URL!r}\n'
    )


class CppWheelTagHook(BuildHookInterface):
    PLUGIN_NAME = "cpp_wheel_tag"
    _original_build_info = None

    def initialize(self, version, build_data):
        project_root = Path(self.root)
        build_info_path = project_root / BUILD_INFO_PATH
        project_version = str(self.metadata.version)
        self._original_build_info = build_info_path.read_text(encoding="utf-8")
        build_info_path.write_text(
            _render_build_info(project_version, _get_commit_id(project_root), _get_build_date()),
            encoding="utf-8",
        )

        if os.environ.get("MSPROBE_CPP_BUILD") != "1":
            return

        build_data["pure_python"] = False

        py_version = f"{sys.version_info.major}{sys.version_info.minor}"
        python_tag = f"cp{py_version}"
        abi_tag = f"cp{py_version}"
        if os.path.isdir(CI_GLIBC_PATH):
            libc_name, libc_version = "glibc", "2.17"
        else:
            libc_name, libc_version = platform.libc_ver()
        if libc_name == "glibc" and libc_version:
            major, minor = libc_version.split(".")
            platform_tag = f"manylinux_{major}_{minor}_{platform.machine()}"
        else:
            platform_tag = f"linux_{platform.machine()}"

        build_data["tag"] = f"{python_tag}-{abi_tag}-{platform_tag}"

    def finalize(self, version, build_data, artifact_path):
        del version, build_data, artifact_path
        original_build_info = getattr(self, "_original_build_info", None)
        if original_build_info is not None:
            (Path(self.root) / BUILD_INFO_PATH).write_text(original_build_info, encoding="utf-8")
