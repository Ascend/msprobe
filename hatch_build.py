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
import sys
import platform

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

CI_GLIBC_PATH = "/opt/gcc11-glibc2.17"


class CppWheelTagHook(BuildHookInterface):
    PLUGIN_NAME = "cpp_wheel_tag"

    def initialize(self, version, build_data):
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
