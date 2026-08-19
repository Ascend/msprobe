# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
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

from msprobe._build_info import BUILD_DATE, COMMIT_ID, REPOSITORY_URL, VERSION


def get_version_info() -> str:
    """Return the unified MindStudio version information."""
    commit_suffix = f" ({COMMIT_ID})" if COMMIT_ID != "unknown" else ""
    return (
        f"msprobe {VERSION}{commit_suffix}\n"
        "Copyright (C) 2026 Huawei Technologies Co., Ltd.\n"
        "License: Mulan PSL v2.\n\n"
        "Build Info:\n"
        f"  Date : {BUILD_DATE}\n"
        f"  Repo : {REPOSITORY_URL}"
    )
