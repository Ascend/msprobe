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

"""Fallback metadata used when running directly from a source checkout.

The build hook temporarily replaces these values while creating a wheel and
restores this file afterwards.
"""

VERSION = "26.0.0"
COMMIT_ID = "unknown"
BUILD_DATE = "unknown"
REPOSITORY_URL = "https://gitcode.com/Ascend/MindStudio-Probe"
