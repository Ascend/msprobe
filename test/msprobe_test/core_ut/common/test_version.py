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

from unittest import TestCase
from unittest.mock import patch

from msprobe import version


class TestVersionInfo(TestCase):
    @patch.object(version, "REPOSITORY_URL", "https://gitcode.com/Ascend/MindStudio-Probe")
    @patch.object(version, "BUILD_DATE", "2026-07-15T10:32:11Z")
    @patch.object(version, "COMMIT_ID", "41e14ec")
    @patch.object(version, "VERSION", "26.1.0")
    def test_get_version_info_contains_build_metadata(self):
        self.assertEqual(
            version.get_version_info(),
            "msprobe 26.1.0 (41e14ec)\n"
            "Copyright (C) 2026 Huawei Technologies Co., Ltd.\n"
            "License: Mulan PSL v2.\n\n"
            "Build Info:\n"
            "  Date : 2026-07-15T10:32:11Z\n"
            "  Repo : https://gitcode.com/Ascend/MindStudio-Probe",
        )

    @patch.object(version, "COMMIT_ID", "unknown")
    def test_get_version_info_omits_unknown_commit(self):
        self.assertNotIn("(unknown)", version.get_version_info())
