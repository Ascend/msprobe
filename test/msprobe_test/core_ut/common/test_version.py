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
