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

import sys
import types
import unittest
from unittest import mock

from msprobe.core.common.output_postprocess import load_pt_helper


class TestLoadPtHelper(unittest.TestCase):
    def setUp(self):
        self.original_loader = load_pt_helper._loader
        load_pt_helper._loader = None

    def tearDown(self):
        load_pt_helper._loader = self.original_loader

    def test_load_pt_file_lazy_import_and_cache(self):
        first_loader = mock.Mock(side_effect=["first", "second"])
        second_loader = mock.Mock(return_value="unused")

        first_module = types.ModuleType("msprobe.pytorch.common.utils")
        first_module.load_pt = first_loader
        second_module = types.ModuleType("msprobe.pytorch.common.utils")
        second_module.load_pt = second_loader

        with mock.patch.dict(sys.modules, {"msprobe.pytorch.common.utils": first_module}):
            first = load_pt_helper.load_pt_file("first.pt", to_cpu=True)

        with mock.patch.dict(sys.modules, {"msprobe.pytorch.common.utils": second_module}):
            second = load_pt_helper.load_pt_file("second.pt")

        self.assertEqual(first, "first")
        self.assertEqual(second, "second")
        first_loader.assert_has_calls([
            mock.call("first.pt", to_cpu=True),
            mock.call("second.pt", to_cpu=False),
        ])
        second_loader.assert_not_called()

    def test_load_pt_file_forwards_default_args(self):
        cached_loader = mock.Mock(return_value="ok")
        load_pt_helper._loader = cached_loader

        result = load_pt_helper.load_pt_file("demo.pt")

        self.assertEqual(result, "ok")
        cached_loader.assert_called_once_with("demo.pt", to_cpu=False)


if __name__ == "__main__":
    unittest.main()
