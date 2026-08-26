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
import tempfile
import unittest

from msprobe.core.common.exceptions import MsprobeException
from msprobe.core.dump.common_config import CommonConfig, LoadConfig


class TestLoadConfig(unittest.TestCase):
    """Unit tests for LoadConfig parsing and validation."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # ---- parsing ----

    def test_load_section_absent(self):
        """config.json without 'load' key -> disabled, no error."""
        cfg = LoadConfig({})
        self.assertFalse(cfg.is_enabled)
        self.assertIsNone(cfg.modules)
        self.assertIsNone(cfg.path)
        self.assertEqual(cfg.step, [])
        self.assertEqual(cfg.rank, [])
        self.assertFalse(cfg.dump_after_load)

    def test_load_section_empty_dict(self):
        """'load': {} -> enabled but modules missing -> error."""
        with self.assertRaises(MsprobeException):
            LoadConfig({"load": {}})

    def test_load_section_empty_modules(self):
        """modules: [] -> error (modules is required, must be non-empty)."""
        with self.assertRaises(MsprobeException) as ctx:
            LoadConfig({"load": {"path": self.tmpdir, "modules": []}})
        self.assertIn("non-empty", str(ctx.exception))

    def test_load_full_config(self):
        """All fields populated."""
        cfg = LoadConfig({
            "load": {
                "path": self.tmpdir,
                "modules": ["Module.blocks.0.attn.A.forward.0", "Module.blocks.1.mlp.B.forward.0"],
                "step": [0],
                "rank": [1],
                "dump_after_load": True,
            }
        })
        self.assertTrue(cfg.is_enabled)
        self.assertEqual(cfg.path, self.tmpdir)
        self.assertEqual(cfg.modules, ["Module.blocks.0.attn.A.forward.0", "Module.blocks.1.mlp.B.forward.0"])
        self.assertEqual(cfg.step, [0])
        self.assertEqual(cfg.rank, [1])
        self.assertTrue(cfg.dump_after_load)

    def test_load_defaults_step_rank(self):
        """step/rank default to [] (auto-align), dump_after_load defaults to False."""
        cfg = LoadConfig({"load": {"path": self.tmpdir, "modules": ["Module.m1.A.forward.0"]}})
        self.assertTrue(cfg.is_enabled)
        self.assertEqual(cfg.step, [])
        self.assertEqual(cfg.rank, [])
        self.assertFalse(cfg.dump_after_load)

    def test_load_step_rank_range_string(self):
        """step/rank support range string like dump's step/rank."""
        cfg = LoadConfig({"load": {"path": self.tmpdir, "modules": ["Module.m1.A.forward.0"], "step": ["0-2"], "rank": ["1-3"]}})
        self.assertEqual(cfg.step, [0, 1, 2])
        self.assertEqual(cfg.rank, [1, 2, 3])

    # ---- validation ----

    def test_check_path_missing(self):
        """path is required."""
        with self.assertRaises(MsprobeException) as ctx:
            LoadConfig({"load": {"modules": ["Module.m1.A.forward.0"]}})
        self.assertIn("load.path is required", str(ctx.exception))

    def test_check_path_not_a_directory(self):
        """path must be an existing directory."""
        with self.assertRaises(MsprobeException) as ctx:
            LoadConfig({"load": {"path": "/nonexistent/path/abc", "modules": ["Module.m1.A.forward.0"]}})
        self.assertIn("not a directory", str(ctx.exception))

    def test_check_modules_not_list(self):
        """modules must be a list."""
        with self.assertRaises(MsprobeException) as ctx:
            LoadConfig({"load": {"path": self.tmpdir, "modules": "blocks.0.attn"}})
        self.assertIn("must be a", str(ctx.exception))

    def test_check_modules_element_not_str(self):
        """modules elements must be str."""
        with self.assertRaises(MsprobeException) as ctx:
            LoadConfig({"load": {"path": self.tmpdir, "modules": [123]}})
        self.assertIn("must be a list", str(ctx.exception))

    def test_check_dump_after_load_not_bool(self):
        """dump_after_load must be bool."""
        with self.assertRaises(MsprobeException) as ctx:
            LoadConfig({"load": {"path": self.tmpdir, "modules": ["Module.m1.A.forward.0"], "dump_after_load": "yes"}})
        self.assertIn("dump_after_load must be bool", str(ctx.exception))

    # ---- integration with CommonConfig ----

    def test_common_config_load_absent(self):
        """CommonConfig without load -> load_config.is_enabled == False."""
        cc = CommonConfig({"task": "tensor", "dump_path": self.tmpdir, "level": "L0"})
        self.assertFalse(cc.load_config.is_enabled)

    def test_common_config_with_load(self):
        """CommonConfig propagates load section."""
        cc = CommonConfig({
            "task": "tensor", "dump_path": self.tmpdir, "level": "L0",
            "load": {"path": self.tmpdir, "modules": ["Module.blocks.0.attn.A.forward.0"], "dump_after_load": True},
        })
        self.assertTrue(cc.load_config.is_enabled)
        self.assertEqual(cc.load_config.modules, ["Module.blocks.0.attn.A.forward.0"])
        self.assertTrue(cc.load_config.dump_after_load)


if __name__ == "__main__":
    unittest.main()
